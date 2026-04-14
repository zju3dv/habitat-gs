"""Tool abstraction layer — `Tool` protocol + `ToolRegistry` singleton.

The goal is to consolidate tool definitions into a single declarative
model so that adding a new tool only requires writing one Tool subclass
and one register call.

Architecture:

  ToolMetadata    — declarative tool definition (name, schema,
                    nav_mode/task_type gates, permission level)
  RoundState      — mutable per-round tracking state that the agent
                    accumulates across multiple tool calls (captured
                    images, last collision, action chain)
  ToolContext     — runtime context handed to every Tool.execute()
                    call: bridge client, session/loop ids, output dir,
                    nav_mode, task_type, plus the agent's RoundState
  ToolResult      — standardized return value (ok / body / captured
                    images / latency / error)
  Tool (Protocol) — what every concrete tool must implement
  ToolRegistry    — global registry; tools register at module import
                    time, callers fetch schemas and dispatch by name

All per-round mutable state lives on `ctx.round_state`. Tool subclasses
are stateless beyond their metadata, which is what makes a single global
Registry safe to share across the nav_agent, chat_agent, and mcp_server.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ToolCategory(Enum):
    """High-level grouping for telemetry and UI."""

    NAVIGATION = "navigation"
    PERCEPTION = "perception"
    MAPPING = "mapping"
    STATUS = "status"
    SESSION = "session"


class PermissionLevel(Enum):
    """How dangerous a tool is, used by approval gates."""

    READ_ONLY = "read_only"        # does not change world state
    MUTATING = "mutating"          # changes state but is recoverable
    DESTRUCTIVE = "destructive"    # cannot be undone


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class ToolMetadata:
    """Declarative tool definition — the single source of truth.

    Every Tool subclass has exactly one ToolMetadata instance as a class
    attribute. The Registry uses these fields to:

    1. generate OpenAI function-calling schemas (`build_openai_schemas`)
    2. filter tools by `nav_mode` / `task_type` (`available_for`)
    3. expose the tool through MCP
    """

    # Identity
    name: str                                     # e.g. "forward"
    category: ToolCategory
    description: str                              # LLM-facing help text

    # LLM interface — JSON Schema for OpenAI function calling
    parameters_schema: Dict[str, Any] = field(default_factory=dict)

    # Availability constraints. Empty `allowed_task_types` means "any
    # task type"; we use a sentinel rather than None so that the
    # filter logic stays a simple set-membership check.
    allowed_nav_modes: Set[str] = field(
        default_factory=lambda: {"navmesh", "mapless"}
    )
    allowed_task_types: Optional[Set[str]] = None  # None → any task type

    # Backward-compat aliases. When a Tool was renamed during the
    # Phase 2 migration (e.g. the legacy `hab_init` MCP tool became
    # `hab_init_scene` because the Tool class is `InitSceneTool`),
    # the Tool declares its old names here so the dynamic MCP
    # registration can expose both the canonical `hab_<new_name>`
    # and the legacy `hab_<old_name>` entry points. External MCP
    # clients that hardcoded the old names continue to work.
    legacy_names: Set[str] = field(default_factory=set)

    # Whether this tool should be exposed via the dynamic MCP server
    # registration. Default True. Set to False for tools that need
    # state only available inside a NavAgent subprocess (loop_id +
    # state_version_ref) which top-level MCP callers cannot provide.
    # `update_nav_status` is the canonical opt-out: it would always
    # fail with a cryptic 'loop_id must be non-empty' bridge error
    # if exposed to MCP, polluting the tool inventory. Orthogonal
    # to allowed_nav_modes / allowed_task_types — those control
    # agent execution context, this controls external API exposure.
    mcp_visible: bool = True

    # Runtime hints (telemetry / planning)
    typical_latency_ms: int = 0
    typical_token_cost: int = 0
    permission: PermissionLevel = PermissionLevel.READ_ONLY
    requires_session: bool = True

    # Provenance metadata (version tracking).
    version: int = 1
    author: str = "human"
    parent_version: Optional[int] = None
    success_rate_history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Sanity-check the most easily-broken fields. Catching these at
        # registration time is much more useful than catching them
        # at the first dispatch.
        if not self.name:
            raise ValueError("ToolMetadata.name must be a non-empty string")
        if not isinstance(self.category, ToolCategory):
            raise TypeError(
                f"ToolMetadata.category must be ToolCategory, "
                f"got {type(self.category).__name__}"
            )
        if not self.description:
            raise ValueError(
                f"ToolMetadata.description for {self.name!r} must not be empty"
            )
        if not isinstance(self.allowed_nav_modes, set):
            raise TypeError(
                f"ToolMetadata.allowed_nav_modes must be a set, "
                f"got {type(self.allowed_nav_modes).__name__}"
            )
        if not self.allowed_nav_modes:
            raise ValueError(
                f"ToolMetadata.allowed_nav_modes for {self.name!r} cannot be empty"
            )
        if self.allowed_task_types is not None and not isinstance(
            self.allowed_task_types, set
        ):
            raise TypeError(
                f"ToolMetadata.allowed_task_types must be a set or None, "
                f"got {type(self.allowed_task_types).__name__}"
            )
        if not isinstance(self.legacy_names, set):
            raise TypeError(
                f"ToolMetadata.legacy_names must be a set, "
                f"got {type(self.legacy_names).__name__}"
            )


# ---------------------------------------------------------------------------
# RoundState — per-round mutable tracking
# ---------------------------------------------------------------------------


@dataclass
class RoundState:
    """Mutable per-round tracking state.

    The legacy `ToolExecutor` stored these as instance attributes, which
    coupled tool execution to a long-lived `ToolExecutor` instance. In
    Phase 2 the same fields live here on `ctx.round_state`, so Tool
    subclasses themselves can stay stateless and the agent owns the
    lifecycle (clearing per-round state, reading captured images, etc.).

    Lifecycle (managed by the agent's turn loop, NOT by individual
    tools, except where noted):

      - `captured_images`: appended by perception/movement tools.
        Agent reads after each dispatch and clears between tool calls
        within a round (current legacy behaviour: cleared at the start
        of each `ToolExecutor.execute`).
      - `last_visual_path`: set by look/panorama tools. Auto-injected
        by `update_nav_status` into mapless patches.
      - `last_collided` / `last_movement_action`: set by movement
        tools (forward/turn/navigate). Auto-injected by
        `update_nav_status` into action history entries.
      - `round_actions`: append-only list of action descriptions
        accumulated during a round. `update_nav_status` reads it
        and clears it as part of its execute() (legacy parity).
    """

    captured_images: List[str] = field(default_factory=list)
    last_visual_path: Optional[str] = None
    last_collided: bool = False
    last_movement_action: Optional[str] = None
    # Human-readable per-step action descriptions
    # (e.g. "forward(0.5m)!", "turn_left(45°)"). Joined with " → "
    # by update_nav_status to produce the action chain string that
    # gets auto-injected into action_history_append entries.
    round_actions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ToolContext — what every tool dispatch sees
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Runtime context passed to every tool execution.

    The agent constructs one of these once and reuses it for the entire
    nav loop / chat session. Per-round mutable state is on
    `round_state`. The Phase 3 memory bundle will plug in via
    `memory_bundle`; until then it stays None.

    Note that `state_version_ref` is a list rather than an int because
    `update_nav_status` needs to mutate it from inside the tool (the
    bridge's response carries the new state_version, and the agent's
    next round must see it). Mirroring the legacy `state_version_ref`
    list-of-one pattern keeps the migration mechanical.
    """

    bridge: Any                              # BridgeClient at runtime
    session_id: str
    loop_id: str
    output_dir: str
    nav_mode: str
    task_type: str
    workspace_host: str = ""
    is_gaussian: bool = False                # mutable: InitSceneTool sets this
    state_version_ref: List[int] = field(default_factory=lambda: [1])
    round_state: RoundState = field(default_factory=RoundState)
    memory_bundle: Optional[Any] = None      # Phase 3: set to MemoryBundle by nav_agent/chat_agent


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Standardized tool output.

    Tools should return `ToolResult.ok=False` with an `error` string for
    expected failures (bridge errors, validation failures, missing
    sessions). Unexpected exceptions are caught by `ToolRegistry.dispatch`
    and converted to the same shape, so callers never need to wrap
    dispatch in try/except.
    """

    ok: bool
    body: Dict[str, Any] = field(default_factory=dict)
    captured_images: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Tool(Protocol):
    """A capability the agent can invoke.

    Subclass-via-Protocol: any class with a `metadata` class attribute
    and an `execute(args, ctx)` method satisfies this. We use
    `runtime_checkable` so `isinstance(x, Tool)` works in tests.
    """

    metadata: ToolMetadata

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Perform the action and return a ToolResult."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Global tool registry — the single place where tools are enumerated.

    Class-level dict because Tool subclasses are stateless and we want
    one canonical registration table that nav_agent / chat_agent /
    mcp_server all read from. Tools register themselves at module
    import time:

        class ForwardTool:
            metadata = ToolMetadata(name="forward", ...)
            def execute(self, args, ctx): ...

        ToolRegistry.register(ForwardTool())

    Phase 2 PR 2/3 introduce the actual Tool subclasses; PR 4 switches
    nav_agent / chat_agent / mcp_server to use this registry as the
    only dispatch path.
    """

    _tools: Dict[str, Tool] = {}

    # ── registration ──────────────────────────────────────────────

    @classmethod
    def register(cls, tool: Tool) -> None:
        """Register a tool. Re-registration with the same name overwrites
        the previous entry."""
        if not hasattr(tool, "metadata"):
            raise TypeError(
                f"register() expects an object with a `metadata` attribute, "
                f"got {type(tool).__name__}"
            )
        if not hasattr(tool, "execute"):
            raise TypeError(
                f"register() expects an object with an `execute` method, "
                f"got {type(tool).__name__}"
            )
        cls._tools[tool.metadata.name] = tool

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a tool. Mainly used by tests to keep the global
        registry isolated between cases."""
        cls._tools.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        """Wipe the registry. Tests use this in fixtures."""
        cls._tools.clear()

    # ── lookup ────────────────────────────────────────────────────

    @classmethod
    def get(cls, name: str) -> Optional[Tool]:
        return cls._tools.get(name)

    @classmethod
    def list_all(cls) -> List[Tool]:
        return list(cls._tools.values())

    @classmethod
    def available_for(cls, nav_mode: str, task_type: str) -> List[Tool]:
        """Return tools that satisfy both gates.

        - `nav_mode` filter: drops tools whose metadata `allowed_nav_modes`
          set does not contain the requested mode (e.g. `navigate` is
          {"navmesh"} only and is filtered out in mapless mode).
        - `task_type` filter: drops tools whose metadata `allowed_task_types`
          is non-None and does not contain the requested task type
          (e.g. session tools are {"chat"} only and are filtered out
          for nav_agent loops).
        """
        out: List[Tool] = []
        for tool in cls._tools.values():
            if nav_mode not in tool.metadata.allowed_nav_modes:
                continue
            if (
                tool.metadata.allowed_task_types is not None
                and task_type not in tool.metadata.allowed_task_types
            ):
                continue
            out.append(tool)
        return out

    # ── schema generation ─────────────────────────────────────────

    @classmethod
    def build_openai_schemas(
        cls, nav_mode: str, task_type: str
    ) -> List[Dict[str, Any]]:
        """Generate OpenAI function-calling schemas from tool metadata.

        Replaces the legacy hand-written `build_tool_schemas`. The same
        nav_mode / task_type filtering as `available_for` applies.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.metadata.name,
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.parameters_schema or {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
            for tool in cls.available_for(nav_mode, task_type)
        ]

    # ── dispatch ──────────────────────────────────────────────────

    @classmethod
    def dispatch(
        cls,
        name: str,
        args: Dict[str, Any],
        ctx: ToolContext,
    ) -> ToolResult:
        """Look up `name` and execute it with `args` against `ctx`.

        Always returns a ToolResult — never raises. Unknown tools and
        unexpected exceptions are converted to ``ok=False`` with an
        ``error`` string so callers don't need to wrap dispatch in
        try/except. The latency_ms field is populated even on failure
        so telemetry stays consistent.
        """
        # Start the latency clock as early as possible so every
        # dispatch outcome — success, exception, unknown-tool,
        # gate-rejection — records a non-zero latency.
        start = time.perf_counter()

        def _elapsed_ms() -> float:
            return (time.perf_counter() - start) * 1000.0

        tool = cls.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                body={},
                error=f"Unknown tool: {name}",
                latency_ms=_elapsed_ms(),
            )

        # Defense-in-depth availability gates. `available_for()` filters
        # what the LLM sees in its schemas, but a hallucinated tool name
        # or a direct call from test code could otherwise bypass that
        # filter. Reject at dispatch time as well, mirroring the exact
        # rules in `available_for`.
        if ctx.nav_mode not in tool.metadata.allowed_nav_modes:
            allowed = ",".join(sorted(tool.metadata.allowed_nav_modes))
            return ToolResult(
                ok=False,
                body={},
                error=(
                    f"Tool {name!r} not allowed in nav_mode={ctx.nav_mode!r} "
                    f"(allowed_nav_modes={{{allowed}}})"
                ),
                latency_ms=_elapsed_ms(),
            )
        if (
            tool.metadata.allowed_task_types is not None
            and ctx.task_type not in tool.metadata.allowed_task_types
        ):
            allowed = ",".join(sorted(tool.metadata.allowed_task_types))
            return ToolResult(
                ok=False,
                body={},
                error=(
                    f"Tool {name!r} not allowed for task_type={ctx.task_type!r} "
                    f"(allowed_task_types={{{allowed}}})"
                ),
                latency_ms=_elapsed_ms(),
            )

        try:
            result = tool.execute(args, ctx)
        except Exception as exc:
            return ToolResult(
                ok=False,
                body={},
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=_elapsed_ms(),
            )

        # Defence in depth: a Tool that accidentally returns a
        # legacy-style dict (or anything else) must not crash the
        # caller with AttributeError when we later touch
        # `result.latency_ms`. Convert non-ToolResult returns into an
        # ok=False ToolResult with a descriptive error so the
        # "never raises" contract holds even for buggy tools.
        if not isinstance(result, ToolResult):
            return ToolResult(
                ok=False,
                body={},
                error=(
                    f"Tool {name!r} returned {type(result).__name__}, "
                    f"expected ToolResult"
                ),
                latency_ms=_elapsed_ms(),
            )

        # Tools may have set their own latency_ms. Only fill it in if
        # they didn't, so handcrafted measurements (e.g. excluding
        # post-processing) win over our outer measurement.
        if result.latency_ms == 0.0:
            result.latency_ms = _elapsed_ms()
        return result


__all__ = [
    "ToolCategory",
    "PermissionLevel",
    "ToolMetadata",
    "RoundState",
    "ToolContext",
    "ToolResult",
    "Tool",
    "ToolRegistry",
]
