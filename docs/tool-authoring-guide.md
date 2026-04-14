# Tool Authoring Guide

How to add a new tool to habitat-gs so that the nav_agent LLM, the
chat_agent LLM, and external MCP clients (Codex / Claude Code /
OpenClaw) can all call it. After Phase 2 of the architecture refactor,
adding a tool means writing **one Python class** and **one
registration call** — no more copy-pasting into
`legacy_schemas.py` / `legacy_executor.py` / `mcp_server.py`.

## TL;DR

1. Pick a category (navigation / perception / mapping / status / session).
2. Create a `class YourTool` with a `metadata` class attribute and an
   `execute(self, args, ctx) -> ToolResult` method in
   `tools/habitat_agent/tools/<category>.py`.
3. Call `ToolRegistry.register(YourTool())` at module import time.
4. Write regression tests in `tests/test_tools_<category>.py`.

That's it. Your tool is now visible to:
- `nav_agent` via `ToolRegistry.build_openai_schemas(nav_mode, task_type)`
- `chat_agent` via the same Registry query
- `mcp_server` via the dynamic wrapper factory that iterates
  `ToolRegistry.list_all()` at startup

## Worked example: ForwardTool

The simplest meaningful tool is `ForwardTool` in
`tools/habitat_agent/tools/navigation.py`:

```python
from ._common import collect_images, visual_payload
from .base import (
    PermissionLevel, Tool, ToolCategory, ToolContext,
    ToolMetadata, ToolRegistry, ToolResult,
)


class ForwardTool:
    """Move the agent forward by distance_m metres."""

    metadata = ToolMetadata(
        name="forward",
        category=ToolCategory.NAVIGATION,
        description=(
            "Move agent forward by `distance_m` metres. The bridge "
            "auto-decomposes the request into 0.25m atomic steps and "
            "stops early on collision."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "distance_m": {
                    "type": "number",
                    "description": (
                        "Distance in metres. Must be a positive multiple "
                        "of the 0.25m atomic step."
                    ),
                    "default": 0.5,
                },
            },
        },
        permission=PermissionLevel.MUTATING,
    )

    def execute(self, args, ctx):
        dist = args.get("distance_m", 0.5)
        payload = {
            "action": "move_forward",
            "distance": dist,
            "include_metrics": True,
        }
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("step_and_capture", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))

        collect_images(result, ctx)
        ctx.round_state.last_collided = bool(result.get("collided", False))
        ctx.round_state.last_movement_action = "move_forward"
        col_tag = "!" if ctx.round_state.last_collided else ""
        ctx.round_state.round_actions.append(f"forward({dist}m){col_tag}")

        return ToolResult(
            ok=True,
            body=result,
            captured_images=list(ctx.round_state.captured_images),
        )


ToolRegistry.register(ForwardTool())
```

## The four concepts

### `ToolMetadata`

A declarative blob describing what the tool is. Fields:

| Field | Purpose |
|---|---|
| `name` | LLM-facing name. Kebab-case (`update_nav_status`) or snake (`forward`). |
| `category` | `ToolCategory` enum — purely for grouping / telemetry. |
| `description` | Human-readable text the LLM sees in its tool list. Keep it under 200 chars. |
| `parameters_schema` | JSON Schema for the tool's arguments. This is what becomes the OpenAI / MCP function-calling schema. |
| `allowed_nav_modes` | Set of strings. Default `{"navmesh", "mapless"}`. Set to `{"navmesh"}` for tools that only work with a navmesh (e.g. `navigate`, `find_path`). |
| `allowed_task_types` | Optional set of strings. `None` = visible to any task. `{"chat"}` for chat-only session tools. |
| `permission` | `READ_ONLY` / `MUTATING` / `DESTRUCTIVE`. Used by future approval gates. |
| `requires_session` | Default `True`. Set to `False` only for `init_scene` (which creates the session). |
| `legacy_names` | Set of backward-compat aliases for MCP registration. Use when you rename a pre-Phase-2 tool. |
| `mcp_visible` | Default `True`. Set `False` for nav_loop-internal tools (like `update_nav_status`) that need context MCP callers cannot provide. |
| `version` / `author` / `parent_version` | Provenance tracking. Defaults are fine for hand-written tools. |

### `ToolContext`

Runtime context passed to every `execute()` call. You read from it and
sometimes write to it. Fields:

- `bridge` — the `BridgeClient` instance. Call bridge RPCs via `ctx.bridge.call("action", payload)`.
- `session_id`, `loop_id`, `output_dir`, `workspace_host`, `nav_mode`, `task_type` — the static context of the current nav loop / chat session.
- `is_gaussian` — mutable. `InitSceneTool` sets this.
- `state_version_ref` — a `List[int]` (list-of-one). `UpdateNavStatusTool` mutates `ctx.state_version_ref[0]` from the bridge response. Pattern preserved from legacy.
- `round_state` — a mutable `RoundState` dataclass with per-round tracking fields. See below.

### `RoundState`

Per-round mutable state that accumulates across multiple tool calls in
the same round. Fields:

- `captured_images: List[str]` — append paths of images the tool captured. The agent (`nav_agent` or `chat_agent`) reads and clears this between tool calls.
- `last_visual_path: Optional[str]` — the most recent color image path. `UpdateNavStatusTool` auto-injects this into mapless patches.
- `last_collided: bool` — set by movement tools from the bridge's `collided` field. `UpdateNavStatusTool` uses this to flag action_history entries.
- `last_movement_action: Optional[str]` — the last `move_forward` / `turn_left` / `turn_right` string. For telemetry.
- `round_actions: List[str]` — append a human-readable tag for every tool call you make (e.g. `"forward(0.5m)!"` with `!` for collision). `UpdateNavStatusTool` joins these with `" → "` to auto-inject the action chain into action_history entries, then clears the list.

### `ToolResult`

What `execute()` returns. Fields:

- `ok: bool` — success flag.
- `body: Dict[str, Any]` — the payload the LLM sees (usually the bridge response dict, optionally shaped).
- `captured_images: List[str]` — copy of `ctx.round_state.captured_images` at the time of return (convenience for callers).
- `latency_ms: float` — optional; Registry fills it in if you don't.
- `error: Optional[str]` — error message when `ok=False`.

Return `ToolResult(ok=False, body={}, error="clear message")` for
expected failures (invalid args, session precondition not met).
**Don't raise exceptions** — Registry will catch them and convert to a
slightly less clear error. Predict your failure modes and return them.

## Nav_mode and task_type filtering

Tools declare **which contexts they're valid in** through metadata:

```python
# navmesh-only (bridge requires a navmesh for this action)
allowed_nav_modes={"navmesh"}

# chat-only (nav_agent's LLM should not see this)
allowed_task_types={"chat"}
```

`ToolRegistry.available_for(nav_mode, task_type)` filters on both
gates. `ToolRegistry.dispatch(name, args, ctx)` re-checks at dispatch
time as a defence-in-depth (so a hallucinated LLM call can't bypass
the filter).

## Testing

Every Tool subclass needs at least:

1. **Happy path test** — feed normal args, verify the tool calls the
   right bridge action with the right payload, and returns `ok=True`.
2. **Bridge error test** — make the `FakeBridge` raise, verify the
   tool returns `ok=False` with the error in `.error`.
3. **(if applicable) Precondition test** — missing required args,
   missing session, invalid enum value, etc.

Use the `FakeBridge` harness from `tests/test_tools_nav_side.py` as a
template:

```python
class FakeBridge:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}
        self.session_id = "s1"

    def call(self, action, payload=None):
        self.calls.append((action, dict(payload or {})))
        return self.responses.get(action, {})
```

Put your tests in `tests/test_tools_<category>.py` alongside the
existing ones for the same category.

## MCP exposure

After you register a tool, the MCP server (`mcp_server.py`) will
**automatically** generate a `hab_<tool.metadata.name>` function,
register it with FastMCP, and expose it over stdio / SSE / HTTP.

The signature of the generated MCP wrapper comes from your
`parameters_schema` via `_extract_params` in `mcp_server.py`:

- JSON Schema `type` → Python type annotation (`"number"` → `float`, etc.)
- Fields in `required` become required parameters
- Fields with `default` become optional with that default

If you need the wrapper to reshape the bridge response (e.g.
legacy-compatible `images` field at the top level instead of nested
under `visuals`), add an entry to `_MCP_RESPONSE_SHAPERS` in
`mcp_server.py`. Most tools don't need shaping — only
`init_scene` / `look` / `turn` / `panorama` / `forward` / `navigate` /
`topdown` have custom shapers to preserve pre-Phase-2 response layouts.

## When NOT to use the Registry

Some operations that look like "tools" shouldn't be Tool subclasses:

- **MCP resources** (e.g. `list_artifacts`) — these are resources, not
  tools. Add them as `@mcp.resource(...)` in `mcp_server.py` directly.
- **One-off helpers used inside another tool** — just inline them or
  put them in `_common.py`.
- **Bridge RPCs the LLM should never call directly** (e.g. internal
  debug endpoints) — leave them as plain bridge calls.

## Common pitfalls

1. **Forgetting to call `ToolRegistry.register(YourTool())`** — the
   tool exists as a class but is invisible to everything. Always
   register at module import time.
2. **Forgetting to add your module to `tools/__init__.py`** — if the
   module is never imported, the `register` line never runs.
3. **Mutating `ctx.round_state.captured_images` in-place** — the agent
   copies this list out between calls. You can `append` to it safely.
   Don't assign a new list to `ctx.round_state.captured_images` from
   inside a tool unless you mean to clear.
4. **Returning a dict instead of a ToolResult** — `ToolRegistry.dispatch`
   will catch it and return an `ok=False` wrapper, but your tool's
   latency / body / captured_images won't flow correctly. Always return
   `ToolResult(...)`.
5. **Raising exceptions instead of returning `ok=False`** — same
   problem, the dispatch layer catches it but with less clear error
   messages. Be explicit.
6. **Mistyping the bridge action name** — if you pass
   `ctx.bridge.call("move_foward", ...)` (typo) the bridge returns an
   `HabitatAdapterError`. The Tool will return `ok=False` with that
   error, but you'll spend time debugging. Cross-reference
   `src_python/habitat_sim/habitat_adapter_internal/mixins_api.py` for
   the canonical action names.

## Reference files

- `tools/habitat_agent/tools/base.py` — all the types and the Registry
- `tools/habitat_agent/tools/navigation.py` — 5 tools, good examples
- `tools/habitat_agent/tools/status.py` — `UpdateNavStatusTool`, the
  most intricate auto-injection logic
- `tools/habitat_agent/tools/session.py` — `NavLoopStartTool`, the
  most validation-heavy tool (contains the codex P1/P2 fix)
- `tests/test_tools_nav_side.py` — testing template
- `tools/habitat_agent/interfaces/mcp_server.py` — dynamic MCP
  wrapper factory and response shapers
