"""Phase 2 PR 1 — unit tests for the Tool abstraction layer.

Covers `ToolMetadata`, `RoundState`, `ToolContext`, `ToolResult`,
the `Tool` Protocol, and `ToolRegistry`. The Registry is a global
singleton, so every test that touches it must restore state via the
`fresh_registry` fixture.
"""

from __future__ import annotations

import os
import sys

import pytest

# Make the tools/ directory importable so `import habitat_agent.tools.base`
# works without an installed package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from habitat_agent.tools.base import (  # noqa: E402
    PermissionLevel,
    RoundState,
    Tool,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_registry():
    """Save / restore the global registry around each test.

    The registry is a class-level dict, so without this fixture tests
    would leak state into each other (and into the rest of the suite,
    once Phase 2 PR 2 starts registering tools at import time).
    """
    saved = dict(ToolRegistry._tools)
    ToolRegistry.clear()
    try:
        yield ToolRegistry
    finally:
        ToolRegistry._tools.clear()
        ToolRegistry._tools.update(saved)


def _make_tool(
    name: str,
    *,
    category: ToolCategory = ToolCategory.NAVIGATION,
    nav_modes=None,
    task_types=None,
    execute_impl=None,
):
    """Helper to construct a minimal Tool subclass for tests."""

    class _Tool:
        metadata = ToolMetadata(
            name=name,
            category=category,
            description=f"Test tool {name}",
            parameters_schema={
                "type": "object",
                "properties": {"x": {"type": "number"}},
            },
            allowed_nav_modes=nav_modes if nav_modes is not None else {"navmesh", "mapless"},
            allowed_task_types=task_types,
        )

        def execute(self, args, ctx):
            if execute_impl is not None:
                return execute_impl(args, ctx)
            return ToolResult(ok=True, body={"name": name, "args": args})

    return _Tool()


def _make_ctx(**overrides):
    defaults = dict(
        bridge=None,
        session_id="s1",
        loop_id="l1",
        output_dir="/tmp/out",
        nav_mode="navmesh",
        task_type="pointnav",
    )
    defaults.update(overrides)
    return ToolContext(**defaults)


# ---------------------------------------------------------------------------
# 1. ToolMetadata validation
# ---------------------------------------------------------------------------


def test_metadata_mcp_visible_defaults_to_true():
    """Codex P1 regression lock — `mcp_visible` field default.

    By default every Tool is visible to the dynamic MCP registration.
    Tools that need a context only available inside a nav_loop
    subprocess (e.g. UpdateNavStatusTool, which needs ctx.loop_id and
    ctx.state_version_ref) opt out by setting mcp_visible=False."""
    m = ToolMetadata(
        name="ok",
        category=ToolCategory.NAVIGATION,
        description="d",
    )
    assert m.mcp_visible is True


def test_metadata_mcp_visible_can_be_set_false():
    """Tools can opt out of MCP exposure explicitly."""
    m = ToolMetadata(
        name="internal",
        category=ToolCategory.STATUS,
        description="d",
        mcp_visible=False,
    )
    assert m.mcp_visible is False


def test_metadata_legacy_names_defaults_to_empty_set():
    """Codex P1 regression lock — ToolMetadata.legacy_names.

    The `legacy_names` field is how a Tool declares the old MCP
    alias(es) it used to have, so the dynamic mcp_server registration
    can expose both the canonical `hab_<new_name>` AND the legacy
    `hab_<old_name>` tools without hard-breaking external clients.
    Default must be an empty set (no aliases) so existing tools
    stay unaffected."""
    m = ToolMetadata(
        name="ok",
        category=ToolCategory.NAVIGATION,
        description="d",
    )
    assert m.legacy_names == set()

    m2 = ToolMetadata(
        name="init_scene",
        category=ToolCategory.SESSION,
        description="d",
        legacy_names={"init"},
    )
    assert m2.legacy_names == {"init"}


def test_metadata_legacy_names_type_check():
    """legacy_names must be a set — a list would work by accident
    but silently allow duplicates and order-dependent behaviour."""
    with pytest.raises(TypeError, match="legacy_names"):
        ToolMetadata(
            name="x",
            category=ToolCategory.SESSION,
            description="d",
            legacy_names=["init"],  # type: ignore[arg-type]
        )


def test_metadata_required_fields():
    """ToolMetadata rejects empty name / wrong category type / empty
    description / wrong allowed_nav_modes type / empty allowed_nav_modes
    set / wrong allowed_task_types type."""

    # Empty name
    with pytest.raises(ValueError, match="name"):
        ToolMetadata(
            name="",
            category=ToolCategory.NAVIGATION,
            description="desc",
        )

    # Wrong category type
    with pytest.raises(TypeError, match="category"):
        ToolMetadata(
            name="t",
            category="navigation",  # type: ignore[arg-type]
            description="desc",
        )

    # Empty description
    with pytest.raises(ValueError, match="description"):
        ToolMetadata(
            name="t",
            category=ToolCategory.NAVIGATION,
            description="",
        )

    # Wrong allowed_nav_modes type
    with pytest.raises(TypeError, match="allowed_nav_modes"):
        ToolMetadata(
            name="t",
            category=ToolCategory.NAVIGATION,
            description="d",
            allowed_nav_modes=["navmesh"],  # type: ignore[arg-type]
        )

    # Empty allowed_nav_modes set
    with pytest.raises(ValueError, match="allowed_nav_modes"):
        ToolMetadata(
            name="t",
            category=ToolCategory.NAVIGATION,
            description="d",
            allowed_nav_modes=set(),
        )

    # Wrong allowed_task_types type
    with pytest.raises(TypeError, match="allowed_task_types"):
        ToolMetadata(
            name="t",
            category=ToolCategory.NAVIGATION,
            description="d",
            allowed_task_types=["chat"],  # type: ignore[arg-type]
        )

    # Defaults are applied correctly
    m = ToolMetadata(
        name="ok",
        category=ToolCategory.NAVIGATION,
        description="d",
    )
    assert m.allowed_nav_modes == {"navmesh", "mapless"}
    assert m.allowed_task_types is None
    assert m.permission == PermissionLevel.READ_ONLY
    assert m.version == 1


# ---------------------------------------------------------------------------
# 2. Registry register / get / list_all
# ---------------------------------------------------------------------------


def test_registry_register_and_get(fresh_registry):
    tool_a = _make_tool("alpha")
    tool_b = _make_tool("beta")
    fresh_registry.register(tool_a)
    fresh_registry.register(tool_b)

    assert fresh_registry.get("alpha") is tool_a
    assert fresh_registry.get("beta") is tool_b
    assert fresh_registry.get("gamma") is None
    assert len(fresh_registry.list_all()) == 2

    # register() rejects garbage objects (no metadata / no execute)
    with pytest.raises(TypeError, match="metadata"):
        fresh_registry.register(object())

    class _NoExecute:
        metadata = ToolMetadata(
            name="no_exec",
            category=ToolCategory.NAVIGATION,
            description="d",
        )
    with pytest.raises(TypeError, match="execute"):
        fresh_registry.register(_NoExecute())

    # Re-registration overwrites the previous entry
    tool_a_v2 = _make_tool("alpha")
    fresh_registry.register(tool_a_v2)
    assert fresh_registry.get("alpha") is tool_a_v2
    assert len(fresh_registry.list_all()) == 2


# ---------------------------------------------------------------------------
# 3. nav_mode filtering
# ---------------------------------------------------------------------------


def test_registry_filter_by_nav_mode(fresh_registry):
    """available_for() drops tools whose allowed_nav_modes does not
    contain the requested mode. Mirrors the legacy
    `if nav_mode != "mapless"` gate for navigate / find_path /
    sample_point / topdown."""
    universal = _make_tool("look")  # both modes
    navmesh_only_a = _make_tool("navigate", nav_modes={"navmesh"})
    navmesh_only_b = _make_tool("find_path", nav_modes={"navmesh"})
    mapless_only = _make_tool("polar_signal", nav_modes={"mapless"})
    fresh_registry.register(universal)
    fresh_registry.register(navmesh_only_a)
    fresh_registry.register(navmesh_only_b)
    fresh_registry.register(mapless_only)

    navmesh_tools = {t.metadata.name for t in fresh_registry.available_for("navmesh", "pointnav")}
    assert navmesh_tools == {"look", "navigate", "find_path"}

    mapless_tools = {t.metadata.name for t in fresh_registry.available_for("mapless", "pointnav")}
    assert mapless_tools == {"look", "polar_signal"}
    assert "navigate" not in mapless_tools
    assert "find_path" not in mapless_tools


# ---------------------------------------------------------------------------
# 4. task_type filtering
# ---------------------------------------------------------------------------


def test_registry_filter_by_task_type(fresh_registry):
    """Tools with `allowed_task_types == {"chat"}` are visible to
    chat_agent but invisible to nav_agent loops. None means "any"."""
    universal = _make_tool("forward")  # task_types=None → any
    chat_only_a = _make_tool(
        "init_scene",
        category=ToolCategory.SESSION,
        task_types={"chat"},
    )
    chat_only_b = _make_tool(
        "nav_loop_start",
        category=ToolCategory.SESSION,
        task_types={"chat"},
    )
    fresh_registry.register(universal)
    fresh_registry.register(chat_only_a)
    fresh_registry.register(chat_only_b)

    pointnav_tools = {
        t.metadata.name for t in fresh_registry.available_for("navmesh", "pointnav")
    }
    assert pointnav_tools == {"forward"}
    assert "init_scene" not in pointnav_tools

    chat_tools = {
        t.metadata.name for t in fresh_registry.available_for("navmesh", "chat")
    }
    assert chat_tools == {"forward", "init_scene", "nav_loop_start"}


# ---------------------------------------------------------------------------
# 5. OpenAI schema generation
# ---------------------------------------------------------------------------


def test_registry_build_openai_schemas_format(fresh_registry):
    """Generated schemas conform to OpenAI function-calling spec."""
    fresh_registry.register(_make_tool("forward"))
    fresh_registry.register(_make_tool("turn"))

    schemas = fresh_registry.build_openai_schemas("navmesh", "pointnav")
    assert len(schemas) == 2
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert "name" in fn and isinstance(fn["name"], str)
        assert "description" in fn and isinstance(fn["description"], str)
        assert "parameters" in fn and isinstance(fn["parameters"], dict)
        assert fn["parameters"]["type"] == "object"

    # Tool with empty parameters_schema → still gets a valid object schema
    class _NoParams:
        metadata = ToolMetadata(
            name="export_video",
            category=ToolCategory.STATUS,
            description="Export video.",
        )
        def execute(self, args, ctx):
            return ToolResult(ok=True)
    fresh_registry.register(_NoParams())
    schemas2 = fresh_registry.build_openai_schemas("navmesh", "pointnav")
    export = next(s for s in schemas2 if s["function"]["name"] == "export_video")
    assert export["function"]["parameters"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# 6. Dispatch — unknown tool
# ---------------------------------------------------------------------------


def test_registry_dispatch_unknown_tool_returns_error(fresh_registry):
    """dispatch() of an unknown name returns ok=False with error,
    never raises."""
    ctx = _make_ctx()
    result = fresh_registry.dispatch("nonexistent", {}, ctx)
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert "Unknown tool" in (result.error or "")
    assert result.body == {}


# ---------------------------------------------------------------------------
# 7. Dispatch — exception wrapping + latency
# ---------------------------------------------------------------------------


def test_registry_dispatch_wraps_exceptions_and_measures_latency(fresh_registry):
    """If a tool's execute() raises, dispatch() converts to
    ToolResult.ok=False with error filled. Latency is always measured."""

    def _boom(args, ctx):
        raise RuntimeError("bridge exploded")

    fresh_registry.register(_make_tool("boom", execute_impl=_boom))
    ctx = _make_ctx()
    result = fresh_registry.dispatch("boom", {}, ctx)
    assert result.ok is False
    assert "RuntimeError" in (result.error or "")
    assert "bridge exploded" in (result.error or "")
    assert result.latency_ms >= 0.0  # always populated

    # Successful dispatch — latency_ms is populated when tool didn't set it
    def _ok(args, ctx):
        return ToolResult(ok=True, body={"got": args})

    fresh_registry.register(_make_tool("ok_tool", execute_impl=_ok))
    result2 = fresh_registry.dispatch("ok_tool", {"x": 1}, ctx)
    assert result2.ok is True
    assert result2.body == {"got": {"x": 1}}
    assert result2.latency_ms > 0.0  # outer timer ran

    # Tool that sets its own latency_ms wins (outer timer doesn't overwrite)
    def _self_timed(args, ctx):
        return ToolResult(ok=True, body={}, latency_ms=42.5)

    fresh_registry.register(_make_tool("timed", execute_impl=_self_timed))
    result3 = fresh_registry.dispatch("timed", {}, ctx)
    assert result3.latency_ms == 42.5


# ---------------------------------------------------------------------------
# 8. RoundState default factory + ToolContext composition
# ---------------------------------------------------------------------------


def test_dispatch_unknown_tool_records_nonzero_latency(fresh_registry):
    """Codex P3 regression lock.

    `dispatch()` promises consistent failure telemetry. The unknown-tool
    early return used to leave `latency_ms=0.0` (dataclass default),
    which is indistinguishable from 'not measured' in metrics.
    Every dispatch path, including rejections, must record a real
    latency measurement."""
    ctx = _make_ctx()
    result = fresh_registry.dispatch("no_such_tool", {}, ctx)
    assert result.ok is False
    assert "Unknown tool" in (result.error or "")
    assert result.latency_ms > 0.0, (
        "unknown-tool rejection still has the default 0.0 latency; "
        "every dispatch outcome should record a non-zero timing"
    )


def test_dispatch_nav_mode_gate_rejection_records_nonzero_latency(fresh_registry):
    """P3 regression lock for the nav_mode gate rejection path."""
    fresh_registry.register(_make_tool("navigate", nav_modes={"navmesh"}))
    ctx = _make_ctx(nav_mode="mapless")
    result = fresh_registry.dispatch("navigate", {}, ctx)
    assert result.ok is False
    assert "nav_mode" in (result.error or "")
    assert result.latency_ms > 0.0


def test_dispatch_task_type_gate_rejection_records_nonzero_latency(fresh_registry):
    """P3 regression lock for the task_type gate rejection path."""
    fresh_registry.register(
        _make_tool(
            "init_scene",
            category=ToolCategory.SESSION,
            task_types={"chat"},
        )
    )
    ctx = _make_ctx(task_type="pointnav")
    result = fresh_registry.dispatch("init_scene", {}, ctx)
    assert result.ok is False
    assert "task_type" in (result.error or "")
    assert result.latency_ms > 0.0


def test_dispatch_wraps_non_tool_result_return_value(fresh_registry):
    """Codex P2 regression lock.

    `ToolRegistry.dispatch` promises to 'never raise'. A tool that
    accidentally returns a legacy-style dict (or anything that isn't
    a ToolResult) MUST NOT crash the caller with an AttributeError
    when dispatch tries to read `result.latency_ms`. Instead,
    dispatch should convert the non-ToolResult return into an
    `ok=False` ToolResult with a descriptive error.

    This matters for the MCP dynamic-wrapper factory (a wrapper that
    forgets to convert a dict back into a ToolResult would silently
    break). Defence in depth."""

    def _buggy_returns_dict(args, ctx):
        return {"ok": True, "result": "this is a legacy-style dict"}

    fresh_registry.register(
        _make_tool("buggy_tool", execute_impl=_buggy_returns_dict)
    )
    ctx = _make_ctx()

    result = fresh_registry.dispatch("buggy_tool", {}, ctx)
    # Must not raise, must be a ToolResult
    assert isinstance(result, ToolResult)
    assert result.ok is False
    # Error should name the tool AND the bad return type so operators
    # debugging a misbehaving variant know what to look for
    assert "buggy_tool" in (result.error or "")
    assert "dict" in (result.error or "")
    assert result.latency_ms >= 0.0

    # Complementary positive case: a tool that returns a legitimate
    # ToolResult still works exactly as before (sanity check that
    # the isinstance guard isn't overly aggressive)
    def _ok(args, ctx):
        return ToolResult(ok=True, body={"fine": True})

    fresh_registry.register(_make_tool("ok_tool", execute_impl=_ok))
    good = fresh_registry.dispatch("ok_tool", {}, ctx)
    assert good.ok is True
    assert good.body == {"fine": True}


def test_dispatch_enforces_allowed_nav_modes_gate(fresh_registry):
    """Codex P1 regression lock.

    `ToolRegistry.dispatch` must refuse to execute a tool whose
    `allowed_nav_modes` does not contain `ctx.nav_mode`, even if the
    caller looks up the tool by name directly (e.g. an LLM hallucinates
    a tool it shouldn't have seen in its schemas, or a test calls
    dispatch with a mismatched context). `available_for()` is the
    schema filter; `dispatch()` is the defense-in-depth gate."""

    navmesh_only = _make_tool("navigate", nav_modes={"navmesh"})
    fresh_registry.register(navmesh_only)

    # In navmesh mode: dispatch succeeds
    ctx_navmesh = _make_ctx(nav_mode="navmesh", task_type="pointnav")
    result_ok = fresh_registry.dispatch("navigate", {}, ctx_navmesh)
    assert result_ok.ok is True

    # In mapless mode: dispatch rejects with a clear error
    ctx_mapless = _make_ctx(nav_mode="mapless", task_type="pointnav")
    result_gated = fresh_registry.dispatch("navigate", {}, ctx_mapless)
    assert result_gated.ok is False
    assert "nav_mode" in (result_gated.error or "")
    assert "mapless" in (result_gated.error or "")


def test_dispatch_enforces_allowed_task_types_gate(fresh_registry):
    """Codex P1 regression lock.

    Mirror of the nav_mode gate: `dispatch` must also enforce
    `allowed_task_types`. A session tool like `init_scene`
    (`allowed_task_types={"chat"}`) must not run when dispatched from
    a nav_agent context (task_type="pointnav")."""

    chat_only = _make_tool(
        "init_scene",
        category=ToolCategory.SESSION,
        task_types={"chat"},
    )
    fresh_registry.register(chat_only)

    # In chat context: dispatch succeeds
    ctx_chat = _make_ctx(task_type="chat")
    result_ok = fresh_registry.dispatch("init_scene", {}, ctx_chat)
    assert result_ok.ok is True

    # In pointnav context: dispatch rejects
    ctx_pointnav = _make_ctx(task_type="pointnav")
    result_gated = fresh_registry.dispatch("init_scene", {}, ctx_pointnav)
    assert result_gated.ok is False
    assert "task_type" in (result_gated.error or "")
    assert "pointnav" in (result_gated.error or "")


def test_dispatch_gate_error_mentions_tool_name(fresh_registry):
    """Quality-of-life: the gate-rejection error should name the tool
    so operators debugging a stuck LLM know which capability was denied."""
    navmesh_only = _make_tool("topdown", nav_modes={"navmesh"})
    fresh_registry.register(navmesh_only)
    ctx = _make_ctx(nav_mode="mapless")
    result = fresh_registry.dispatch("topdown", {}, ctx)
    assert result.ok is False
    assert "topdown" in (result.error or "")


def test_round_state_and_context_independence():
    """Each ToolContext gets its own RoundState (no shared default
    list), and each RoundState's lists are independent."""
    ctx_a = _make_ctx(loop_id="loop_a")
    ctx_b = _make_ctx(loop_id="loop_b")

    # Mutate one — the other must NOT see it
    ctx_a.round_state.captured_images.append("/img/a.png")
    ctx_a.round_state.last_collided = True
    ctx_a.round_state.round_actions.append("forward(0.5m)")

    assert ctx_b.round_state.captured_images == []
    assert ctx_b.round_state.last_collided is False
    assert ctx_b.round_state.round_actions == []

    # state_version_ref is also independent (separate list-of-one)
    ctx_a.state_version_ref[0] = 99
    assert ctx_b.state_version_ref == [1]

    # is_gaussian is mutable for InitSceneTool
    assert ctx_a.is_gaussian is False
    ctx_a.is_gaussian = True
    assert ctx_b.is_gaussian is False


# ---------------------------------------------------------------------------
# 9. Tool Protocol runtime check
# ---------------------------------------------------------------------------


def test_tool_protocol_runtime_check():
    """`isinstance(x, Tool)` works for compliant classes and rejects
    things that look like Tools but lack the required attributes."""
    class _Compliant:
        metadata = ToolMetadata(
            name="ok",
            category=ToolCategory.NAVIGATION,
            description="d",
        )

        def execute(self, args, ctx):
            return ToolResult(ok=True)

    assert isinstance(_Compliant(), Tool)

    class _MissingExecute:
        metadata = ToolMetadata(
            name="bad",
            category=ToolCategory.NAVIGATION,
            description="d",
        )

    assert not isinstance(_MissingExecute(), Tool)
