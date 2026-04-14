"""Phase 2 PR 3 — unit tests for the 5 chat-only session Tool subclasses.

Covers `InitSceneTool`, `CloseSessionTool`, `NavLoopStartTool`,
`NavLoopStatusTool`, `NavLoopStopTool`.

The codex P1/P2 regression locks live in the NavLoopStartTool section
at the top of the file because they are the most load-bearing
invariants in the whole Phase 2 migration — breaking either one
silently invalidates benchmark runs. These tests exactly mirror the
two regression tests in `tests/test_nav_eval_metrics.py`
(`test_mcp_non_pointnav_with_explicit_position_goaltype_does_not_leak`
and `test_mcp_pointnav_with_instruction_goaltype_still_requires_coords`)
so both the Registry path (PR 4) and the current mcp_server path stay
green against the same scenarios.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

# Importing the package triggers ToolRegistry.register() for every tool.
import habitat_agent.tools  # noqa: F401,E402

from habitat_agent.tools.base import (  # noqa: E402
    ToolContext,
    ToolRegistry,
    ToolResult,
)
from habitat_agent.tools.session import (  # noqa: E402
    CloseSessionTool,
    InitSceneTool,
    NavLoopStartTool,
    NavLoopStatusTool,
    NavLoopStopTool,
)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


class FakeBridge:
    """Records every (action, payload) tuple; returns queued responses.

    Unlike the nav-side FakeBridge, this one exposes `session_id` as a
    mutable attribute because InitSceneTool / CloseSessionTool need to
    read/write it the way BridgeClient does.
    """

    def __init__(self, responses: Dict[str, Any] = None):
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.responses: Dict[str, Any] = responses or {}
        self.session_id: str = ""
        self.raise_next: bool = False

    def call(self, action: str, payload: Dict[str, Any]) -> Any:
        self.calls.append((action, dict(payload)))
        if self.raise_next:
            self.raise_next = False
            raise RuntimeError("fake bridge error")
        return self.responses.get(action, {})


def _make_ctx(
    bridge: FakeBridge = None,
    *,
    session_id: str = "",
    output_dir: str = "/tmp/artifacts",
    task_type: str = "chat",
) -> ToolContext:
    b = bridge or FakeBridge()
    if session_id:
        b.session_id = session_id
    return ToolContext(
        bridge=b,
        session_id=session_id,
        loop_id="",
        output_dir=output_dir,
        nav_mode="navmesh",
        task_type=task_type,
    )


# ---------------------------------------------------------------------------
# Codex P1/P2 regression locks — the most important tests in this file
# ---------------------------------------------------------------------------


def test_p1_non_pointnav_with_position_goaltype_does_not_leak_coordinates():
    """Codex P1 regression lock.

    A caller passing task_type=objectnav with goal_type=position (an
    attempt to force position routing for a coordinate-blind task)
    MUST NOT cause `goal_position` to appear in the payload sent to
    the nav_agent. Coordinates may still go into `eval_goal_position`
    for GT scoring, but never into the agent-facing `goal_position`.
    """
    bridge = FakeBridge(
        responses={"start_nav_loop": {"loop_id": "navloop-1", "status": "started"}}
    )
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)

    result = NavLoopStartTool().execute(
        {
            "task_type": "objectnav",
            "goal_description": "Find kitchen",
            "goal_type": "position",  # caller tries to force position routing
            "goal_x": 5.0,
            "goal_y": 0.2,
            "goal_z": 3.0,
        },
        ctx,
    )
    assert result.ok, f"unexpected error: {result.error}"
    assert len(bridge.calls) == 1
    action, payload = bridge.calls[0]
    assert action == "start_nav_loop"
    # Agent-facing goal_position must NOT be set for non-pointnav
    assert "goal_position" not in payload, (
        f"P1 LEAK: goal_position={payload.get('goal_position')} "
        f"was forwarded to the agent prompt for task_type=objectnav"
    )
    # Eval-only GT is allowed
    assert payload["eval_goal_position"] == [5.0, 0.2, 3.0]
    assert payload["has_ground_truth"] is True


def test_p2_pointnav_with_instruction_goaltype_still_requires_coords():
    """Codex P2 regression lock.

    A caller passing task_type=pointnav with goal_type=instruction (an
    attempt to bypass the PointNav coordinate requirement) MUST be
    rejected. Without this check a caller could spin up a PointNav
    loop with no target at all and waste an evaluation slot.
    """
    bridge = FakeBridge(
        responses={"start_nav_loop": {"loop_id": "navloop-x"}}
    )
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)

    result = NavLoopStartTool().execute(
        {
            "task_type": "pointnav",
            "goal_description": "Go somewhere",
            "goal_type": "instruction",  # caller tries to bypass coord requirement
            # NO goal_x/y/z
        },
        ctx,
    )
    assert result.ok is False, (
        f"P2 BYPASS: pointnav with goal_type=instruction and no coords "
        f"should have been rejected, but execute() returned ok=True"
    )
    assert result.error is not None
    assert "pointnav requires" in result.error.lower()
    # Bridge was NOT called
    assert bridge.calls == []


def test_pointnav_with_complete_coords_succeeds_and_sets_goal_position():
    """Complementary positive case: the normal PointNav path DOES set
    goal_position and eval_goal_position."""
    bridge = FakeBridge(responses={"start_nav_loop": {"loop_id": "nl"}})
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)
    result = NavLoopStartTool().execute(
        {
            "task_type": "pointnav",
            "goal_x": 3.0,
            "goal_y": 0.2,
            "goal_z": 4.0,
        },
        ctx,
    )
    assert result.ok
    _, payload = bridge.calls[0]
    assert payload["goal_type"] == "position"
    assert payload["goal_position"] == [3.0, 0.2, 4.0]
    assert payload["eval_goal_position"] == [3.0, 0.2, 4.0]
    assert payload["has_ground_truth"] is True


def test_pointnav_origin_requires_explicit_zeros():
    """Caller must pass goal_x=0, goal_y=0, goal_z=0 explicitly when
    the target really is the origin. Omitting them is NOT interpreted
    as 'origin' — it's a rejection."""
    bridge = FakeBridge(responses={"start_nav_loop": {"loop_id": "nl"}})
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)
    result = NavLoopStartTool().execute(
        {
            "task_type": "pointnav",
            "goal_x": 0.0,
            "goal_y": 0.0,
            "goal_z": 0.0,
        },
        ctx,
    )
    assert result.ok
    _, payload = bridge.calls[0]
    assert payload["goal_position"] == [0.0, 0.0, 0.0]


def test_partial_coords_rejected():
    """2-of-3 coords is a clear input error, not a silent fill."""
    bridge = FakeBridge()
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)
    result = NavLoopStartTool().execute(
        {"task_type": "objectnav", "goal_x": 1.0, "goal_y": 0.2},
        ctx,
    )
    assert result.ok is False
    assert "together" in (result.error or "").lower()
    assert bridge.calls == []


def test_has_ground_truth_without_coords_rejected():
    """has_ground_truth=True but no goal_x/y/z → reject rather than
    silently assume origin."""
    bridge = FakeBridge()
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)
    result = NavLoopStartTool().execute(
        {
            "task_type": "objectnav",
            "goal_description": "Find chair",
            "has_ground_truth": True,
        },
        ctx,
    )
    assert result.ok is False
    assert "has_ground_truth" in (result.error or "")
    assert bridge.calls == []


# ---------------------------------------------------------------------------
# NavLoopStartTool — preconditions and non-coordinate behaviour
# ---------------------------------------------------------------------------


def test_nav_loop_start_requires_active_session():
    bridge = FakeBridge()
    bridge.session_id = ""  # no session
    ctx = _make_ctx(bridge)
    result = NavLoopStartTool().execute(
        {"task_type": "pointnav", "goal_x": 1.0, "goal_y": 0.2, "goal_z": 2.0}, ctx
    )
    assert result.ok is False
    assert "No active session" in (result.error or "")
    assert bridge.calls == []


def test_imagenav_requires_reference_image():
    bridge = FakeBridge()
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)
    result = NavLoopStartTool().execute(
        {"task_type": "imagenav", "goal_description": "Find this view"}, ctx
    )
    assert result.ok is False
    assert "reference_image" in (result.error or "")
    assert bridge.calls == []


def test_imagenav_with_reference_image_succeeds():
    bridge = FakeBridge(responses={"start_nav_loop": {"loop_id": "nl"}})
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)
    result = NavLoopStartTool().execute(
        {
            "task_type": "imagenav",
            "goal_description": "Find this view",
            "reference_image": "/tmp/ref.png",
        },
        ctx,
    )
    assert result.ok
    _, payload = bridge.calls[0]
    assert payload["reference_image"] == "/tmp/ref.png"
    assert payload["goal_type"] == "instruction"  # non-pointnav default
    assert "goal_position" not in payload


def test_objectnav_without_coords_has_no_eval_goal_and_no_has_ground_truth():
    """Pure instruction-mode objectnav: no coordinates provided → no
    eval_goal_position, no has_ground_truth flag, no goal_position."""
    bridge = FakeBridge(responses={"start_nav_loop": {"loop_id": "nl"}})
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)
    result = NavLoopStartTool().execute(
        {"task_type": "objectnav", "goal_description": "Find kitchen"}, ctx
    )
    assert result.ok
    _, payload = bridge.calls[0]
    assert payload["goal_type"] == "instruction"
    assert "goal_position" not in payload
    assert "eval_goal_position" not in payload
    assert "has_ground_truth" not in payload


def test_nav_loop_start_passes_max_iterations_and_nav_mode():
    bridge = FakeBridge(responses={"start_nav_loop": {"loop_id": "nl"}})
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)
    NavLoopStartTool().execute(
        {
            "task_type": "pointnav",
            "goal_x": 1.0,
            "goal_y": 0.2,
            "goal_z": 1.0,
            "max_iterations": 75,
            "nav_mode": "mapless",
            "success_distance_threshold": 0.2,
        },
        ctx,
    )
    _, payload = bridge.calls[0]
    assert payload["max_iterations"] == 75
    assert payload["nav_mode"] == "mapless"
    assert payload["success_distance_threshold"] == 0.2


def test_nav_loop_start_default_goal_description():
    bridge = FakeBridge(responses={"start_nav_loop": {"loop_id": "nl"}})
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)
    NavLoopStartTool().execute(
        {"task_type": "pointnav", "goal_x": 1.0, "goal_y": 0.2, "goal_z": 1.0}, ctx
    )
    _, payload = bridge.calls[0]
    assert payload["goal_description"] == "Navigate (pointnav)"


def test_has_ground_truth_string_false_treated_as_false():
    """Codex P2 regression lock — LLM string boolean bug.

    Some LLMs emit JSON string booleans ("false", "true") even when
    the schema declares a boolean parameter. Python's `bool("false")`
    returns True (non-empty strings are truthy), so a naive
    `bool(args.get(...))` would treat a caller saying "I do NOT have
    ground truth" as "I DO have ground truth" and incorrectly trigger
    the strict GT coordinate-required check.

    The fix: parse `has_ground_truth` via `_parse_bool_flag` which
    recognises "false"/"0"/"no" strings as False. This test locks in
    the correct semantics: `"false"` + pure-instruction objectnav
    (no coords) MUST be accepted, not rejected as "GT without coords"."""
    bridge = FakeBridge(responses={"start_nav_loop": {"loop_id": "nl"}})
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)
    result = NavLoopStartTool().execute(
        {
            "task_type": "objectnav",
            "goal_description": "Find kitchen",
            "has_ground_truth": "false",  # string boolean from LLM
        },
        ctx,
    )
    assert result.ok, (
        f"string 'false' was misinterpreted as truthy: {result.error}"
    )
    _, payload = bridge.calls[0]
    # Pure instruction mode: no eval_goal_position, no has_ground_truth
    assert "has_ground_truth" not in payload
    assert "eval_goal_position" not in payload


def test_has_ground_truth_string_true_treated_as_true():
    """Complement to the P2 lock: string 'true' MUST still require
    coords (original GT semantics preserved)."""
    bridge = FakeBridge()
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)
    result = NavLoopStartTool().execute(
        {
            "task_type": "objectnav",
            "goal_description": "Find chair",
            "has_ground_truth": "true",  # string boolean from LLM
            # no goal_x/y/z
        },
        ctx,
    )
    assert result.ok is False
    assert "has_ground_truth" in (result.error or "")
    assert bridge.calls == []


def test_has_ground_truth_native_bool_preserved():
    """Native Python bool still works — the new parser must not
    break the existing happy path."""
    bridge = FakeBridge(responses={"start_nav_loop": {"loop_id": "nl"}})
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)

    # False case: no coord requirement
    result_f = NavLoopStartTool().execute(
        {"task_type": "objectnav", "has_ground_truth": False}, ctx
    )
    assert result_f.ok
    assert "has_ground_truth" not in bridge.calls[-1][1]

    # True case with coords: succeeds, eval_goal_position is set
    bridge.calls.clear()
    result_t = NavLoopStartTool().execute(
        {
            "task_type": "objectnav",
            "has_ground_truth": True,
            "goal_x": 1.0,
            "goal_y": 0.2,
            "goal_z": 1.0,
        },
        ctx,
    )
    assert result_t.ok
    _, payload = bridge.calls[0]
    assert payload["has_ground_truth"] is True
    assert payload["eval_goal_position"] == [1.0, 0.2, 1.0]


def test_has_ground_truth_int_coerces_correctly():
    """0/1 int values are treated the same as False/True."""
    bridge = FakeBridge()
    bridge.session_id = "s1"
    ctx = _make_ctx(bridge)

    # 0 → False: pure instruction mode, no coord requirement
    result_0 = NavLoopStartTool().execute(
        {"task_type": "objectnav", "has_ground_truth": 0}, ctx
    )
    assert result_0.ok

    # 1 → True: coord required, reject without coords
    result_1 = NavLoopStartTool().execute(
        {"task_type": "objectnav", "has_ground_truth": 1}, ctx
    )
    assert result_1.ok is False
    assert "has_ground_truth" in (result_1.error or "")


def test_nav_loop_start_drops_unwritable_output_dir_from_payload(
    tmp_path, monkeypatch
):
    """Codex P2 regression lock.

    `_build_payload` tolerates an unwritable artifacts dir (bridge
    may be running in a container where `/tmp/artifacts` is read-only)
    by catching the OSError from os.makedirs. The fallback semantics
    MUST be: omit `output_dir` from the bridge payload entirely so
    the bridge picks its own default. The previous implementation
    still wrote the unwritable path into the payload, defeating the
    fallback.

    Note on the patch target: we MUST patch
    `habitat_agent.tools.session.os.makedirs` — the canonical import
    path used by `NavLoopStartTool` — and NOT the alternate
    `tools.habitat_agent.tools.session` path. Those can end up as
    two distinct sys.modules entries on runners where the repo root
    is also on sys.path, and patching the alias may silently stop
    affecting the tool under test. Using the dotted-path form of
    `monkeypatch.setattr` guarantees Python resolves the exact
    module object the tool imports from.
    """
    bridge = FakeBridge(responses={"start_nav_loop": {"loop_id": "nl"}})
    bridge.session_id = "s1"

    unwritable = "/proc/1/root/nonexistent/nav_artifacts"  # would fail anyway
    ctx = _make_ctx(bridge, output_dir=unwritable)

    # Force the OSError branch on every host (root included) by
    # patching through the canonical module path.
    def _boom_makedirs(*args, **kwargs):
        raise OSError(13, "Permission denied", unwritable)

    monkeypatch.setattr(
        "habitat_agent.tools.session.os.makedirs", _boom_makedirs
    )

    result = NavLoopStartTool().execute(
        {
            "task_type": "pointnav",
            "goal_x": 1.0,
            "goal_y": 0.2,
            "goal_z": 1.0,
        },
        ctx,
    )
    assert result.ok, f"unexpected error: {result.error}"
    _, payload = bridge.calls[0]
    assert "output_dir" not in payload, (
        f"unwritable output_dir was still forwarded to the bridge: "
        f"{payload.get('output_dir')!r}. The fallback should drop it "
        f"so the bridge can pick its own default."
    )


def test_nav_loop_start_keeps_writable_output_dir_in_payload(tmp_path):
    """Complementary positive case: when makedirs succeeds, the
    output_dir stays in the payload (the drop only happens on the
    OSError fallback path)."""
    bridge = FakeBridge(responses={"start_nav_loop": {"loop_id": "nl"}})
    bridge.session_id = "s1"
    writable = str(tmp_path / "artifacts")
    ctx = _make_ctx(bridge, output_dir=writable)
    result = NavLoopStartTool().execute(
        {
            "task_type": "pointnav",
            "goal_x": 1.0,
            "goal_y": 0.2,
            "goal_z": 1.0,
        },
        ctx,
    )
    assert result.ok
    _, payload = bridge.calls[0]
    assert payload["output_dir"] == writable


def test_nav_loop_start_bridge_exception_returns_error():
    bridge = FakeBridge()
    bridge.session_id = "s1"
    bridge.raise_next = True
    ctx = _make_ctx(bridge)
    result = NavLoopStartTool().execute(
        {"task_type": "pointnav", "goal_x": 1.0, "goal_y": 0.2, "goal_z": 1.0}, ctx
    )
    assert result.ok is False
    assert "fake bridge error" in (result.error or "")


# ---------------------------------------------------------------------------
# InitSceneTool
# ---------------------------------------------------------------------------


def test_init_scene_mutates_context_session_id_and_is_gaussian():
    bridge = FakeBridge(
        responses={
            "init_scene": {
                "session_id": "new-session-42",
                "is_gaussian": True,
                "scene": "apartment_1",
            }
        }
    )
    ctx = _make_ctx(bridge)
    result = InitSceneTool().execute({"scene": "apartment_1"}, ctx)
    assert result.ok
    action, payload = bridge.calls[0]
    assert action == "init_scene"
    assert payload["scene"] == "apartment_1"
    assert payload["sensor"]["color_sensor"] is True
    assert payload["sensor"]["depth_sensor"] is True
    assert payload["sensor"]["semantic_sensor"] is False
    # Context was mutated
    assert bridge.session_id == "new-session-42"
    assert ctx.session_id == "new-session-42"
    assert ctx.is_gaussian is True


def test_init_scene_not_is_gaussian_when_response_says_false():
    bridge = FakeBridge(
        responses={"init_scene": {"session_id": "s1", "is_gaussian": False}}
    )
    ctx = _make_ctx(bridge)
    InitSceneTool().execute({}, ctx)
    assert ctx.is_gaussian is False


def test_init_scene_forwards_custom_dataset_config():
    bridge = FakeBridge(responses={"init_scene": {"session_id": "s1"}})
    ctx = _make_ctx(bridge)
    InitSceneTool().execute(
        {"scene_dataset_config_file": "/custom/path.json"}, ctx
    )
    _, payload = bridge.calls[0]
    assert payload["scene_dataset_config_file"] == "/custom/path.json"


def test_init_scene_bridge_exception_leaves_session_id_cleared():
    """Legacy behaviour: session_id is cleared BEFORE the bridge call
    (so a re-init doesn't look like 'same session'). If the bridge
    then fails, session_id stays cleared."""
    bridge = FakeBridge()
    bridge.session_id = "old"
    bridge.raise_next = True
    ctx = _make_ctx(bridge, session_id="old")
    result = InitSceneTool().execute({}, ctx)
    assert result.ok is False
    assert bridge.session_id is None


def test_init_scene_depth_default_is_true():
    """Codex P2 regression lock — preserve legacy hab_init defaults.

    Legacy `mcp_server.hab_init` had `depth: bool = True`. The
    migrated InitSceneTool must keep the same default so callers
    that omit the param see no behavioural change."""
    bridge = FakeBridge(responses={"init_scene": {"session_id": "s1"}})
    ctx = _make_ctx(bridge)
    InitSceneTool().execute({}, ctx)
    _, payload = bridge.calls[0]
    assert payload["sensor"]["depth_sensor"] is True


def test_init_scene_semantic_default_is_false():
    """Legacy default: semantic=False."""
    bridge = FakeBridge(responses={"init_scene": {"session_id": "s1"}})
    ctx = _make_ctx(bridge)
    InitSceneTool().execute({}, ctx)
    _, payload = bridge.calls[0]
    assert payload["sensor"]["semantic_sensor"] is False


def test_init_scene_explicit_depth_false_disables_depth_sensor():
    """Codex P2 regression lock — explicit depth=False MUST be
    forwarded to the sensor config. The previous migration hardcoded
    depth_sensor=True regardless of caller intent, silently corrupting
    callers that explicitly disabled depth."""
    bridge = FakeBridge(responses={"init_scene": {"session_id": "s1"}})
    ctx = _make_ctx(bridge)
    InitSceneTool().execute({"depth": False}, ctx)
    _, payload = bridge.calls[0]
    assert payload["sensor"]["depth_sensor"] is False, (
        f"depth=False was ignored — sensor config: {payload['sensor']}"
    )


def test_init_scene_explicit_semantic_true_enables_semantic_sensor():
    """Mirror P2 lock for semantic=True."""
    bridge = FakeBridge(responses={"init_scene": {"session_id": "s1"}})
    ctx = _make_ctx(bridge)
    InitSceneTool().execute({"semantic": True}, ctx)
    _, payload = bridge.calls[0]
    assert payload["sensor"]["semantic_sensor"] is True


def test_init_scene_depth_string_false_handled_via_parse_bool_flag():
    """LLM-emitted string boolean ('false') for depth must parse
    correctly, just like has_ground_truth in NavLoopStartTool. Reuses
    the parse_bool_flag helper from PR3 round 3."""
    bridge = FakeBridge(responses={"init_scene": {"session_id": "s1"}})
    ctx = _make_ctx(bridge)
    InitSceneTool().execute({"depth": "false"}, ctx)
    _, payload = bridge.calls[0]
    assert payload["sensor"]["depth_sensor"] is False


# ---------------------------------------------------------------------------
# Third-person sensor support — bug: implementation exists in the bridge
# (mixins_session_scene._simulator_factory_with_third_person) but InitSceneTool
# hardcoded the sensor dict without the third_person_color_sensor field, so
# MCP callers had no way to enable it.
# ---------------------------------------------------------------------------


def test_init_scene_third_person_default_is_false():
    """When the caller omits `third_person`, the sensor dict must NOT
    request the third-person view (preserves backward compatibility)."""
    bridge = FakeBridge(responses={"init_scene": {"session_id": "s1"}})
    ctx = _make_ctx(bridge)
    InitSceneTool().execute({}, ctx)
    _, payload = bridge.calls[0]
    assert payload["sensor"].get("third_person_color_sensor", False) is False


def test_init_scene_explicit_third_person_true_enables_sensor():
    """When caller passes third_person=True, the sensor dict must carry
    third_person_color_sensor=True so the bridge's third_person_color_sensor
    branch fires and the extra CameraSensorSpec gets injected before
    Simulator() construction."""
    bridge = FakeBridge(responses={"init_scene": {"session_id": "s1"}})
    ctx = _make_ctx(bridge)
    InitSceneTool().execute({"third_person": True}, ctx)
    _, payload = bridge.calls[0]
    assert payload["sensor"].get("third_person_color_sensor") is True, (
        f"third_person=True was dropped — sensor config: {payload['sensor']}"
    )


def test_init_scene_third_person_string_true_handled_via_parse_bool_flag():
    """Mirror the depth/semantic string-boolean tolerance for LLM-emitted
    'true' strings."""
    bridge = FakeBridge(responses={"init_scene": {"session_id": "s1"}})
    ctx = _make_ctx(bridge)
    InitSceneTool().execute({"third_person": "true"}, ctx)
    _, payload = bridge.calls[0]
    assert payload["sensor"].get("third_person_color_sensor") is True


def test_init_scene_third_person_exposed_in_mcp_schema():
    """The MCP tool parameters_schema must advertise `third_person` so
    LLM agents (and `describe_api` introspection callers) can discover
    the capability. Previously hidden behind the implementation."""
    schema = InitSceneTool.metadata.parameters_schema
    props = schema.get("properties", {})
    assert "third_person" in props, (
        f"third_person missing from InitSceneTool parameters_schema: "
        f"{list(props.keys())}"
    )
    assert props["third_person"].get("type") == "boolean"


def test_init_scene_uses_hab_default_scene_env_var(monkeypatch):
    """Codex P1 regression lock — InitSceneTool must provide a
    default scene when the caller omits one.

    The bridge's init_scene action REQUIRES a non-empty `payload.scene`
    string. The legacy mcp_server.hab_init function had a fallback
    chain: caller arg → HAB_DEFAULT_SCENE env → hardcoded
    "interior_0405_840145". When we migrated InitSceneTool we dropped
    the whole chain, so a no-arg `init_scene()` call now sends no
    scene key at all and the bridge returns an immediate validation
    error.

    This test locks in: caller omits `scene`, but HAB_DEFAULT_SCENE
    is set in the env → payload.scene == env value."""
    monkeypatch.setenv("HAB_DEFAULT_SCENE", "test_scene_env")
    bridge = FakeBridge(responses={"init_scene": {"session_id": "s1"}})
    ctx = _make_ctx(bridge)
    result = InitSceneTool().execute({}, ctx)
    assert result.ok
    _, payload = bridge.calls[0]
    assert payload.get("scene") == "test_scene_env", (
        f"scene field missing or wrong: payload={payload}"
    )


def test_init_scene_falls_back_to_hardcoded_default_without_env(monkeypatch):
    """P1 regression lock: when no arg AND no env var, fall back to
    the legacy hardcoded default so no-arg init_scene() still works."""
    monkeypatch.delenv("HAB_DEFAULT_SCENE", raising=False)
    bridge = FakeBridge(responses={"init_scene": {"session_id": "s1"}})
    ctx = _make_ctx(bridge)
    result = InitSceneTool().execute({}, ctx)
    assert result.ok
    _, payload = bridge.calls[0]
    assert payload.get("scene"), (
        f"no-arg init_scene must still populate payload.scene, got: {payload}"
    )
    # Match the legacy hardcoded default exactly
    assert payload["scene"] == "interior_0405_840145"


def test_init_scene_explicit_scene_wins_over_env_and_default(monkeypatch):
    """Explicit arg > env > hardcoded default (fallback order)."""
    monkeypatch.setenv("HAB_DEFAULT_SCENE", "env_scene")
    bridge = FakeBridge(responses={"init_scene": {"session_id": "s1"}})
    ctx = _make_ctx(bridge)
    result = InitSceneTool().execute({"scene": "explicit_scene"}, ctx)
    assert result.ok
    _, payload = bridge.calls[0]
    assert payload["scene"] == "explicit_scene"


def test_init_scene_tool_declares_legacy_name_init():
    """P1 regression lock — InitSceneTool must declare `{"init"}`
    as a legacy name so mcp_server's dynamic registration can
    expose both `hab_init_scene` AND `hab_init` (the name old MCP
    clients hardcoded before Phase 2)."""
    assert "init" in InitSceneTool.metadata.legacy_names


def test_close_session_tool_declares_legacy_name_close():
    """P1 regression lock — CloseSessionTool must declare `{"close"}`
    as a legacy name so `hab_close` stays exposed."""
    assert "close" in CloseSessionTool.metadata.legacy_names


def test_init_scene_failure_also_clears_ctx_session_id():
    """Codex P3 regression lock.

    Previously the upfront clearing only touched `ctx.bridge.session_id`;
    `ctx.session_id` stayed stale on the failure path. After a failed
    re-init the context could advertise an old session while the bridge
    had none — any caller reading `ctx.session_id` to decide whether a
    session was active would see an inconsistent view.

    Both fields MUST be cleared upfront so the failure path leaves a
    coherent 'no active session' state."""
    bridge = FakeBridge()
    bridge.session_id = "old-session"
    bridge.raise_next = True
    ctx = _make_ctx(bridge, session_id="old-session")
    assert ctx.session_id == "old-session"  # precondition
    result = InitSceneTool().execute({}, ctx)
    assert result.ok is False
    # BOTH sides of the session view must be cleared after a failed re-init
    assert bridge.session_id is None
    assert ctx.session_id == "", (
        f"ctx.session_id leaked the stale 'old-session' value after a "
        f"failed re-init; got {ctx.session_id!r}"
    )


# ---------------------------------------------------------------------------
# CloseSessionTool
# ---------------------------------------------------------------------------


def test_close_session_clears_ctx_session_id():
    bridge = FakeBridge(responses={"close_session": {"ok": True}})
    ctx = _make_ctx(bridge, session_id="s1")
    result = CloseSessionTool().execute({}, ctx)
    assert result.ok
    assert bridge.calls[0] == ("close_session", {})
    assert bridge.session_id is None
    assert ctx.session_id == ""


def test_close_session_bridge_exception_returns_error():
    bridge = FakeBridge()
    bridge.raise_next = True
    ctx = _make_ctx(bridge, session_id="s1")
    result = CloseSessionTool().execute({}, ctx)
    assert result.ok is False
    assert "fake bridge error" in (result.error or "")


# ---------------------------------------------------------------------------
# NavLoopStatusTool / NavLoopStopTool + _resolve_loop_id fallback
# ---------------------------------------------------------------------------


def test_nav_loop_status_with_explicit_loop_id():
    bridge = FakeBridge(
        responses={
            "get_nav_loop_status": {"loop_id": "nl-1", "status": "in_progress"}
        }
    )
    ctx = _make_ctx(bridge, session_id="s1")
    result = NavLoopStatusTool().execute({"loop_id": "nl-1"}, ctx)
    assert result.ok
    action, payload = bridge.calls[0]
    assert action == "get_nav_loop_status"
    assert payload == {"loop_id": "nl-1", "include_nav_status": True}


def test_nav_loop_status_fallback_to_most_recent_active_loop():
    """When loop_id is empty, resolve via get_runtime_status → last
    active nav_loop."""
    bridge = FakeBridge(
        responses={
            "get_runtime_status": {
                "nav_loops": [
                    {"loop_id": "nl-old"},
                    {"loop_id": "nl-current"},
                ],
                "recently_closed_nav_loops": [],
            },
            "get_nav_loop_status": {"status": "in_progress"},
        }
    )
    ctx = _make_ctx(bridge, session_id="s1")
    result = NavLoopStatusTool().execute({}, ctx)
    assert result.ok
    # Two bridge calls: runtime lookup, then status fetch
    assert len(bridge.calls) == 2
    assert bridge.calls[0][0] == "get_runtime_status"
    assert bridge.calls[1][0] == "get_nav_loop_status"
    assert bridge.calls[1][1]["loop_id"] == "nl-current"


def test_nav_loop_status_fallback_to_closed_loop():
    """If no active loops, fall back to most-recent closed loop."""
    bridge = FakeBridge(
        responses={
            "get_runtime_status": {
                "nav_loops": [],
                "recently_closed_nav_loops": [{"loop_id": "nl-done"}],
            },
            "get_nav_loop_status": {"status": "reached"},
        }
    )
    ctx = _make_ctx(bridge, session_id="s1")
    NavLoopStatusTool().execute({}, ctx)
    assert bridge.calls[1][1]["loop_id"] == "nl-done"


def test_nav_loop_status_no_loops_returns_error():
    bridge = FakeBridge(
        responses={
            "get_runtime_status": {
                "nav_loops": [],
                "recently_closed_nav_loops": [],
            }
        }
    )
    ctx = _make_ctx(bridge, session_id="s1")
    result = NavLoopStatusTool().execute({}, ctx)
    assert result.ok is False
    assert "No active nav loop" in (result.error or "")


def test_nav_loop_stop_with_explicit_loop_id():
    bridge = FakeBridge(responses={"stop_nav_loop": {"returncode": 0}})
    ctx = _make_ctx(bridge, session_id="s1")
    result = NavLoopStopTool().execute({"loop_id": "nl-1"}, ctx)
    assert result.ok
    action, payload = bridge.calls[0]
    assert action == "stop_nav_loop"
    assert payload == {"loop_id": "nl-1"}


def test_nav_loop_stop_no_loops_returns_error():
    bridge = FakeBridge(
        responses={
            "get_runtime_status": {
                "nav_loops": [],
                "recently_closed_nav_loops": [],
            }
        }
    )
    ctx = _make_ctx(bridge, session_id="s1")
    result = NavLoopStopTool().execute({}, ctx)
    assert result.ok is False
    assert "No active nav loop" in (result.error or "")


# ---------------------------------------------------------------------------
# Registry visibility — session tools are chat-only
# ---------------------------------------------------------------------------


def test_session_tools_visible_to_chat_task_type_only():
    """Session tools have `allowed_task_types={"chat"}`, so they are
    visible to chat_agent (task_type='chat') and invisible to nav_agent
    task types (pointnav, objectnav, etc.)."""
    chat_names = {
        t.metadata.name
        for t in ToolRegistry.available_for("navmesh", "chat")
    }
    for name in (
        "init_scene",
        "close_session",
        "nav_loop_start",
        "nav_loop_status",
        "nav_loop_stop",
    ):
        assert name in chat_names, f"{name} should be visible in chat mode"

    pointnav_names = {
        t.metadata.name
        for t in ToolRegistry.available_for("navmesh", "pointnav")
    }
    for name in (
        "init_scene",
        "close_session",
        "nav_loop_start",
        "nav_loop_status",
        "nav_loop_stop",
    ):
        assert name not in pointnav_names, (
            f"{name} should NOT be visible in pointnav mode"
        )


def test_registry_totals_after_pr3():
    """After PR 3, the Registry should hold 16 tools (original):
    11 nav-side (PR 2) + 5 session (PR 3).
    SceneGraphQueryTool is the 17th added in the rerun/SG sprint."""
    assert len(ToolRegistry._tools) >= 16


# ---------------------------------------------------------------------------
# SceneGraphQueryTool tests
# ---------------------------------------------------------------------------

_SAMPLE_SG = {
    "nodes": [
        {"type": "room", "label": "kitchen", "id": "r1", "centroid_xyz": [1.0, 0.0, 2.0]},
        {"type": "room", "label": "bedroom", "id": "r2", "centroid_xyz": [5.0, 0.0, 3.0]},
        {"type": "object", "label": "chair", "id": "o1", "position_xyz": [1.5, 0.0, 2.5], "room_id": "r1"},
        {"type": "object", "label": "table", "id": "o2", "position_xyz": [2.0, 0.0, 2.0], "room_id": "r1"},
    ],
    "rooms": [
        {"id": "r1", "label": "kitchen", "centroid_xyz": [1.0, 0.0, 2.0], "area": 12.0},
        {"id": "r2", "label": "bedroom", "centroid_xyz": [5.0, 0.0, 3.0], "area": 10.0},
    ],
}


def _make_sg_ctx(sg_data=None):
    """Build a ToolContext whose bridge returns a get_scene_graph response."""
    response = {
        "session_id": "s1",
        "scene_id": "test_scene",
        "nodes": sg_data.get("nodes", []) if sg_data else [],
        "rooms": sg_data.get("rooms", []) if sg_data else [],
        "total_matched": len(sg_data.get("nodes", [])) if sg_data else 0,
        "scene_graph_available": sg_data is not None,
    }

    class _FakeBridge:
        calls: List[Tuple] = []

        def call(self, action, payload=None):
            self.calls.append((action, dict(payload or {})))
            return response

    bridge = _FakeBridge()
    ctx = _make_sg_bare_ctx(bridge)
    return ctx, bridge


def _make_sg_bare_ctx(bridge):
    from habitat_agent.tools.base import ToolContext, RoundState
    return ToolContext(
        bridge=bridge,
        session_id="s1",
        loop_id="",
        output_dir="/tmp",
        workspace_host="",
        nav_mode="navmesh",
        task_type="chat",
        round_state=RoundState(),
    )


def test_scene_graph_query_rooms():
    """Query room type returns matching room nodes."""
    from habitat_agent.tools.session import SceneGraphQueryTool
    ctx, bridge = _make_sg_ctx(_SAMPLE_SG)
    tool = SceneGraphQueryTool()
    result = tool.execute({"query_type": "room", "room_type": "kitchen"}, ctx)
    assert result.ok
    assert bridge.calls[-1][0] == "get_scene_graph"
    assert bridge.calls[-1][1]["query_type"] == "room"


def test_scene_graph_query_objects():
    """Query object label passes correct payload to bridge."""
    from habitat_agent.tools.session import SceneGraphQueryTool
    ctx, bridge = _make_sg_ctx(_SAMPLE_SG)
    tool = SceneGraphQueryTool()
    result = tool.execute({"query_type": "object", "object_label": "chair"}, ctx)
    assert result.ok
    assert bridge.calls[-1][1]["object_label"] == "chair"


def test_scene_graph_no_sg_returns_ok_empty():
    """When no SG is loaded, bridge still returns ok with empty nodes."""
    from habitat_agent.tools.session import SceneGraphQueryTool
    class _NoBridge:
        def call(self, action, payload=None):
            return {
                "session_id": "s1", "scene_id": "s", "nodes": [],
                "rooms": [], "total_matched": 0, "scene_graph_available": False,
            }
    ctx = _make_sg_bare_ctx(_NoBridge())
    tool = SceneGraphQueryTool()
    result = tool.execute({"query_type": "all"}, ctx)
    assert result.ok
    assert result.body["nodes"] == []
    assert result.body["scene_graph_available"] is False


def test_scene_graph_bridge_exception_returns_error():
    """Bridge exception surfaces as ok=False ToolResult."""
    from habitat_agent.tools.session import SceneGraphQueryTool
    class _BadBridge:
        def call(self, action, payload=None):
            raise RuntimeError("bridge down")
    ctx = _make_sg_bare_ctx(_BadBridge())
    tool = SceneGraphQueryTool()
    result = tool.execute({"query_type": "room"}, ctx)
    assert not result.ok
    assert "bridge down" in result.error


def test_scene_graph_tool_is_mcp_visible():
    """SceneGraphQueryTool must be mcp_visible so it's exported via MCP."""
    from habitat_agent.tools.session import SceneGraphQueryTool
    assert SceneGraphQueryTool.metadata.mcp_visible is True


def test_scene_graph_tool_in_registry():
    """SceneGraphQueryTool is registered under name 'scene_graph'."""
    from habitat_agent.tools.base import ToolRegistry
    tool = ToolRegistry.get("scene_graph")
    assert tool is not None
    assert tool.metadata.name == "scene_graph"


def test_scene_graph_blocked_in_mapless_mode():
    """SceneGraphQueryTool must not be available in mapless nav mode.

    SG provides absolute 3D positions for every object, which is equivalent
    to having a complete scene map — incompatible with the mapless constraint.
    """
    from habitat_agent.tools.base import ToolRegistry
    navmesh_tools = {t.metadata.name for t in ToolRegistry.available_for("navmesh", "pointnav")}
    mapless_tools = {t.metadata.name for t in ToolRegistry.available_for("mapless", "pointnav")}
    assert "scene_graph" in navmesh_tools, "scene_graph must be available in navmesh mode"
    assert "scene_graph" not in mapless_tools, "scene_graph must be blocked in mapless mode"
