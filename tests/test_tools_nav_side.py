"""Phase 2 PR 2 — unit tests for the 11 nav-side Tool subclasses.

Covers `ForwardTool`, `TurnTool`, `NavigateTool`, `FindPathTool`,
`SamplePointTool`, `LookTool`, `PanoramaTool`, `DepthAnalyzeTool`,
`TopdownTool`, `UpdateNavStatusTool`, `ExportVideoTool`.

Each test uses a `FakeBridge` that records calls and returns stub
responses, so we can assert on:
  - the exact bridge action name a tool invokes
  - the payload shape (keys + types + derived fields)
  - RoundState mutations (captured_images, last_collided, etc.)
  - error-path behaviour (ToolResult.ok=False when bridge raises)

The most involved tests cover `UpdateNavStatusTool`'s three legacy
behaviours that MUST be preserved: flat-arg coercion, action-chain
auto-injection, and terminal-status synthesis.
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

# Importing the package triggers ToolRegistry.register() for all 11 tools.
import habitat_agent.tools  # noqa: F401,E402  — registration side effect

from habitat_agent.tools.base import (  # noqa: E402
    RoundState,
    ToolContext,
    ToolRegistry,
    ToolResult,
)
from habitat_agent.tools.mapping import TopdownTool  # noqa: E402
from habitat_agent.tools.navigation import (  # noqa: E402
    FindPathTool,
    ForwardTool,
    NavigateTool,
    SamplePointTool,
    TurnTool,
)
from habitat_agent.tools.perception import (  # noqa: E402
    DepthAnalyzeTool,
    LookTool,
    PanoramaTool,
)
from habitat_agent.tools.status import (  # noqa: E402
    ExportVideoTool,
    UpdateNavStatusTool,
)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


class FakeBridge:
    """Records every (action, payload) tuple; returns queued responses."""

    def __init__(self, responses: Dict[str, Any] = None):
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.responses: Dict[str, Any] = responses or {}
        self.raise_next: bool = False
        self.raise_message: str = "fake bridge exploded"

    def call(self, action: str, payload: Dict[str, Any]) -> Any:
        self.calls.append((action, dict(payload)))
        if self.raise_next:
            self.raise_next = False
            raise RuntimeError(self.raise_message)
        return self.responses.get(action, {})


def _make_ctx(
    bridge: FakeBridge = None,
    *,
    nav_mode: str = "navmesh",
    task_type: str = "pointnav",
    loop_id: str = "navloop-abc",
    output_dir: str = "/tmp/artifacts",
) -> ToolContext:
    return ToolContext(
        bridge=bridge or FakeBridge(),
        session_id="s1",
        loop_id=loop_id,
        output_dir=output_dir,
        nav_mode=nav_mode,
        task_type=task_type,
    )


# ---------------------------------------------------------------------------
# ForwardTool
# ---------------------------------------------------------------------------


def test_forward_calls_step_and_capture_with_correct_payload():
    bridge = FakeBridge(
        responses={
            "step_and_capture": {
                "collided": False,
                "position": [1.0, 0.2, 0.5],
                "visuals": {
                    "color_sensor": {"path": "/tmp/artifacts/frame1_color.png"},
                    "depth_sensor": {"path": "/tmp/artifacts/frame1_depth.png"},
                },
            }
        }
    )
    ctx = _make_ctx(bridge)
    result = ForwardTool().execute({"distance_m": 0.5}, ctx)

    assert result.ok
    assert len(bridge.calls) == 1
    action, payload = bridge.calls[0]
    assert action == "step_and_capture"
    assert payload["action"] == "move_forward"
    assert payload["distance"] == 0.5
    assert payload["include_metrics"] is True
    assert payload["output_dir"] == "/tmp/artifacts"
    # RoundState mutations
    assert "/tmp/artifacts/frame1_color.png" in ctx.round_state.captured_images
    assert "/tmp/artifacts/frame1_depth.png" in ctx.round_state.captured_images
    assert ctx.round_state.last_visual_path == "/tmp/artifacts/frame1_color.png"
    assert ctx.round_state.last_collided is False
    assert ctx.round_state.last_movement_action == "move_forward"
    assert ctx.round_state.round_actions == ["forward(0.5m)"]
    # captured_images also copied into ToolResult for the caller
    assert "/tmp/artifacts/frame1_color.png" in result.captured_images


def test_forward_tags_collision_in_round_actions():
    """When the bridge reports collided=True, round_actions entry gets
    a trailing '!' marker so action_history shows the collision."""
    bridge = FakeBridge(
        responses={"step_and_capture": {"collided": True, "visuals": {}}}
    )
    ctx = _make_ctx(bridge)
    ForwardTool().execute({"distance_m": 0.25}, ctx)
    assert ctx.round_state.last_collided is True
    assert ctx.round_state.round_actions == ["forward(0.25m)!"]


def test_forward_bridge_exception_returns_error_result():
    bridge = FakeBridge()
    bridge.raise_next = True
    ctx = _make_ctx(bridge)
    result = ForwardTool().execute({"distance_m": 0.5}, ctx)
    assert result.ok is False
    assert result.error is not None
    assert "fake bridge exploded" in result.error
    # RoundState should NOT be mutated on error (we short-circuited
    # before the collect_images / last_collided / round_actions writes)
    assert ctx.round_state.captured_images == []
    assert ctx.round_state.round_actions == []


# ---------------------------------------------------------------------------
# TurnTool
# ---------------------------------------------------------------------------


def test_turn_left_builds_turn_left_action():
    bridge = FakeBridge(
        responses={"step_and_capture": {"collided": False, "visuals": {}}}
    )
    ctx = _make_ctx(bridge)
    result = TurnTool().execute({"direction": "left", "degrees": 45}, ctx)
    assert result.ok
    action, payload = bridge.calls[0]
    assert action == "step_and_capture"
    assert payload["action"] == "turn_left"
    assert payload["degrees"] == 45
    assert ctx.round_state.last_movement_action == "turn_left"
    assert ctx.round_state.round_actions == ["turn_left(45°)"]


def test_turn_rejects_invalid_direction():
    """Codex round-7 P2 regression lock — defensive enum validation.

    Legacy mcp_server.hab_turn had an explicit `if direction not in
    ("left", "right"): return error` check. The Phase 2 PR 4 cutover
    migrated TurnTool from legacy_executor.py (which had no such
    check, because nav_agent's LLM was strictly prompted) and lost
    the MCP-side defensive validation.

    Without this check, an LLM that hallucinates `direction="up"`
    sends `action="turn_up"` to the bridge, which rejects it with
    a cryptic 'Unknown action' error. Restore the early-return
    error so MCP callers (and any LLM that doesn't strictly follow
    the schema enum) get a clear client-side error."""
    bridge = FakeBridge(
        responses={"step_and_capture": {"collided": False, "visuals": {}}}
    )
    ctx = _make_ctx(bridge)

    result = TurnTool().execute({"direction": "up", "degrees": 360}, ctx)
    assert result.ok is False, (
        "TurnTool accepted direction='up' — bridge will receive "
        "action='turn_up' and fail with an opaque error"
    )
    assert result.error is not None
    assert "left" in result.error.lower()
    assert "right" in result.error.lower()
    # The bridge MUST NOT be called when validation fails
    assert bridge.calls == [], (
        f"bridge was called despite invalid direction: {bridge.calls}"
    )


def test_turn_error_message_includes_invalid_value():
    """Quality-of-life: the error message names the bad value so
    the LLM (or human operator) can self-correct."""
    bridge = FakeBridge()
    ctx = _make_ctx(bridge)
    result = TurnTool().execute({"direction": "around"}, ctx)
    assert result.ok is False
    assert "around" in (result.error or "")


def test_turn_right_builds_turn_right_action():
    bridge = FakeBridge(
        responses={"step_and_capture": {"collided": False, "visuals": {}}}
    )
    ctx = _make_ctx(bridge)
    TurnTool().execute({"direction": "right", "degrees": 10}, ctx)
    action, payload = bridge.calls[0]
    assert payload["action"] == "turn_right"
    assert ctx.round_state.last_movement_action == "turn_right"


# ---------------------------------------------------------------------------
# NavigateTool / FindPathTool / SamplePointTool
# ---------------------------------------------------------------------------


def test_navigate_forwards_goal_and_max_steps():
    bridge = FakeBridge(responses={"navigate_step": {"visuals": {}, "reached": False}})
    ctx = _make_ctx(bridge)
    result = NavigateTool().execute(
        {"x": 3.0, "y": 0.2, "z": 4.0, "max_steps": 5}, ctx
    )
    assert result.ok
    action, payload = bridge.calls[0]
    assert action == "navigate_step"
    assert payload["goal"] == [3.0, 0.2, 4.0]
    assert payload["max_steps"] == 5
    assert payload["include_metrics"] is True


def test_navigate_missing_coordinate_returns_error():
    """KeyError for missing coords becomes ToolResult.ok=False.
    Bridge must NOT be called."""
    bridge = FakeBridge()
    ctx = _make_ctx(bridge)
    result = NavigateTool().execute({"x": 1.0, "y": 2.0}, ctx)  # z missing
    assert result.ok is False
    assert "navigate requires x, y, z" in result.error
    assert bridge.calls == []


def test_find_path_sends_end_coordinate():
    bridge = FakeBridge(
        responses={
            "find_shortest_path": {"waypoints": [], "geodesic_distance": 5.0}
        }
    )
    ctx = _make_ctx(bridge)
    result = FindPathTool().execute({"x": 1.0, "y": 0.2, "z": 2.0}, ctx)
    assert result.ok
    action, payload = bridge.calls[0]
    assert action == "find_shortest_path"
    assert payload == {"end": [1.0, 0.2, 2.0]}
    assert result.body["geodesic_distance"] == 5.0


def test_sample_point_takes_no_args():
    bridge = FakeBridge(
        responses={"sample_navigable_point": {"point": [1.0, 0.2, 1.0]}}
    )
    ctx = _make_ctx(bridge)
    result = SamplePointTool().execute({}, ctx)
    assert result.ok
    action, payload = bridge.calls[0]
    assert action == "sample_navigable_point"
    assert payload == {}


def test_navmesh_only_tools_filtered_in_mapless_mode():
    """Registry filtering honours allowed_nav_modes={"navmesh"} for the
    4 navmesh-only tools."""
    mapless_tools = {t.metadata.name for t in ToolRegistry.available_for("mapless", "pointnav")}
    for name in ("navigate", "find_path", "sample_point", "topdown"):
        assert name not in mapless_tools, f"{name} should be filtered in mapless"

    navmesh_tools = {t.metadata.name for t in ToolRegistry.available_for("navmesh", "pointnav")}
    for name in ("navigate", "find_path", "sample_point", "topdown"):
        assert name in navmesh_tools, f"{name} should be present in navmesh"


# ---------------------------------------------------------------------------
# LookTool / PanoramaTool / DepthAnalyzeTool
# ---------------------------------------------------------------------------


def test_look_captures_images_and_updates_last_visual():
    bridge = FakeBridge(
        responses={
            "get_visuals": {
                "visuals": {
                    "color_sensor": {"path": "/o/look_color.png"},
                    "depth_sensor": {"path": "/o/look_depth.png"},
                }
            }
        }
    )
    ctx = _make_ctx(bridge, output_dir="/o")
    result = LookTool().execute({}, ctx)
    assert result.ok
    assert bridge.calls[0][0] == "get_visuals"
    assert "/o/look_color.png" in ctx.round_state.captured_images
    assert ctx.round_state.last_visual_path == "/o/look_color.png"
    assert ctx.round_state.round_actions == ["look"]


def test_panorama_picks_front_image_when_no_color_sensor_key():
    """Panorama returns an `images` list; front-direction entry becomes
    last_visual_path if no color_sensor key exists."""
    bridge = FakeBridge(
        responses={
            "get_panorama": {
                "images": [
                    {"path": "/o/pano_front.png", "direction": "front"},
                    {"path": "/o/pano_right.png", "direction": "right"},
                    {"path": "/o/pano_back.png", "direction": "back"},
                    {"path": "/o/pano_left.png", "direction": "left"},
                ]
            }
        }
    )
    ctx = _make_ctx(bridge, output_dir="/o")
    result = PanoramaTool().execute({"include_depth": False}, ctx)
    assert result.ok
    action, payload = bridge.calls[0]
    assert action == "get_panorama"
    assert payload["include_depth_analysis"] is False
    assert len(ctx.round_state.captured_images) == 4
    # First image (color priority fallback) becomes last_visual_path
    assert ctx.round_state.last_visual_path == "/o/pano_front.png"
    assert ctx.round_state.round_actions == ["panorama"]


def test_depth_analyze_is_pure_passthrough():
    bridge = FakeBridge(
        responses={
            "analyze_depth": {
                "front_left": {"min": 1.2, "mean": 2.1},
                "front_center": {"min": 0.5, "mean": 1.0},
                "front_right": {"min": 1.8, "mean": 2.5},
            }
        }
    )
    ctx = _make_ctx(bridge)
    result = DepthAnalyzeTool().execute({}, ctx)
    assert result.ok
    assert bridge.calls[0] == ("analyze_depth", {})
    assert result.body["front_center"]["min"] == 0.5
    # depth_analyze does NOT append to round_actions (matches legacy)
    assert ctx.round_state.round_actions == []
    # no images captured
    assert ctx.round_state.captured_images == []


# ---------------------------------------------------------------------------
# TopdownTool
# ---------------------------------------------------------------------------


def test_topdown_with_goal_builds_goal_field():
    bridge = FakeBridge(responses={"get_topdown_map": {"image_path": "/o/top.png"}})
    ctx = _make_ctx(bridge, output_dir="/o")
    result = TopdownTool().execute(
        {"goal_x": 1.0, "goal_y": 0.2, "goal_z": 2.0, "show_path": True}, ctx
    )
    assert result.ok
    action, payload = bridge.calls[0]
    assert action == "get_topdown_map"
    assert payload["goal"] == [1.0, 0.2, 2.0]
    assert payload["show_path"] is True


def test_topdown_without_goal_omits_goal_field():
    bridge = FakeBridge(responses={"get_topdown_map": {}})
    ctx = _make_ctx(bridge)
    TopdownTool().execute({}, ctx)
    _, payload = bridge.calls[0]
    assert "goal" not in payload
    assert "show_path" not in payload


# ---------------------------------------------------------------------------
# UpdateNavStatusTool — the complex one
# ---------------------------------------------------------------------------


def test_update_nav_status_normalises_flat_patch_args():
    """LLM sends patch fields flat; executor reconstructs patch dict."""
    bridge = FakeBridge(
        responses={
            "update_nav_loop_status": {"nav_status": {"state_version": 42}}
        }
    )
    ctx = _make_ctx(bridge)
    UpdateNavStatusTool().execute(
        {
            "status": "in_progress",
            "total_steps": 5,
            "nav_phase": "navigating",
        },
        ctx,
    )
    action, bridge_payload = bridge.calls[0]
    assert action == "update_nav_loop_status"
    patch = bridge_payload["patch"]
    assert patch["status"] == "in_progress"
    assert patch["total_steps"] == 5
    assert patch["nav_phase"] == "navigating"


def test_update_nav_status_accepts_nested_patch_arg():
    """If the LLM does the right thing and nests under 'patch', that
    shape wins."""
    bridge = FakeBridge(
        responses={"update_nav_loop_status": {"nav_status": {"state_version": 1}}}
    )
    ctx = _make_ctx(bridge)
    UpdateNavStatusTool().execute(
        {"patch": {"status": "in_progress", "total_steps": 3}}, ctx
    )
    _, bridge_payload = bridge.calls[0]
    assert bridge_payload["patch"] == {"status": "in_progress", "total_steps": 3}


def test_update_nav_status_injects_last_visual_from_round_state():
    """Mapless mode auto-inject: if patch has no last_visual but
    RoundState does, inject it."""
    bridge = FakeBridge(
        responses={"update_nav_loop_status": {"nav_status": {"state_version": 1}}}
    )
    ctx = _make_ctx(bridge, nav_mode="mapless")
    ctx.round_state.last_visual_path = "/o/look_123.png"
    UpdateNavStatusTool().execute({"total_steps": 1}, ctx)
    _, bridge_payload = bridge.calls[0]
    assert bridge_payload["patch"]["last_visual"] == {"path": "/o/look_123.png"}


def test_update_nav_status_respects_explicit_last_visual():
    """If the LLM supplied last_visual, do NOT overwrite it."""
    bridge = FakeBridge(
        responses={"update_nav_loop_status": {"nav_status": {"state_version": 1}}}
    )
    ctx = _make_ctx(bridge, nav_mode="mapless")
    ctx.round_state.last_visual_path = "/o/auto.png"
    UpdateNavStatusTool().execute(
        {"total_steps": 1, "last_visual": {"path": "/o/explicit.png"}}, ctx
    )
    _, bridge_payload = bridge.calls[0]
    assert bridge_payload["patch"]["last_visual"] == {"path": "/o/explicit.png"}


def test_update_nav_status_injects_action_chain_into_history():
    """action_history_append entries get collided + action auto-filled
    from the accumulated round_actions."""
    bridge = FakeBridge(
        responses={"update_nav_loop_status": {"nav_status": {"state_version": 2}}}
    )
    ctx = _make_ctx(bridge)
    ctx.round_state.round_actions = ["forward(0.5m)", "turn_left(45°)", "look"]
    ctx.round_state.last_collided = True

    UpdateNavStatusTool().execute(
        {
            "total_steps": 5,
            "action_history_append": [
                {"perception": "A wall", "analysis": "blocked", "decision": "turn left"}
            ],
        },
        ctx,
    )
    _, bridge_payload = bridge.calls[0]
    entry = bridge_payload["patch"]["action_history_append"][0]
    assert entry["perception"] == "A wall"
    assert entry["collided"] is True  # auto-injected
    assert entry["action"] == "forward(0.5m) → turn_left(45°) → look"  # auto-injected


def test_update_nav_status_clears_round_actions_after_call():
    """round_actions is cleared after update_nav_status so the next
    cycle starts fresh (matches legacy `self._round_actions = []`)."""
    bridge = FakeBridge(
        responses={"update_nav_loop_status": {"nav_status": {"state_version": 3}}}
    )
    ctx = _make_ctx(bridge)
    ctx.round_state.round_actions = ["forward(0.5m)"]
    UpdateNavStatusTool().execute({"total_steps": 1}, ctx)
    assert ctx.round_state.round_actions == []


def test_update_nav_status_synthesizes_terminal_history():
    """When status is terminal and the LLM didn't provide
    action_history_append, synthesize a minimal one from `finding`
    so the bridge's mapless validation accepts the patch."""
    bridge = FakeBridge(
        responses={"update_nav_loop_status": {"nav_status": {"state_version": 10}}}
    )
    ctx = _make_ctx(bridge, nav_mode="mapless")
    ctx.round_state.round_actions = ["forward(0.5m)", "look"]
    ctx.round_state.last_collided = False

    UpdateNavStatusTool().execute(
        {"status": "reached", "finding": "Target bedroom visible"}, ctx
    )
    _, bridge_payload = bridge.calls[0]
    ah = bridge_payload["patch"]["action_history_append"]
    assert isinstance(ah, list)
    assert len(ah) == 1
    entry = ah[0]
    assert entry["perception"] == "Target bedroom visible"
    assert entry["analysis"] == "Terminal status: reached"
    assert entry["decision"] == "Set status=reached"
    assert entry["action"] == "forward(0.5m) → look"
    assert entry["collided"] is False


def test_update_nav_status_does_not_overwrite_existing_terminal_history():
    """If the LLM *did* provide action_history_append on a terminal
    patch, we must not clobber it."""
    bridge = FakeBridge(
        responses={"update_nav_loop_status": {"nav_status": {"state_version": 9}}}
    )
    ctx = _make_ctx(bridge)
    llm_entries = [
        {"perception": "Saw exit", "analysis": "Done", "decision": "Stop"}
    ]
    UpdateNavStatusTool().execute(
        {
            "status": "reached",
            "finding": "Exit visible",
            "action_history_append": llm_entries,
        },
        ctx,
    )
    _, bridge_payload = bridge.calls[0]
    ah = bridge_payload["patch"]["action_history_append"]
    assert len(ah) == 1
    # The LLM's perception text survives (synthesis would have replaced it
    # with the finding). Note that the injection pass still adds
    # collided + action defaults to the existing entry, which is expected.
    assert ah[0]["perception"] == "Saw exit"


def test_update_nav_status_propagates_state_version_from_response():
    """The response's nav_status.state_version is written back into
    ctx.state_version_ref[0] so the next round's expected_version
    matches."""
    bridge = FakeBridge(
        responses={"update_nav_loop_status": {"nav_status": {"state_version": 77}}}
    )
    ctx = _make_ctx(bridge)
    ctx.state_version_ref[0] = 5
    result = UpdateNavStatusTool().execute({"total_steps": 1}, ctx)
    assert result.ok
    _, bridge_payload = bridge.calls[0]
    assert bridge_payload["expected_version"] == 5  # pre-call value
    assert ctx.state_version_ref[0] == 77  # post-call value
    assert result.body["state_version"] == 77


def test_update_nav_status_bridge_exception_still_clears_round_actions():
    """Legacy semantics: round_actions is cleared BEFORE the bridge
    call, so a bridge failure does NOT leak stale actions into the
    next cycle."""
    bridge = FakeBridge()
    bridge.raise_next = True
    ctx = _make_ctx(bridge)
    ctx.round_state.round_actions = ["forward(0.5m)"]
    result = UpdateNavStatusTool().execute({"total_steps": 1}, ctx)
    assert result.ok is False
    assert ctx.round_state.round_actions == []


# ---------------------------------------------------------------------------
# ExportVideoTool
# ---------------------------------------------------------------------------


def test_update_nav_status_metadata_marks_mcp_invisible():
    """Codex P1 regression lock.

    UpdateNavStatusTool is the only nav-side tool that requires
    state only available inside a NavAgent subprocess (loop_id +
    state_version_ref). It MUST be excluded from dynamic MCP
    registration so MCP top-level callers don't see a tool that
    is guaranteed to fail with a cryptic 'loop_id must be non-empty'
    error from the bridge."""
    assert UpdateNavStatusTool.metadata.mcp_visible is False, (
        "update_nav_status must NOT be exposed via MCP — it needs "
        "ctx.loop_id which top-level MCP callers cannot provide"
    )


def test_other_nav_side_tools_remain_mcp_visible():
    """Sanity check: only update_nav_status flips the flag.
    All other nav-side tools (forward, turn, look, panorama, etc.)
    work fine with the MCP top-level context, so they keep the
    default mcp_visible=True."""
    for tool_cls in (
        ForwardTool, TurnTool, NavigateTool, FindPathTool, SamplePointTool,
        LookTool, PanoramaTool, DepthAnalyzeTool, TopdownTool, ExportVideoTool,
    ):
        assert tool_cls.metadata.mcp_visible is True, (
            f"{tool_cls.__name__} should be MCP-visible"
        )


def test_export_video_sends_visual_payload():
    bridge = FakeBridge(
        responses={"export_video_trace": {"video_path": "/o/trace.mp4"}}
    )
    ctx = _make_ctx(bridge, output_dir="/o")
    result = ExportVideoTool().execute({}, ctx)
    assert result.ok
    action, payload = bridge.calls[0]
    assert action == "export_video_trace"
    assert payload["output_dir"] == "/o"
    assert result.body["video_path"] == "/o/trace.mp4"


# ---------------------------------------------------------------------------
# End-to-end cycle sanity: multiple tool calls accumulate state correctly
# ---------------------------------------------------------------------------


def test_multi_step_cycle_accumulates_state_like_legacy():
    """Simulate forward → turn → forward → update_nav_status and verify
    the action chain, collision tracking, and state clearing match the
    legacy ToolExecutor behaviour."""
    bridge = FakeBridge(
        responses={
            "step_and_capture": {
                "collided": False,
                "visuals": {"color_sensor": {"path": "/o/f.png"}},
            },
            "update_nav_loop_status": {"nav_status": {"state_version": 42}},
        }
    )
    ctx = _make_ctx(bridge)

    # Step 1: forward
    ForwardTool().execute({"distance_m": 0.5}, ctx)
    assert ctx.round_state.round_actions == ["forward(0.5m)"]
    assert ctx.round_state.last_movement_action == "move_forward"

    # Step 2: turn right 90
    TurnTool().execute({"direction": "right", "degrees": 90}, ctx)
    assert ctx.round_state.round_actions == ["forward(0.5m)", "turn_right(90°)"]
    assert ctx.round_state.last_movement_action == "turn_right"

    # Step 3: forward again; simulate collision this time
    bridge.responses["step_and_capture"]["collided"] = True
    ForwardTool().execute({"distance_m": 0.25}, ctx)
    assert ctx.round_state.round_actions == [
        "forward(0.5m)",
        "turn_right(90°)",
        "forward(0.25m)!",  # collision marker
    ]
    assert ctx.round_state.last_collided is True

    # Step 4: update_nav_status — chain gets injected, then cleared
    UpdateNavStatusTool().execute(
        {
            "total_steps": 3,
            "collisions": 1,
            "action_history_append": [
                {"perception": "p", "analysis": "a", "decision": "d"}
            ],
        },
        ctx,
    )
    _, payload = bridge.calls[-1]
    entry = payload["patch"]["action_history_append"][0]
    assert entry["action"] == "forward(0.5m) → turn_right(90°) → forward(0.25m)!"
    assert entry["collided"] is True
    # Chain cleared for next cycle
    assert ctx.round_state.round_actions == []
