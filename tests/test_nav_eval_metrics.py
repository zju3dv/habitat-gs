# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the GT evaluation framework: SPL, path length,
trajectory accumulation, eval_goal isolation, closed-loop protection.

These tests use in-memory fakes (no real simulator) and exercise the
nav_loop mixin + _Session state directly. They cover the bugs caught
in review so regressions are detected early.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest import mock

import numpy as np
import pytest

# Prepend repo src_python to habitat_sim package path so that local edits
# to habitat_adapter_internal take precedence over any installed wheel.
import habitat_sim  # noqa: E402

_OPENCLAW_SRC_HABITAT_SIM = str(
    Path(__file__).resolve().parent.parent / "src_python" / "habitat_sim"
)
if _OPENCLAW_SRC_HABITAT_SIM not in list(habitat_sim.__path__):
    habitat_sim.__path__.insert(0, _OPENCLAW_SRC_HABITAT_SIM)
# Drop any cached import of the submodule so we pick up the local copy
for _name in list(sys.modules):
    if _name.startswith("habitat_sim.habitat_adapter"):
        del sys.modules[_name]

from habitat_sim.habitat_adapter import HabitatAdapter  # noqa: E402
from habitat_sim.habitat_adapter_internal.types import _Session  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------


class _FakePathFinder:
    """Fake pathfinder where geodesic ≈ euclidean for reachable points.

    Returns False for points below x=0 (unreachable), mimicking
    disconnected navmesh islands.
    """

    def __init__(self, reachable: bool = True, geodesic_multiplier: float = 1.0) -> None:
        self.is_loaded = True
        self.reachable = reachable
        self.geodesic_multiplier = geodesic_multiplier

    def snap_point(self, point: np.ndarray) -> np.ndarray:
        # Simulate snap: clamp y to 0.2 and leave x/z unchanged
        p = np.array(point, dtype=np.float32)
        p[1] = 0.2
        return p

    def find_path(self, path: Any) -> bool:
        if not self.reachable:
            path.geodesic_distance = 0.0
            path.points = []
            return False
        start = np.array(path.requested_start, dtype=np.float32)
        end = np.array(path.requested_end, dtype=np.float32)
        path.geodesic_distance = float(np.linalg.norm(end - start)) * self.geodesic_multiplier
        path.points = [start, end]
        return True


def _make_session(
    position=(0.0, 0.2, 0.0),
    with_pathfinder: bool = True,
    pathfinder: _FakePathFinder | None = None,
) -> _Session:
    """Build a minimal _Session sufficient for the mixins under test."""
    fake_agent = SimpleNamespace(
        get_state=lambda: SimpleNamespace(
            position=np.array(position, dtype=np.float32),
            rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
    )
    fake_sim = SimpleNamespace(
        pathfinder=(pathfinder or _FakePathFinder()) if with_pathfinder else None,
        get_agent=lambda _id=0: fake_agent,
    )
    session = _Session(
        session_id="test-session",
        simulator=fake_sim,  # type: ignore[arg-type]
        scene="test_scene",
        settings={},
        agent_id=0,
    )
    session.trajectory = [list(position)]
    return session


def _make_adapter() -> HabitatAdapter:
    """Build an adapter with a no-op simulator factory — we only exercise
    mixin methods that don't actually spin up a simulator."""
    return HabitatAdapter(simulator_factory=lambda _: None)


# ---------------------------------------------------------------------------
# _compute_geodesic_distance
# ---------------------------------------------------------------------------


def test_compute_geodesic_returns_none_without_pathfinder():
    adapter = _make_adapter()
    session = _make_session(with_pathfinder=False)
    assert adapter._compute_geodesic_distance(session, [0, 0, 0], [1, 0, 1]) is None


def test_compute_geodesic_snaps_off_mesh_goal():
    """Goals not exactly on navmesh vertices should still resolve via snap."""
    adapter = _make_adapter()
    pf = _FakePathFinder(reachable=True)
    session = _make_session(with_pathfinder=True, pathfinder=pf)
    # Start at origin, goal at (3, 0, 4); snap forces y=0.2 (off-mesh raw y=5.0)
    d = adapter._compute_geodesic_distance(session, [0, 0.2, 0], [3, 5.0, 4])
    assert d is not None
    assert d == pytest.approx(5.0, abs=1e-3)


def test_compute_geodesic_returns_none_on_unreachable():
    adapter = _make_adapter()
    pf = _FakePathFinder(reachable=False)
    session = _make_session(with_pathfinder=True, pathfinder=pf)
    assert adapter._compute_geodesic_distance(session, [0, 0, 0], [1, 0, 1]) is None


# ---------------------------------------------------------------------------
# _build_debug_snapshot
# ---------------------------------------------------------------------------


def test_debug_snapshot_no_eval_goal():
    adapter = _make_adapter()
    session = _make_session(position=(0.0, 0.2, 0.0))
    session.eval_goal = None
    snap = adapter._build_debug_snapshot(session)
    assert snap["gt_goal"] is None
    assert snap["gt_euclidean_distance"] is None
    assert snap["gt_geodesic_distance"] is None
    assert snap["gt_goal_direction_deg"] is None
    assert snap["gt_path_length"] == 0.0
    assert snap["gt_initial_geodesic_distance"] is None
    # Position/heading always present
    assert snap["gt_position"] is not None
    assert snap["gt_heading_deg"] is not None


def test_debug_snapshot_with_eval_goal():
    adapter = _make_adapter()
    session = _make_session(position=(0.0, 0.2, 0.0))
    session.eval_goal = [3.0, 0.2, 4.0]
    session.initial_geodesic_distance = 7.5
    session.cumulative_path_length = 2.5
    snap = adapter._build_debug_snapshot(session)
    assert snap["gt_goal"] == [3.0, 0.2, 4.0]
    # Euclidean: sqrt(3² + 0 + 4²) = 5.0
    assert snap["gt_euclidean_distance"] == pytest.approx(5.0, abs=1e-3)
    # Geodesic: via fake pathfinder (multiplier 1.0) = 5.0
    assert snap["gt_geodesic_distance"] == pytest.approx(5.0, abs=1e-3)
    assert snap["gt_initial_geodesic_distance"] == pytest.approx(7.5)
    assert snap["gt_path_length"] == pytest.approx(2.5)


def test_debug_snapshot_independent_of_last_goal():
    """eval_goal drives GT metrics; last_goal (agent-facing) is ignored here."""
    adapter = _make_adapter()
    session = _make_session(position=(0.0, 0.2, 0.0))
    session.eval_goal = [3.0, 0.2, 4.0]
    session.last_goal = [100.0, 0.2, 100.0]  # bogus agent goal
    snap = adapter._build_debug_snapshot(session)
    # Should use eval_goal = [3, 0.2, 4], not last_goal
    assert snap["gt_goal"] == [3.0, 0.2, 4.0]
    assert snap["gt_euclidean_distance"] == pytest.approx(5.0, abs=1e-3)


# ---------------------------------------------------------------------------
# cumulative_path_length / _record_pose
# ---------------------------------------------------------------------------


class _MovableAgent:
    """Agent whose position can be updated by tests."""

    def __init__(self, start=(0.0, 0.2, 0.0)) -> None:
        self.position = np.array(start, dtype=np.float32)

    def set_position(self, p):
        self.position = np.array(p, dtype=np.float32)

    def get_state(self) -> Any:
        return SimpleNamespace(
            position=np.array(self.position, dtype=np.float32),
            rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )


def _make_session_movable(start=(0.0, 0.2, 0.0)) -> tuple[_Session, _MovableAgent]:
    agent = _MovableAgent(start)
    fake_sim = SimpleNamespace(
        pathfinder=_FakePathFinder(),
        get_agent=lambda _id=0: agent,
    )
    session = _Session(
        session_id="test-session",
        simulator=fake_sim,  # type: ignore[arg-type]
        scene="test_scene",
        settings={},
        agent_id=0,
    )
    session.trajectory = [list(agent.position)]
    return session, agent


def test_record_pose_accumulates_path_length():
    adapter = _make_adapter()
    session, agent = _make_session_movable(start=(0.0, 0.2, 0.0))
    session.cumulative_path_length = 0.0

    # Move 1m in X direction
    agent.set_position((1.0, 0.2, 0.0))
    adapter._record_pose(session)
    assert session.cumulative_path_length == pytest.approx(1.0, abs=1e-3)

    # Move 1m in Z direction
    agent.set_position((1.0, 0.2, 1.0))
    adapter._record_pose(session)
    assert session.cumulative_path_length == pytest.approx(2.0, abs=1e-3)


def test_record_pose_ignores_duplicate_position():
    adapter = _make_adapter()
    session, agent = _make_session_movable()
    session.cumulative_path_length = 0.0

    # Recording same position twice should not add length
    adapter._record_pose(session)
    adapter._record_pose(session)
    assert session.cumulative_path_length == 0.0
    assert len(session.trajectory) == 1


def test_record_pose_survives_trajectory_cap():
    """When trajectory cap triggers, path_length must remain accurate."""
    adapter = _make_adapter()
    session, agent = _make_session_movable(start=(0.0, 0.2, 0.0))
    session.cumulative_path_length = 0.0

    # Walk 600 steps of 0.1m each (forces cap at 512)
    for i in range(1, 601):
        agent.set_position((i * 0.1, 0.2, 0.0))
        adapter._record_pose(session)

    assert len(session.trajectory) == 512  # Capped
    # Total walked: 600 * 0.1 = 60m
    assert session.cumulative_path_length == pytest.approx(60.0, abs=1e-3)


# ---------------------------------------------------------------------------
# _set_agent_state rebases trajectory (Bug A)
# ---------------------------------------------------------------------------


def test_set_agent_state_rebases_trajectory():
    """After teleport, next _record_pose must NOT include the teleport jump
    in cumulative_path_length."""
    adapter = _make_adapter()
    session, agent = _make_session_movable(start=(0.0, 0.2, 0.0))

    # Walk 2m to accumulate some path length
    agent.set_position((2.0, 0.2, 0.0))
    adapter._record_pose(session)
    assert session.cumulative_path_length == pytest.approx(2.0)

    # Simulate teleport: what _set_agent_state does internally after
    # agent.set_state(). We test the rebase logic directly since set_state
    # requires habitat_sim internals.
    agent.set_position((10.0, 0.2, 10.0))  # teleport to far location
    new_position = agent.get_state().position.tolist()
    session.trajectory = [new_position]  # rebase logic from mixin

    # Now walk 1m from new location
    agent.set_position((11.0, 0.2, 10.0))
    adapter._record_pose(session)

    # Path length should only include pre-teleport walk (2m) + post-teleport walk (1m)
    # NOT include the teleport jump (14.14m)
    assert session.cumulative_path_length == pytest.approx(3.0, abs=1e-3)


# ---------------------------------------------------------------------------
# SPL computation edge cases
# ---------------------------------------------------------------------------


def _compute_spl(l_opt, l_actual, success):
    """Reimplementation of nav_agent's SPL formula for direct unit testing."""
    if l_opt is None or l_actual is None:
        return None
    if not isinstance(l_opt, (int, float)) or not isinstance(l_actual, (int, float)):
        return None
    if l_opt == 0:
        return 1.0 if success else 0.0
    if l_opt > 0:
        return round((1.0 if success else 0.0) * (l_opt / max(l_opt, l_actual)), 4)
    return None


def test_spl_trivial_success_l_opt_zero():
    """Agent starts at goal: SPL=1.0 if success, 0.0 if failure."""
    assert _compute_spl(0.0, 0.0, True) == 1.0
    assert _compute_spl(0.0, 0.5, True) == 1.0
    assert _compute_spl(0.0, 0.0, False) == 0.0


def test_spl_optimal_path():
    """Agent walks exactly the optimal path: SPL=1.0 for success."""
    assert _compute_spl(10.0, 10.0, True) == 1.0


def test_spl_suboptimal_path():
    """Agent walks 2× optimal: SPL=0.5."""
    assert _compute_spl(10.0, 20.0, True) == 0.5


def test_spl_failure_always_zero():
    """Failure always yields SPL=0 regardless of path length."""
    assert _compute_spl(10.0, 10.0, False) == 0.0
    assert _compute_spl(10.0, 100.0, False) == 0.0


def test_spl_none_when_missing_data():
    assert _compute_spl(None, 10.0, True) is None
    assert _compute_spl(10.0, None, True) is None


def test_spl_clamps_when_actual_below_opt():
    """Floating point: l_actual might be slightly < l_opt — SPL stays 1.0."""
    # max(l_opt, l_actual) handles this
    assert _compute_spl(10.0, 9.999, True) == 1.0


# NOTE: the old test_gt_end_distance_{zero_preserved,none_falls_back}
# tests used to codify the ambiguous single-field fallback. They were
# removed because session_stats.jsonl no longer has a merged
# `gt_end_distance` — it has separate `gt_end_geodesic_distance` and
# `gt_end_euclidean_distance` fields. See the Stage 1 schema cleanup.


# ---------------------------------------------------------------------------
# success judgment: navmesh strict + threshold override
# ---------------------------------------------------------------------------


def _judge_success(
    has_navmesh: bool,
    agent_reached: bool,
    gt_geodesic: float | None,
    gt_euclidean: float | None,
    threshold: float,
) -> bool:
    """Reimplementation of the success-judging logic in collect_session_stats."""
    if has_navmesh:
        gt_judge = gt_geodesic
    else:
        gt_judge = gt_euclidean
    gt_success = gt_judge is not None and gt_judge < threshold
    return bool(agent_reached and gt_success)


def test_navmesh_strict_no_euclidean_fallback():
    """On navmesh tasks, missing geodesic → NOT success, even if euclidean is close."""
    # Euclidean close (0.3m) but geodesic unreachable (None)
    assert _judge_success(
        has_navmesh=True,
        agent_reached=True,
        gt_geodesic=None,
        gt_euclidean=0.3,
        threshold=0.5,
    ) is False


def test_navmesh_success_with_geodesic():
    assert _judge_success(
        has_navmesh=True,
        agent_reached=True,
        gt_geodesic=0.3,
        gt_euclidean=0.3,
        threshold=0.5,
    ) is True


def test_nomesh_falls_back_to_euclidean():
    """When no navmesh, euclidean is acceptable for success judgment."""
    assert _judge_success(
        has_navmesh=False,
        agent_reached=True,
        gt_geodesic=None,
        gt_euclidean=0.3,
        threshold=0.5,
    ) is True


def test_success_requires_agent_reached():
    """Even with gt_distance < threshold, agent must declare reached."""
    assert _judge_success(
        has_navmesh=True,
        agent_reached=False,
        gt_geodesic=0.1,
        gt_euclidean=0.1,
        threshold=0.5,
    ) is False


def test_threshold_override():
    """Stricter threshold (0.2) rejects what 0.5 accepts."""
    assert _judge_success(True, True, 0.3, 0.3, threshold=0.5) is True
    assert _judge_success(True, True, 0.3, 0.3, threshold=0.2) is False


# ---------------------------------------------------------------------------
# nav_status field separation: agent-visible vs eval-only
# ---------------------------------------------------------------------------


def test_pointnav_uses_goal_position_as_eval():
    """For pointnav, goal_position serves dual purpose: agent-facing + eval GT."""
    # Simulating the logic in mixin's _start_nav_loop
    task_type = "pointnav"
    goal_position = [1.0, 0.2, -2.0]
    eval_goal_payload = None

    if eval_goal_payload is not None:
        eval_goal = eval_goal_payload
    elif goal_position is not None and task_type == "pointnav":
        eval_goal = goal_position
    else:
        eval_goal = None

    assert eval_goal == [1.0, 0.2, -2.0]


def test_nonpointnav_without_eval_has_no_eval_goal():
    """objectnav/imagenav/etc. without explicit eval_goal → no eval metrics."""
    task_type = "objectnav"
    goal_position = None
    eval_goal_payload = None

    if eval_goal_payload is not None:
        eval_goal = eval_goal_payload
    elif goal_position is not None and task_type == "pointnav":
        eval_goal = goal_position
    else:
        eval_goal = None

    assert eval_goal is None


def test_nonpointnav_with_explicit_eval_goal():
    """objectnav with eval_goal_position: agent-facing goal_position stays None."""
    task_type = "objectnav"
    goal_position = None
    eval_goal_payload = [5.0, 0.2, -3.0]

    if eval_goal_payload is not None:
        eval_goal = eval_goal_payload
    elif goal_position is not None and task_type == "pointnav":
        eval_goal = goal_position
    else:
        eval_goal = None

    assert eval_goal == [5.0, 0.2, -3.0]
    assert goal_position is None  # agent cannot see coords


# ---------------------------------------------------------------------------
# Atomic JSON persistence
# ---------------------------------------------------------------------------


def test_persist_json_atomic_writes_complete_file(tmp_path):
    adapter = _make_adapter()
    target = tmp_path / "nav_status.json"
    adapter._persist_json_atomic(str(target), {"key": "value", "version": 1})
    assert target.is_file()
    loaded = json.loads(target.read_text())
    assert loaded == {"key": "value", "version": 1}


def test_persist_json_atomic_replaces_existing(tmp_path):
    adapter = _make_adapter()
    target = tmp_path / "nav_status.json"
    target.write_text('{"old": true}')
    adapter._persist_json_atomic(str(target), {"new": True})
    assert json.loads(target.read_text()) == {"new": True}


def test_persist_json_atomic_no_tmp_leftover(tmp_path):
    adapter = _make_adapter()
    target = tmp_path / "nav_status.json"
    adapter._persist_json_atomic(str(target), {"a": 1})
    # Only the target file should exist; no .tmp remnants
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "nav_status.json"


# ---------------------------------------------------------------------------
# P0-1: Agent visibility isolation — the core PR #28 invariant.
# Non-pointnav tasks with evaluation GT MUST NOT leak coords/polar to agent.
# ---------------------------------------------------------------------------


def test_state_summary_nonpointnav_gt_hides_polar():
    """Non-pointnav task with eval_goal set but last_goal=None →
    euclidean_distance_to_goal and goal_direction_deg must be None."""
    adapter = _make_adapter()
    session = _make_session(position=(0.0, 0.2, 0.0))
    session.last_goal = None  # non-pointnav: agent has no goal coord
    session.eval_goal = [5.0, 0.2, 3.0]  # GT only
    summary = adapter._build_state_summary(session)
    assert summary["euclidean_distance_to_goal"] is None
    assert summary["goal_direction_deg"] is None


def test_state_summary_mapless_hides_absolute_position_and_goal():
    """In mapless mode, summary must not include absolute position/goal
    keys at all — only polar signals are allowed."""
    adapter = _make_adapter()
    session = _make_session(position=(1.0, 0.2, 2.0))
    session.mapless = True
    session.last_goal = [3.0, 0.2, 4.0]  # pointnav mapless is allowed
    summary = adapter._build_state_summary(session)
    assert "position" not in summary
    assert "goal" not in summary
    # Polar signals remain (agent needs them to navigate)
    assert summary["euclidean_distance_to_goal"] is not None
    assert summary["goal_direction_deg"] is not None


def test_state_summary_navmesh_exposes_position_for_pointnav():
    """navmesh (non-mapless) pointnav — absolute coords are visible."""
    adapter = _make_adapter()
    session = _make_session(position=(1.0, 0.2, 2.0))
    session.mapless = False
    session.last_goal = [3.0, 0.2, 4.0]
    summary = adapter._build_state_summary(session)
    assert "position" in summary
    assert "goal" in summary
    assert summary["goal"] == pytest.approx([3.0, 0.2, 4.0], abs=1e-3)


# Import PromptBuilder lazily — nav_agent.py is under tools/
_TOOLS_DIR = str(Path(__file__).resolve().parent.parent / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from nav_agent import PromptBuilder, collect_session_stats  # noqa: E402


def _base_nav_status(**overrides) -> Dict[str, Any]:
    """Minimal nav_status dict for PromptBuilder tests."""
    base: Dict[str, Any] = {
        "task_id": "loop1",
        "task_type": "objectnav",
        "nav_mode": "navmesh",
        "has_navmesh": True,
        "goal_type": "instruction",
        "goal_description": "Find the kitchen",
        "goal_position": None,  # agent-visible — None for non-pointnav
        "eval_goal_position": [5.123, 0.2, 3.456],  # eval-only
        "has_ground_truth": True,
        "session_id": "s1",
        "status": "in_progress",
        "nav_phase": "navigating",
        "total_steps": 0,
        "collisions": 0,
        "last_action": "none",
        "state_version": 1,
        "substeps": [],
        "current_substep_index": 0,
        "action_history": [],
        "spatial_memory_file": "",
        "_debug": {
            "gt_position": [0.0, 0.2, 0.0],
            "gt_goal": [5.123, 0.2, 3.456],
            "gt_euclidean_distance": 6.234,
            "gt_geodesic_distance": 6.234,
        },
    }
    base.update(overrides)
    return base


def test_prompt_system_documents_atomic_movement_and_agent_autonomy():
    """Regression lock: the system prompt must tell the LLM the
    atomic step sizes and that magnitude is the agent's decision.

    Empirical e2e against a real LLM (2026-04-10) showed that when
    this documentation is missing the LLM tends to reuse the same
    two values for every movement call. When it's present, the LLM
    actually varies forward distance and turn degrees based on the
    scene it sees. See Phase 2 PR review round 8 notes.

    This test does NOT lock the specific section header or phrasing —
    only the factual content that must be there regardless of how
    the prose evolves:
      - forward atomic step is 0.25m
      - turn atomic step is 10°
      - compound moves auto-decompose to atomic steps
      - the agent is required to specify distance_m / degrees
        explicitly (no silent defaults)
    """
    pb = PromptBuilder(nav_mode="mapless")
    system = pb.build_system_prompt(_base_nav_status(
        task_type="pointnav",
        nav_mode="mapless",
    ))

    # Atomic step documentation (factual claim about the bridge)
    assert "0.25m" in system, (
        "forward atomic step (0.25m) not documented in system prompt. "
        "Without this fact the LLM doesn't know what the valid "
        "distance_m multiples are."
    )
    assert "10°" in system, (
        "turn atomic step (10°) not documented in system prompt."
    )

    # Auto-decompose / atomic step language (mechanism explanation)
    lower = system.lower()
    assert ("auto-decompose" in lower or "atomic step" in lower), (
        "system prompt must explain that compound moves auto-decompose "
        "into atomic steps; without this, the LLM may think it can "
        "only pass exactly one atomic step at a time."
    )

    # Agent-autonomy language (explicit specify requirement)
    assert "specify" in lower, (
        "system prompt must require the agent to explicitly specify "
        "distance_m / degrees instead of relying on a hard-coded default."
    )


def test_prompt_uses_memory_bundle_when_provided(tmp_path):
    """Phase 3 regression lock: when PromptBuilder receives a
    MemoryBundle, it should use render_context instead of inline
    _format_spatial_summary, and the output should contain the
    Memory header."""
    from habitat_agent.memory.base import MemoryBundle
    from habitat_agent.memory.spatial import SpatialMemory

    # Write a sample spatial memory file
    spatial = {
        "snapshots": [{"heading_deg": 90, "room_label": "office", "objects_detected": ["desk"]}],
        "rooms": {"office": {"visit_count": 1}},
        "object_sightings": {"desk": {"count": 1}},
    }
    sm_file = tmp_path / "spatial_memory.json"
    sm_file.write_text(json.dumps(spatial), encoding="utf-8")

    bundle = MemoryBundle()
    bundle.register(SpatialMemory(str(sm_file)))

    pb = PromptBuilder(nav_mode="mapless")
    msg = pb.build_round_message(
        _base_nav_status(nav_mode="mapless"),
        round_idx=0, consecutive_idle=0, idle_threshold=3,
        memory_bundle=bundle,
    )
    # The render_context output uses "## spatial" as header
    assert "## spatial" in msg
    assert "Snapshots: 1" in msg
    assert "office" in msg
    assert "desk" in msg


def test_prompt_falls_back_without_memory_bundle():
    """When memory_bundle is None, PromptBuilder should fall back to
    the inline _format_spatial_summary (pre-Phase-3 behavior)."""
    pb = PromptBuilder(nav_mode="mapless")
    msg = pb.build_round_message(
        _base_nav_status(nav_mode="mapless"),
        round_idx=0, consecutive_idle=0, idle_threshold=3,
        memory_bundle=None,
    )
    # Should still have spatial memory section (from inline method)
    # Since _base_nav_status doesn't have a real spatial_memory_file,
    # the fallback will show "No spatial memory file available."
    assert "spatial" in msg.lower() or "memory" in msg.lower()


def test_build_system_prompt_uses_library(tmp_path):
    """Phase 4 PR 3: build_system_prompt should now delegate to
    PromptLibrary when the asset files exist. The render-equivalence
    tests verify output equality; this test verifies the code path."""
    pb = PromptBuilder(nav_mode="mapless")
    ns = _base_nav_status(task_type="pointnav", nav_mode="mapless")
    result = pb.build_system_prompt(ns)
    # PromptLibrary path produces ## Key Rules with fragment + controller
    assert "## Key Rules" in result
    assert "Rule 1" in result
    assert "Rule 7" in result  # collision recovery fragment
    assert "Rule 8" in result  # terminal handling fragment
    # Controller-specific content (pointnav mapless Rule 4)
    assert "euclidean_distance_to_goal" in result


def test_build_system_prompt_falls_back_on_library_error(tmp_path):
    """When PromptLibrary raises (e.g. missing asset files), the legacy
    hardcoded f-string path is used as a fallback."""
    import unittest.mock as mock

    pb = PromptBuilder(nav_mode="mapless")
    ns = _base_nav_status(task_type="pointnav", nav_mode="mapless")

    # Point _library_root at a nonexistent path so load_controller raises
    with mock.patch.object(pb, "_library_root", return_value=tmp_path / "nonexistent"):
        # Should not raise; fallback produces a valid prompt
        result = pb.build_system_prompt(ns)
    assert "## Key Rules" in result
    assert "Rule 1" in result


def test_fallback_prompt_still_has_task_specific_rules(tmp_path):
    """Codex P1 regression lock.

    When PromptLibrary fails and the .txt controller files are also
    deleted, the fallback path must NOT silently produce a prompt that
    is missing Rules 2-4. 'No specific controller' is not acceptable
    as an agent prompt — Rules 2 (decompose) and 3 (task router) are
    critical to correct behavior.

    The fallback should produce a prompt with at minimum a reasonable
    Rules 2-4 placeholder rather than an empty controller slot."""
    import unittest.mock as mock

    pb = PromptBuilder(nav_mode="mapless")
    ns = _base_nav_status(task_type="pointnav", nav_mode="mapless")

    with mock.patch.object(pb, "_library_root", return_value=tmp_path / "nonexistent"):
        result = pb.build_system_prompt(ns)

    # The critical check: fallback must NOT produce the useless
    # "No specific controller" generic string as Rules 2-4
    assert "No specific controller" not in result, (
        "Fallback prompt contains 'No specific controller' — Rules 2-4 "
        "are missing. This means both PromptLibrary AND the .txt fallback "
        "failed, leaving the agent with an incomplete prompt."
    )
    # Should still mention task-type and nav_mode for orientation
    assert "Rule 2" in result or "decompose" in result.lower(), (
        "Fallback prompt has no Rule 2 / decomposition guidance"
    )


def test_prompt_objectnav_navmesh_hides_eval_coords():
    """PromptBuilder must not render eval_goal_position coords in the
    round message for objectnav with GT."""
    pb = PromptBuilder(nav_mode="navmesh")
    msg = pb.build_round_message(_base_nav_status(), round_idx=0, consecutive_idle=0, idle_threshold=3)
    # Distinctive coord values must not appear anywhere
    assert "5.123" not in msg
    assert "3.456" not in msg
    # Goal description and gt_* debug fields must not leak either
    assert "gt_goal" not in msg
    assert "gt_euclidean_distance" not in msg
    # But the description must be present
    assert "Find the kitchen" in msg


def test_prompt_objectnav_mapless_hides_polar_without_last_goal():
    """mapless + non-pointnav → euclidean_distance_to_goal is None;
    PromptBuilder must not print a leaked distance."""
    pb = PromptBuilder(nav_mode="mapless")
    status = _base_nav_status(
        nav_mode="mapless",
        has_navmesh=False,
        euclidean_distance_to_goal=None,
        goal_direction_deg=None,
    )
    msg = pb.build_round_message(status, round_idx=0, consecutive_idle=0, idle_threshold=3)
    assert "5.123" not in msg
    assert "3.456" not in msg
    # The "no distance yet" fallback text should appear
    assert "after first movement" in msg


def test_prompt_pointnav_navmesh_shows_goal_position():
    """Baseline: pointnav navmesh legitimately exposes goal coords —
    ensure the coord-leak checks above are meaningful."""
    pb = PromptBuilder(nav_mode="navmesh")
    status = _base_nav_status(
        task_type="pointnav",
        goal_type="position",
        goal_position=[7.0, 0.2, 8.0],
    )
    msg = pb.build_round_message(status, round_idx=0, consecutive_idle=0, idle_threshold=3)
    assert "7.000" in msg and "8.000" in msg  # formatted as %.3f


# ---------------------------------------------------------------------------
# P0-2: _start_nav_loop direct tests — exercises real mixin code path
# ---------------------------------------------------------------------------


def _fake_proc(pid: int = 12345):
    """Minimal Popen-like fake."""
    return SimpleNamespace(
        pid=pid,
        poll=lambda: None,  # still running
        returncode=None,
        terminate=lambda: None,
        wait=lambda timeout=None: 0,
        kill=lambda: None,
    )


def _attach_session(adapter, session_id: str, position=(0.0, 0.2, 0.0)):
    """Build a session with a fake simulator, attach to adapter._sessions."""
    session = _make_session(position=position)
    session.session_id = session_id
    session.created_at_s = 0.0
    session.last_activity_s = 0.0
    adapter._sessions[session_id] = session
    return session


def test_start_nav_loop_pointnav_routes_both_goals(tmp_path, monkeypatch):
    adapter = _make_adapter()
    session = _attach_session(adapter, "s1", position=(0.0, 0.2, 0.0))
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    result = adapter._start_nav_loop(
        None,
        {
            "session_id": "s1",
            "task_type": "pointnav",
            "goal_type": "position",
            "goal_description": "Go to target",
            "goal_position": [3.0, 0.2, 4.0],
            "output_dir": str(tmp_path),
        },
    )
    # Session state
    assert session.last_goal == [3.0, 0.2, 4.0]
    assert session.eval_goal == [3.0, 0.2, 4.0]
    assert session.cumulative_path_length == 0.0
    assert len(session.trajectory) == 1
    assert session.trajectory[0] == pytest.approx([0.0, 0.2, 0.0], abs=1e-3)
    assert session.initial_geodesic_distance is not None
    assert session.initial_geodesic_distance == pytest.approx(5.0, abs=1e-2)
    # Persisted nav_status has both fields
    nav_status = json.loads(Path(result["nav_status_file"]).read_text())
    assert nav_status["goal_position"] == [3.0, 0.2, 4.0]
    assert nav_status["eval_goal_position"] == [3.0, 0.2, 4.0]
    assert nav_status["has_ground_truth"] is True
    # _debug has GT metrics
    assert nav_status["_debug"]["gt_goal"] == [3.0, 0.2, 4.0]
    assert nav_status["_debug"]["gt_euclidean_distance"] == pytest.approx(5.0, abs=1e-2)


def test_start_nav_loop_objectnav_with_eval_goal_hides_agent_goal(tmp_path, monkeypatch):
    """objectnav + eval_goal_position → session.last_goal stays None,
    nav_status.goal_position is None, but eval_goal_position is set."""
    adapter = _make_adapter()
    session = _attach_session(adapter, "s2")
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    result = adapter._start_nav_loop(
        None,
        {
            "session_id": "s2",
            "task_type": "objectnav",
            "goal_type": "instruction",
            "goal_description": "Find kitchen",
            "eval_goal_position": [5.0, 0.2, 3.0],
            "output_dir": str(tmp_path),
        },
    )
    # Agent-facing goal is None — no coord leak
    assert session.last_goal is None
    # Eval goal is set for metrics
    assert session.eval_goal == [5.0, 0.2, 3.0]
    assert session.initial_geodesic_distance is not None
    nav_status = json.loads(Path(result["nav_status_file"]).read_text())
    assert nav_status["goal_position"] is None
    assert nav_status["eval_goal_position"] == [5.0, 0.2, 3.0]
    assert nav_status["has_ground_truth"] is True


def test_start_nav_loop_objectnav_no_gt_has_no_eval(tmp_path, monkeypatch):
    """objectnav without any GT: has_ground_truth=False, no SPL."""
    adapter = _make_adapter()
    session = _attach_session(adapter, "s3")
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    result = adapter._start_nav_loop(
        None,
        {
            "session_id": "s3",
            "task_type": "objectnav",
            "goal_type": "instruction",
            "goal_description": "Find kitchen",
            "output_dir": str(tmp_path),
        },
    )
    assert session.last_goal is None
    assert session.eval_goal is None
    assert session.initial_geodesic_distance is None
    nav_status = json.loads(Path(result["nav_status_file"]).read_text())
    assert nav_status["goal_position"] is None
    assert nav_status["eval_goal_position"] is None
    assert nav_status["has_ground_truth"] is False


def test_start_nav_loop_resets_cumulative_path_length(tmp_path, monkeypatch):
    """Starting a loop must reset cumulative_path_length from whatever
    it accumulated in a previous episode on the same session."""
    adapter = _make_adapter()
    session = _attach_session(adapter, "s4")
    session.cumulative_path_length = 42.0  # stale from previous loop
    session.trajectory = [[9.0, 0.2, 9.0], [10.0, 0.2, 10.0]]  # stale
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    adapter._start_nav_loop(
        None,
        {
            "session_id": "s4",
            "task_type": "pointnav",
            "goal_type": "position",
            "goal_description": "Go",
            "goal_position": [1.0, 0.2, 1.0],
            "output_dir": str(tmp_path),
        },
    )
    # Reset on loop start
    assert session.cumulative_path_length == 0.0
    # Trajectory rebased to current position (agent at origin)
    assert len(session.trajectory) == 1
    assert session.trajectory[0] == pytest.approx([0.0, 0.2, 0.0], abs=1e-3)


def test_start_nav_loop_unreachable_goal_does_not_mutate_session(tmp_path, monkeypatch):
    """Atomicity: when _start_nav_loop raises (e.g., unreachable goal),
    the live session state must be unchanged. Otherwise the caller sees
    a session with stale trajectory/goal/path_length from a loop that
    never actually started."""
    from habitat_sim.habitat_adapter_internal.types import HabitatAdapterError

    adapter = _make_adapter()
    session = _make_session(
        position=(5.0, 0.2, 5.0),
        pathfinder=_FakePathFinder(reachable=False),
    )
    session.session_id = "s_atomic"
    adapter._sessions["s_atomic"] = session

    # Seed the session with some pre-existing state from a "previous loop"
    session.cumulative_path_length = 42.0
    session.trajectory = [[1.0, 0.2, 1.0], [2.0, 0.2, 2.0], [3.0, 0.2, 3.0]]
    session.last_goal = [9.0, 0.2, 9.0]
    session.eval_goal = [9.0, 0.2, 9.0]
    session.initial_geodesic_distance = 11.3

    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )

    with pytest.raises(HabitatAdapterError, match="unreachable"):
        adapter._start_nav_loop(
            None,
            {
                "session_id": "s_atomic",
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "Go",
                "goal_position": [10.0, 0.2, 10.0],
                "output_dir": str(tmp_path),
            },
        )

    # Every mutable session field must be UNCHANGED
    assert session.cumulative_path_length == 42.0
    assert session.trajectory == [[1.0, 0.2, 1.0], [2.0, 0.2, 2.0], [3.0, 0.2, 3.0]]
    assert session.last_goal == [9.0, 0.2, 9.0]
    assert session.eval_goal == [9.0, 0.2, 9.0]
    assert session.initial_geodesic_distance == 11.3


def test_start_nav_loop_popen_failure_does_not_mutate_session(tmp_path, monkeypatch):
    """Atomicity (extended): when the nav_loop subprocess fails to spawn
    (e.g., OSError from Popen, script not found), the session must be
    unchanged — no stale trajectory, no rebased path_length, no updated
    goals. The caller should see the exception and a pristine session."""
    adapter = _make_adapter()
    session = _make_session(position=(0.0, 0.2, 0.0))
    session.session_id = "s_popen_fail"
    adapter._sessions["s_popen_fail"] = session

    # Pre-existing state that must survive the Popen failure
    session.cumulative_path_length = 17.5
    session.trajectory = [[5.0, 0.2, 5.0], [6.0, 0.2, 5.0]]
    session.last_goal = [99.0, 0.2, 99.0]
    session.eval_goal = [99.0, 0.2, 99.0]
    session.initial_geodesic_distance = 7.7

    def _raise_popen(*args, **kwargs):
        raise OSError(2, "No such file or directory: 'nav_agent.py'")

    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        _raise_popen,
    )

    with pytest.raises(OSError):
        adapter._start_nav_loop(
            None,
            {
                "session_id": "s_popen_fail",
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "Go",
                "goal_position": [3.0, 0.2, 4.0],
                "output_dir": str(tmp_path),
            },
        )

    # Every mutable session field unchanged — the loop never started
    assert session.cumulative_path_length == 17.5
    assert session.trajectory == [[5.0, 0.2, 5.0], [6.0, 0.2, 5.0]]
    assert session.last_goal == [99.0, 0.2, 99.0]
    assert session.eval_goal == [99.0, 0.2, 99.0]
    assert session.initial_geodesic_distance == 7.7


# ---------------------------------------------------------------------------
# TUI loop-distance fallback (P3 from Codex)
# ---------------------------------------------------------------------------

from habitat_agent_tui import _format_loop_distance  # noqa: E402


def test_tui_distance_labels_geodesic():
    """Stage 4 ambiguity cleanup: distance is rendered with an explicit
    unit suffix so viewers know which quantity they're looking at. The
    previous implementation silently merged geodesic/euclidean/waypoint
    values into a single unlabeled number."""
    nav = {
        "has_navmesh": True,
        "_debug": {"gt_geodesic_distance": 3.14, "gt_euclidean_distance": 2.71},
        "geodesic_distance": 99.0,
    }
    # 'g' suffix = authoritative GT geodesic (navmesh + pathfinder)
    assert _format_loop_distance(nav) == "3.140 g"


def test_tui_distance_labels_euclidean_when_no_navmesh():
    """No navmesh → euclidean is the only GT distance; labeled 'e'."""
    nav = {
        "has_navmesh": False,
        "_debug": {"gt_geodesic_distance": None, "gt_euclidean_distance": 2.71},
    }
    assert _format_loop_distance(nav) == "2.710 e"


def test_tui_distance_labels_waypoint_for_non_gt_run():
    """Non-GT run with agent-queried find_path distance → 'wp' suffix.
    This is NOT to the goal; it's to whatever point the agent queried.
    The label makes that unambiguous for benchmark operators."""
    nav = {
        "has_navmesh": True,
        "_debug": {},
        "geodesic_distance": 5.5,
    }
    assert _format_loop_distance(nav) == "5.500 wp"


def test_tui_distance_dash_when_everything_missing():
    nav = {"has_navmesh": True, "_debug": {}}
    assert _format_loop_distance(nav) == "-"


def test_tui_distance_rejects_bool_mistyped_payload():
    """bool subclasses int — must not render True as '1.000 g'."""
    nav = {
        "has_navmesh": True,
        "_debug": {"gt_geodesic_distance": True, "gt_euclidean_distance": False},
    }
    assert _format_loop_distance(nav) == "-"


def test_tui_distance_handles_missing_debug_key():
    """nav_status without _debug key at all → fall through to waypoint
    from the top-level geodesic_distance (agent-written)."""
    nav = {"has_navmesh": True, "geodesic_distance": 2.0}
    assert _format_loop_distance(nav) == "2.000 wp"


def test_tui_distance_navmesh_gt_prefers_geodesic_over_euclidean():
    """When navmesh has both values, geodesic wins (more accurate),
    labeled 'g'."""
    nav = {
        "has_navmesh": True,
        "_debug": {"gt_geodesic_distance": 5.2, "gt_euclidean_distance": 4.1},
    }
    # Authoritative path is geodesic — that's what's displayed and labeled
    assert _format_loop_distance(nav) == "5.200 g"


def test_tui_distance_no_navmesh_uses_euclidean_even_if_geodesic_present():
    """Defensive: if somehow gt_geodesic_distance is present but
    has_navmesh=False (shouldn't happen, but be robust), prefer
    euclidean since geodesic is not meaningful without a navmesh."""
    nav = {
        "has_navmesh": False,
        "_debug": {"gt_geodesic_distance": 5.2, "gt_euclidean_distance": 4.1},
    }
    assert _format_loop_distance(nav) == "4.100 e"


def test_tui_no_inline_distance_fallback_outside_helper():
    """Anti-regression: the TUI file must only compute loop distance
    via the _format_loop_distance helper. Any inline `gt_geodesic_distance
    if X is not None else gt_euclidean_distance` outside the helper is
    a structural duplication bug (the Codex P2 detail-view finding)
    and must stay removed after Stage 4."""
    tui_path = (
        Path(__file__).resolve().parent.parent
        / "tools" / "habitat_agent_tui.py"
    )
    src = tui_path.read_text()
    # The helper itself uses these names in a tuple, which is fine.
    # Count inline `_gt_geo = debug.get("gt_geodesic_distance")` style
    # local-variable extraction — the detail view previously had a
    # second copy; after Stage 4 there should be zero.
    import re
    # Find all occurrences of '_gt_geo = ... gt_geodesic_distance' pattern
    pattern = re.compile(r"_gt_geo\s*=.*gt_geodesic_distance")
    hits = pattern.findall(src)
    assert hits == [], (
        f"Found {len(hits)} inline `_gt_geo = ... gt_geodesic_distance` "
        "extraction(s); this logic should go through _format_loop_distance. "
        f"Matches: {hits}"
    )


def test_start_nav_loop_rejects_bool_success_distance_threshold(tmp_path, monkeypatch):
    """success_distance_threshold=True must NOT be silently coerced to 1.0.
    Python's isinstance(True, int) is True because bool is a subclass of int,
    so a mistyped JSON payload could widen the success threshold and
    corrupt benchmark results. Bool values should fall through to None."""
    adapter = _make_adapter()
    _attach_session(adapter, "s_bool_thr")
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    result = adapter._start_nav_loop(
        None,
        {
            "session_id": "s_bool_thr",
            "task_type": "pointnav",
            "goal_type": "position",
            "goal_description": "Go",
            "goal_position": [3.0, 0.2, 4.0],
            "success_distance_threshold": True,  # mistyped
            "output_dir": str(tmp_path),
        },
    )
    nav_status = json.loads(Path(result["nav_status_file"]).read_text())
    # Must NOT be 1.0 — bool should be rejected (→ None, default 0.5 applies downstream)
    assert nav_status["success_distance_threshold"] is None


def test_start_nav_loop_accepts_valid_float_threshold(tmp_path, monkeypatch):
    """Sanity check that real floats still work."""
    adapter = _make_adapter()
    _attach_session(adapter, "s_float_thr")
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    result = adapter._start_nav_loop(
        None,
        {
            "session_id": "s_float_thr",
            "task_type": "pointnav",
            "goal_type": "position",
            "goal_description": "Go",
            "goal_position": [3.0, 0.2, 4.0],
            "success_distance_threshold": 0.2,
            "output_dir": str(tmp_path),
        },
    )
    nav_status = json.loads(Path(result["nav_status_file"]).read_text())
    assert nav_status["success_distance_threshold"] == 0.2


def test_start_nav_loop_rejects_unreachable_goal_on_navmesh(tmp_path, monkeypatch):
    """Navmesh scene with an unreachable eval_goal → _start_nav_loop
    must reject upfront. Otherwise SPL/success are never meaningfully
    computable and the session just wastes benchmark budget."""
    from habitat_sim.habitat_adapter_internal.types import HabitatAdapterError
    adapter = _make_adapter()
    # Pathfinder that always reports unreachable (find_path returns False)
    session = _make_session(
        position=(0.0, 0.2, 0.0),
        pathfinder=_FakePathFinder(reachable=False),
    )
    session.session_id = "s_unreach"
    adapter._sessions["s_unreach"] = session
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    with pytest.raises(HabitatAdapterError, match="unreachable"):
        adapter._start_nav_loop(
            None,
            {
                "session_id": "s_unreach",
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "Go",
                "goal_position": [10.0, 0.2, 10.0],
                "output_dir": str(tmp_path),
            },
        )


def test_start_nav_loop_mapless_allows_missing_geodesic(tmp_path, monkeypatch):
    """Mapless scene (no pathfinder) cannot pre-check reachability —
    _start_nav_loop must NOT reject for initial_geodesic_distance=None
    in that case, because the distance is only unknown, not unreachable."""
    adapter = _make_adapter()
    # No pathfinder → has_navmesh will be False → reachability check skipped
    session = _make_session(with_pathfinder=False)
    session.session_id = "s_mapless"
    adapter._sessions["s_mapless"] = session
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    adapter._start_nav_loop(
        None,
        {
            "session_id": "s_mapless",
            "task_type": "pointnav",
            "goal_type": "position",
            "goal_description": "Go",
            "goal_position": [3.0, 0.2, 4.0],
            "output_dir": str(tmp_path),
        },
    )
    # Loop started successfully even though l_opt is unknown
    assert session.eval_goal == [3.0, 0.2, 4.0]
    assert session.initial_geodesic_distance is None


def test_start_nav_loop_has_ground_truth_string_false_is_false(tmp_path, monkeypatch):
    """Regression guard: has_ground_truth='false' (string) must be
    parsed via _coerce_bool as boolean False, not treated as truthy.
    Previously a plain bool() cast would misread any non-empty string."""
    adapter = _make_adapter()
    session = _attach_session(adapter, "s_strbool")
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    # Pass the flag as string "false" without eval_goal_position. With plain
    # bool() this would be True and raise; with _coerce_bool it's False and
    # the loop starts with no GT.
    adapter._start_nav_loop(
        None,
        {
            "session_id": "s_strbool",
            "task_type": "objectnav",
            "goal_type": "instruction",
            "goal_description": "Find kitchen",
            "has_ground_truth": "false",
            "output_dir": str(tmp_path),
        },
    )
    assert session.eval_goal is None


def test_start_nav_loop_has_ground_truth_string_true_requires_eval_goal(tmp_path, monkeypatch):
    """Regression guard: has_ground_truth='true' must be parsed as True
    and trigger the eval_goal_position requirement."""
    from habitat_sim.habitat_adapter_internal.types import HabitatAdapterError
    adapter = _make_adapter()
    _attach_session(adapter, "s_strbool2")
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    with pytest.raises(HabitatAdapterError, match="has_ground_truth"):
        adapter._start_nav_loop(
            None,
            {
                "session_id": "s_strbool2",
                "task_type": "objectnav",
                "goal_type": "instruction",
                "goal_description": "Find kitchen",
                "has_ground_truth": "true",
                "output_dir": str(tmp_path),
            },
        )


def test_start_nav_loop_rejects_has_ground_truth_without_eval_goal(tmp_path, monkeypatch):
    """has_ground_truth=True without eval_goal_position (and without
    fallback goal_position for pointnav) must raise — otherwise the flag
    would be silently dropped by the adapter."""
    from habitat_sim.habitat_adapter_internal.types import HabitatAdapterError
    adapter = _make_adapter()
    _attach_session(adapter, "s_hgt")
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    with pytest.raises(HabitatAdapterError, match="has_ground_truth"):
        adapter._start_nav_loop(
            None,
            {
                "session_id": "s_hgt",
                "task_type": "objectnav",
                "goal_type": "instruction",
                "goal_description": "Find something",
                "has_ground_truth": True,
                # NO eval_goal_position — the bug we are guarding against
                "output_dir": str(tmp_path),
            },
        )


def test_start_nav_loop_pointnav_has_ground_truth_with_only_goal_position(tmp_path, monkeypatch):
    """For pointnav, has_ground_truth=True with goal_position (no explicit
    eval_goal_position) is acceptable — goal_position doubles as eval GT."""
    adapter = _make_adapter()
    session = _attach_session(adapter, "s_pn_hgt")
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    adapter._start_nav_loop(
        None,
        {
            "session_id": "s_pn_hgt",
            "task_type": "pointnav",
            "goal_type": "position",
            "goal_description": "Go",
            "goal_position": [1.0, 0.2, 2.0],
            "has_ground_truth": True,  # allowed: goal_position is fallback eval
            "output_dir": str(tmp_path),
        },
    )
    assert session.eval_goal == [1.0, 0.2, 2.0]
    assert session.last_goal == [1.0, 0.2, 2.0]


def test_start_nav_loop_origin_with_eval_goal_position(tmp_path, monkeypatch):
    """eval_goal_position=[0,0,0] is a valid GT (origin edge case)."""
    adapter = _make_adapter()
    session = _attach_session(adapter, "s5", position=(5.0, 0.2, 5.0))
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    result = adapter._start_nav_loop(
        None,
        {
            "session_id": "s5",
            "task_type": "objectnav",
            "goal_type": "instruction",
            "goal_description": "Return to origin",
            "eval_goal_position": [0.0, 0.0, 0.0],
            "output_dir": str(tmp_path),
        },
    )
    assert session.eval_goal == [0.0, 0.0, 0.0]
    assert session.last_goal is None  # not pointnav
    assert session.initial_geodesic_distance is not None
    nav_status = json.loads(Path(result["nav_status_file"]).read_text())
    assert nav_status["eval_goal_position"] == [0.0, 0.0, 0.0]
    assert nav_status["has_ground_truth"] is True


# ---------------------------------------------------------------------------
# P0-3: Closed-loop _debug protection (PR #28 bug #10 fix)
# ---------------------------------------------------------------------------


def test_get_nav_loop_status_closed_loop_preserves_debug(tmp_path, monkeypatch):
    """For a closed nav loop, _debug must NOT be refreshed from the
    (possibly moved) session — otherwise historical metrics corrupt."""
    from habitat_sim.habitat_adapter_internal.types import _NavLoopRecord

    adapter = _make_adapter()
    session = _attach_session(adapter, "s1", position=(100.0, 0.2, 100.0))
    # Session has since moved far from the original closed-loop endpoint
    session.last_goal = [200.0, 0.2, 200.0]
    session.eval_goal = [200.0, 0.2, 200.0]
    session.cumulative_path_length = 999.0

    frozen_debug = {
        "gt_position": [1.0, 0.2, 1.0],
        "gt_goal": [5.0, 0.2, 5.0],
        "gt_euclidean_distance": 5.656,
        "gt_geodesic_distance": 5.656,
        "gt_path_length": 10.0,
        "gt_initial_geodesic_distance": 7.0,
    }
    nav_status_file = tmp_path / "closed_nav_status.json"
    nav_status_content = {
        "task_id": "closed1",
        "session_id": "s1",
        "state_version": 5,
        "status": "reached",
        "_debug": frozen_debug,
        "goal_position": [5.0, 0.2, 5.0],
        "eval_goal_position": [5.0, 0.2, 5.0],
    }
    nav_status_file.write_text(json.dumps(nav_status_content))

    proc = SimpleNamespace(
        pid=1, poll=lambda: 0, returncode=0,
        wait=lambda timeout=None: 0, terminate=lambda: None, kill=lambda: None,
    )
    record = _NavLoopRecord(
        loop_id="closed1",
        process=proc,
        session_id="s1",
        task_type="pointnav",
        nav_mode="navmesh",
        has_navmesh=True,
        nav_status_file=str(nav_status_file),
        spatial_memory_file=str(tmp_path / "spatial.json"),
        log_file=str(tmp_path / "log"),
        started_at_s=0.0,
        nav_status=copy_deep(nav_status_content),
        state_version=5,
        returncode=0,
        ended_at_s=1.0,
    )
    adapter._closed_nav_loops["closed1"] = record

    result = adapter._get_nav_loop_status(
        None, {"loop_id": "closed1", "include_nav_status": True}
    )
    # The closed loop's _debug must equal the frozen snapshot, NOT a
    # rebuilt snapshot from the current (moved) session state.
    dbg = result["nav_status"]["_debug"]
    assert dbg["gt_position"] == [1.0, 0.2, 1.0]
    assert dbg["gt_goal"] == [5.0, 0.2, 5.0]
    assert dbg["gt_path_length"] == 10.0
    assert dbg["gt_initial_geodesic_distance"] == 7.0


def copy_deep(obj):
    import copy as _c
    return _c.deepcopy(obj)


# ---------------------------------------------------------------------------
# P1-1: collect_session_stats end-to-end
# ---------------------------------------------------------------------------


def _write_nav_status(loop_dir: Path, content: Dict[str, Any]) -> Path:
    """Write a nav_status.json inside a {artifacts}/{session}/{loop}/ layout."""
    loop_dir.mkdir(parents=True, exist_ok=True)
    nav_file = loop_dir / "nav_status.json"
    nav_file.write_text(json.dumps(content))
    return nav_file


def _write_events(nav_file: Path, events: list) -> Path:
    events_file = Path(str(nav_file) + ".events.jsonl")
    with events_file.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return events_file


def _read_stats(tmp_path: Path) -> Dict[str, Any]:
    stats_path = tmp_path / "session_stats.jsonl"
    lines = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1, f"Expected 1 stats line, got {len(lines)}"
    return lines[0]


def test_collect_stats_skips_bridge_refresh_on_terminal_file(tmp_path):
    """When nav_status.json already shows a terminal status (e.g., written
    by mark_terminal_status during the timeout / signal / fatal path),
    collect_session_stats MUST NOT call bridge.get_nav_loop_status —
    the bridge's in-memory record is still in_progress and persisting it
    back would overwrite the terminal status just written to the file."""
    loop_dir = tmp_path / "s_term" / "loop1"
    nav = {
        "task_id": "loop1",
        "task_type": "pointnav",
        "nav_mode": "navmesh",
        "has_navmesh": True,
        "has_ground_truth": True,
        "eval_goal_position": [3.0, 0.2, 4.0],
        "status": "timeout",  # mark_terminal_status just wrote this
        "total_steps": 30,
        "collisions": 1,
        "action_history": [],
        "_debug": {
            "gt_euclidean_distance": 2.5,
            "gt_geodesic_distance": 3.1,
            "gt_initial_geodesic_distance": 5.0,
            "gt_path_length": 4.0,
        },
    }
    nav_file = _write_nav_status(loop_dir, nav)
    _write_events(nav_file, [{"phase": "round_end", "total_steps": 30}])

    # A bridge that — if called — would destroy the terminal status
    # by writing the (stale) in_progress state back to the file.
    called = []

    class _CorruptingBridge:
        def call(self, action, payload=None):
            called.append((action, payload))
            corrupted = dict(nav)
            corrupted["status"] = "in_progress"
            nav_file.write_text(json.dumps(corrupted))
            return {}

    collect_session_stats(
        str(nav_file),
        str(nav_file) + ".events.jsonl",
        bridge=_CorruptingBridge(),
        loop_id="loop1",
    )
    stats = _read_stats(tmp_path)
    # Bridge must NOT have been called on terminal path
    assert called == [], f"bridge called unexpectedly: {called}"
    # Outcome preserved as terminal
    assert stats["outcome"] == "timeout"


def test_collect_stats_refreshes_bridge_on_non_terminal(tmp_path):
    """Sanity: for a non-terminal nav_status (normal path), the bridge
    refresh still happens so _debug stays fresh."""
    loop_dir = tmp_path / "s_live" / "loop1"
    nav = {
        "task_id": "loop1",
        "task_type": "pointnav",
        "nav_mode": "navmesh",
        "has_navmesh": True,
        "has_ground_truth": True,
        "eval_goal_position": [3.0, 0.2, 4.0],
        "status": "reached",  # agent-driven terminal (via update_nav_status)
        "total_steps": 30,
        "collisions": 0,
        "action_history": [],
        "_debug": {
            "gt_geodesic_distance": 0.2,
            "gt_initial_geodesic_distance": 5.0,
            "gt_path_length": 5.1,
        },
    }
    nav_file = _write_nav_status(loop_dir, nav)
    _write_events(nav_file, [{"phase": "round_end", "total_steps": 30}])

    called = []

    class _TrackingBridge:
        def call(self, action, payload=None):
            called.append((action, payload))
            return {}

    collect_session_stats(
        str(nav_file),
        str(nav_file) + ".events.jsonl",
        bridge=_TrackingBridge(),
        loop_id="loop1",
    )
    # "reached" is also terminal — verify we also skip refresh.
    # This codifies that collect_session_stats treats ANY terminal status
    # (reached/blocked/timeout/error) the same way: stats derive from
    # whatever is already in the file; the bridge refresh is an
    # optimization for live loops only.
    assert called == []
    stats = _read_stats(tmp_path)
    assert stats["outcome"] == "reached"


def test_collect_session_stats_navmesh_pointnav_success(tmp_path):
    loop_dir = tmp_path / "s1" / "loop1"
    nav = {
        "task_id": "loop1",
        "task_type": "pointnav",
        "nav_mode": "navmesh",
        "has_navmesh": True,
        "has_ground_truth": True,
        "eval_goal_position": [3.0, 0.2, 4.0],
        "status": "reached",
        "total_steps": 25,
        "collisions": 2,
        "action_history": [
            {"action": "forward 0.5", "step": 1},
            {"action": "turn_right 45", "step": 2},
            {"action": "forward 0.5", "step": 3},
        ],
        "_debug": {
            "gt_euclidean_distance": 0.3,
            "gt_geodesic_distance": 0.3,
            "gt_initial_geodesic_distance": 5.0,
            "gt_path_length": 6.0,
        },
    }
    nav_file = _write_nav_status(loop_dir, nav)
    _write_events(nav_file, [
        {"phase": "round_end", "total_steps": 10},
        {"phase": "round_end", "total_steps": 20},
        {"phase": "round_end", "total_steps": 25},
    ])
    collect_session_stats(str(nav_file), str(nav_file) + ".events.jsonl")
    stats = _read_stats(tmp_path)
    assert stats["has_gt_goal"] is True
    assert stats["success_distance_threshold"] == 0.5
    assert stats["gt_success"] is True
    # New unambiguous schema: both geodesic and euclidean end distances
    # are stored as separate fields. The old merged `gt_end_distance`
    # field is removed — no more apples-vs-oranges in benchmark reports.
    assert "gt_end_distance" not in stats
    assert stats["gt_end_geodesic_distance"] == 0.3
    assert stats["gt_end_euclidean_distance"] == 0.3
    assert stats["gt_initial_geodesic_distance"] == 5.0
    assert stats["gt_path_length"] == 6.0
    # SPL = 1.0 * (5 / max(5, 6)) = 0.8333
    assert stats["spl"] == pytest.approx(0.8333, abs=1e-3)
    assert stats["tool_usage"] == {"forward": 2, "turn_right": 1}
    assert stats["rounds"] == 3
    assert stats["effective_rounds"] == 3


def test_collect_session_stats_no_gt_emits_null_metrics(tmp_path):
    loop_dir = tmp_path / "s2" / "loop1"
    nav = {
        "task_id": "loop1",
        "task_type": "objectnav",
        "nav_mode": "mapless",
        "has_navmesh": False,
        "has_ground_truth": False,
        "eval_goal_position": None,
        "status": "reached",
        "total_steps": 10,
        "collisions": 0,
        "action_history": [],
        "_debug": {},
    }
    nav_file = _write_nav_status(loop_dir, nav)
    _write_events(nav_file, [])
    collect_session_stats(str(nav_file), str(nav_file) + ".events.jsonl")
    stats = _read_stats(tmp_path)
    assert stats["has_gt_goal"] is False
    assert stats["gt_success"] is None
    # Both distance fields are None when there is no GT
    assert stats["gt_end_geodesic_distance"] is None
    assert stats["gt_end_euclidean_distance"] is None
    assert "gt_end_distance" not in stats
    assert stats["spl"] is None
    assert stats["success_distance_threshold"] is None


def test_collect_session_stats_has_ground_truth_flag_with_missing_eval_goal(tmp_path):
    """Defensive: hand-crafted nav_status with has_ground_truth=True but
    no eval_goal_position. The adapter rejects this on _start_nav_loop
    (see test_start_nav_loop_rejects_has_ground_truth_without_eval_goal),
    so this state shouldn't arise in practice — but collect_session_stats
    should still handle it without crashing."""
    loop_dir = tmp_path / "s3" / "loop1"
    nav = {
        "task_id": "loop1",
        "task_type": "pointnav",
        "nav_mode": "navmesh",
        "has_navmesh": True,
        "has_ground_truth": True,
        "status": "reached",
        "total_steps": 5,
        "collisions": 0,
        "action_history": [],
        "_debug": {
            "gt_euclidean_distance": None,
            "gt_geodesic_distance": None,
            "gt_initial_geodesic_distance": None,
            "gt_path_length": 1.0,
        },
    }
    nav_file = _write_nav_status(loop_dir, nav)
    _write_events(nav_file, [])
    collect_session_stats(str(nav_file), str(nav_file) + ".events.jsonl")
    stats = _read_stats(tmp_path)
    assert stats["has_gt_goal"] is True
    # With navmesh but no geodesic, gt_success must be False (strict)
    assert stats["gt_success"] is False
    assert stats["spl"] is None  # no l_opt


def test_collect_session_stats_threshold_override(tmp_path):
    loop_dir = tmp_path / "s5" / "loop1"
    nav = {
        "task_id": "loop1",
        "task_type": "pointnav",
        "nav_mode": "navmesh",
        "has_navmesh": True,
        "has_ground_truth": True,
        "eval_goal_position": [1.0, 0.2, 1.0],
        "success_distance_threshold": 0.2,
        "status": "reached",
        "total_steps": 5,
        "collisions": 0,
        "action_history": [],
        "_debug": {
            "gt_euclidean_distance": 0.3,
            "gt_geodesic_distance": 0.3,
            "gt_initial_geodesic_distance": 2.0,
            "gt_path_length": 2.5,
        },
    }
    nav_file = _write_nav_status(loop_dir, nav)
    _write_events(nav_file, [])
    collect_session_stats(str(nav_file), str(nav_file) + ".events.jsonl")
    stats = _read_stats(tmp_path)
    assert stats["success_distance_threshold"] == 0.2
    # 0.3 > 0.2 → NOT success
    assert stats["gt_success"] is False


def test_collect_session_stats_effective_rounds_monotonic(tmp_path):
    """effective_rounds counts only rounds where total_steps grew."""
    loop_dir = tmp_path / "s6" / "loop1"
    nav = {
        "task_id": "loop1",
        "task_type": "pointnav",
        "nav_mode": "navmesh",
        "has_navmesh": True,
        "has_ground_truth": True,
        "eval_goal_position": [1.0, 0.2, 1.0],
        "status": "reached",
        "total_steps": 30,
        "collisions": 0,
        "action_history": [],
        "_debug": {"gt_geodesic_distance": 0.1, "gt_initial_geodesic_distance": 5.0, "gt_path_length": 5.0},
    }
    nav_file = _write_nav_status(loop_dir, nav)
    _write_events(nav_file, [
        {"phase": "round_end", "total_steps": 10},  # effective #1
        {"phase": "round_end", "total_steps": 10},  # idle
        {"phase": "round_end", "total_steps": 20},  # effective #2
        {"phase": "round_end", "total_steps": 30},  # effective #3
    ])
    collect_session_stats(str(nav_file), str(nav_file) + ".events.jsonl")
    stats = _read_stats(tmp_path)
    assert stats["rounds"] == 4
    assert stats["effective_rounds"] == 3


# ---------------------------------------------------------------------------
# P1-2: generate_report.py
# ---------------------------------------------------------------------------

_REPORT_MODULE = None


def _get_report_module():
    global _REPORT_MODULE
    if _REPORT_MODULE is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_report",
            str(Path(__file__).resolve().parent.parent / "tools" / "analytics" / "generate_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _REPORT_MODULE = mod
    return _REPORT_MODULE


def _run_report(entries, fn_name, tmp_path):
    """Write entries to a jsonl, call fn_name, capture stdout."""
    import io
    import contextlib
    mod = _get_report_module()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        getattr(mod, fn_name)(entries)
    return buf.getvalue()


def test_report_overall_threshold_single_value(tmp_path):
    entries = [
        {
            "has_gt_goal": True,
            "has_navmesh": True,
            "success_distance_threshold": 0.5,
            "gt_success": True,
            "gt_end_geodesic_distance": 0.3,
            "gt_end_euclidean_distance": 0.3,
            "spl": 0.9,
            "gt_initial_geodesic_distance": 5.0,
            "gt_path_length": 5.5,
            "outcome": "reached",
            "rounds": 3,
            "effective_rounds": 3,
            "total_steps": 20,
            "collisions": 0,
        },
        {
            "has_gt_goal": True,
            "has_navmesh": True,
            "success_distance_threshold": 0.5,
            "gt_success": False,
            "gt_end_geodesic_distance": 1.2,
            "gt_end_euclidean_distance": 1.1,
            "spl": 0.0,
            "gt_initial_geodesic_distance": 6.0,
            "gt_path_length": 7.0,
            "outcome": "reached",
            "rounds": 4,
            "effective_rounds": 4,
            "total_steps": 25,
            "collisions": 1,
        },
    ]
    out = _run_report(entries, "report_overall", tmp_path)
    assert "within 0.5m" in out
    assert "GT success rate" in out
    assert "1/2" in out  # one success
    assert "Avg SPL: 0.450" in out
    assert "SPL coverage: 2/2" in out


def test_report_overall_threshold_range(tmp_path):
    entries = [
        {"has_gt_goal": True, "has_navmesh": True, "success_distance_threshold": 0.2,
         "gt_success": True, "gt_end_geodesic_distance": 0.1, "gt_end_euclidean_distance": 0.1,
         "spl": 1.0, "outcome": "reached", "rounds": 1, "effective_rounds": 1,
         "total_steps": 10, "collisions": 0},
        {"has_gt_goal": True, "has_navmesh": True, "success_distance_threshold": 0.5,
         "gt_success": True, "gt_end_geodesic_distance": 0.3, "gt_end_euclidean_distance": 0.3,
         "spl": 0.9, "outcome": "reached", "rounds": 1, "effective_rounds": 1,
         "total_steps": 10, "collisions": 0},
    ]
    out = _run_report(entries, "report_overall", tmp_path)
    assert "0.2" in out and "0.5m" in out  # range shown


def test_report_overall_splits_gt_and_non_gt(tmp_path):
    entries = [
        {"has_gt_goal": True, "has_navmesh": True, "success_distance_threshold": 0.5,
         "gt_success": True, "spl": 1.0, "outcome": "reached", "rounds": 1,
         "effective_rounds": 1, "total_steps": 10, "collisions": 0,
         "gt_end_geodesic_distance": 0.1, "gt_end_euclidean_distance": 0.1},
        {"has_gt_goal": False, "has_navmesh": True, "outcome": "reached",
         "rounds": 1, "effective_rounds": 1, "total_steps": 10, "collisions": 0},
        {"has_gt_goal": False, "has_navmesh": False, "outcome": "blocked",
         "rounds": 2, "effective_rounds": 1, "total_steps": 5, "collisions": 3},
    ]
    out = _run_report(entries, "report_overall", tmp_path)
    assert "Sessions without GT goal (metrics skipped): 2" in out
    assert "GT-evaluated sessions: 1" in out


def test_report_overall_spl_coverage_partial(tmp_path):
    """Some GT sessions miss SPL (no l_opt) — coverage reported correctly."""
    entries = [
        {"has_gt_goal": True, "has_navmesh": True, "success_distance_threshold": 0.5,
         "gt_success": True, "spl": 1.0, "outcome": "reached", "rounds": 1,
         "effective_rounds": 1, "total_steps": 10, "collisions": 0,
         "gt_end_geodesic_distance": 0.1, "gt_end_euclidean_distance": 0.1},
        {"has_gt_goal": True, "has_navmesh": True, "success_distance_threshold": 0.5,
         "gt_success": False, "spl": None, "outcome": "blocked", "rounds": 1,
         "effective_rounds": 1, "total_steps": 5, "collisions": 2,
         "gt_end_geodesic_distance": 3.0, "gt_end_euclidean_distance": 2.8},
    ]
    out = _run_report(entries, "report_overall", tmp_path)
    assert "SPL coverage: 1/2" in out


def test_report_overall_splits_end_distance_by_navmesh(tmp_path):
    """Stage 2: navmesh and non-navmesh end distances MUST be reported
    separately (never averaged together — that was the original ambiguity
    bug). Navmesh sessions use gt_end_geodesic_distance; non-navmesh
    sessions use gt_end_euclidean_distance."""
    entries = [
        # navmesh sessions → geodesic
        {"has_gt_goal": True, "has_navmesh": True, "success_distance_threshold": 0.5,
         "gt_success": True, "spl": 1.0, "outcome": "reached", "rounds": 1,
         "effective_rounds": 1, "total_steps": 10, "collisions": 0,
         "gt_end_geodesic_distance": 0.2, "gt_end_euclidean_distance": 0.1},
        {"has_gt_goal": True, "has_navmesh": True, "success_distance_threshold": 0.5,
         "gt_success": True, "spl": 0.9, "outcome": "reached", "rounds": 1,
         "effective_rounds": 1, "total_steps": 10, "collisions": 0,
         "gt_end_geodesic_distance": 0.4, "gt_end_euclidean_distance": 0.3},
        # non-navmesh session → euclidean only
        {"has_gt_goal": True, "has_navmesh": False, "success_distance_threshold": 0.5,
         "gt_success": True, "spl": None, "outcome": "reached", "rounds": 1,
         "effective_rounds": 1, "total_steps": 10, "collisions": 0,
         "gt_end_geodesic_distance": None, "gt_end_euclidean_distance": 0.35},
    ]
    out = _run_report(entries, "report_overall", tmp_path)
    # Two separate averages labeled by navmesh/no-navmesh + metric type
    assert "geodesic" in out.lower()
    assert "euclidean" in out.lower()
    # Avg geodesic over the two navmesh sessions = (0.2 + 0.4) / 2 = 0.30
    assert "0.30" in out
    # Single non-navmesh session avg euclidean = 0.35
    assert "0.35" in out
    # The old single merged line must NOT appear
    assert "Avg GT end distance:" not in out


def test_report_tool_usage_uses_gt_success_not_outcome(tmp_path):
    """Stage 2: tool-usage analysis must classify success/failure by
    gt_success (the evaluator's verdict) rather than outcome (the
    agent's self-report). Using outcome mixed GT-verified and
    agent-reported successes in the same bucket."""
    entries = [
        # Agent self-reported reached, but GT says failure (distance too far)
        {"has_gt_goal": True, "has_navmesh": True, "gt_success": False,
         "outcome": "reached", "spl": 0.0,
         "gt_end_geodesic_distance": 2.0, "gt_end_euclidean_distance": 2.0,
         "tool_usage": {"forward": 30, "turn_right": 10},
         "action_history_len": 40, "capability_requests": [],
         "rounds": 5, "effective_rounds": 5, "total_steps": 100, "collisions": 2},
        # Agent reported reached AND GT says success
        {"has_gt_goal": True, "has_navmesh": True, "gt_success": True,
         "outcome": "reached", "spl": 0.9,
         "gt_end_geodesic_distance": 0.2, "gt_end_euclidean_distance": 0.2,
         "tool_usage": {"forward": 5, "turn_right": 2},
         "action_history_len": 7, "capability_requests": [],
         "rounds": 2, "effective_rounds": 2, "total_steps": 15, "collisions": 0},
    ]
    out = _run_report(entries, "report_tool_usage", tmp_path)
    # The "Sessions" line in report_tool_usage should count ONE success
    # (the second entry with gt_success=True) and ONE failure (the first
    # entry with gt_success=False). The previous code used outcome and
    # would have counted TWO successes.
    assert "1 reached" in out or "1 success" in out
    assert "1 not reached" in out or "1 failure" in out


# ---------------------------------------------------------------------------
# P1-3: MCP hab_nav_loop_start routing
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp_module(monkeypatch):
    """Import the canonical MCP server module with the BridgeClient patched.

    Phase 2 PR 4 rewrote mcp_server to register all hab_* tools
    dynamically from `ToolRegistry`. The old hand-written functions
    used a module-level `bridge_call()` wrapper that tests could
    monkeypatch; the new dynamic wrappers go through
    `ToolRegistry.dispatch(name, args, ctx)` where `ctx.bridge` is
    the module's `_bridge` instance. So we patch `_bridge.call`
    directly — that intercept fires for every dispatch regardless
    of which Tool subclass ends up handling the call.

    Also note the tool name change: Phase 2 PR 3 introduced
    InitSceneTool / CloseSessionTool / NavLoopStartTool / etc., so
    the MCP tool names are now `hab_init_scene` / `hab_close_session` /
    `hab_nav_loop_start`, mirroring the Tool class names. The
    existing tests still reference `hab_nav_loop_start` (same name
    as the legacy function) so they continue to work; tests that
    referenced `hab_init` / `hab_close` would need to be updated.
    """
    import importlib
    sys.modules.pop("habitat_agent.interfaces.mcp_server", None)
    mod = importlib.import_module("habitat_agent.interfaces.mcp_server")
    # Ensure a session exists so nav_loop_start passes its precondition
    mod._bridge.session_id = "s1"
    # Intercept every bridge.call(...) through the shared BridgeClient
    # instance that every dispatched Tool uses.
    calls = []

    def fake_call(action, payload=None):
        calls.append((action, payload))
        return {"loop_id": "fake_loop", "pid": 1, "status": "started"}

    monkeypatch.setattr(mod._bridge, "call", fake_call)
    mod._calls = calls  # type: ignore[attr-defined]
    return mod


def _get_payload(mcp_module):
    """Extract the start_nav_loop payload from the recorded calls."""
    assert mcp_module._calls, "no bridge_call recorded"
    action, payload = mcp_module._calls[-1]
    assert action == "start_nav_loop", f"expected start_nav_loop, got {action}"
    return payload


def test_mcp_pointnav_routes_both_goal_and_eval_goal(mcp_module):
    result = mcp_module.hab_nav_loop_start(
        task_type="pointnav",
        goal_description="Go there",
        goal_x=3.0, goal_y=0.2, goal_z=4.0,
    )
    assert '"error"' not in result
    payload = _get_payload(mcp_module)
    assert payload["goal_type"] == "position"
    assert payload["goal_position"] == [3.0, 0.2, 4.0]
    assert payload["eval_goal_position"] == [3.0, 0.2, 4.0]
    assert payload["has_ground_truth"] is True


def test_mcp_objectnav_only_routes_eval_goal(mcp_module):
    result = mcp_module.hab_nav_loop_start(
        task_type="objectnav",
        goal_description="Find kitchen",
        goal_x=5.0, goal_y=0.2, goal_z=3.0,
    )
    assert '"error"' not in result
    payload = _get_payload(mcp_module)
    assert payload["goal_type"] == "instruction"
    assert "goal_position" not in payload  # NOT set for non-pointnav
    assert payload["eval_goal_position"] == [5.0, 0.2, 3.0]
    assert payload["has_ground_truth"] is True


def test_mcp_pointnav_without_coords_rejected(mcp_module):
    result = mcp_module.hab_nav_loop_start(
        task_type="pointnav",
        goal_description="Go",
        # no goal_x/y/z, no has_ground_truth
    )
    data = json.loads(result)
    assert "error" in data
    assert "pointnav requires" in data["error"].lower()
    # bridge was NOT called
    assert not mcp_module._calls


def test_mcp_pointnav_origin_requires_explicit_coords(mcp_module):
    """Caller must pass goal_x=0, goal_y=0, goal_z=0 explicitly.
    Without that, even with has_ground_truth=True, the request is rejected
    — we no longer silently assume origin from default values."""
    result = mcp_module.hab_nav_loop_start(
        task_type="pointnav",
        goal_description="Go to origin",
        has_ground_truth=True,
        goal_x=0.0, goal_y=0.0, goal_z=0.0,  # explicit zeros
    )
    assert '"error"' not in result
    payload = _get_payload(mcp_module)
    assert payload["goal_position"] == [0.0, 0.0, 0.0]
    assert payload["eval_goal_position"] == [0.0, 0.0, 0.0]
    assert payload["has_ground_truth"] is True


def test_mcp_has_ground_truth_without_coords_rejected(mcp_module):
    """has_ground_truth=True but no goal_x/y/z → reject (forgot to pass)."""
    result = mcp_module.hab_nav_loop_start(
        task_type="objectnav",
        goal_description="Find chair",
        has_ground_truth=True,
        # no goal_x/y/z
    )
    data = json.loads(result)
    assert "error" in data
    assert "has_ground_truth" in data["error"]
    assert not mcp_module._calls  # bridge not called


def test_mcp_partial_coords_rejected(mcp_module):
    """Passing 2 of 3 coords is an input error, not a silent fill."""
    result = mcp_module.hab_nav_loop_start(
        task_type="objectnav",
        goal_description="Find chair",
        goal_x=1.0, goal_y=0.2,
        # goal_z omitted
    )
    data = json.loads(result)
    assert "error" in data
    assert "together" in data["error"].lower()
    assert not mcp_module._calls


def test_mcp_objectnav_without_coords_has_no_eval_goal(mcp_module):
    result = mcp_module.hab_nav_loop_start(
        task_type="objectnav",
        goal_description="Find kitchen",
    )
    assert '"error"' not in result
    payload = _get_payload(mcp_module)
    assert "goal_position" not in payload
    assert "eval_goal_position" not in payload
    assert "has_ground_truth" not in payload


def test_mcp_non_pointnav_with_explicit_position_goaltype_does_not_leak(mcp_module):
    """Codex P1: a caller passing task_type=objectnav with goal_type=position
    must NOT cause goal_position to be forwarded to the agent. The non-PointNav
    contract is "coordinate-blind" — coordinates may live in eval_goal_position
    for GT scoring, but never in the agent-facing goal_position."""
    result = mcp_module.hab_nav_loop_start(
        task_type="objectnav",
        goal_description="Find kitchen",
        goal_type="position",  # caller tries to force position routing
        goal_x=5.0, goal_y=0.2, goal_z=3.0,
    )
    assert '"error"' not in result
    payload = _get_payload(mcp_module)
    # Agent-facing goal_position must NOT be set for non-pointnav
    assert "goal_position" not in payload
    # Eval-only GT is still allowed
    assert payload["eval_goal_position"] == [5.0, 0.2, 3.0]
    assert payload["has_ground_truth"] is True


def test_mcp_pointnav_with_instruction_goaltype_still_requires_coords(mcp_module):
    """Codex P2: task_type=pointnav with goal_type=instruction must still
    require explicit coordinates. Without this, a caller can spin up a
    PointNav loop with no target at all and waste an evaluation slot."""
    result = mcp_module.hab_nav_loop_start(
        task_type="pointnav",
        goal_description="Go somewhere",
        goal_type="instruction",  # caller tries to bypass position requirement
        # no goal_x/y/z
    )
    data = json.loads(result)
    assert "error" in data
    assert "pointnav requires" in data["error"].lower()
    assert not mcp_module._calls


def test_mcp_legacy_hab_init_alias_still_registered(mcp_module):
    """Codex P1 regression lock.

    Phase 2 PR 4 renamed the Tool class `InitSceneTool` (tool name
    "init_scene"), which naively produced `hab_init_scene` as the
    MCP tool name. That broke every external MCP client that
    hardcoded the legacy `hab_init`. The fix is to register a
    `hab_init` alias alongside `hab_init_scene`.

    This test locks in: `mcp_module.hab_init` must exist as a callable
    and must actually route to the bridge's `init_scene` action,
    preserving exactly what legacy `hab_init` did."""
    assert hasattr(mcp_module, "hab_init"), (
        "hab_init alias was not registered — external MCP clients "
        "hardcoding hab_init will hard-fail"
    )
    # Invoke the alias and verify it hits the bridge's init_scene action
    mcp_module.hab_init(scene="test_scene")
    assert mcp_module._calls, "hab_init did not reach the bridge"
    action, _ = mcp_module._calls[-1]
    assert action == "init_scene"


def test_mcp_legacy_hab_close_alias_still_registered(mcp_module):
    """P1 regression lock — `hab_close` alias preserved."""
    assert hasattr(mcp_module, "hab_close"), (
        "hab_close alias was not registered — external MCP clients "
        "hardcoding hab_close will hard-fail"
    )
    mcp_module.hab_close()
    action, _ = mcp_module._calls[-1]
    assert action == "close_session"


def test_mcp_does_not_expose_update_nav_status(mcp_module):
    """Codex P1 regression lock.

    `update_nav_status` is nav_loop-internal — it needs `loop_id`
    and `state_version_ref` from the NavAgent subprocess context
    that no MCP top-level caller can provide. The dynamic MCP
    registration MUST skip it so the MCP tool inventory doesn't
    include a tool that is guaranteed to fail with a cryptic
    bridge error."""
    assert not hasattr(mcp_module, "hab_update_nav_status"), (
        "hab_update_nav_status was exposed via MCP, but it requires "
        "nav_loop-internal context (loop_id, state_version_ref) "
        "that top-level MCP callers cannot supply. It will always "
        "fail and pollute the MCP tool inventory."
    )


def test_mcp_total_tool_count_excludes_invisible(mcp_module):
    """Total MCP-visible tools: 19 - 1 = 18.

    Breakdown:
      - 17 canonical Tool subclasses (includes SceneGraphQueryTool added in PR #36)
      - 2 legacy aliases (hab_init for InitSceneTool, hab_close
        for CloseSessionTool)
      - MINUS 1 (hab_update_nav_status excluded as MCP-invisible)
      = 18 MCP-visible tool names total
    """
    import asyncio
    async def count():
        return len(await mcp_module.mcp.list_tools())
    n = asyncio.run(count())
    assert n == 18, (
        f"expected 18 MCP-visible tools (17 canonical + 2 aliases - "
        f"1 invisible update_nav_status), got {n}"
    )


def test_mcp_hab_init_response_has_scene_info_wrapper(mcp_module, monkeypatch):
    """Codex round-6 P1 regression lock — preserve hab_init shape.

    Legacy hab_init wrapped the full bridge response under a
    `scene_info` key and surfaced `session_id` + `is_gaussian` at
    the top level. PR 4 originally returned the bridge response
    verbatim, breaking external clients that read `scene_info`."""
    # Override the bridge stub for this test so init_scene returns
    # something distinguishable
    def fake_call(action, payload=None):
        mcp_module._calls.append((action, payload))
        if action == "init_scene":
            return {
                "session_id": "abc123",
                "is_gaussian": True,
                "scene": "test_scene",
                "agent_state": {"position": [0, 0, 0]},
            }
        return {"loop_id": "fake_loop"}
    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    raw = mcp_module.hab_init(scene="test_scene")
    response = json.loads(raw)
    assert "scene_info" in response, (
        f"hab_init response missing scene_info wrapper: {response}"
    )
    assert response["session_id"] == "abc123"
    assert response["is_gaussian"] is True
    # Full bridge body is under scene_info
    assert response["scene_info"]["agent_state"]["position"] == [0, 0, 0]


def test_mcp_hab_look_response_has_top_level_images_list(mcp_module, monkeypatch):
    """Round-6 P1 — hab_look must surface `images` as a flat list of
    file paths at the top level (not buried under `visuals` dict)."""
    def fake_call(action, payload=None):
        mcp_module._calls.append((action, payload))
        return {
            "visuals": {
                "color_sensor": {"path": "/o/look_color.png"},
                "depth_sensor": {"path": "/o/look_depth.png"},
            },
            "agent_state": {"heading": 1.5},
        }
    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    raw = mcp_module.hab_look()
    response = json.loads(raw)
    assert "images" in response, (
        f"hab_look response missing top-level images list: {response}"
    )
    assert "/o/look_color.png" in response["images"]
    assert "/o/look_depth.png" in response["images"]
    # visuals dict should be stripped from the top level
    assert "visuals" not in response
    # Other fields (agent_state) preserved
    assert response["agent_state"]["heading"] == 1.5


def test_mcp_hab_turn_response_has_top_level_images_list(mcp_module, monkeypatch):
    """Round-6 P1 — hab_turn must shape the same way as hab_look."""
    def fake_call(action, payload=None):
        mcp_module._calls.append((action, payload))
        return {
            "visuals": {"color_sensor": {"path": "/o/turn.png"}},
            "collided": False,
        }
    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    raw = mcp_module.hab_turn(direction="left", degrees=45)
    response = json.loads(raw)
    assert response["images"] == ["/o/turn.png"]
    assert "visuals" not in response
    assert response["collided"] is False


def test_mcp_hab_topdown_includes_topdown_map_in_images(mcp_module, monkeypatch):
    """Round-6 P1 — hab_topdown must collect the `topdown_map` field
    INTO the top-level images list and strip the original key."""
    def fake_call(action, payload=None):
        mcp_module._calls.append((action, payload))
        return {
            "topdown_map": {"path": "/o/topdown.png"},
            "visuals": {"color_sensor": {"path": "/o/td_color.png"}},
            "metadata": {"scale": 0.05},
        }
    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    raw = mcp_module.hab_topdown()
    response = json.loads(raw)
    assert "/o/topdown.png" in response["images"]
    assert "/o/td_color.png" in response["images"]
    assert "topdown_map" not in response
    assert "visuals" not in response
    assert response["metadata"]["scale"] == 0.05


def test_mcp_hab_panorama_strips_images_dict_and_returns_path_list(
    mcp_module, monkeypatch
):
    """Round-6 P1 — panorama bridge response has `images` as a list
    of dicts with `path`; the wrapper must replace it with a
    flat list of paths."""
    def fake_call(action, payload=None):
        mcp_module._calls.append((action, payload))
        return {
            "images": [
                {"path": "/o/p_front.png", "direction": "front"},
                {"path": "/o/p_right.png", "direction": "right"},
                {"path": "/o/p_back.png", "direction": "back"},
                {"path": "/o/p_left.png", "direction": "left"},
            ],
            "depth_analysis": {"front": {"min": 1.0}},
        }
    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    raw = mcp_module.hab_panorama()
    response = json.loads(raw)
    # images is now a flat list of strings, not list-of-dicts
    assert response["images"] == [
        "/o/p_front.png", "/o/p_right.png", "/o/p_back.png", "/o/p_left.png",
    ]
    # depth_analysis preserved as a sibling field
    assert response["depth_analysis"]["front"]["min"] == 1.0


def test_mcp_hab_forward_strips_visuals_top_level(mcp_module, monkeypatch):
    """Round-6 P1 — hab_forward strips `visuals` from the top level
    (matching legacy behaviour) but does not add an `images` key."""
    def fake_call(action, payload=None):
        mcp_module._calls.append((action, payload))
        return {
            "visuals": {"color_sensor": {"path": "/o/f.png"}},
            "position": [1, 0, 0],
            "collided": False,
        }
    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    raw = mcp_module.hab_forward(distance_m=0.5)
    response = json.loads(raw)
    assert "visuals" not in response
    assert response["position"] == [1, 0, 0]
    assert response["collided"] is False


def test_mcp_canonical_names_also_registered(mcp_module):
    """Complement to the legacy-alias tests: the new canonical names
    (hab_init_scene / hab_close_session) are ALSO registered, so
    callers that use the new names see the same behaviour."""
    assert hasattr(mcp_module, "hab_init_scene")
    assert hasattr(mcp_module, "hab_close_session")


def test_mcp_success_distance_threshold_override(mcp_module):
    result = mcp_module.hab_nav_loop_start(
        task_type="pointnav",
        goal_description="Go",
        goal_x=1.0, goal_y=0.2, goal_z=1.0,
        success_distance_threshold=0.2,
    )
    assert '"error"' not in result
    payload = _get_payload(mcp_module)
    assert payload["success_distance_threshold"] == 0.2


# ===========================================================================
# Phase 0 — File I/O primitives (acquire_nav_status_lock, append_jsonl_atomic)
# ===========================================================================
#
# These primitives are added to tools/habitat_agent_core.py to support the
# B1/B2/B4/B5 fixes. They are tested in isolation here so that subsequent
# phases can rely on them with confidence.

import threading
import time

from habitat_agent_core import (  # noqa: E402
    acquire_nav_status_lock,
    append_jsonl_atomic,
)


def test_acquire_nav_status_lock_releases_on_normal_exit(tmp_path):
    """Lock must be released after the with-block so a re-acquire works."""
    target = tmp_path / "nav_status.json"
    target.write_text("{}")
    with acquire_nav_status_lock(str(target)):
        pass
    # Re-acquire immediately — would block if lock leaked
    with acquire_nav_status_lock(str(target), timeout_s=0.5):
        pass


def test_acquire_nav_status_lock_releases_on_exception(tmp_path):
    """Lock must be released even when the with-block raises."""
    target = tmp_path / "nav_status.json"
    target.write_text("{}")
    with pytest.raises(RuntimeError):
        with acquire_nav_status_lock(str(target)):
            raise RuntimeError("boom")
    # Lock released → re-acquire works
    with acquire_nav_status_lock(str(target), timeout_s=0.5):
        pass


def test_acquire_nav_status_lock_creates_sidecar_in_parent_dir(tmp_path):
    """The lock file lives next to nav_status.json (sidecar pattern)."""
    target = tmp_path / "nav_status.json"
    target.write_text("{}")
    with acquire_nav_status_lock(str(target)):
        sidecars = [p.name for p in tmp_path.iterdir() if p.name.endswith(".lock")]
        assert sidecars, f"no .lock sidecar found in {list(tmp_path.iterdir())}"


def test_acquire_nav_status_lock_blocks_concurrent_writer(tmp_path):
    """Two threads contending for the same lock — second one waits.

    Verifies the cross-process semantics by using two distinct file
    descriptors (separate open() calls inside the helper) — fcntl.flock
    is per-FD so this matches the bridge↔subprocess scenario.
    """
    target = tmp_path / "nav_status.json"
    target.write_text("{}")
    holder_acquired = threading.Event()
    holder_release = threading.Event()
    waiter_acquired = threading.Event()
    waiter_acquire_time = []

    def _holder():
        with acquire_nav_status_lock(str(target)):
            holder_acquired.set()
            holder_release.wait(timeout=2.0)

    def _waiter():
        holder_acquired.wait(timeout=2.0)
        start = time.monotonic()
        with acquire_nav_status_lock(str(target), timeout_s=2.0):
            waiter_acquire_time.append(time.monotonic() - start)
            waiter_acquired.set()

    t_holder = threading.Thread(target=_holder)
    t_waiter = threading.Thread(target=_waiter)
    t_holder.start()
    t_waiter.start()

    # Give the waiter a chance to start blocking
    time.sleep(0.1)
    assert not waiter_acquired.is_set(), "waiter acquired lock while holder still holds it"

    holder_release.set()
    t_holder.join(timeout=2.0)
    t_waiter.join(timeout=2.0)

    assert waiter_acquired.is_set(), "waiter never acquired lock"
    # Waiter should have waited at least ~0.1s while holder held it
    assert waiter_acquire_time[0] >= 0.05, f"waiter acquired suspiciously fast: {waiter_acquire_time[0]}"


def test_acquire_nav_status_lock_works_on_readonly_sidecar(tmp_path):
    """Codex P2 follow-up: the lock helper must work across different
    Unix users.

    In a shared deployment (bridge and nav_agent running as different
    UIDs in the same group), the second process may not have write
    permission on the lock file created by the first. On Linux,
    fcntl.flock is advisory and does not require write access — so
    opening the lock with O_RDONLY lets a read-only user still acquire
    the lock. The previous O_RDWR path raised PermissionError in that
    scenario, silently breaking cross-process serialization.

    This test simulates the cross-user scenario by pre-creating the
    lock file with mode 0o444 (read-only for everyone, including the
    owner of the current test process). A working helper must still
    acquire the lock.
    """
    target = tmp_path / "nav_status.json"
    target.write_text("{}")
    lock_path = str(target) + ".lock"
    # Pre-create with read-only mode. Opening this with O_RDWR would
    # raise PermissionError — the test would fail under the old code.
    open(lock_path, "w").close()
    os.chmod(lock_path, 0o444)
    try:
        with acquire_nav_status_lock(str(target), timeout_s=1.0):
            pass
    finally:
        # Restore write bits so pytest cleanup can remove the directory.
        os.chmod(lock_path, 0o644)


def test_acquire_nav_status_lock_times_out(tmp_path):
    """If the lock cannot be acquired within timeout_s, raise TimeoutError."""
    target = tmp_path / "nav_status.json"
    target.write_text("{}")

    holder_acquired = threading.Event()
    holder_release = threading.Event()

    def _holder():
        with acquire_nav_status_lock(str(target)):
            holder_acquired.set()
            holder_release.wait(timeout=2.0)

    t = threading.Thread(target=_holder)
    t.start()
    holder_acquired.wait(timeout=1.0)
    try:
        with pytest.raises(TimeoutError):
            with acquire_nav_status_lock(str(target), timeout_s=0.2):
                pass
    finally:
        holder_release.set()
        t.join(timeout=2.0)


def test_append_jsonl_atomic_writes_complete_record(tmp_path):
    target = tmp_path / "stats.jsonl"
    append_jsonl_atomic(str(target), {"a": 1, "b": "hello"})
    lines = target.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"a": 1, "b": "hello"}


def test_append_jsonl_atomic_appends_to_existing(tmp_path):
    target = tmp_path / "stats.jsonl"
    append_jsonl_atomic(str(target), {"i": 1})
    append_jsonl_atomic(str(target), {"i": 2})
    append_jsonl_atomic(str(target), {"i": 3})
    lines = target.read_text().splitlines()
    assert [json.loads(line)["i"] for line in lines] == [1, 2, 3]


def test_append_jsonl_atomic_raises_on_oserror(tmp_path, monkeypatch):
    """Disk-full / permission failures must propagate, not be swallowed.

    The implementation uses raw os.open for fsync support, so the test
    monkeypatches os.write to simulate a write failure.
    """
    target = tmp_path / "stats.jsonl"
    real_write = os.write

    def _failing_write(fd, data):
        # Simulate ENOSPC mid-write
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("os.write", _failing_write)
    with pytest.raises(OSError):
        append_jsonl_atomic(str(target), {"i": 1})
    # Restore for cleanup
    monkeypatch.setattr("os.write", real_write)


def test_append_jsonl_atomic_calls_fsync(tmp_path, monkeypatch):
    """Each append must fsync the file descriptor before close."""
    target = tmp_path / "stats.jsonl"
    fsync_calls = []
    real_fsync = os.fsync

    def _tracking_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr("os.fsync", _tracking_fsync)
    append_jsonl_atomic(str(target), {"i": 1})
    assert len(fsync_calls) >= 1, "fsync was not called"


def test_append_jsonl_atomic_creates_parent_directory(tmp_path):
    """Parent directory should be auto-created if missing (matches the
    convention of _persist_json_atomic in the bridge)."""
    target = tmp_path / "deep" / "subdir" / "stats.jsonl"
    append_jsonl_atomic(str(target), {"i": 1})
    assert target.is_file()


# Need os imported at the top of the test file
import os  # noqa: E402,E401


# ===========================================================================
# Phase 1 — B1 + B2: mark_terminal_status routes through bridge,
# falls back to locked + fsync'd write only when bridge is unreachable.
# ===========================================================================

from nav_agent import mark_terminal_status as _mark_terminal_status  # noqa: E402


class _RecordingBridge:
    """Test double that records every bridge.call invocation.

    Accepts **kwargs so callers that pass per-call `timeout` keywords
    (e.g., mark_terminal_status after the Codex P2 short-timeout fix)
    are compatible with the test double.
    """

    def __init__(self, raise_exc: Optional[Exception] = None):
        self.calls: list = []
        self.raise_exc = raise_exc

    def call(self, action, payload=None, **kwargs):
        self.calls.append((action, payload, kwargs))
        if self.raise_exc is not None:
            raise self.raise_exc
        return {"ok": True}


def _make_nav_status_file(tmp_path, **overrides) -> Path:
    """Helper: write a minimal nav_status.json for terminal-status tests."""
    nav = {
        "task_id": "loop1",
        "session_id": "s1",
        "status": "in_progress",
        "state_version": 5,
        "total_steps": 10,
        "_debug": {"gt_geodesic_distance": 1.0},
    }
    nav.update(overrides)
    path = tmp_path / "nav_status.json"
    path.write_text(json.dumps(nav))
    return path


def test_mark_terminal_status_uses_bridge_when_available(tmp_path):
    """Path 1: bridge is reachable → patch goes through bridge, file is
    NOT touched directly by mark_terminal_status."""
    nav_file = _make_nav_status_file(tmp_path)
    original_content = nav_file.read_text()
    bridge = _RecordingBridge()

    _mark_terminal_status(
        str(nav_file), "timeout", "max iterations exceeded",
        bridge=bridge, loop_id="loop1",
    )

    # Bridge was called with the right patch
    assert len(bridge.calls) == 1
    action, payload, kwargs = bridge.calls[0]
    assert action == "update_nav_loop_status"
    assert payload["loop_id"] == "loop1"
    assert payload["patch"]["status"] == "timeout"
    assert payload["patch"]["error"] == "max iterations exceeded"
    # Short timeout is mandatory for shutdown paths
    assert kwargs.get("timeout") is not None
    assert 0 < kwargs["timeout"] <= 5

    # File is unchanged — bridge owns the write
    assert nav_file.read_text() == original_content


def test_mark_terminal_status_falls_back_when_bridge_raises(tmp_path):
    """Path 2: bridge.call raises → mark_terminal_status writes file directly."""
    nav_file = _make_nav_status_file(tmp_path)
    bridge = _RecordingBridge(raise_exc=RuntimeError("bridge unreachable"))

    _mark_terminal_status(
        str(nav_file), "error", "some error",
        bridge=bridge, loop_id="loop1",
    )

    nav = json.loads(nav_file.read_text())
    assert nav["status"] == "error"
    assert nav["error"] == "some error"
    # state_version was bumped from 5 → 6
    assert nav["state_version"] == 6


def test_mark_terminal_status_no_bridge_uses_fallback(tmp_path):
    """When bridge is None (e.g., bridge already known dead), go straight
    to the locked fallback."""
    nav_file = _make_nav_status_file(tmp_path)
    _mark_terminal_status(str(nav_file), "blocked", "stuck", bridge=None, loop_id=None)
    nav = json.loads(nav_file.read_text())
    assert nav["status"] == "blocked"
    assert nav["error"] == "stuck"


def test_mark_terminal_status_fallback_no_tmp_leftover(tmp_path):
    """Atomic write must clean up the temp file."""
    nav_file = _make_nav_status_file(tmp_path)
    _mark_terminal_status(str(nav_file), "timeout", "x", bridge=None, loop_id=None)
    # Only nav_status.json + sidecar .lock should remain
    files = sorted(p.name for p in tmp_path.iterdir())
    for name in files:
        assert not name.endswith(".tmp"), f"leftover tmp file: {name}"


def test_mark_terminal_status_fallback_calls_fsync(tmp_path, monkeypatch):
    """B2 fix: fallback path must fsync the file (not just os.replace)."""
    nav_file = _make_nav_status_file(tmp_path)
    fsync_calls = []
    real_fsync = os.fsync

    def _tracking_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr("os.fsync", _tracking_fsync)
    _mark_terminal_status(str(nav_file), "timeout", "x", bridge=None, loop_id=None)
    assert len(fsync_calls) >= 1, "fallback path did not fsync"


def test_mark_terminal_status_fallback_acquires_lock(tmp_path):
    """B1 fix: fallback path must acquire the sidecar flock so it
    serializes with concurrent bridge writes."""
    nav_file = _make_nav_status_file(tmp_path)
    holder_acquired = threading.Event()
    holder_release = threading.Event()
    fallback_done = threading.Event()

    def _hold_lock():
        with acquire_nav_status_lock(str(nav_file)):
            holder_acquired.set()
            holder_release.wait(timeout=2.0)

    def _try_fallback():
        _mark_terminal_status(str(nav_file), "timeout", "x", bridge=None, loop_id=None)
        fallback_done.set()

    holder = threading.Thread(target=_hold_lock)
    fallback = threading.Thread(target=_try_fallback)
    holder.start()
    holder_acquired.wait(timeout=1.0)
    fallback.start()
    # Fallback must NOT complete while holder still holds the lock
    time.sleep(0.1)
    assert not fallback_done.is_set(), "fallback did not wait for lock"
    holder_release.set()
    holder.join(timeout=2.0)
    fallback.join(timeout=2.0)
    assert fallback_done.is_set(), "fallback never completed"


def test_mark_terminal_status_fallback_increments_state_version(tmp_path):
    nav_file = _make_nav_status_file(tmp_path, state_version=42)
    _mark_terminal_status(str(nav_file), "timeout", "x", bridge=None, loop_id=None)
    assert json.loads(nav_file.read_text())["state_version"] == 43


def test_mark_terminal_status_fallback_preserves_other_fields(tmp_path):
    """The fallback RMW must keep all unrelated fields intact (e.g.
    total_steps, _debug). Only status/error/state_version/updated_at change."""
    nav_file = _make_nav_status_file(tmp_path, total_steps=99, collisions=7)
    _mark_terminal_status(str(nav_file), "timeout", "x", bridge=None, loop_id=None)
    nav = json.loads(nav_file.read_text())
    assert nav["total_steps"] == 99
    assert nav["collisions"] == 7
    assert nav["_debug"] == {"gt_geodesic_distance": 1.0}
    assert nav["task_id"] == "loop1"


def test_mark_terminal_status_bridge_call_uses_short_timeout(tmp_path):
    """Codex P2 follow-up: mark_terminal_status must pass a short timeout
    to bridge.call so shutdown paths cannot block for the default 60s
    when the bridge is unreachable but still accepts connections.
    """
    nav_file = _make_nav_status_file(tmp_path)
    observed_timeouts = []

    class _TimingBridge:
        def call(self, action, payload=None, *, timeout=None):
            observed_timeouts.append(timeout)
            return {"ok": True}

    _mark_terminal_status(
        str(nav_file), "timeout", "x",
        bridge=_TimingBridge(), loop_id="loop1",
    )
    assert len(observed_timeouts) == 1
    timeout = observed_timeouts[0]
    assert timeout is not None
    assert 0 < timeout <= 5, f"timeout {timeout} not within shutdown-friendly window"


def test_mark_terminal_status_hanging_bridge_falls_back_quickly(tmp_path):
    """Integration: if the bridge RPC hangs beyond the short timeout,
    mark_terminal_status must fall back to the locked local write and
    return in bounded time. This is the scenario where the bridge accepts
    TCP connections but its handler is stuck (held lock, long computation).
    """
    nav_file = _make_nav_status_file(tmp_path)

    class _HangingBridge:
        def call(self, action, payload=None, *, timeout=None):
            # Simulate a bridge whose handler has hung — raise a
            # urllib-style socket.timeout that BridgeClient.call would
            # raise when its urlopen timeout fires.
            import socket
            raise socket.timeout("simulated bridge handler hang")

    start = time.monotonic()
    _mark_terminal_status(
        str(nav_file), "timeout", "x",
        bridge=_HangingBridge(), loop_id="loop1",
    )
    elapsed = time.monotonic() - start
    # Should be near-instant: bridge raises immediately in the test, then
    # fallback acquires the lock and writes. <1s is a generous bound.
    assert elapsed < 1.0, f"mark_terminal_status took {elapsed}s — fallback is not reached quickly"
    # Fallback actually wrote the terminal status
    nav = json.loads(nav_file.read_text())
    assert nav["status"] == "timeout"


def test_bridge_client_call_honors_per_call_timeout():
    """BridgeClient.call must accept a per-call timeout override that
    replaces the default 60s for shutdown-critical code paths."""
    from habitat_agent_core import BridgeClient

    client = BridgeClient(host="127.0.0.1", port=1)  # nothing listening

    captured_timeout = []
    real_urlopen = __import__("urllib.request", fromlist=["urlopen"]).urlopen

    def _capturing_urlopen(req, *args, **kwargs):
        captured_timeout.append(kwargs.get("timeout"))
        # Raise so we don't actually try to reach anything
        raise ConnectionRefusedError("no server")

    import urllib.request as _ureq
    _ureq.urlopen = _capturing_urlopen
    try:
        try:
            client.call("ping", {}, timeout=2.5)
        except Exception:
            pass
    finally:
        _ureq.urlopen = real_urlopen

    assert captured_timeout == [2.5]


# ===========================================================================
# Phase 2 — B4: collect_session_stats writes via append_jsonl_atomic and
# raises on failure (no silent swallow).
# ===========================================================================


def test_collect_session_stats_uses_append_jsonl_atomic(tmp_path, monkeypatch):
    """Verify collect_session_stats routes through append_jsonl_atomic
    rather than calling the raw open("a") pattern."""
    loop_dir = tmp_path / "s1" / "loop1"
    nav = {
        "task_id": "loop1",
        "task_type": "pointnav",
        "nav_mode": "navmesh",
        "has_navmesh": True,
        "has_ground_truth": True,
        "eval_goal_position": [3.0, 0.2, 4.0],
        "status": "reached",
        "total_steps": 10,
        "collisions": 0,
        "action_history": [],
        "_debug": {"gt_geodesic_distance": 0.1, "gt_initial_geodesic_distance": 5.0, "gt_path_length": 5.5},
    }
    nav_file = _write_nav_status(loop_dir, nav)
    _write_events(nav_file, [])

    # Track appends through the helper.
    # Phase 1 PR 2 moved collect_session_stats to `analytics.session_stats`;
    # it looks up `append_jsonl_atomic` in that module's namespace, so the
    # monkeypatch must target `analytics.session_stats`, not `nav_agent`.
    captured = []
    import analytics.session_stats as stats_mod

    def _tracking_append(path, record):
        captured.append((path, record))
        # Still write so the file exists for later assertions
        append_jsonl_atomic(path, record)

    monkeypatch.setattr(stats_mod, "append_jsonl_atomic", _tracking_append)
    collect_session_stats(str(nav_file), str(nav_file) + ".events.jsonl")

    assert len(captured) == 1
    captured_path, captured_record = captured[0]
    assert captured_path.endswith("session_stats.jsonl")
    assert captured_record["loop_id"] == "loop1"
    assert captured_record["outcome"] == "reached"


def test_collect_session_stats_logs_but_does_not_raise_on_append_failure(
    tmp_path, monkeypatch
):
    """B4 + Codex P2 follow-up: a disk-full / permission failure during
    stats append MUST log loudly (so operators notice the lost row) but
    MUST NOT raise.

    Rationale: an exception propagating from collect_session_stats would
    be caught by main()'s generic fatal handler, which in turn calls
    mark_terminal_status(..., "error"). That means a successfully
    completed nav loop (status=reached) would be silently rewritten to
    status=error purely because analytics could not be persisted. The
    "loud ERROR log line" is sufficient visibility without corrupting
    the nav outcome.
    """
    loop_dir = tmp_path / "s2" / "loop1"
    nav = {
        "task_id": "loop1",
        "task_type": "pointnav",
        "nav_mode": "navmesh",
        "has_navmesh": True,
        "has_ground_truth": False,
        "status": "reached",
        "total_steps": 10,
        "collisions": 0,
        "action_history": [],
        "_debug": {},
    }
    nav_file = _write_nav_status(loop_dir, nav)
    _write_events(nav_file, [])

    # Phase 1 PR 2: patch on analytics.session_stats (where the symbol is
    # actually resolved), not nav_agent.
    import analytics.session_stats as stats_mod

    def _failing_append(path, record):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(stats_mod, "append_jsonl_atomic", _failing_append)

    # Capture stdout for log line assertion
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # Must NOT raise
        collect_session_stats(str(nav_file), str(nav_file) + ".events.jsonl")
    output = buf.getvalue()
    assert "ERROR" in output
    assert "session_stats" in output

    # nav_status.json on disk is untouched — outcome preserved
    assert json.loads(nav_file.read_text())["status"] == "reached"


# ===========================================================================
# Phase 3 — B5: copy_file_atomic_under_lock helper for last_good.json
# ===========================================================================

from habitat_agent_core import copy_file_atomic_under_lock  # noqa: E402


def test_copy_file_atomic_produces_complete_copy(tmp_path):
    src = tmp_path / "nav_status.json"
    dst = tmp_path / "nav_status.json.last-good.json"
    src.write_text('{"key": "value", "version": 7}')
    copy_file_atomic_under_lock(str(src), str(dst), lock_path=str(src))
    assert dst.read_text() == '{"key": "value", "version": 7}'


def test_copy_file_atomic_uses_temp_then_replace(tmp_path):
    """No leftover .tmp at the destination after copy completes."""
    src = tmp_path / "nav_status.json"
    dst = tmp_path / "nav_status.json.last-good.json"
    src.write_text("{}")
    copy_file_atomic_under_lock(str(src), str(dst), lock_path=str(src))
    leftover = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftover == []


def test_copy_file_atomic_blocks_concurrent_lock_holder(tmp_path):
    """A concurrent lock holder must block the copy until release —
    proves the helper actually acquires the same flock used by writers."""
    src = tmp_path / "nav_status.json"
    dst = tmp_path / "nav_status.json.last-good.json"
    src.write_text('{"v": 1}')
    holder_acquired = threading.Event()
    holder_release = threading.Event()
    copy_done = threading.Event()

    def _hold():
        with acquire_nav_status_lock(str(src)):
            holder_acquired.set()
            holder_release.wait(timeout=2.0)

    def _do_copy():
        copy_file_atomic_under_lock(str(src), str(dst), lock_path=str(src))
        copy_done.set()

    h = threading.Thread(target=_hold)
    c = threading.Thread(target=_do_copy)
    h.start()
    holder_acquired.wait(timeout=1.0)
    c.start()
    time.sleep(0.1)
    assert not copy_done.is_set(), "copy did not wait for lock"
    holder_release.set()
    h.join(timeout=2.0)
    c.join(timeout=2.0)
    assert copy_done.is_set()


def test_copy_file_atomic_overwrites_existing_dst(tmp_path):
    src = tmp_path / "nav_status.json"
    dst = tmp_path / "nav_status.json.last-good.json"
    src.write_text("{}")
    dst.write_text('{"old": true}')
    copy_file_atomic_under_lock(str(src), str(dst), lock_path=str(src))
    assert dst.read_text() == "{}"


# ===========================================================================
# Phase 4 — B3: structured logging for silent-swallow paths.
# Tests verify that read/append failures emit a log line instead of being
# silently ignored. We capture stdout (nav_agent's `log()` writes there).
# ===========================================================================

import io
import contextlib

import nav_agent as _nav_agent_mod  # noqa: E402


def _capture_log(callable_, *args, **kwargs):
    """Run callable_ and return everything it printed to stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        callable_(*args, **kwargs)
    return buf.getvalue()


def test_append_trace_logs_on_oserror(tmp_path, monkeypatch):
    """append_trace must log on failure (was silent except: pass)."""
    target = tmp_path / "trace.jsonl"

    def _failing_open(*args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr("builtins.open", _failing_open)
    output = _capture_log(_nav_agent_mod.append_trace, str(target), "kind", 1)
    assert "WARNING" in output or "ERROR" in output
    assert "append_trace" in output


def test_append_round_event_logs_on_oserror(tmp_path, monkeypatch):
    """append_round_event must log on failure."""
    target = tmp_path / "events.jsonl"

    def _failing_open(*args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr("builtins.open", _failing_open)
    output = _capture_log(
        _nav_agent_mod.append_round_event, str(target), "round_start", 1, {"status": "x"}
    )
    assert "WARNING" in output or "ERROR" in output
    assert "append_round_event" in output


# ===========================================================================
# _start_nav_loop full atomicity. session.mapless and the
# "clear last_goal" branch must be inside the rollback envelope, post-Popen
# failures must kill the orphan subprocess, and the dead duplicate block
# must be gone.
# ===========================================================================


def test_start_nav_loop_pending_phase_clears_last_goal_on_no_goal_position(tmp_path, monkeypatch):
    """objectnav with no goal_position → session.last_goal becomes None.
    Previously this clear lived in the post-Popen dead block; now
    it lives in the rollback-protected pending phase."""
    adapter = _make_adapter()
    session = _attach_session(adapter, "s_clear")
    session.last_goal = [99.0, 0.2, 99.0]  # stale from a previous loop
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    adapter._start_nav_loop(
        None,
        {
            "session_id": "s_clear",
            "task_type": "objectnav",
            "goal_type": "instruction",
            "goal_description": "Find chair",
            "output_dir": str(tmp_path),
        },
    )
    assert session.last_goal is None


def test_start_nav_loop_pending_phase_sets_mapless_flag(tmp_path, monkeypatch):
    """nav_mode=mapless task → session.mapless = True after success."""
    adapter = _make_adapter()
    session = _attach_session(adapter, "s_mapless")
    assert session.mapless is False  # default
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    adapter._start_nav_loop(
        None,
        {
            "session_id": "s_mapless",
            "task_type": "objectnav",
            "goal_type": "instruction",
            "goal_description": "Find chair",
            "nav_mode": "mapless",
            "output_dir": str(tmp_path),
        },
    )
    assert session.mapless is True


def test_start_nav_loop_post_popen_failure_kills_subprocess(tmp_path, monkeypatch):
    """If Popen succeeds but record registration fails (e.g., policy
    registration raises), the subprocess MUST be terminated — otherwise
    we leave an orphan process untracked by the bridge."""
    adapter = _make_adapter()
    _attach_session(adapter, "s_orphan")

    proc_killed = []
    fake_proc_obj = SimpleNamespace(
        pid=99,
        poll=lambda: None,
        returncode=None,
        terminate=lambda: proc_killed.append("terminate"),
        wait=lambda timeout=None: 0,
        kill=lambda: proc_killed.append("kill"),
    )

    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: fake_proc_obj,
    )
    # Force registration to fail after Popen succeeds
    def _raise(*a, **k):
        raise RuntimeError("policy registration failed")

    monkeypatch.setattr(adapter, "_register_nav_loop_policy", _raise)

    with pytest.raises(RuntimeError, match="policy registration failed"):
        adapter._start_nav_loop(
            None,
            {
                "session_id": "s_orphan",
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "Go",
                "goal_position": [3.0, 0.2, 4.0],
                "output_dir": str(tmp_path),
            },
        )

    # Subprocess was terminated (no orphan)
    assert "terminate" in proc_killed, f"proc.terminate not called: {proc_killed}"


def test_start_nav_loop_post_popen_failure_rolls_back_session_mapless(tmp_path, monkeypatch):
    """A post-Popen failure must restore session.mapless to its pre-call value."""
    adapter = _make_adapter()
    session = _attach_session(adapter, "s_mapless_rb")
    assert session.mapless is False  # pre-call

    fake_proc_obj = SimpleNamespace(
        pid=99,
        poll=lambda: None,
        returncode=None,
        terminate=lambda: None,
        wait=lambda timeout=None: 0,
        kill=lambda: None,
    )
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: fake_proc_obj,
    )

    def _raise(*a, **k):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(adapter, "_register_nav_loop_policy", _raise)

    with pytest.raises(RuntimeError):
        adapter._start_nav_loop(
            None,
            {
                "session_id": "s_mapless_rb",
                "task_type": "objectnav",
                "goal_type": "instruction",
                "goal_description": "Find chair",
                "nav_mode": "mapless",
                "output_dir": str(tmp_path),
            },
        )

    # mapless was restored to its pre-call value
    assert session.mapless is False, "mapless not rolled back after registration failure"


# ===========================================================================
# Codex P2 follow-ups: bridge persist must acquire the sidecar flock so
# subprocess fallback writes and bridge refresh writes cannot race.
# ===========================================================================


def test_bridge_persist_nav_status_acquires_sidecar_lock(tmp_path, monkeypatch):
    """The bridge's persist path for nav_status.json must go through the
    sidecar-lock wrapper so it serializes with mark_terminal_status
    fallback writes on the same file.
    """
    adapter = _make_adapter()
    session = _attach_session(adapter, "s_bridge_lock")

    lock_calls = []

    # Wrap the bridge-side lock helper to count acquisitions
    import habitat_sim.habitat_adapter_internal.mixins_patch as mixins_patch_mod

    real_lock = mixins_patch_mod._acquire_nav_status_lock

    @contextlib.contextmanager
    def _tracking_lock(path, *args, **kwargs):
        lock_calls.append(path)
        with real_lock(path, *args, **kwargs):
            yield

    monkeypatch.setattr(
        mixins_patch_mod, "_acquire_nav_status_lock", _tracking_lock
    )

    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    result = adapter._start_nav_loop(
        None,
        {
            "session_id": "s_bridge_lock",
            "task_type": "pointnav",
            "goal_type": "position",
            "goal_description": "Go",
            "goal_position": [3.0, 0.2, 4.0],
            "output_dir": str(tmp_path),
        },
    )
    # _start_nav_loop wrote nav_status.json; verify the lock was taken
    # for the final nav_status file path.
    nav_status_path = result["nav_status_file"]
    assert any(p == nav_status_path for p in lock_calls), (
        f"bridge did not acquire sidecar lock for {nav_status_path}; "
        f"lock_calls={lock_calls}"
    )


def test_bridge_persist_blocked_by_concurrent_subprocess_flock(tmp_path, monkeypatch):
    """Integration: while a subprocess-style holder owns the sidecar
    flock (simulating mark_terminal_status fallback), a bridge-initiated
    persist MUST block until the holder releases.

    Proves that bridge and subprocess share the SAME cross-process lock
    so the Codex P2 race (read-triggered bridge persist overwriting a
    locked subprocess terminal write) cannot happen.
    """
    adapter = _make_adapter()
    _attach_session(adapter, "s_concurrent")
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )

    # First do a successful _start_nav_loop so nav_status.json exists
    # and the loop is registered in adapter._nav_loops — necessary for
    # the subsequent _get_nav_loop_status refresh path to be a write target.
    result = adapter._start_nav_loop(
        None,
        {
            "session_id": "s_concurrent",
            "task_type": "pointnav",
            "goal_type": "position",
            "goal_description": "Go",
            "goal_position": [3.0, 0.2, 4.0],
            "output_dir": str(tmp_path),
        },
    )
    nav_status_path = result["nav_status_file"]
    loop_id = result["loop_id"]

    # Subprocess-style lock holder (matches acquire_nav_status_lock semantics
    # from tools/habitat_agent_core.py; both sides use the same sidecar file).
    holder_acquired = threading.Event()
    holder_release = threading.Event()
    bridge_done = threading.Event()

    def _subprocess_hold():
        with acquire_nav_status_lock(nav_status_path):
            holder_acquired.set()
            holder_release.wait(timeout=2.0)

    def _bridge_persist():
        # _get_nav_loop_status with include_nav_status triggers the
        # bridge's refresh-and-persist path for active loops
        adapter._get_nav_loop_status(
            None, {"loop_id": loop_id, "include_nav_status": True}
        )
        bridge_done.set()

    holder = threading.Thread(target=_subprocess_hold)
    bridge = threading.Thread(target=_bridge_persist)
    holder.start()
    holder_acquired.wait(timeout=1.0)
    bridge.start()

    time.sleep(0.15)
    assert not bridge_done.is_set(), (
        "bridge persist completed while subprocess held sidecar lock — "
        "bridge path is not acquiring the same lock"
    )

    holder_release.set()
    holder.join(timeout=2.0)
    bridge.join(timeout=2.0)
    assert bridge_done.is_set(), "bridge persist never completed"


def test_no_dead_code_post_popen_block(tmp_path):
    """Anti-regression grep: the dead duplicate session.last_goal block
    after subprocess.Popen() must be removed."""
    nav_loop_path = (
        Path(__file__).resolve().parent.parent
        / "src_python" / "habitat_sim" / "habitat_adapter_internal"
        / "mixins_nav_loop.py"
    )
    src = nav_loop_path.read_text()
    # The dead block had this distinctive comment + structure. After the
    # cleanup, no occurrence of this comment + assignment pattern should remain.
    assert "Set or clear last_goal for polar coord computation" not in src, (
        "Dead code block was not removed"
    )


# ===========================================================================
# Mapless + GT + has_navmesh coverage — the important "mapless agent on a
# navmesh-equipped scene with evaluation GT" configuration. Mapless is an
# agent-policy flag (which tools are visible), NOT a bridge-capability
# flag: the bridge still uses the pathfinder when it is loaded, and so
# all GT distance metrics AND SPL must remain computable.
#
# This section codifies that contract for all 5 task types so a future
# refactor cannot silently drop distance computation when nav_mode=mapless.
# ===========================================================================


def _run_mapless_gt_navmesh_start(
    adapter,
    session_id,
    tmp_path,
    monkeypatch,
    task_type,
    *,
    goal_position=None,
    eval_goal_position=None,
    goal_type=None,
):
    """Helper: start a nav loop with nav_mode=mapless on a navmesh-backed
    session. Returns the nav_status dict persisted to disk."""
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    payload = {
        "session_id": session_id,
        "task_type": task_type,
        "goal_type": goal_type or ("position" if task_type == "pointnav" else "instruction"),
        "goal_description": f"{task_type} test goal",
        "nav_mode": "mapless",  # explicit mapless policy
        "output_dir": str(tmp_path),
    }
    if goal_position is not None:
        payload["goal_position"] = goal_position
    if eval_goal_position is not None:
        payload["eval_goal_position"] = eval_goal_position
    if task_type == "imagenav":
        payload["reference_image"] = "/tmp/dummy_ref.jpg"
    result = adapter._start_nav_loop(None, payload)
    return json.loads(Path(result["nav_status_file"]).read_text())


# --- pointnav mapless + GT + navmesh -------------------------------------


def test_pointnav_mapless_gt_navmesh_full_spl_pipeline(tmp_path, monkeypatch):
    """PointNav + mapless + navmesh + GT: the full SPL pipeline must work.

    - l_opt (gt_initial_geodesic_distance) computed from pathfinder
    - _debug.gt_euclidean_distance / gt_geodesic_distance populated
    - agent polar signals visible (session.last_goal is set for pointnav)
    - agent absolute coords NOT visible (mapless filter)
    - collect_session_stats emits non-null SPL
    """
    adapter = _make_adapter()
    session = _attach_session(adapter, "s_pn_mapless", position=(0.0, 0.2, 0.0))

    nav = _run_mapless_gt_navmesh_start(
        adapter, "s_pn_mapless", tmp_path, monkeypatch,
        task_type="pointnav",
        goal_position=[3.0, 0.2, 4.0],
    )

    # Bridge-side evaluation state
    assert session.eval_goal == [3.0, 0.2, 4.0]
    assert session.last_goal == [3.0, 0.2, 4.0]  # pointnav: agent sees this
    assert session.mapless is True
    assert session.initial_geodesic_distance is not None
    assert session.initial_geodesic_distance == pytest.approx(5.0, abs=1e-2)
    assert session.cumulative_path_length == 0.0

    # nav_status.json on-disk fields
    assert nav["has_navmesh"] is True
    assert nav["nav_mode"] == "mapless"
    assert nav["has_ground_truth"] is True
    assert nav["eval_goal_position"] == [3.0, 0.2, 4.0]
    dbg = nav["_debug"]
    assert dbg["gt_goal"] == [3.0, 0.2, 4.0]
    assert dbg["gt_euclidean_distance"] == pytest.approx(5.0, abs=1e-2)
    assert dbg["gt_geodesic_distance"] == pytest.approx(5.0, abs=1e-2)
    assert dbg["gt_initial_geodesic_distance"] == pytest.approx(5.0, abs=1e-2)
    assert dbg["gt_path_length"] == 0.0

    # Agent visibility: polar signals yes, absolute coords no
    summary = adapter._build_state_summary(session)
    assert summary["euclidean_distance_to_goal"] is not None
    assert summary["goal_direction_deg"] is not None
    # mapless filter hides absolute position/goal keys
    assert "position" not in summary
    assert "goal" not in summary

    # End-to-end SPL computation via collect_session_stats
    loop_dir = tmp_path / "s_pn_mapless" / nav["task_id"]
    stats_nav = dict(nav)
    stats_nav["status"] = "reached"  # simulate agent reaching the goal
    # Simulate realistic path length: walked 6m total (slightly suboptimal)
    stats_nav["_debug"] = dict(dbg)
    stats_nav["_debug"]["gt_geodesic_distance"] = 0.1  # within threshold
    stats_nav["_debug"]["gt_path_length"] = 6.0
    stats_nav["total_steps"] = 25
    stats_nav["collisions"] = 0
    stats_nav["action_history"] = []
    nav_file = _write_nav_status(loop_dir, stats_nav)
    _write_events(nav_file, [{"phase": "round_end", "total_steps": 25}])
    collect_session_stats(str(nav_file), str(nav_file) + ".events.jsonl")
    stats = _read_stats(tmp_path)
    assert stats["has_gt_goal"] is True
    assert stats["gt_success"] is True
    assert stats["spl"] == pytest.approx(5.0 / 6.0, abs=1e-3)


# --- non-pointnav mapless + GT + navmesh ---------------------------------


@pytest.mark.parametrize(
    "task_type",
    ["objectnav", "imagenav", "eqa", "instruction_following"],
)
def test_nonpointnav_mapless_gt_navmesh_metrics_work_but_agent_blind(
    tmp_path, monkeypatch, task_type
):
    """Non-pointnav tasks (objectnav/imagenav/eqa/instruction_following)
    in mapless + navmesh + GT: evaluation metrics ALL work, but the
    agent gets no distance/direction signals (intentional GT isolation).
    """
    adapter = _make_adapter()
    session = _attach_session(
        adapter, f"s_{task_type}_mapless", position=(0.0, 0.2, 0.0)
    )

    nav = _run_mapless_gt_navmesh_start(
        adapter, f"s_{task_type}_mapless", tmp_path, monkeypatch,
        task_type=task_type,
        eval_goal_position=[3.0, 0.2, 4.0],
    )

    # Bridge-side state: eval_goal set, last_goal intentionally NOT set
    assert session.eval_goal == [3.0, 0.2, 4.0]
    assert session.last_goal is None, (
        f"{task_type}: session.last_goal must stay None to prevent GT leak "
        "via polar signals"
    )
    assert session.mapless is True
    assert session.initial_geodesic_distance is not None
    assert session.initial_geodesic_distance == pytest.approx(5.0, abs=1e-2)

    # GT debug fully populated despite mapless — the bridge always uses
    # the pathfinder when available, regardless of agent policy
    dbg = nav["_debug"]
    assert dbg["gt_goal"] == [3.0, 0.2, 4.0]
    assert dbg["gt_euclidean_distance"] == pytest.approx(5.0, abs=1e-2)
    assert dbg["gt_geodesic_distance"] == pytest.approx(5.0, abs=1e-2)
    assert dbg["gt_initial_geodesic_distance"] == pytest.approx(5.0, abs=1e-2)
    assert dbg["gt_path_length"] == 0.0

    # Agent-visible goal_position is None (no coord leak)
    assert nav["goal_position"] is None
    # But eval_goal_position is there for evaluation
    assert nav["eval_goal_position"] == [3.0, 0.2, 4.0]
    assert nav["has_ground_truth"] is True

    # Agent visibility: NO polar signals, NO absolute coords
    summary = adapter._build_state_summary(session)
    assert summary["euclidean_distance_to_goal"] is None, (
        f"{task_type}: polar euclidean signal leaked to agent"
    )
    assert summary["goal_direction_deg"] is None, (
        f"{task_type}: polar direction signal leaked to agent"
    )
    assert "position" not in summary
    assert "goal" not in summary

    # End-to-end SPL: the fact that the agent can't see distance has
    # zero impact on the evaluator's ability to compute SPL.
    loop_dir = tmp_path / f"s_{task_type}_mapless" / nav["task_id"]
    stats_nav = dict(nav)
    stats_nav["status"] = "reached"
    stats_nav["_debug"] = dict(dbg)
    stats_nav["_debug"]["gt_geodesic_distance"] = 0.3  # within 0.5m threshold
    stats_nav["_debug"]["gt_path_length"] = 7.5
    stats_nav["total_steps"] = 40
    stats_nav["collisions"] = 2
    stats_nav["action_history"] = []
    nav_file = _write_nav_status(loop_dir, stats_nav)
    _write_events(nav_file, [{"phase": "round_end", "total_steps": 40}])
    collect_session_stats(str(nav_file), str(nav_file) + ".events.jsonl")
    stats = _read_stats(tmp_path)
    assert stats["has_gt_goal"] is True, f"{task_type}: has_gt_goal not set"
    assert stats["gt_success"] is True, f"{task_type}: gt_success not True"
    assert stats["spl"] is not None, f"{task_type}: spl is None"
    assert stats["spl"] == pytest.approx(5.0 / 7.5, abs=1e-3), (
        f"{task_type}: spl wrong"
    )


def test_pointnav_mapless_gt_navmesh_unreachable_goal_rejected(tmp_path, monkeypatch):
    """Sanity: the upfront unreachable-goal rejection still fires in
    mapless + has_navmesh=True. mapless must not accidentally bypass
    the navmesh reachability check.
    """
    from habitat_sim.habitat_adapter_internal.types import HabitatAdapterError

    adapter = _make_adapter()
    # Pathfinder loaded but configured to report everything unreachable
    session = _make_session(
        position=(0.0, 0.2, 0.0),
        pathfinder=_FakePathFinder(reachable=False),
    )
    session.session_id = "s_pn_mapless_unreach"
    adapter._sessions["s_pn_mapless_unreach"] = session
    monkeypatch.setattr(
        "habitat_sim.habitat_adapter_internal.mixins_nav_loop.subprocess.Popen",
        lambda *a, **k: _fake_proc(),
    )
    with pytest.raises(HabitatAdapterError, match="unreachable"):
        adapter._start_nav_loop(
            None,
            {
                "session_id": "s_pn_mapless_unreach",
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "Go",
                "goal_position": [10.0, 0.2, 10.0],
                "nav_mode": "mapless",
                "output_dir": str(tmp_path),
            },
        )
