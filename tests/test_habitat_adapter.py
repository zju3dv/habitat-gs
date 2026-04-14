# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest

# Installed wheels may omit habitat_adapter*; merge repo src_python into the
# existing habitat_sim package search path so submodules resolve (importlib
# loading only habitat_adapter.py breaks relative imports to habitat_adapter_internal).
_OPENCLAW_SRC_HABITAT_SIM = (
    Path(__file__).resolve().parent.parent / "src_python" / "habitat_sim"
)
try:
    from habitat_sim.habitat_adapter import HabitatAdapter
except ModuleNotFoundError:
    import habitat_sim

    _extra = str(_OPENCLAW_SRC_HABITAT_SIM)
    if _extra not in list(habitat_sim.__path__):
        habitat_sim.__path__.append(_extra)
    from habitat_sim.habitat_adapter import HabitatAdapter


class _FakeAgent:
    def __init__(self) -> None:
        self._state = SimpleNamespace(
            position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        self.agent_config = SimpleNamespace(
            action_space={
                "move_forward": SimpleNamespace(name="move_forward"),
                "turn_left": SimpleNamespace(name="turn_left"),
                "turn_right": SimpleNamespace(name="turn_right"),
            }
        )

    def get_state(self) -> Any:
        return SimpleNamespace(
            position=np.array(self._state.position, dtype=np.float32),
            rotation=np.array(self._state.rotation, dtype=np.float32),
        )

    def set_state(self, state: Any, infer_sensor_states: bool = True) -> None:
        del infer_sensor_states
        self._state = SimpleNamespace(
            position=np.array(state.position, dtype=np.float32),
            rotation=np.array(state.rotation, dtype=np.float32),
        )


class _FakePathFinder:
    def __init__(self, is_loaded: bool = True) -> None:
        self.is_loaded = is_loaded
        self.navigable_area = 12.5
        self._last_seed = None
        self.loaded_navmesh_path: str | None = None

    def seed(self, seed: int) -> None:
        self._last_seed = seed

    def get_bounds(self) -> Any:
        return (
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([4.0, 2.0, 4.0], dtype=np.float32),
        )

    def load_nav_mesh(self, navmesh_path: str) -> None:
        self.loaded_navmesh_path = navmesh_path
        self.is_loaded = True

    def get_random_navigable_point(self) -> np.ndarray:
        return np.array([1.25, 0.0, 1.75], dtype=np.float32)

    def get_random_navigable_point_near(
        self, point: np.ndarray, distance: float, max_tries: int = 100
    ) -> np.ndarray:
        del max_tries
        return np.array(
            [float(point[0]) + min(distance, 0.5), float(point[1]), float(point[2])],
            dtype=np.float32,
        )

    def snap_point(self, point: np.ndarray) -> np.ndarray:
        return np.array(point, dtype=np.float32)

    def is_navigable(self, point: np.ndarray) -> bool:
        return bool(point[0] >= 0.0)

    def island_radius(self, point: np.ndarray) -> float:
        del point
        return 3.5

    def find_path(self, path: Any) -> bool:
        start = np.array(path.requested_start, dtype=np.float32)
        end = np.array(path.requested_end, dtype=np.float32)
        if end[0] < 0.0:
            path.geodesic_distance = 0.0
            path.points = []
            return False

        path.geodesic_distance = float(np.linalg.norm(end - start))
        path.points = [start, ((start + end) / 2.0), end]
        return True

    def get_topdown_view(self, meters_per_pixel: float, height: float) -> np.ndarray:
        del meters_per_pixel, height
        return np.array(
            [
                [0, 1, 1, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.uint8,
        )


class _FakeFollower:
    def __init__(self, agent: _FakeAgent, goal_radius: float | None) -> None:
        self._agent = agent
        self._goal_radius = 0.25 if goal_radius is None else goal_radius

    def next_action_along(self, goal: np.ndarray) -> Any:
        position = self._agent.get_state().position
        if np.linalg.norm(np.asarray(goal, dtype=np.float32) - position) <= self._goal_radius:
            return None
        return "move_forward"


class _FakeSimulator:
    def __init__(self, pathfinder: _FakePathFinder | None = None) -> None:
        self._agent = _FakeAgent()
        self.pathfinder = pathfinder or _FakePathFinder()
        self._frame = 0
        self._previous_step_time = 0.004
        self.closed = False
        self.closed_destroy: bool | None = None

    def reset(self) -> Dict[str, Any]:
        return {
            "color_sensor": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth_sensor": np.full((2, 2), 1.0, dtype=np.float32),
            "collided": False,
        }

    def step(self, action: Any, dt: float = 1.0 / 60.0) -> Dict[str, Any]:
        del dt
        self._frame += 1
        state = self._agent.get_state()
        if action == "move_forward":
            next_state = SimpleNamespace(
                position=np.array(
                    [state.position[0] + 0.5, state.position[1], state.position[2]],
                    dtype=np.float32,
                ),
                rotation=state.rotation,
            )
            self._agent.set_state(next_state)
        return {
            "color_sensor": np.full((2, 2, 3), self._frame, dtype=np.uint8),
            "depth_sensor": np.full((2, 2), 2.0 + self._frame, dtype=np.float32),
            "collided": bool(action == "turn_left"),
        }

    def get_sensor_observations(self, agent_ids: int = 0) -> Dict[str, Any]:
        del agent_ids
        return {
            "color_sensor": np.full((2, 2, 3), 7, dtype=np.uint8),
            "depth_sensor": np.full((2, 2), 3.0, dtype=np.float32),
            "collided": False,
        }

    def get_agent(self, agent_id: int) -> _FakeAgent:
        del agent_id
        return self._agent

    def make_greedy_follower(
        self, agent_id: int = 0, goal_radius: float | None = None
    ) -> _FakeFollower:
        del agent_id
        return _FakeFollower(self._agent, goal_radius)

    def close(self, destroy: bool = True) -> None:
        self.closed = True
        self.closed_destroy = destroy


class _FakePopen:
    def __init__(self, args: Any, env: Dict[str, Any], stdout: Any, stderr: Any, close_fds: bool) -> None:
        self.args = args
        self.env = env
        self.stdout = stdout
        self.stderr = stderr
        self.close_fds = close_fds
        self.pid = 4321
        self.returncode = None

    def poll(self) -> Any:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0
        return 0


def _patch_popen(monkeypatch: pytest.MonkeyPatch) -> list[_FakePopen]:
    import importlib

    # Prefer the refactored internal module (src_python version); fall back to
    # the legacy monolithic facade which imports subprocess at top level.
    try:
        target_mod = importlib.import_module(
            "habitat_sim.habitat_adapter_internal.mixins_nav_loop"
        )
    except ModuleNotFoundError:
        target_mod = sys.modules[HabitatAdapter.__module__]

    procs: list[_FakePopen] = []

    def _factory(args: Any, env: Dict[str, Any], stdout: Any, stderr: Any, close_fds: bool) -> _FakePopen:
        proc = _FakePopen(args, env, stdout, stderr, close_fds)
        procs.append(proc)
        return proc

    monkeypatch.setattr(target_mod.subprocess, "Popen", _factory)
    return procs


def test_describe_api_contract() -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    response = adapter.handle_request({"action": "describe_api", "payload": {}})

    assert response["ok"] is True
    assert response["action"] == "describe_api"
    supported = response["result"]["supported_actions"]
    assert "init_scene" in supported
    assert "get_scene_info" in supported
    assert "find_shortest_path" in supported
    assert "navigate_step" in supported
    assert "step_action" in supported
    assert "step_and_capture" in supported
    assert "get_visuals" in supported
    assert "export_video_trace" in supported
    init_payload = response["result"]["actions"]["init_scene"]["payload"]
    assert "default_agent_navmesh" in init_payload
    topdown_payload = response["result"]["actions"]["get_topdown_map"]["payload"]
    assert "meters_per_pixel" in topdown_payload
    navigate_payload = response["result"]["actions"]["navigate_step"]["payload"]
    assert "goal" in navigate_payload
    assert "max_steps" in navigate_payload
    visuals_payload = response["result"]["actions"]["get_visuals"]["payload"]
    assert "include_metrics" in visuals_payload
    video_payload = response["result"]["actions"]["export_video_trace"]["payload"]
    assert "sensor" in video_payload
    assert "fps" in video_payload
    capture_payload = response["result"]["actions"]["step_and_capture"]["payload"]
    assert "include_publish_hints" in capture_payload
    nav_loop_payload = response["result"]["actions"]["start_nav_loop"]["payload"]
    assert "task_type" in nav_loop_payload
    assert "nav_mode" in nav_loop_payload
    assert "forbids agent map access" in nav_loop_payload["nav_mode"]
    update_payload = response["result"]["actions"]["update_nav_loop_status"]["payload"]
    assert "patch" in update_payload
    assert "expected_version" in update_payload


def test_session_lifecycle_and_metrics() -> None:
    captured_settings: Dict[str, Any] = {}

    def _factory(settings: Dict[str, Any]) -> _FakeSimulator:
        captured_settings.update(settings)
        return _FakeSimulator()

    adapter = HabitatAdapter(simulator_factory=_factory)
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "request_id": "req-init",
            "action": "init_scene",
            "payload": {
                "scene": existing_scene,
                "sensor": {"width": 320, "height": 240, "depth_sensor": True},
            },
        }
    )

    assert init_response["ok"] is True
    session_id = init_response["session_id"]
    assert isinstance(session_id, str)
    assert captured_settings["scene"] == existing_scene
    assert captured_settings["width"] == 320
    assert captured_settings["height"] == 240
    assert captured_settings["depth_sensor"] is True

    step_response = adapter.handle_request(
        {
            "request_id": "req-step",
            "action": "step_action",
            "session_id": session_id,
            "payload": {"action": "turn_left"},
        }
    )
    assert step_response["ok"] is True
    assert step_response["result"]["step_count"] == 1
    assert step_response["result"]["collided"] is True

    metrics_response = adapter.handle_request(
        {
            "request_id": "req-metrics",
            "action": "get_metrics",
            "session_id": session_id,
            "payload": {},
        }
    )
    assert metrics_response["ok"] is True
    metrics = metrics_response["result"]
    assert metrics["step_count"] == 1
    assert metrics["agent_state"]["position"] == [1.0, 2.0, 3.0]
    assert metrics["step_time_s"] == 0.004

    close_response = adapter.handle_request(
        {
            "request_id": "req-close",
            "action": "close_session",
            "session_id": session_id,
            "payload": {},
        }
    )
    assert close_response["ok"] is True
    assert close_response["result"]["closed"] is True

    invalid_after_close = adapter.handle_request(
        {
            "request_id": "req-after-close",
            "action": "get_metrics",
            "session_id": session_id,
            "payload": {},
        }
    )
    assert invalid_after_close["ok"] is False
    assert invalid_after_close["error"]["type"] == "HabitatAdapterError"
    assert adapter._sessions == {}


def test_close_session_uses_destroy_true() -> None:
    simulator = _FakeSimulator()
    adapter = HabitatAdapter(simulator_factory=lambda _: simulator)
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]

    close_response = adapter.handle_request(
        {
            "action": "close_session",
            "session_id": session_id,
            "payload": {},
        }
    )

    assert close_response["ok"] is True
    assert simulator.closed is True
    assert simulator.closed_destroy is True


def test_observation_data_truncation() -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {
                "scene": existing_scene,
                "include_observation_data": True,
                "max_observation_elements": 5,
            },
        }
    )
    observation = init_response["result"]["observation"]["color_sensor"]

    assert observation["num_elements"] == 12
    assert observation["truncated"] is True
    assert len(observation["data"]) == 5


def test_get_visuals_exports_png_files(tmp_path: Path) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]

    visuals_response = adapter.handle_request(
        {
            "action": "get_visuals",
            "session_id": session_id,
            "payload": {
                "output_dir": str(tmp_path),
            },
        }
    )

    assert visuals_response["ok"] is True
    visuals = visuals_response["result"]["visuals"]

    color_item = visuals["color_sensor"]
    depth_item = visuals["depth_sensor"]

    assert color_item["ok"] is True
    assert depth_item["ok"] is True
    assert color_item["mime_type"] == "image/png"
    assert depth_item["mime_type"] == "image/png"
    assert color_item["path"].endswith(".png")
    assert depth_item["path"].endswith(".png")
    assert Path(color_item["path"]).exists()
    assert Path(depth_item["path"]).exists()


def test_navigation_actions_and_topdown_map(tmp_path: Path) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]

    scene_info = adapter.handle_request(
        {
            "action": "get_scene_info",
            "session_id": session_id,
            "payload": {},
        }
    )
    assert scene_info["ok"] is True
    assert scene_info["result"]["navmesh_loaded"] is True
    assert scene_info["result"]["navigable_area"] == pytest.approx(12.5)

    sampled = adapter.handle_request(
        {
            "action": "sample_navigable_point",
            "session_id": session_id,
            "payload": {
                "near": [1.0, 2.0, 3.0],
                "distance": 0.5,
                "seed": 7,
            },
        }
    )
    assert sampled["ok"] is True
    assert sampled["result"]["is_navigable"] is True
    assert sampled["result"]["point"] == pytest.approx([1.5, 2.0, 3.0])

    set_state = adapter.handle_request(
        {
            "action": "set_agent_state",
            "session_id": session_id,
            "payload": {
                "position": [0.5, 2.0, 3.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "include_observation_data": True,
            },
        }
    )
    assert set_state["ok"] is True
    assert set_state["result"]["agent_state"]["position"] == pytest.approx(
        [0.5, 2.0, 3.0]
    )

    shortest_path = adapter.handle_request(
        {
            "action": "find_shortest_path",
            "session_id": session_id,
            "payload": {"end": [2.0, 2.0, 3.0]},
        }
    )
    assert shortest_path["ok"] is True
    assert shortest_path["result"]["reachable"] is True
    assert len(shortest_path["result"]["path_points"]) == 3

    topdown = adapter.handle_request(
        {
            "action": "get_topdown_map",
            "session_id": session_id,
            "payload": {
                "output_dir": str(tmp_path),
                "goal": [2.0, 2.0, 3.0],
                "path_points": shortest_path["result"]["path_points"],
            },
        }
    )
    assert topdown["ok"] is True
    topdown_item = topdown["result"]["topdown_map"]
    assert Path(topdown_item["path"]).exists()

    navigate = adapter.handle_request(
        {
            "action": "navigate_step",
            "session_id": session_id,
            "payload": {
                "goal": [2.0, 2.0, 3.0],
                "include_visuals": True,
                "include_metrics": True,
                "include_publish_hints": True,
                "output_dir": str(tmp_path),
            },
        }
    )
    assert navigate["ok"] is True
    assert navigate["result"]["nav_status"] == "en_route"
    assert navigate["result"]["action"] == "move_forward"
    assert navigate["result"]["metrics"]["agent_state"]["position"] == pytest.approx(
        [1.0, 2.0, 3.0]
    )
    assert "publish_hints" in navigate["result"]


def test_navigate_step_unreachable() -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]

    response = adapter.handle_request(
        {
            "action": "navigate_step",
            "session_id": session_id,
            "payload": {"goal": [-1.0, 0.0, 0.0]},
        }
    )

    assert response["ok"] is True
    assert response["result"]["nav_status"] == "unreachable"
    assert response["result"]["reachable"] is False


def test_navigate_step_no_visuals_still_records_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]

    navigate = adapter.handle_request(
        {
            "action": "navigate_step",
            "session_id": session_id,
            "payload": {
                "goal": [2.0, 2.0, 3.0],
                "include_visuals": False,
                "include_publish_hints": False,
                "output_dir": str(tmp_path),
            },
        }
    )
    assert navigate["ok"] is True
    navigate_result = navigate["result"]
    assert navigate_result["nav_status"] == "en_route"
    assert "visuals" not in navigate_result
    assert "publish_hints" not in navigate_result

    color_frames = sorted(tmp_path.glob(f"{session_id}_step*_color_sensor.png"))
    depth_frames = sorted(tmp_path.glob(f"{session_id}_step*_depth_sensor.png"))
    assert len(color_frames) == 1
    assert len(depth_frames) == 0

    def _fake_encode_video_trace(
        frame_paths: list[str], video_path: str, fps: float
    ) -> None:
        assert len(frame_paths) == 1
        assert fps == pytest.approx(4.0)
        Path(video_path).write_bytes(b"fake-mp4")

    monkeypatch.setattr(adapter, "_encode_video_trace", _fake_encode_video_trace)

    exported = adapter.handle_request(
        {
            "action": "export_video_trace",
            "session_id": session_id,
            "payload": {
                "output_dir": str(tmp_path),
                "sensor": "color_sensor",
                "fps": 4.0,
            },
        }
    )
    assert exported["ok"] is True
    assert exported["result"]["video"]["frame_count"] == 1


def test_navigate_step_honors_max_steps() -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]

    navigate = adapter.handle_request(
        {
            "action": "navigate_step",
            "session_id": session_id,
            "payload": {
                "goal": [4.0, 2.0, 3.0],
                "max_steps": 3,
                "include_metrics": True,
            },
        }
    )

    assert navigate["ok"] is True
    result = navigate["result"]
    assert result["nav_status"] == "en_route"
    assert result["max_steps"] == 3
    assert result["steps_executed"] == 3
    assert result["actions_taken"] == ["move_forward", "move_forward", "move_forward"]
    assert result["action"] == "move_forward"
    assert result["step_count"] == 3
    assert result["metrics"]["agent_state"]["position"] == pytest.approx([2.5, 2.0, 3.0])


def test_resolve_n_steps_compound_params(tmp_path: Path) -> None:
    """_resolve_n_steps converts degrees/distance to step counts correctly."""
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {"action": "init_scene", "payload": {"scene": existing_scene}}
    )
    session_id = init_response["session_id"]

    # turn_right with degrees=90 → 9 steps, no collision → steps_taken == 9
    resp = adapter.handle_request(
        {
            "action": "step_action",
            "session_id": session_id,
            "payload": {"action": "turn_right", "degrees": 90},
        }
    )
    assert resp["ok"] is True
    assert resp["result"]["steps_taken"] == 9
    assert resp["result"]["collided"] is False

    # move_forward with distance=1.0 → 4 steps, no collision → steps_taken == 4
    resp = adapter.handle_request(
        {
            "action": "step_action",
            "session_id": session_id,
            "payload": {"action": "move_forward", "distance": 1.0},
        }
    )
    assert resp["ok"] is True
    assert resp["result"]["steps_taken"] == 4
    assert resp["result"]["collided"] is False

    # turn_left with degrees=30 → 3 steps, but turn_left always collides → stops at 1
    resp = adapter.handle_request(
        {
            "action": "step_action",
            "session_id": session_id,
            "payload": {"action": "turn_left", "degrees": 30},
        }
    )
    assert resp["ok"] is True
    assert resp["result"]["steps_taken"] == 1
    assert resp["result"]["collided"] is True

    # negative degrees → abs() → same as positive; turn_right -90 → 9 steps
    resp = adapter.handle_request(
        {
            "action": "step_action",
            "session_id": session_id,
            "payload": {"action": "turn_right", "degrees": -90},
        }
    )
    assert resp["ok"] is True
    assert resp["result"]["steps_taken"] == 9
    assert resp["result"]["collided"] is False

    # no degrees/distance → 1 step
    resp = adapter.handle_request(
        {
            "action": "step_action",
            "session_id": session_id,
            "payload": {"action": "turn_right"},
        }
    )
    assert resp["ok"] is True
    assert resp["result"]["steps_taken"] == 1

    # step_and_capture with distance=0.5 → 2 steps
    resp = adapter.handle_request(
        {
            "action": "step_and_capture",
            "session_id": session_id,
            "payload": {
                "action": "move_forward",
                "distance": 0.5,
                "output_dir": str(tmp_path),
            },
        }
    )
    assert resp["ok"] is True
    assert resp["result"]["steps_taken"] == 2
    assert resp["result"]["collided"] is False


def test_step_action_reports_collisions_without_navmesh() -> None:
    adapter = HabitatAdapter(
        simulator_factory=lambda _: _FakeSimulator(
            pathfinder=_FakePathFinder(is_loaded=False)
        )
    )
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {"action": "init_scene", "payload": {"scene": existing_scene}}
    )
    session_id = init_response["session_id"]

    response = adapter.handle_request(
        {
            "action": "step_action",
            "session_id": session_id,
            "payload": {"action": "turn_left"},
        }
    )

    assert response["ok"] is True
    assert response["result"]["collided"] is True


def test_validation_errors() -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())

    bad_action = adapter.handle_request({"action": "unknown", "payload": {}})
    assert bad_action["ok"] is False
    assert bad_action["error"]["type"] == "ValueError"

    bad_payload = adapter.handle_request({"action": "init_scene", "payload": []})
    assert bad_payload["ok"] is False
    assert bad_payload["error"]["type"] == "ValueError"


def test_init_scene_disables_default_navmesh_for_gaussian_dataset(
    tmp_path: Path,
) -> None:
    dataset_cfg = tmp_path / "train.scene_dataset_config.json"
    navmesh_dir = tmp_path / "train" / "interior_0405_840145"
    navmesh_dir.mkdir(parents=True)
    navmesh_path = navmesh_dir / "interior_0405_840145.navmesh"
    navmesh_path.write_bytes(b"navmesh")
    dataset_cfg.write_text(
        """
{
  "stages": {
    "paths": {
      ".gs.ply": [
        "train/interior_0405_840145/*.gs.ply"
      ]
    },
    "default_attributes": {
      "render_asset_type": "gaussian_splatting"
    }
  },
  "navmesh_instances": {
    "interior_0405_840145": "train/interior_0405_840145/interior_0405_840145.navmesh"
  }
}
""".strip(),
        encoding="utf-8",
    )

    captured_settings: Dict[str, Any] = {}
    fake_pathfinder = _FakePathFinder(is_loaded=False)

    def _factory(settings: Dict[str, Any]) -> _FakeSimulator:
        captured_settings.update(settings)
        return _FakeSimulator(pathfinder=fake_pathfinder)

    adapter = HabitatAdapter(simulator_factory=_factory)
    response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {
                "scene": "interior_0405_840145",
                "scene_dataset_config_file": str(dataset_cfg),
            },
        }
    )

    assert response["ok"] is True
    assert captured_settings["default_agent_navmesh"] is False
    assert captured_settings["enable_physics"] is False
    assert fake_pathfinder.loaded_navmesh_path == str(navmesh_path.resolve())


def test_init_scene_physics_default_non_gaussian_matches_builtin_settings(
    tmp_path: Path,
) -> None:
    from habitat_sim.utils.settings import default_sim_settings

    dataset_cfg = tmp_path / "mesh.scene_dataset_config.json"
    dataset_cfg.write_text(
        """
{
  "stages": {
    "paths": {
      ".glb": ["models/scene.glb"]
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    captured_settings: Dict[str, Any] = {}

    def _factory(settings: Dict[str, Any]) -> _FakeSimulator:
        captured_settings.update(settings)
        return _FakeSimulator(pathfinder=_FakePathFinder(is_loaded=False))

    adapter = HabitatAdapter(simulator_factory=_factory)
    existing_scene = str(Path(__file__).resolve())
    response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {
                "scene": existing_scene,
                "scene_dataset_config_file": str(dataset_cfg),
            },
        }
    )

    assert response["ok"] is True
    assert captured_settings["enable_physics"] == default_sim_settings["enable_physics"]


def test_start_nav_loop_initializes_extended_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    procs = _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    response = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "navigate to target",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["task_type"] == "pointnav"
    assert result["nav_mode"] == "navmesh"
    assert result["has_navmesh"] is True
    assert result["state_version"] == 1
    assert len(procs) == 1
    assert procs[0].args[1].endswith("nav_agent.py")

    nav_status = json.loads(nav_status_path.read_text())
    assert nav_status["task_type"] == "pointnav"
    assert nav_status["nav_mode"] == "navmesh"
    assert nav_status["has_navmesh"] is True
    assert nav_status["substeps"] == []
    assert nav_status["current_substep_index"] == 0
    assert nav_status["nav_phase"] == "decomposing"
    assert nav_status["rooms_discovered"] == []
    assert nav_status["goal_position"] == [2.0, 2.0, 3.0]
    assert nav_status["state_version"] == 1

    spatial_memory_path = Path(result["spatial_memory_file"])
    assert spatial_memory_path.exists()
    assert spatial_memory_path.name == f"spatial_memory_{result['loop_id']}.json"
    assert nav_status["spatial_memory_file"] == str(spatial_memory_path)
    spatial_memory = json.loads(spatial_memory_path.read_text())
    assert spatial_memory["snapshots"] == []
    assert spatial_memory["rooms"] == {}
    assert spatial_memory["object_sightings"] == {}


def test_start_nav_loop_uses_distinct_spatial_memory_file_per_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    _patch_popen(monkeypatch)

    start_one = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(tmp_path / "nav_status_1.json"),
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "navigate to target 1",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )
    start_two = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(tmp_path / "nav_status_2.json"),
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "navigate to target 2",
                "goal_position": [4.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )

    assert start_one["ok"] is True
    assert start_two["ok"] is True

    result_one = start_one["result"]
    result_two = start_two["result"]
    spatial_one = Path(result_one["spatial_memory_file"])
    spatial_two = Path(result_two["spatial_memory_file"])

    assert spatial_one != spatial_two
    assert spatial_one.name == f"spatial_memory_{result_one['loop_id']}.json"
    assert spatial_two.name == f"spatial_memory_{result_two['loop_id']}.json"
    assert spatial_one.exists()
    assert spatial_two.exists()

    memory_one = json.loads(spatial_one.read_text())
    memory_two = json.loads(spatial_two.read_text())
    assert memory_one["task_id"] == result_one["loop_id"]
    assert memory_two["task_id"] == result_two["loop_id"]


def test_start_nav_loop_defaults_to_mapless_without_navmesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(
        simulator_factory=lambda _: _FakeSimulator(pathfinder=_FakePathFinder(is_loaded=False))
    )
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    response = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "goal_type": "instruction",
                "goal_description": "find the doorway",
                "session_id": session_id,
            },
        }
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["nav_mode"] == "mapless"
    assert result["has_navmesh"] is False

    nav_status = json.loads(nav_status_path.read_text())
    assert nav_status["nav_mode"] == "mapless"
    assert nav_status["has_navmesh"] is False


def test_start_nav_loop_uses_gateway_env_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    procs = _patch_popen(monkeypatch)
    monkeypatch.setenv("OPENCLAW_GATEWAY_HOST", "gateway.internal")
    monkeypatch.setenv("OPENCLAW_GATEWAY_PORT", "22334")
    nav_status_path = tmp_path / "nav_status.json"

    response = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "use gateway env defaults",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )

    assert response["ok"] is True
    assert len(procs) == 1
    assert procs[0].env["OPENCLAW_GATEWAY_HOST"] == "gateway.internal"
    assert procs[0].env["OPENCLAW_GATEWAY_PORT"] == "22334"


def test_start_nav_loop_respects_explicit_nav_mode_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    response = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "nav_mode": "mapless",
                "goal_type": "position",
                "goal_description": "force mapless mode",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["nav_mode"] == "mapless"
    assert result["has_navmesh"] is True

    nav_status = json.loads(nav_status_path.read_text())
    assert nav_status["nav_mode"] == "mapless"
    assert nav_status["has_navmesh"] is True


def test_mapless_policy_is_released_after_loop_process_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    procs = _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    start = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "nav_mode": "mapless",
                "goal_type": "position",
                "goal_description": "mapless test",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )
    assert start["ok"] is True
    assert len(procs) == 1

    blocked = adapter.handle_request(
        {
            "action": "find_shortest_path",
            "session_id": session_id,
            "payload": {"end": [2.0, 2.0, 3.0]},
        }
    )
    assert blocked["ok"] is False
    assert blocked["error"]["type"] == "HabitatAdapterError"
    assert "forbidden while session" in blocked["error"]["message"]

    procs[0].returncode = 0

    allowed = adapter.handle_request(
        {
            "action": "find_shortest_path",
            "session_id": session_id,
            "payload": {"end": [2.0, 2.0, 3.0]},
        }
    )
    assert allowed["ok"] is True
    assert allowed["result"]["reachable"] is True


def test_reap_idle_sessions_zero_timeout_still_cleans_finished_loops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    procs = _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    start = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "nav_mode": "mapless",
                "goal_type": "position",
                "goal_description": "mapless test",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )
    assert start["ok"] is True
    assert session_id in adapter._session_mapless_loop_ids

    procs[0].returncode = 0
    expired = adapter.reap_idle_sessions(0)

    assert expired == []
    assert session_id not in adapter._session_mapless_loop_ids


def test_update_nav_loop_status_applies_patch_and_bumps_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    start = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "navigate to target",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )
    assert start["ok"] is True
    loop_id = start["result"]["loop_id"]

    update = adapter.handle_request(
        {
            "action": "update_nav_loop_status",
            "payload": {
                "loop_id": loop_id,
                "expected_version": 1,
                "patch": {
                    "substeps": [
                        {
                            "action": "navigate_to",
                            "goal_position": [2.0, 2.0, 3.0],
                            "status": "in_progress",
                        },
                        {"action": "verify_arrival", "status": "pending"},
                    ],
                    "nav_phase": "navigating",
                    "total_steps": 3,
                    "current_position": [1.5, 2.0, 3.0],
                    "geodesic_distance": 1.25,
                    "action_history_append": [{"action": "navigate"}],
                },
            },
        }
    )

    assert update["ok"] is True
    result = update["result"]
    assert result["state_version"] == 2
    assert result["nav_status"]["state_version"] == 2
    assert result["nav_status"]["nav_phase"] == "navigating"
    assert result["nav_status"]["total_steps"] == 3
    history = result["nav_status"]["action_history"]
    assert len(history) == 1
    assert history[0]["action"] == "navigate"
    assert history[0]["step"] == 3
    assert "step_count" not in history[0]
    assert history[0]["pos"] == pytest.approx([1.5, 2.0, 3.0])

    nav_status = json.loads(nav_status_path.read_text())
    assert nav_status["state_version"] == 2
    assert nav_status["current_position"] == [1.5, 2.0, 3.0]
    assert nav_status["geodesic_distance"] == pytest.approx(1.25)


def test_update_nav_loop_status_autogenerates_history_entry_from_last_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    start = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "navigate to target",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )
    assert start["ok"] is True
    loop_id = start["result"]["loop_id"]

    update = adapter.handle_request(
        {
            "action": "update_nav_loop_status",
            "payload": {
                "loop_id": loop_id,
                "expected_version": 1,
                "patch": {
                    "nav_phase": "navigating",
                    "total_steps": 1,
                    "current_position": [1.5, 2.0, 3.0],
                    "last_action": {"last_action": "turn_left", "collided": False},
                },
            },
        }
    )
    assert update["ok"] is True
    history = update["result"]["nav_status"]["action_history"]
    assert len(history) == 1
    assert history[0]["step"] == 1
    assert history[0]["action"] == "turn_left"
    assert history[0]["collided"] is False
    assert history[0]["pos"] == pytest.approx([1.5, 2.0, 3.0])


def test_update_nav_loop_status_rejects_mapless_motion_without_visual_grounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    start = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "nav_mode": "mapless",
                "goal_type": "position",
                "goal_description": "mapless visual-first",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )
    assert start["ok"] is True
    loop_id = start["result"]["loop_id"]

    rejected = adapter.handle_request(
        {
            "action": "update_nav_loop_status",
            "payload": {
                "loop_id": loop_id,
                "expected_version": 1,
                "patch": {
                    "nav_phase": "navigating",
                    "total_steps": 1,
                    "current_position": [1.5, 2.0, 3.0],
                    "action_history_append": [
                        {"step": 1, "action": "forward_0.5m", "collided": False}
                    ],
                },
            },
        }
    )

    assert rejected["ok"] is False
    assert rejected["error"]["type"] == "HabitatAdapterError"
    assert 'must include "last_visual"' in rejected["error"]["message"]


def test_update_nav_loop_status_accepts_mapless_motion_with_visual_grounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    start = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "nav_mode": "mapless",
                "goal_type": "position",
                "goal_description": "mapless visual-first",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )
    assert start["ok"] is True
    loop_id = start["result"]["loop_id"]

    accepted = adapter.handle_request(
        {
            "action": "update_nav_loop_status",
            "payload": {
                "loop_id": loop_id,
                "expected_version": 1,
                "patch": {
                    "nav_phase": "navigating",
                    "total_steps": 1,
                    "current_position": [1.5, 2.0, 3.0],
                    "last_visual": {
                        "sensor": "color_sensor",
                        "path": "/tmp/frame_0001.png",
                    },
                    "last_action": {
                        "step": 1,
                        "action": "forward_0.5m",
                        "collided": False,
                        "saw": "front corridor looks clear for one short move",
                    },
                    "action_history_append": [
                        {
                            "step": 1,
                            "action": "forward_0.5m",
                            "collided": False,
                            "saw": "front corridor looks clear for one short move",
                        }
                    ],
                },
            },
        }
    )

    assert accepted["ok"] is True
    history = accepted["result"]["nav_status"]["action_history"]
    assert len(history) == 1
    assert history[0]["action"] == "forward_0.5m"
    assert history[0]["saw"] == "front corridor looks clear for one short move"
    assert accepted["result"]["nav_status"]["last_visual"]["path"] == "/tmp/frame_0001.png"


def test_update_nav_loop_status_rejects_closed_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    start = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "navigate to target",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )
    assert start["ok"] is True
    loop_id = start["result"]["loop_id"]

    stop = adapter.handle_request(
        {
            "action": "stop_nav_loop",
            "payload": {"loop_id": loop_id},
        }
    )
    assert stop["ok"] is True

    closed_update = adapter.handle_request(
        {
            "action": "update_nav_loop_status",
            "payload": {
                "loop_id": loop_id,
                "patch": {"nav_phase": "navigating"},
            },
        }
    )

    assert closed_update["ok"] is False
    assert closed_update["error"]["type"] == "HabitatAdapterError"
    assert "not running" in closed_update["error"]["message"]


def test_stop_nav_loop_forces_terminal_status_for_in_progress_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    start = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "navigate to target",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )
    assert start["ok"] is True
    loop_id = start["result"]["loop_id"]

    stop = adapter.handle_request(
        {
            "action": "stop_nav_loop",
            "payload": {"loop_id": loop_id},
        }
    )
    assert stop["ok"] is True

    status = adapter.handle_request(
        {
            "action": "get_nav_loop_status",
            "payload": {"loop_id": loop_id, "include_nav_status": True},
        }
    )
    assert status["ok"] is True
    nav_status = status["result"]["nav_status"]
    assert nav_status["status"] == "error"
    assert "stop_nav_loop" in nav_status["error"]
    assert nav_status["state_version"] >= 2

    file_nav_status = json.loads(nav_status_path.read_text())
    assert file_nav_status["status"] == "error"
    assert "stop_nav_loop" in file_nav_status["error"]
    assert file_nav_status["state_version"] >= 2


def test_update_nav_loop_status_rejects_version_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    start = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "navigate to target",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )
    assert start["ok"] is True
    loop_id = start["result"]["loop_id"]

    conflict = adapter.handle_request(
        {
            "action": "update_nav_loop_status",
            "payload": {
                "loop_id": loop_id,
                "expected_version": 2,
                "patch": {"nav_phase": "navigating"},
            },
        }
    )

    assert conflict["ok"] is False
    assert conflict["error"]["type"] == "HabitatAdapterError"
    assert "expected_version mismatch" in conflict["error"]["message"]


def test_update_nav_loop_status_rejects_immutable_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    start = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "navigate to target",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )
    assert start["ok"] is True
    loop_id = start["result"]["loop_id"]

    immutable_error = adapter.handle_request(
        {
            "action": "update_nav_loop_status",
            "payload": {
                "loop_id": loop_id,
                "expected_version": 1,
                "patch": {"goal_description": "mutate immutable"},
            },
        }
    )

    assert immutable_error["ok"] is False
    assert immutable_error["error"]["type"] == "HabitatAdapterError"
    assert "immutable fields" in immutable_error["error"]["message"]


def test_get_nav_loop_status_returns_canonical_when_file_is_corrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {
            "action": "init_scene",
            "payload": {"scene": existing_scene},
        }
    )
    session_id = init_response["session_id"]
    _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    start = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "navigate to target",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )
    assert start["ok"] is True
    loop_id = start["result"]["loop_id"]

    update = adapter.handle_request(
        {
            "action": "update_nav_loop_status",
            "payload": {
                "loop_id": loop_id,
                "expected_version": 1,
                "patch": {"nav_phase": "navigating"},
            },
        }
    )
    assert update["ok"] is True

    nav_status_path.write_text('{"broken":', encoding="utf-8")

    status = adapter.handle_request(
        {
            "action": "get_nav_loop_status",
            "payload": {"loop_id": loop_id, "include_nav_status": True},
        }
    )

    assert status["ok"] is True
    result = status["result"]
    assert result["state_version"] == 2
    assert result["nav_status"]["task_id"] == loop_id
    assert result["nav_status"]["state_version"] == 2
    assert "nav_status_file_error" in result


def test_update_nav_loop_status_spatial_memory_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())
    init_response = adapter.handle_request(
        {"action": "init_scene", "payload": {"scene": existing_scene}}
    )
    session_id = init_response["session_id"]
    _patch_popen(monkeypatch)
    nav_status_path = tmp_path / "nav_status.json"

    start = adapter.handle_request(
        {
            "action": "start_nav_loop",
            "payload": {
                "nav_status_file": str(nav_status_path),
                "task_type": "pointnav",
                "goal_type": "position",
                "goal_description": "navigate to target",
                "goal_position": [2.0, 2.0, 3.0],
                "session_id": session_id,
            },
        }
    )
    assert start["ok"] is True
    loop_id = start["result"]["loop_id"]
    spatial_memory_file = start["result"]["spatial_memory_file"]
    assert os.path.isfile(spatial_memory_file)

    # First append: kitchen observation
    update1 = adapter.handle_request(
        {
            "action": "update_nav_loop_status",
            "payload": {
                "loop_id": loop_id,
                "expected_version": 1,
                "patch": {
                    "nav_phase": "navigating",
                    "spatial_memory_append": [
                        {
                            "position": [1.0, 0.2, -1.5],
                            "heading_deg": 90.0,
                            "scene_description": "Kitchen with dining table",
                            "room_label": "kitchen",
                            "objects_detected": ["table", "chair"],
                        }
                    ],
                },
            },
        }
    )
    assert update1["ok"] is True

    memory = json.loads(Path(spatial_memory_file).read_text())
    assert len(memory["snapshots"]) == 1
    assert memory["snapshots"][0]["scene_description"] == "Kitchen with dining table"
    assert memory["snapshots"][0]["room_label"] == "kitchen"
    assert "kitchen" in memory["rooms"]
    assert memory["rooms"]["kitchen"]["visit_count"] == 1
    assert memory["object_sightings"]["table"]["count"] == 1
    assert memory["object_sightings"]["chair"]["count"] == 1

    # Second append: same room, new objects
    update2 = adapter.handle_request(
        {
            "action": "update_nav_loop_status",
            "payload": {
                "loop_id": loop_id,
                "expected_version": 2,
                "patch": {
                    "spatial_memory_append": [
                        {
                            "position": [1.5, 0.2, -1.0],
                            "heading_deg": 180.0,
                            "scene_description": "Kitchen from another angle, microwave visible",
                            "room_label": "kitchen",
                            "objects_detected": ["microwave"],
                        }
                    ],
                },
            },
        }
    )
    assert update2["ok"] is True

    memory2 = json.loads(Path(spatial_memory_file).read_text())
    assert len(memory2["snapshots"]) == 2
    assert memory2["rooms"]["kitchen"]["visit_count"] == 2
    assert memory2["object_sightings"]["microwave"]["count"] == 1
    assert memory2["object_sightings"]["table"]["count"] == 1  # preserved from first


# ---------------------------------------------------------------------------
# Regression: init_scene sensor-capture failure must not leak orphaned session
# (Codex review P2 — mixins_session_scene.py:160)
# ---------------------------------------------------------------------------


def test_init_scene_capture_failure_does_not_leak_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """If _capture_sensor_observations raises during init_scene, the session
    must be removed from _sessions (no orphaned entry leaking resources)."""
    from unittest.mock import patch

    adapter = HabitatAdapter(simulator_factory=lambda _: _FakeSimulator())
    existing_scene = str(Path(__file__).resolve())

    # Patch _capture_sensor_observations on the adapter instance to simulate a
    # renderer/sensor failure during the first capture in init_scene.
    with patch.object(
        adapter,
        "_capture_sensor_observations",
        side_effect=RuntimeError("simulated sensor capture failure"),
    ):
        response = adapter.handle_request(
            {"action": "init_scene", "payload": {"scene": existing_scene}}
        )

    assert response["ok"] is False
    # The orphaned session must have been cleaned up — registry must be empty.
    assert adapter._sessions == {}, (
        "init_scene left an orphaned session after capture failure"
    )


# ---------------------------------------------------------------------------
# Regression: step_action / navigate_step must not call get_sensor_observations
# after step() when no visual robot is active (Codex review P1 — mixins_navigation.py:747)
# ---------------------------------------------------------------------------


def test_step_action_skips_get_sensor_observations_when_no_visual_robot() -> None:
    """When visual robot is disabled (the default), step_action must reuse the
    observations already returned by simulator.step() and must NOT call
    get_sensor_observations(), which would trigger a redundant second render."""
    fake_sim = _FakeSimulator()
    get_sensor_call_count: list[int] = [0]
    _original_gso = fake_sim.get_sensor_observations

    def _spied_get_sensor(agent_ids: int = 0) -> Dict[str, Any]:
        get_sensor_call_count[0] += 1
        return _original_gso(agent_ids)

    fake_sim.get_sensor_observations = _spied_get_sensor  # type: ignore[method-assign]

    adapter = HabitatAdapter(simulator_factory=lambda _: fake_sim)
    existing_scene = str(Path(__file__).resolve())

    init_resp = adapter.handle_request(
        {"action": "init_scene", "payload": {"scene": existing_scene}}
    )
    assert init_resp["ok"] is True
    session_id = init_resp["session_id"]
    calls_after_init = get_sensor_call_count[0]

    # Execute one step_action; visual robot is not active by default.
    adapter.handle_request(
        {
            "action": "step_action",
            "session_id": session_id,
            "payload": {"action": "move_forward"},
        }
    )

    extra_calls = get_sensor_call_count[0] - calls_after_init
    assert extra_calls == 0, (
        f"step_action triggered {extra_calls} redundant get_sensor_observations "
        "call(s) — step() output should be reused directly when no visual robot is active"
    )


def test_navigate_step_skips_get_sensor_observations_when_no_visual_robot() -> None:
    """Same redundant-render check as above but for the navigate_step hot loop."""
    fake_sim = _FakeSimulator()
    get_sensor_call_count: list[int] = [0]
    _original_gso = fake_sim.get_sensor_observations

    def _spied_get_sensor(agent_ids: int = 0) -> Dict[str, Any]:
        get_sensor_call_count[0] += 1
        return _original_gso(agent_ids)

    fake_sim.get_sensor_observations = _spied_get_sensor  # type: ignore[method-assign]

    adapter = HabitatAdapter(simulator_factory=lambda _: fake_sim)
    existing_scene = str(Path(__file__).resolve())

    init_resp = adapter.handle_request(
        {"action": "init_scene", "payload": {"scene": existing_scene}}
    )
    assert init_resp["ok"] is True
    session_id = init_resp["session_id"]
    calls_after_init = get_sensor_call_count[0]

    # Navigate one step toward a nearby goal (reachable in max_steps=1).
    adapter.handle_request(
        {
            "action": "navigate_step",
            "session_id": session_id,
            "payload": {
                "goal": [1.5, 2.0, 3.0],
                "max_steps": 1,
            },
        }
    )

    extra_calls = get_sensor_call_count[0] - calls_after_init
    assert extra_calls == 0, (
        f"navigate_step triggered {extra_calls} redundant get_sensor_observations "
        "call(s) during navigation loop — step() output should be reused directly"
    )
