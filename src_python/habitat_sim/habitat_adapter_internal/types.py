from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import habitat_sim

_DEFAULT_MAX_OBSERVATION_ELEMENTS = 2048
_DEFAULT_DEPTH_VIS_MAX = 10.0
_DEFAULT_VISUAL_OUTPUT_DIR = "/tmp/habitat_gs_visuals"
_DEFAULT_TOPDOWN_METERS_PER_PIXEL = 0.05
_DEFAULT_VIDEO_FPS = 6.0
_API_VERSION = "habitat-gs/v1"
_MAX_CLOSED_NAV_LOOPS = 64
_MAX_NAV_ACTION_HISTORY = 200
_MAX_NAVIGATE_STEP_BURST = 32
_NAV_STATUS_ALLOWED_VALUES = frozenset(
    {"in_progress", "reached", "blocked", "error", "timeout"}
)
_NAV_STATUS_PATCH_FIELDS = frozenset(
    {
        "substeps",
        "current_substep_index",
        "status",
        "nav_phase",
        "total_steps",
        "collisions",
        "current_position",
        "geodesic_distance",
        "rooms_discovered",
        "last_visual",
        "last_action",
        "action_history_append",
        "spatial_memory_file",
        "spatial_memory_append",
        "finding",
        "error",
        "capability_request",
    }
)
_NAV_STATUS_IMMUTABLE_FIELDS = frozenset(
    {
        "task_id",
        "task_type",
        "nav_mode",
        "has_navmesh",
        "is_gaussian",
        "goal_type",
        "goal_description",
        "goal_position",
        "target_object",
        "reference_image",
        "session_id",
        "state_version",
        "updated_at",
    }
)
_NAV_LOOP_SCRIPT = str(Path(__file__).resolve().parents[3] / "tools" / "nav_agent.py")

SUPPORTED_ACTIONS = (
    "describe_api",
    "init_scene",
    "get_scene_info",
    "set_agent_state",
    "sample_navigable_point",
    "find_shortest_path",
    "get_topdown_map",
    "navigate_step",
    "step_action",
    "step_and_capture",
    "get_observation",
    "get_visuals",
    "export_video_trace",
    "get_metrics",
    "get_runtime_status",
    "close_session",
    "start_nav_loop",
    "get_nav_loop_status",
    "update_nav_loop_status",
    "stop_nav_loop",
    "get_panorama",
    "analyze_depth",
    "query_depth",
    "get_scene_graph",
)


class HabitatAdapterError(RuntimeError):
    """Raised for user-facing protocol errors in bridge requests."""


@dataclass
class _Session:
    session_id: str
    simulator: habitat_sim.Simulator
    scene: str
    settings: Dict[str, Any]
    agent_id: int = 0
    step_count: int = 0
    last_action: Optional[str] = None
    last_sensor_obs: Optional[Dict[str, Any]] = None
    created_at_s: float = 0.0
    last_activity_s: float = 0.0
    trajectory: List[List[float]] = field(default_factory=list)
    collision_points: List[Dict[str, Any]] = field(default_factory=list)
    last_goal: Optional[List[float]] = None
    last_collision: Optional[Dict[str, Any]] = None
    mapless: bool = False
    is_gaussian: bool = False  # True if scene uses GS rendering
    capture_counter: int = 0  # unique per-capture counter for image filenames
    # SPL metrics: accumulate real path length independently of trajectory cap
    cumulative_path_length: float = 0.0
    # Start-of-episode geodesic distance to goal (l_opt in SPL formula).
    # Set when start_nav_loop is called and not updated afterwards.
    initial_geodesic_distance: Optional[float] = None
    # Evaluation ground-truth goal — invisible to the agent.
    # Used only by _build_debug_snapshot and SPL computation.
    # For pointnav this equals last_goal; for other tasks this is independent
    # so the agent never sees eval coordinates or polar signals derived from them.
    eval_goal: Optional[List[float]] = None
    # Precomputed scene graph loaded from room_object_scene_graph.json at init_scene.
    # None if the file was not found for the current scene.
    scene_graph: Optional[Dict[str, Any]] = None
    # Visual robot (third-person camera proxy)
    visual_robot_enabled: bool = False
    visual_robot_kind: Optional[str] = None  # "articulated" | "rigid"
    visual_robot_obj: Any = None
    visual_robot_template_handle: Optional[str] = None


@dataclass
class _NavLoopRecord:
    loop_id: str
    process: subprocess.Popen[Any]
    session_id: str
    task_type: str
    nav_mode: str
    has_navmesh: bool
    nav_status_file: str
    spatial_memory_file: str
    log_file: str
    started_at_s: float
    nav_status: Dict[str, Any]
    state_version: int
    ended_at_s: Optional[float] = None
    returncode: Optional[int] = None
