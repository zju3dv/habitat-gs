from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .types import _API_VERSION


class HabitatAdapterApiMixin:
    """Public adapter API description and contract metadata."""

    def _describe_api(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        del session_id, payload
        return {
            "api_version": _API_VERSION,
            "actions": {
                "describe_api": {
                    "session_required": False,
                    "payload": {},
                },
                "init_scene": {
                    "session_required": False,
                    "payload": {
                        "scene": "str (required)",
                        "scene_dataset_config_file": "str (optional)",
                        "default_agent_navmesh": "bool (optional; auto-disabled for GS dataset configs)",
                        "default_agent": "int (optional, default 0)",
                        "seed": "int (optional)",
                        "frustum_culling": "bool (optional)",
                        "enable_physics": (
                            "bool (optional; default from habitat_sim default_sim_settings, "
                            "typically built_with_bullet; auto false for GS dataset configs when omitted)"
                        ),
                        "sensor": {
                            "width": "int",
                            "height": "int",
                            "sensor_height": "float",
                            "hfov": "float",
                            "zfar": "float",
                            "color_sensor": "bool",
                            "depth_sensor": "bool",
                            "semantic_sensor": "bool",
                            "third_person_color_sensor": (
                                "bool (optional, default false; enables an "
                                "over-the-shoulder RGB camera at ~1.5m behind "
                                "and 1.2m above the agent)"
                            ),
                            "third_person_sensor_uuid": (
                                "str (optional, default 'third_rgb_sensor'; "
                                "name of the third-person sensor in visuals "
                                "and observation payloads)"
                            ),
                        },
                    },
                },
                "get_scene_info": {
                    "session_required": True,
                    "payload": {
                        "refresh_observation": "bool (optional, default false)",
                    },
                },
                "set_agent_state": {
                    "session_required": True,
                    "payload": {
                        "position": "list[float] length 3 (required)",
                        "rotation": "list[float] length 4 (optional)",
                        "snap_to_navmesh": "bool (optional, default false)",
                        "infer_sensor_states": "bool (optional, default true)",
                        "include_observation_data": "bool (optional, default false)",
                        "max_observation_elements": "int (optional, default 2048)",
                    },
                },
                "sample_navigable_point": {
                    "session_required": True,
                    "payload": {
                        "near": "list[float] length 3 (optional)",
                        "distance": "float (optional, default 3.0)",
                        "max_tries": "int (optional, default 100)",
                        "seed": "int (optional)",
                    },
                },
                "find_shortest_path": {
                    "session_required": True,
                    "payload": {
                        "start": "list[float] length 3 (optional, default current pose)",
                        "end": "list[float] length 3 (required)",
                        "snap_start": "bool (optional, default true)",
                        "snap_end": "bool (optional, default true)",
                    },
                },
                "get_topdown_map": {
                    "session_required": True,
                    "payload": {
                        "output_dir": "str (optional, default /tmp/habitat_gs_visuals)",
                        "height": "float (optional, default current agent y)",
                        "meters_per_pixel": "float (optional, default 0.05)",
                        "goal": "list[float] length 3 (optional)",
                        "path_points": "list[list[float]] (optional)",
                    },
                },
                "navigate_step": {
                    "session_required": True,
                    "payload": {
                        "goal": "list[float] length 3 (required)",
                        "goal_radius": "float (optional)",
                        "max_steps": "int (optional, default 1) — greedy actions to execute in one call, clamped to 32",
                        "dt": "float (optional, default 1/60)",
                        "include_observation_data": "bool (optional, default false)",
                        "max_observation_elements": "int (optional, default 2048)",
                        "include_metrics": "bool (optional, default true)",
                        "include_visuals": "bool (optional, default false) — include exported visuals in response; trace frames are still recorded for replay",
                        "include_publish_hints": "bool (optional, default false)",
                        "output_dir": "str (optional, default /tmp/habitat_gs_visuals)",
                        "sensors": "list[str] (optional)",
                        "depth_max": "float (optional, default 10.0)",
                    },
                },
                "step_action": {
                    "session_required": True,
                    "payload": {
                        "action": "str|int (required) — move_forward | turn_left | turn_right",
                        "degrees": "float (optional) — for turn_left/turn_right: rotate this many degrees, auto-decomposed into 10°/step atomic turns (e.g. degrees=30 → 3 steps)",
                        "distance": "float (optional) — for move_forward: advance this many metres, auto-decomposed into 0.25m/step atomic steps (e.g. distance=1.0 → 4 steps); stops early on collision",
                        "dt": "float (optional, default 1/60)",
                        "include_observation_data": "bool (optional, default false)",
                        "max_observation_elements": "int (optional, default 2048)",
                    },
                },
                "step_and_capture": {
                    "session_required": True,
                    "payload": {
                        "action": "str|int (required) — move_forward | turn_left | turn_right",
                        "degrees": "float (optional) — for turn_left/turn_right: rotate this many degrees, auto-decomposed into 10°/step atomic turns",
                        "distance": "float (optional) — for move_forward: advance this many metres, auto-decomposed into 0.25m/step atomic steps; stops early on collision",
                        "dt": "float (optional, default 1/60)",
                        "include_observation_data": "bool (optional, default false)",
                        "max_observation_elements": "int (optional, default 2048)",
                        "output_dir": "str (optional, default /tmp/habitat_gs_visuals)",
                        "sensors": "list[str] (optional)",
                        "depth_max": "float (optional, default 10.0)",
                        "include_metrics": "bool (optional, default true)",
                        "include_publish_hints": "bool (optional, default true)",
                    },
                },
                "get_observation": {
                    "session_required": True,
                    "payload": {
                        "refresh": "bool (optional, default false)",
                        "include_observation_data": "bool (optional, default false)",
                        "max_observation_elements": "int (optional, default 2048)",
                    },
                },
                "get_visuals": {
                    "session_required": True,
                    "payload": {
                        "refresh": "bool (optional, default false)",
                        "output_dir": "str (optional, default /tmp/habitat_gs_visuals)",
                        "sensors": "list[str] (optional)",
                        "depth_max": "float (optional, default 10.0)",
                        "include_metrics": "bool (optional, default false)",
                    },
                },
                "export_video_trace": {
                    "session_required": True,
                    "payload": {
                        "output_dir": "str (optional, default /tmp/habitat_gs_visuals)",
                        "sensor": "str (optional, default color_sensor)",
                        "fps": "float (optional, default 6.0)",
                        "step_start": "int (optional)",
                        "step_end": "int (optional)",
                        "include_metrics": "bool (optional, default true)",
                        "include_publish_hints": "bool (optional, default true)",
                    },
                },
                "get_metrics": {
                    "session_required": True,
                    "payload": {},
                },
                "get_runtime_status": {
                    "session_required": False,
                    "payload": {
                        "include_nav_status": "bool (optional, default false) — include canonical nav_status for active and recently closed loops",
                    },
                },
                "get_scene_graph": {
                    "session_required": True,
                    "payload": {
                        "query_type": "str (required) — 'all', 'room', or 'object'",
                        "room_type": "str (optional) — filter rooms by type label (e.g. 'kitchen', 'bedroom')",
                        "object_label": "str (optional) — filter objects by label (e.g. 'chair', 'table')",
                        "max_results": "int (optional, default 10) — max nodes returned",
                    },
                },
                "close_session": {
                    "session_required": True,
                    "payload": {},
                },
                "start_nav_loop": {
                    "session_required": False,
                    "payload": {
                        "task_type": "str (optional, default 'pointnav') — pointnav, objectnav, imagenav, instruction_following, eqa",
                        "nav_mode": "str (optional) — 'navmesh' enables map-planning tools; 'mapless' forbids agent map access even when navmesh exists; defaults from current scene navmesh availability",
                        "goal_type": "str (required) — 'position' or 'instruction'",
                        "goal_description": "str (required) — human-readable goal description",
                        "goal_position": "list[float] length 3 (optional) — target coordinates for position mode; required when goal_type='position'",
                        "session_id": "str (required) — active bridge session id",
                        "max_iterations": "int (optional, default 50) — max nav loop iterations before timeout",
                        "agent_timeout": "int (optional, default 120) — seconds to wait per agent round",
                        "workspace_host": "str (optional) — host workspace root; auto-derived from output_dir/session_id/loop_id/nav_status.json when omitted",
                        "reference_image": "str (optional) — path to reference image for imagenav tasks",
                    },
                },
                "get_nav_loop_status": {
                    "session_required": False,
                    "payload": {
                        "loop_id": "str (required) — loop id returned by start_nav_loop",
                        "include_nav_status": "bool (optional, default false) — include canonical nav_status in response",
                    },
                },
                "update_nav_loop_status": {
                    "session_required": False,
                    "payload": {
                        "loop_id": "str (required) — loop id returned by start_nav_loop",
                        "expected_version": "int (optional) — optimistic concurrency check against nav_status.state_version",
                        "patch": {
                            "substeps": "list (optional)",
                            "current_substep_index": "int >= 0 (optional)",
                            "status": "str (optional) — in_progress|reached|blocked|error|timeout",
                            "nav_phase": "str (optional)",
                            "total_steps": "int >= 0 (optional)",
                            "collisions": "int >= 0 (optional)",
                            "current_position": "list[float] length 3 or null (optional)",
                            "geodesic_distance": "float or null (optional)",
                            "rooms_discovered": "list (optional)",
                            "last_visual": "json value (optional)",
                            "last_action": "json value (optional)",
                            "action_history_append": "list (optional) — appended to existing history",
                            "spatial_memory_file": "str (optional)",
                            "spatial_memory_append": "list (optional) — observation entries appended to spatial memory file",
                            "finding": "json value (optional)",
                            "error": "str|null (optional)",
                        },
                    },
                },
                "stop_nav_loop": {
                    "session_required": False,
                    "payload": {
                        "loop_id": "str (required) — loop id returned by start_nav_loop",
                    },
                },
                "get_panorama": {
                    "session_required": True,
                    "payload": {
                        "include_depth_analysis": "bool (optional, default false) — include per-direction depth analysis",
                        "clearance_threshold": "float (optional, default 0.5) — clearance in meters for depth analysis",
                    },
                },
                "analyze_depth": {
                    "session_required": True,
                    "payload": {
                        "clearance_threshold": "float (optional, default 0.5) — min distance in meters to consider clear",
                    },
                },
                "query_depth": {
                    "session_required": True,
                    "payload": {
                        "points": "list of [u, v] pixel coordinates (optional)",
                        "bbox": "[x1, y1, x2, y2] pixel region (optional)",
                    },
                },
            },
            "supported_actions": list(self.SUPPORTED_ACTIONS),
        }
