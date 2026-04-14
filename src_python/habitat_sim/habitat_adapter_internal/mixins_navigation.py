from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import numpy as np

import habitat_sim

from .types import (
    _DEFAULT_DEPTH_VIS_MAX,
    _DEFAULT_MAX_OBSERVATION_ELEMENTS,
    _DEFAULT_TOPDOWN_METERS_PER_PIXEL,
    _DEFAULT_VISUAL_OUTPUT_DIR,
    _MAX_NAVIGATE_STEP_BURST,
    HabitatAdapterError,
    _Session,
)


class HabitatAdapterNavigationMixin:
    """Navigation actions and kinematic control endpoints."""

    def _set_agent_state(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        position = np.asarray(
            self._coerce_float_list(payload.get("position"), 3, "payload.position"),
            dtype=np.float32,
        )
        rotation_payload = payload.get("rotation")
        snap_to_navmesh = self._coerce_bool(
            payload.get("snap_to_navmesh", False),
            field_name="payload.snap_to_navmesh",
        )
        infer_sensor_states = self._coerce_bool(
            payload.get("infer_sensor_states", True),
            field_name="payload.infer_sensor_states",
        )
        include_data = self._coerce_bool(
            payload.get("include_observation_data", False),
            field_name="payload.include_observation_data",
        )
        max_elements = self._coerce_int(
            payload.get(
                "max_observation_elements", _DEFAULT_MAX_OBSERVATION_ELEMENTS
            ),
            field_name="payload.max_observation_elements",
        )

        agent = session.simulator.get_agent(session.agent_id)
        current_state = agent.get_state()
        rotation = (
            self._coerce_float_list(rotation_payload, 4, "payload.rotation")
            if rotation_payload is not None
            else current_state.rotation
        )

        if snap_to_navmesh:
            pathfinder = self._require_pathfinder_loaded(session)
            position = np.asarray(pathfinder.snap_point(position), dtype=np.float32)

        next_state = habitat_sim.AgentState()
        next_state.position = position
        next_state.rotation = rotation
        agent.set_state(next_state, infer_sensor_states=infer_sensor_states)

        # Rebase trajectory after teleport: otherwise the next _record_pose
        # will compute the segment distance from the stale pre-teleport point
        # to the new position, polluting cumulative_path_length (and SPL).
        new_position = self._current_position(session).tolist()
        session.trajectory = [new_position]

        session.last_sensor_obs = self._capture_sensor_observations(session)
        return {
            "session_id": session.session_id,
            "agent_state": self._build_metrics(session)["agent_state"],
            "observation": self._serialize_observation(
                session.last_sensor_obs,
                include_data=include_data,
                max_elements=max_elements,
            ),
        }

    def _sample_navigable_point(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        self._assert_map_planning_allowed(session, action_name="sample_navigable_point")
        pathfinder = self._require_pathfinder_loaded(session)
        seed_value = payload.get("seed")
        if seed_value is not None:
            pathfinder.seed(self._coerce_int(seed_value, "payload.seed"))

        near_value = payload.get("near")
        if near_value is not None:
            near = np.asarray(
                self._coerce_float_list(near_value, 3, "payload.near"),
                dtype=np.float32,
            )
            distance = self._coerce_float(
                payload.get("distance", 3.0), field_name="payload.distance"
            )
            max_tries = self._coerce_int(
                payload.get("max_tries", 100), field_name="payload.max_tries"
            )
            point = pathfinder.get_random_navigable_point_near(
                near, distance, max_tries=max_tries
            )
        else:
            point = pathfinder.get_random_navigable_point()

        point_np = np.asarray(point, dtype=np.float32)
        response = {
            "session_id": session.session_id,
            "point": point_np.tolist(),
            "is_navigable": bool(pathfinder.is_navigable(point_np)),
        }
        island_radius = getattr(pathfinder, "island_radius", None)
        if callable(island_radius):
            response["island_radius"] = float(island_radius(point_np))
        return response

    def _find_shortest_path(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        self._assert_map_planning_allowed(session, action_name="find_shortest_path")
        pathfinder = self._require_pathfinder_loaded(session)
        snap_start = self._coerce_bool(
            payload.get("snap_start", True), field_name="payload.snap_start"
        )
        snap_end = self._coerce_bool(
            payload.get("snap_end", True), field_name="payload.snap_end"
        )
        end = np.asarray(
            self._coerce_float_list(payload.get("end"), 3, "payload.end"),
            dtype=np.float32,
        )
        start_payload = payload.get("start")
        if start_payload is None:
            start = np.asarray(
                session.simulator.get_agent(session.agent_id).get_state().position,
                dtype=np.float32,
            )
        else:
            start = np.asarray(
                self._coerce_float_list(start_payload, 3, "payload.start"),
                dtype=np.float32,
            )

        if snap_start:
            start = np.asarray(pathfinder.snap_point(start), dtype=np.float32)
        if snap_end:
            end = np.asarray(pathfinder.snap_point(end), dtype=np.float32)

        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = end
        reachable = bool(pathfinder.find_path(path))
        path_points = []
        geodesic_distance = None
        if reachable:
            geodesic_distance = float(path.geodesic_distance)
            path_points = [
                np.asarray(point, dtype=np.float32).tolist() for point in path.points
            ]

        return {
            "session_id": session.session_id,
            "reachable": reachable,
            "start": self._to_numeric_list(start),
            "end": self._to_numeric_list(end),
            "geodesic_distance": geodesic_distance,
            "path_points": path_points,
        }

    def _current_position(self, session: _Session) -> np.ndarray:
        return np.asarray(
            session.simulator.get_agent(session.agent_id).get_state().position,
            dtype=np.float32,
        )

    def _current_rotation(self, session: _Session) -> list:
        rotation = self._to_numeric_list(
            session.simulator.get_agent(session.agent_id).get_state().rotation
        )
        if isinstance(rotation, list) and len(rotation) == 4:
            return [float(item) for item in rotation]
        return [1.0, 0.0, 0.0, 0.0]

    def _forward_vector(self, session: _Session) -> np.ndarray:
        """Return the agent's forward direction in world XZ plane.

        Habitat's move_forward advances along the agent's local -Z axis,
        so we rotate [0, 0, -1] by the agent quaternion [w, x, y, z].
        """
        w, x, y, z = self._current_rotation(session)
        # Rotate [0, 0, -1] by quaternion [w, x, y, z]
        forward = np.array(
            [
                -2.0 * (x * z + y * w),
                -2.0 * (y * z - x * w),
                -(1.0 - 2.0 * (x * x + y * y)),
            ],
            dtype=np.float32,
        )
        forward[1] = 0.0  # Project onto horizontal plane
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return forward / norm

    def _heading_degrees(self, session: _Session) -> float:
        forward = self._forward_vector(session)
        return float(np.degrees(np.arctan2(forward[2], forward[0])))

    @staticmethod
    def _rotated_heading(direction: np.ndarray, delta_deg: float) -> np.ndarray:
        rad = np.deg2rad(delta_deg)
        direction = np.asarray(direction, dtype=np.float32)
        rotated = np.array(
            [
                direction[0] * np.cos(rad) - direction[2] * np.sin(rad),
                0.0,
                direction[0] * np.sin(rad) + direction[2] * np.cos(rad),
            ],
            dtype=np.float32,
        )
        norm = np.linalg.norm(rotated)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return rotated / norm

    def _record_pose(self, session: _Session) -> None:
        position = self._current_position(session).tolist()
        if session.trajectory and np.allclose(session.trajectory[-1], position):
            return
        # Accumulate cumulative path length (for SPL) before the cap may drop
        # old trajectory points. Use previous position from trajectory if
        # available, else skip (first pose).
        if session.trajectory:
            prev = np.asarray(session.trajectory[-1], dtype=np.float32)
            curr = np.asarray(position, dtype=np.float32)
            session.cumulative_path_length += float(np.linalg.norm(curr - prev))
        session.trajectory.append(position)
        if len(session.trajectory) > 512:
            session.trajectory = session.trajectory[-512:]

    def _record_collision(self, session: _Session, action: Optional[str] = None) -> None:
        """Update session collision bookkeeping after a collision is detected."""
        position = self._current_position(session).tolist()
        heading = self._heading_degrees(session)
        # Infer which side was hit from the action that caused the collision
        if action in ("turn_left",):
            side = "left"
        elif action in ("turn_right",):
            side = "right"
        else:
            side = "front"
        record: Dict[str, Any] = {
            "position": position,
            "heading_deg": round(heading, 3),
            "collision_side": side,
            "step_count": session.step_count,
        }
        session.collision_points.append(record)
        session.last_collision = record

    def _build_state_summary(
        self,
        session: _Session,
        goal: Optional[np.ndarray] = None,
        collided: Optional[bool] = None,
    ) -> Dict[str, Any]:
        position = self._current_position(session)
        goal_value = goal
        if goal_value is None and session.last_goal is not None:
            goal_value = np.asarray(session.last_goal, dtype=np.float32)
        euclidean_distance_to_goal = None
        goal_direction_deg = None
        if goal_value is not None:
            delta = goal_value - position
            euclidean_distance_to_goal = float(np.linalg.norm(delta))
            # Compute relative bearing: how many degrees the agent must
            # call turn_right to face the goal.
            # Positive = call turn_right (which visually rotates LEFT)
            # Negative = call turn_left (which visually rotates RIGHT)
            goal_bearing = float(np.degrees(np.arctan2(delta[2], delta[0])))
            heading = self._heading_degrees(session)
            rel = goal_bearing - heading
            # Wrap to [-180, 180]
            rel = (rel + 180.0) % 360.0 - 180.0
            goal_direction_deg = round(rel, 1)
        heading_deg = round(self._heading_degrees(session), 3)
        summary: Dict[str, Any] = {
            "heading_deg": heading_deg,
            "euclidean_distance_to_goal": (
                round(euclidean_distance_to_goal, 3) if euclidean_distance_to_goal is not None else None
            ),
            "goal_direction_deg": goal_direction_deg,
            "trajectory_length": len(session.trajectory),
            "collision_count": len(session.collision_points),
            "region_label": None,
        }
        # Only expose absolute coordinates in non-mapless mode
        if not session.mapless:
            summary["position"] = self._to_numeric_list(position)
            summary["goal"] = (
                self._to_numeric_list(goal_value) if goal_value is not None else None
            )
        if collided is not None:
            summary["collided"] = collided
        if session.last_collision is not None:
            summary["last_collision_side"] = session.last_collision.get("collision_side")
        return summary

    # ── Depth sensing ────────────────────────────────────────────

    _DEPTH_CLEARANCE_THRESHOLD_M = 0.5

    def _get_depth_array(self, session: _Session) -> np.ndarray:
        """Return the current depth sensor array [H, W] in meters (float32)."""
        obs = self._capture_sensor_observations(session)
        if "depth_sensor" not in obs:
            raise HabitatAdapterError(
                "depth_sensor not available — ensure depth_sensor is enabled in init_scene"
            )
        depth = np.asarray(obs["depth_sensor"], dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return depth

    def _analyze_depth_array(
        self, depth: np.ndarray, clearance: float
    ) -> Dict[str, Any]:
        """Analyze a depth array split into left/front/right thirds."""
        h, w = depth.shape
        third = w // 3
        # Split the camera FOV (default hfov=90°) into three vertical strips.
        # All three regions are within the agent's forward-facing view:
        #   front_left  ≈ forward 15°~45° to the left
        #   front_center ≈ forward ±15° (straight ahead)
        #   front_right ≈ forward 15°~45° to the right
        regions = {
            "front_left": depth[:, :third],
            "front_center": depth[:, third : 2 * third],
            "front_right": depth[:, 2 * third :],
        }
        result: Dict[str, Any] = {}
        closest_dist = float("inf")
        closest_dir = "front_center"
        for name, region in regions.items():
            valid = region[np.isfinite(region)]
            if valid.size == 0:
                result[name] = {
                    "min_dist": None,
                    "mean_dist": None,
                    "clear": False,
                }
                continue
            min_d = float(np.min(valid))
            mean_d = float(np.mean(valid))
            clear = bool(min_d >= clearance)
            result[name] = {
                "min_dist": round(min_d, 3),
                "mean_dist": round(mean_d, 3),
                "clear": clear,
            }
            if min_d < closest_dist:
                closest_dist = min_d
                closest_dir = name
        result["closest_obstacle_dist"] = (
            round(closest_dist, 3) if np.isfinite(closest_dist) else None
        )
        result["closest_obstacle_direction"] = closest_dir
        # Recommend action based on clearance
        center_clear = result.get("front_center", {}).get("clear", False)
        fl_clear = result.get("front_left", {}).get("clear", False)
        fr_clear = result.get("front_right", {}).get("clear", False)
        if center_clear:
            rec = "forward"
        elif fr_clear and not fl_clear:
            rec = "turn_right"
        elif fl_clear and not fr_clear:
            rec = "turn_left"
        elif fr_clear and fl_clear:
            fr_mean = result.get("front_right", {}).get("mean_dist", 0)
            fl_mean = result.get("front_left", {}).get("mean_dist", 0)
            rec = "turn_right" if (fr_mean or 0) >= (fl_mean or 0) else "turn_left"
        else:
            rec = "turn_around"
        result["recommended_action"] = rec
        result["depth_unit"] = "meters"
        result["clearance_threshold"] = clearance
        result["note"] = "All regions are within the camera forward FOV (~90 deg). Use get_panorama with include_depth_analysis=true for full 360-degree obstacle awareness."
        return result

    def _analyze_depth(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        clearance = float(
            payload.get("clearance_threshold", self._DEPTH_CLEARANCE_THRESHOLD_M)
        )
        depth = self._get_depth_array(session)
        result = self._analyze_depth_array(depth, clearance)
        result["image_shape"] = list(depth.shape)
        return result

    def _query_depth(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        depth = self._get_depth_array(session)
        h, w = depth.shape
        result: Dict[str, Any] = {"depth_unit": "meters", "image_shape": [h, w]}

        points = payload.get("points")
        bbox = payload.get("bbox")

        if points is not None:
            if not isinstance(points, list):
                raise HabitatAdapterError('"points" must be a list of [u, v] pairs')
            depths = []
            for i, pt in enumerate(points):
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    raise HabitatAdapterError(
                        f'"points[{i}]" must be [u, v] (pixel coordinates)'
                    )
                u, v = int(pt[0]), int(pt[1])
                if not (0 <= u < w and 0 <= v < h):
                    depths.append(None)
                else:
                    d = float(depth[v, u])
                    depths.append(round(d, 4) if np.isfinite(d) else None)
            result["depths"] = depths

        if bbox is not None:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                raise HabitatAdapterError(
                    '"bbox" must be [x1, y1, x2, y2] (pixel coordinates)'
                )
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))
            if x2 <= x1 or y2 <= y1:
                raise HabitatAdapterError(
                    '"bbox" must have x2 > x1 and y2 > y1'
                )
            region = depth[y1:y2, x1:x2]
            valid = region[np.isfinite(region)]
            if valid.size == 0:
                result["bbox_result"] = {
                    "min_depth": None,
                    "max_depth": None,
                    "mean_depth": None,
                    "pixel_count": int(region.size),
                    "valid_pixel_count": 0,
                }
            else:
                result["bbox_result"] = {
                    "min_depth": round(float(np.min(valid)), 4),
                    "max_depth": round(float(np.max(valid)), 4),
                    "mean_depth": round(float(np.mean(valid)), 4),
                    "pixel_count": int(region.size),
                    "valid_pixel_count": int(valid.size),
                }

        if points is None and bbox is None:
            raise HabitatAdapterError(
                'query_depth requires "points" and/or "bbox" in payload'
            )

        return result

    def _get_topdown_map(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        self._assert_map_planning_allowed(session, action_name="get_topdown_map")
        pathfinder = self._require_pathfinder_loaded(session)
        output_dir = payload.get("output_dir", _DEFAULT_VISUAL_OUTPUT_DIR)
        if not isinstance(output_dir, str) or not output_dir:
            raise HabitatAdapterError(
                'Field "payload.output_dir" must be a non-empty str'
            )
        meters_per_pixel = self._coerce_float(
            payload.get("meters_per_pixel", _DEFAULT_TOPDOWN_METERS_PER_PIXEL),
            field_name="payload.meters_per_pixel",
        )
        if meters_per_pixel <= 0:
            raise HabitatAdapterError('Field "payload.meters_per_pixel" must be > 0')

        show_pose = self._coerce_bool(
            payload.get("show_pose", True), field_name="payload.show_pose"
        )
        show_traj = self._coerce_bool(
            payload.get("show_traj", False), field_name="payload.show_traj"
        )
        show_collisions = self._coerce_bool(
            payload.get("show_collisions", False),
            field_name="payload.show_collisions",
        )
        show_path = self._coerce_bool(
            payload.get("show_path", False), field_name="payload.show_path"
        )
        collision_limit = self._coerce_int(
            payload.get("collision_limit", 10), field_name="payload.collision_limit"
        )
        if collision_limit <= 0:
            raise HabitatAdapterError('Field "payload.collision_limit" must be > 0')

        current_position = self._current_position(session)
        height = payload.get("height", float(current_position[1]))
        height_value = self._coerce_float(height, "payload.height")
        raw_map = np.asarray(pathfinder.get_topdown_view(meters_per_pixel, height_value))
        image = self._to_rgb_topdown(raw_map)

        goal: Optional[np.ndarray] = None
        goal_payload = payload.get("goal")
        if goal_payload is not None:
            goal = np.asarray(
                self._coerce_float_list(goal_payload, 3, "payload.goal"),
                dtype=np.float32,
            )
            session.last_goal = self._to_numeric_list(goal)
        elif show_path and session.last_goal is not None:
            goal = np.asarray(session.last_goal, dtype=np.float32)

        path_points_payload = payload.get("path_points")
        path_points: list = []
        if path_points_payload is not None:
            if not isinstance(path_points_payload, list):
                raise HabitatAdapterError(
                    'Field "payload.path_points" must be list[list[float]]'
                )
            path_points = [
                self._coerce_float_list(point, 3, "payload.path_points[]")
                for point in path_points_payload
            ]
        elif show_path and goal is not None:
            path_obj = habitat_sim.ShortestPath()
            path_obj.requested_start = current_position
            path_obj.requested_end = goal
            if bool(pathfinder.find_path(path_obj)):
                path_points = [
                    np.asarray(p, dtype=np.float32).tolist() for p in path_obj.points
                ]

        if path_points:
            self._draw_topdown_path(image, pathfinder, path_points, meters_per_pixel)
            self._draw_topdown_marker(
                image, pathfinder, path_points[0], meters_per_pixel, (80, 200, 255)
            )
            self._draw_topdown_marker(
                image, pathfinder, path_points[-1], meters_per_pixel, (0, 0, 255)
            )

        if show_traj and session.trajectory:
            self._draw_topdown_path(
                image,
                pathfinder,
                session.trajectory,
                meters_per_pixel,
                color=(80, 220, 120),
            )

        if show_collisions:
            for collision in session.collision_points[-collision_limit:]:
                self._draw_topdown_marker(
                    image,
                    pathfinder,
                    collision["position"],
                    meters_per_pixel,
                    (255, 215, 0),
                    radius=2,
                )

        if goal is not None:
            self._draw_topdown_marker(
                image, pathfinder, goal, meters_per_pixel, (0, 0, 255), radius=3
            )

        self._draw_topdown_marker(
            image, pathfinder, current_position, meters_per_pixel, (0, 255, 0), radius=3
        )
        if show_pose:
            self._draw_topdown_arrow(
                image,
                pathfinder,
                current_position,
                self._forward_vector(session),
                meters_per_pixel,
                color=(0, 255, 255),
            )

        topdown_session_dir = self._session_output_dir(output_dir, session.session_id)
        item = self._write_png_image(
            image=image,
            output_dir=topdown_session_dir,
            name="topdown_map",
            session_id=session.session_id,
            step_count=session.step_count,
        )
        bounds = pathfinder.get_bounds()
        return {
            "session_id": session.session_id,
            "height": height_value,
            "meters_per_pixel": meters_per_pixel,
            "topdown_map": item,
            "map_bounds": [list(bounds[0]), list(bounds[1])],
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "goal": self._to_numeric_list(goal) if goal is not None else None,
            "path_points": path_points,
            "trajectory_points": list(session.trajectory) if show_traj else [],
            "collision_points": (
                list(session.collision_points[-collision_limit:]) if show_collisions else []
            ),
            "state_summary": self._build_state_summary(session, goal=goal),
        }

    def _navigate_step(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        self._assert_map_planning_allowed(session, action_name="navigate_step")
        pathfinder = self._require_pathfinder_loaded(session)
        goal = np.asarray(
            self._coerce_float_list(payload.get("goal"), 3, "payload.goal"),
            dtype=np.float32,
        )
        goal_radius_value = payload.get("goal_radius")
        goal_radius = (
            self._coerce_float(goal_radius_value, "payload.goal_radius")
            if goal_radius_value is not None
            else None
        )
        max_steps = self._coerce_int(
            payload.get("max_steps", 1), field_name="payload.max_steps"
        )
        if max_steps < 1:
            raise HabitatAdapterError('Field "payload.max_steps" must be >= 1')
        max_steps = min(max_steps, _MAX_NAVIGATE_STEP_BURST)
        dt = self._coerce_float(payload.get("dt", 1.0 / 60.0), field_name="payload.dt")
        include_data = self._coerce_bool(
            payload.get("include_observation_data", False),
            field_name="payload.include_observation_data",
        )
        max_elements = self._coerce_int(
            payload.get(
                "max_observation_elements", _DEFAULT_MAX_OBSERVATION_ELEMENTS
            ),
            field_name="payload.max_observation_elements",
        )
        include_metrics = self._coerce_bool(
            payload.get("include_metrics", True),
            field_name="payload.include_metrics",
        )
        include_visuals = self._coerce_bool(
            payload.get("include_visuals", False),
            field_name="payload.include_visuals",
        )
        include_publish_hints = self._coerce_bool(
            payload.get("include_publish_hints", False),
            field_name="payload.include_publish_hints",
        )

        session.last_goal = self._to_numeric_list(goal)

        current_position = self._current_position(session)
        path = habitat_sim.ShortestPath()
        path.requested_start = current_position
        path.requested_end = goal
        reachable = bool(pathfinder.find_path(path))
        path_points: list = (
            [np.asarray(p, dtype=np.float32).tolist() for p in path.points]
            if reachable
            else []
        )
        if not reachable:
            response: Dict[str, Any] = {
                "session_id": session.session_id,
                "nav_status": "unreachable",
                "goal": self._to_numeric_list(goal),
                "reachable": False,
                "action": None,
                "collided": False,
                "step_count": session.step_count,
                "geodesic_distance": None,
                "observation": self._serialize_observation(
                    session.last_sensor_obs,
                    include_data=include_data,
                    max_elements=max_elements,
                ),
            }
            if include_metrics:
                response["metrics"] = self._build_metrics(session)
            return response

        try:
            follower = session.simulator.make_greedy_follower(
                agent_id=session.agent_id,
                goal_radius=goal_radius,
            )
            next_action = follower.next_action_along(goal)
        except Exception as exc:  # noqa: BLE001
            response = {
                "session_id": session.session_id,
                "nav_status": "error",
                "goal": self._to_numeric_list(goal),
                "reachable": True,
                "action": None,
                "collided": False,
                "step_count": session.step_count,
                "error": str(exc),
                "observation": self._serialize_observation(
                    session.last_sensor_obs,
                    include_data=include_data,
                    max_elements=max_elements,
                ),
            }
            if include_metrics:
                response["metrics"] = self._build_metrics(session)
            return response

        # Parse the visual-export payload once, BEFORE the navigation
        # loop, so we can emit a trace frame after every atomic
        # simulator step. Previously `_export_visuals` was called a
        # single time after the whole burst finished, meaning rerun /
        # export_video only saw the navigate endpoint and missed every
        # intermediate frame — the agent looked teleported even though
        # the simulator was actually stepping one atomic action at a
        # time.
        visuals_payload = self._parse_visual_payload(payload)
        trace_sensors = visuals_payload["sensors"]
        # NOTE: we intentionally do NOT narrow trace_sensors to
        # ["color_sensor"] when include_visuals is False. A legacy
        # fallback here used to do that as a size optimization, back
        # when navigate() emitted only one trace frame for the whole
        # burst. Now that we emit one frame per atomic step and the
        # live rerun viewer tails every sensor it finds, filtering out
        # depth leaves the rerun depth panel frozen on the last frame
        # that *did* include depth — typically a pre-navigate look()
        # or panorama(). Keeping trace_sensors as-None here makes
        # `_export_visuals` auto-detect every image-like key in the
        # current observation, so depth / third_rgb / semantic all
        # stay in sync with color during navigate.
        nav_session_dir = self._session_output_dir(
            visuals_payload["output_dir"], session.session_id
        )
        depth_max = visuals_payload["depth_max"]
        trace_visuals: Dict[str, Dict[str, Any]] = {}

        def _emit_trace_frame(obs: Optional[Mapping[str, Any]]) -> None:
            """Write a per-step PNG set for `obs` into the session dir."""
            nonlocal trace_visuals
            if obs is None:
                return
            session.capture_counter += 1
            trace_visuals = self._export_visuals(
                observation=obs,
                output_dir=nav_session_dir,
                sensors=trace_sensors,
                depth_max=depth_max,
                session_id=session.session_id,
                step_count=session.step_count,
                capture_seq=session.capture_counter,
            )

        action_taken: Optional[str]
        actions_taken: list[str] = []
        collided = False
        nav_status = "en_route"
        # See _step_and_capture for the full reasoning: if capture
        # raises after simulator.step has already advanced the sim,
        # session bookkeeping must still track the move so the caller
        # does not retry and double-step. On failure we break out of
        # the loop with observation_error set; the response is still
        # ok=True (a partial success) so clients know the move
        # happened and should re-fetch via get_visuals on the next
        # round instead of retrying navigate().
        observation_error: Optional[str] = None

        def _capture_step_observation(raw_obs: Mapping[str, Any]) -> Any:
            """Read the step-result observation, optionally re-capturing.

            When a visual-robot proxy is attached AND configured to
            hide from first-person sensors, the dict returned by
            `simulator.step(...)` still contains the proxy mesh in
            first-person frames, so we re-capture via
            `_capture_sensor_observations` (which runs the visual
            media path that hides the proxy). Otherwise
            `simulator.step` already rendered everything we need —
            just strip the `collided` flag and return its dict.
            """
            if (
                session.visual_robot_obj is not None
                and self._env_bool("HAB_VISUAL_ROBOT_HIDE_FROM_FIRST_PERSON", True)
            ):
                return self._capture_sensor_observations(session)
            return {k: v for k, v in raw_obs.items() if k != "collided"}

        if next_action is None:
            action_taken = None
            nav_status = "reached"
            try:
                observation = (
                    session.last_sensor_obs
                    if session.last_sensor_obs is not None
                    else self._capture_sensor_observations(session)
                )
            except Exception as exc:  # noqa: BLE001
                observation_error = (
                    f"capture_sensor_observations failed for zero-step "
                    f"navigate (step_count={session.step_count}): {exc}"
                )
                self._logger.warning(
                    "_navigate_step: zero-step capture failed at "
                    "step_count=%s: %s",
                    session.step_count,
                    exc,
                )
                observation = session.last_sensor_obs
            else:
                session.last_sensor_obs = observation
                # Zero-step navigate (already at goal): still emit one frame
                # so replayable video + rerun viewer see the current state.
                _emit_trace_frame(observation)
        else:
            observation = session.last_sensor_obs
            for _ in range(max_steps):
                action_taken = str(next_action)
                raw_observation = session.simulator.step(next_action, dt=dt)
                # Bookkeeping BEFORE capture so session counters stay
                # in sync with simulator state even if capture fails.
                session.step_count += 1
                session.last_action = action_taken
                actions_taken.append(action_taken)
                self._record_pose(session)
                collided = bool(raw_observation.get("collided", False))

                try:
                    observation = _capture_step_observation(raw_observation)
                except Exception as exc:  # noqa: BLE001
                    observation_error = (
                        f"capture_sensor_observations failed inside "
                        f"_navigate_step loop (step_count="
                        f"{session.step_count}): {exc}"
                    )
                    self._logger.warning(
                        "_navigate_step: observation capture failed at "
                        "step_count=%s action=%s: %s",
                        session.step_count,
                        action_taken,
                        exc,
                    )
                    observation = session.last_sensor_obs
                    # Without a fresh observation we can't ask the
                    # follower where to go next; abort the burst.
                    # Reflect collision if it was raised this step
                    # before breaking, so the status is honest about
                    # the reason progress stopped.
                    if collided:
                        self._record_collision(session, action=action_taken)
                        nav_status = "blocked"
                    break

                session.last_sensor_obs = observation
                # Per-step trace frame. rerun's disk-tailing viewer and
                # export_video_trace both depend on this to reproduce
                # navigate() motion smoothly; without it they would only
                # see the final frame of each navigate burst.
                _emit_trace_frame(observation)
                if collided:
                    self._record_collision(session, action=action_taken)
                    nav_status = "blocked"
                    break

                next_action = follower.next_action_along(goal)
                if next_action is None:
                    nav_status = "reached"
                    break
            else:
                nav_status = "en_route"

        action_taken = actions_taken[-1] if actions_taken else None

        metrics = self._build_metrics(session) if include_metrics else None
        refreshed_position = self._current_position(session)
        refreshed_path = habitat_sim.ShortestPath()
        refreshed_path.requested_start = refreshed_position
        refreshed_path.requested_end = goal
        if bool(pathfinder.find_path(refreshed_path)):
            geodesic_distance = float(refreshed_path.geodesic_distance)
            path_points = [
                np.asarray(p, dtype=np.float32).tolist() for p in refreshed_path.points
            ]
        else:
            geodesic_distance = None

        response = {
            "session_id": session.session_id,
            "nav_status": nav_status,
            "goal": self._to_numeric_list(goal),
            "reachable": reachable,
            "action": action_taken,
            "actions_taken": actions_taken,
            "steps_executed": len(actions_taken),
            "max_steps": max_steps,
            "collided": collided,
            "step_count": session.step_count,
            "geodesic_distance": geodesic_distance,
            "path_points": path_points,
            "observation": self._serialize_observation(
                session.last_sensor_obs,
                include_data=include_data,
                max_elements=max_elements,
            ),
            "state_summary": self._build_state_summary(session, goal=goal, collided=collided),
        }
        if observation_error is not None:
            response["observation_error"] = observation_error
        if metrics is not None:
            response["metrics"] = metrics

        if include_visuals:
            response["visuals"] = trace_visuals
            if include_publish_hints:
                response["publish_hints"] = self._build_publish_hints(
                    session=session,
                    action=action_taken or "reached",
                    collided=collided,
                    visuals=trace_visuals,
                    metrics=metrics,
                )
        return response

    def _resolve_n_steps(
        self, action: Any, payload: Mapping[str, Any]
    ) -> int:
        """Return number of atomic steps implied by degrees/distance params."""
        action_str = str(action)
        degrees = payload.get("degrees")
        distance = payload.get("distance")
        if degrees is not None and action_str in ("turn_left", "turn_right"):
            deg = abs(self._coerce_float(degrees, field_name="payload.degrees"))
            return max(1, math.ceil(deg / 10.0))
        if distance is not None and action_str == "move_forward":
            dist = abs(self._coerce_float(distance, field_name="payload.distance"))
            return max(1, math.ceil(dist / 0.25))
        return 1

    def _step_action(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        action = payload.get("action")
        if not isinstance(action, (str, int)):
            raise HabitatAdapterError('Field "payload.action" must be str or int')
        dt = self._coerce_float(payload.get("dt", 1.0 / 60.0), field_name="payload.dt")
        include_data = self._coerce_bool(
            payload.get("include_observation_data", False),
            field_name="payload.include_observation_data",
        )
        max_elements = self._coerce_int(
            payload.get(
                "max_observation_elements", _DEFAULT_MAX_OBSERVATION_ELEMENTS
            ),
            field_name="payload.max_observation_elements",
        )

        n_steps = self._resolve_n_steps(action, payload)
        any_collided = False
        steps_taken = 0
        observation: Any = session.last_sensor_obs
        for _ in range(n_steps):
            raw_observation = session.simulator.step(action, dt=dt)
            # Same optimization as _navigate_step: skip redundant re-render when
            # visual robot is not in use.
            if (
                session.visual_robot_obj is not None
                and self._env_bool("HAB_VISUAL_ROBOT_HIDE_FROM_FIRST_PERSON", True)
            ):
                observation = self._capture_sensor_observations(session)
            else:
                observation = {k: v for k, v in raw_observation.items() if k != "collided"}
            session.step_count += 1
            steps_taken += 1
            session.last_action = str(action)
            session.last_sensor_obs = observation
            self._record_pose(session)
            if raw_observation.get("collided", False):
                any_collided = True
                self._record_collision(session, action=str(action))
                break

        return {
            "session_id": session.session_id,
            "step_count": session.step_count,
            "action": action,
            "steps_taken": steps_taken,
            "collided": any_collided,
            "observation": self._serialize_observation(
                observation,
                include_data=include_data,
                max_elements=max_elements,
            ),
            "state_summary": self._build_state_summary(session, collided=any_collided),
        }

    def _step_and_capture(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        action = payload.get("action")
        if not isinstance(action, (str, int)):
            raise HabitatAdapterError('Field "payload.action" must be str or int')

        dt = self._coerce_float(payload.get("dt", 1.0 / 60.0), field_name="payload.dt")
        include_data = self._coerce_bool(
            payload.get("include_observation_data", False),
            field_name="payload.include_observation_data",
        )
        max_elements = self._coerce_int(
            payload.get(
                "max_observation_elements", _DEFAULT_MAX_OBSERVATION_ELEMENTS
            ),
            field_name="payload.max_observation_elements",
        )

        include_metrics = self._coerce_bool(
            payload.get("include_metrics", True),
            field_name="payload.include_metrics",
        )
        include_publish_hints = self._coerce_bool(
            payload.get("include_publish_hints", True),
            field_name="payload.include_publish_hints",
        )
        output_dir = payload.get("output_dir", _DEFAULT_VISUAL_OUTPUT_DIR)
        if not isinstance(output_dir, str) or not output_dir:
            raise HabitatAdapterError(
                'Field "payload.output_dir" must be a non-empty str'
            )
        depth_max = self._coerce_float(
            payload.get("depth_max", _DEFAULT_DEPTH_VIS_MAX),
            field_name="payload.depth_max",
        )
        if depth_max <= 0:
            raise HabitatAdapterError('Field "payload.depth_max" must be > 0')

        sensors_payload = payload.get("sensors")
        sensors: Optional[list[str]]
        if sensors_payload is None:
            sensors = None
        elif isinstance(sensors_payload, list) and all(
            isinstance(item, str) for item in sensors_payload
        ):
            sensors = sensors_payload
        else:
            raise HabitatAdapterError(
                'Field "payload.sensors" must be list[str] when provided'
            )

        session_dir = self._session_output_dir(output_dir, session.session_id)
        n_steps = self._resolve_n_steps(action, payload)
        any_collided = False
        steps_taken = 0
        observation: Any = session.last_sensor_obs
        # If sensor capture raises after simulator.step() has already
        # advanced the sim, we would previously leave session bookkeeping
        # (step_count, last_action, trajectory) stuck behind the real
        # simulator state and the client would retry on top of an
        # already-moved agent. Now bookkeeping happens BEFORE capture,
        # and capture is wrapped in try/except; on failure the session
        # is still consistent with the simulator and the response
        # carries an `observation_error` so the caller knows the move
        # succeeded but the observation is stale — do NOT retry the
        # movement.
        observation_error: Optional[str] = None
        for i in range(n_steps):
            raw_observation = session.simulator.step(action, dt=dt)
            # Bookkeeping first — must track the simulator even if
            # capture below explodes.
            session.step_count += 1
            steps_taken += 1
            session.last_action = str(action)
            self._record_pose(session)
            collided_this_step = bool(raw_observation.get("collided", False))

            try:
                observation = self._capture_sensor_observations(session)
            except Exception as exc:  # noqa: BLE001
                observation_error = (
                    f"capture_sensor_observations failed after "
                    f"simulator.step (step_count={session.step_count}): {exc}"
                )
                self._logger.warning(
                    "_step_and_capture: observation capture failed after "
                    "step_count=%s action=%s: %s",
                    session.step_count,
                    action,
                    exc,
                )
                # Leave session.last_sensor_obs pointing at the
                # previous (stale) frame rather than overwriting with
                # None — downstream serialization tolerates an
                # observation dict but some clients iterate without
                # None checks. The observation_error field is the
                # authoritative signal that the payload is stale.
                observation = session.last_sensor_obs
                # Abort the remainder of the atomic-step burst: if GL
                # just failed, piling on more captures is unlikely to
                # help and we want the response to reflect how far we
                # actually got.
                if collided_this_step:
                    any_collided = True
                    self._record_collision(session, action=str(action))
                break

            session.last_sensor_obs = observation

            # Capture every intermediate step for video assembly
            if i < n_steps - 1 and not collided_this_step:
                session.capture_counter += 1
                self._export_visuals(
                    observation=observation,
                    output_dir=session_dir,
                    sensors=sensors,
                    depth_max=depth_max,
                    session_id=session.session_id,
                    step_count=session.step_count,
                    capture_seq=session.capture_counter,
                )

            if collided_this_step:
                any_collided = True
                self._record_collision(session, action=str(action))
                break
        collided = any_collided

        # Final frame — this is the one returned in visuals (for agent
        # injection). Skip the export when capture failed above: there
        # is no fresh observation to write, and the stale last-known
        # frame was already exported by the previous successful step.
        if observation_error is None and observation is not None:
            session.capture_counter += 1
            visuals = self._export_visuals(
                observation=observation,
                output_dir=session_dir,
                sensors=sensors,
                depth_max=depth_max,
                session_id=session.session_id,
                step_count=session.step_count,
                capture_seq=session.capture_counter,
            )
        else:
            visuals = {}

        response: Dict[str, Any] = {
            "session_id": session.session_id,
            "step_count": session.step_count,
            "action": action,
            "steps_taken": steps_taken,
            "collided": collided,
            "output_dir": output_dir,
            "observation": self._serialize_observation(
                observation,
                include_data=include_data,
                max_elements=max_elements,
            ),
            "visuals": visuals,
            "state_summary": self._build_state_summary(session, collided=collided),
        }
        if observation_error is not None:
            response["observation_error"] = observation_error
        if include_metrics:
            metrics = self._build_metrics(session)
            response["metrics"] = metrics
        else:
            metrics = None
        if include_publish_hints:
            response["publish_hints"] = self._build_publish_hints(
                session=session,
                action=str(action),
                collided=collided,
                visuals=visuals,
                metrics=metrics,
            )
        return response

