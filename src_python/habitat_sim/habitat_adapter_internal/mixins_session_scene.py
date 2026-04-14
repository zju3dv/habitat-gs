from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np

import habitat_sim
import habitat_sim.utils.settings

from .types import (
    _DEFAULT_MAX_OBSERVATION_ELEMENTS,
    _Session,
    HabitatAdapterError,
)


class HabitatAdapterSessionSceneMixin:
    """Session/scenario lifecycle and simulator bootstrap helpers."""

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        if value.startswith(".") or value.startswith("/"):
            return True
        if "/" in value or "\\" in value:
            return True
        return value.endswith(
            (
                ".glb",
                ".gltf",
                ".ply",
                ".json",
                ".scene_instance",
                ".stage_config",
                ".scene_dataset_config",
            )
        )

    def _init_scene(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        del session_id
        scene = payload.get("scene")
        if not isinstance(scene, str) or not scene:
            raise HabitatAdapterError('Field "payload.scene" must be a non-empty str')
        if self._looks_like_path(scene) and not os.path.exists(scene):
            raise HabitatAdapterError(f'Scene path does not exist: "{scene}"')

        settings = habitat_sim.utils.settings.default_sim_settings.copy()
        settings["scene"] = scene

        dataset_cfg = payload.get("scene_dataset_config_file")
        dataset_cfg_path: Optional[str] = None
        if dataset_cfg is not None:
            if not isinstance(dataset_cfg, str):
                raise HabitatAdapterError(
                    'Field "payload.scene_dataset_config_file" must be str'
                )
            if dataset_cfg != "default" and not os.path.exists(dataset_cfg):
                raise HabitatAdapterError(
                    "scene_dataset_config_file path does not exist: " f'"{dataset_cfg}"'
                )
            settings["scene_dataset_config_file"] = dataset_cfg
            if dataset_cfg != "default":
                dataset_cfg_path = dataset_cfg

        self._maybe_update_bool(settings, "default_agent_navmesh", payload)
        if (
            dataset_cfg_path is not None
            and "default_agent_navmesh" not in payload
            and self._is_gaussian_dataset_config(dataset_cfg_path)
        ):
            # GS scenes do not expose a joined collision mesh, so the default
            # navmesh recomputation path fails during simulator init.
            settings["default_agent_navmesh"] = False

        self._maybe_update_int(settings, "default_agent", payload)
        self._maybe_update_int(settings, "seed", payload)
        self._maybe_update_bool(settings, "frustum_culling", payload)
        # Inherit enable_physics from default_sim_settings (typically built_with_bullet).
        # Gaussian dataset configs lack collision mesh for Bullet; force off when the client
        # omits enable_physics (same class of issue as default_agent_navmesh). Without a
        # scene_dataset_config_file path we cannot detect GS here—clients must pass false.
        if (
            dataset_cfg_path is not None
            and "enable_physics" not in payload
            and self._is_gaussian_dataset_config(dataset_cfg_path)
        ):
            settings["enable_physics"] = False
        self._maybe_update_bool(settings, "enable_physics", payload)

        sensor_cfg = payload.get("sensor", {})
        if not isinstance(sensor_cfg, Mapping):
            raise HabitatAdapterError('Field "payload.sensor" must be an object')
        self._apply_sensor_overrides(settings, sensor_cfg)

        sim = self._simulator_factory_with_third_person(settings)
        self._maybe_load_dataset_navmesh(
            simulator=sim,
            scene=scene,
            dataset_cfg_path=dataset_cfg_path,
        )
        observation = sim.reset()

        new_session_id = str(uuid.uuid4())
        agent_id = int(settings.get("default_agent", 0))
        now = time.monotonic()
        is_gs = (
            dataset_cfg_path is not None
            and self._is_gaussian_dataset_config(dataset_cfg_path)
        )
        session = _Session(
            session_id=new_session_id,
            simulator=sim,
            scene=scene,
            settings=dict(settings),
            agent_id=agent_id,
            last_sensor_obs=observation,
            created_at_s=now,
            last_activity_s=now,
            is_gaussian=is_gs,
        )
        self._maybe_attach_visual_robot(session)
        session.trajectory.append(self._current_position(session).tolist())

        # Optionally load precomputed scene graph (room_object_scene_graph.json).
        # Reuses _resolve_dataset_navmesh_path to find the scene directory — this
        # handles both direct navmesh_instances and nested scene_instances formats.
        # Falls back to the dataset config directory itself.
        # Silently skipped when not found — scene graph is optional.
        if dataset_cfg_path is not None:
            import pathlib
            sg_path = None

            # Primary: derive scene dir from resolved navmesh path (same logic used
            # for loading the navmesh, avoids duplicating dataset config parsing).
            navmesh_abs = self._resolve_dataset_navmesh_path(dataset_cfg_path, scene)
            if navmesh_abs:
                candidate = pathlib.Path(navmesh_abs).parent / "room_object_scene_graph.json"
                if candidate.exists():
                    sg_path = candidate

            # Fallback: look in dataset config directory
            if sg_path is None:
                cfg_dir = pathlib.Path(dataset_cfg_path).parent
                candidate = cfg_dir / "room_object_scene_graph.json"
                if candidate.exists():
                    sg_path = candidate

            if sg_path is not None:
                try:
                    session.scene_graph = json.loads(sg_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass  # malformed SG — treat as absent

        self._sessions[new_session_id] = session
        try:
            session.last_sensor_obs = self._capture_sensor_observations(session)
        except Exception:
            # Roll back the registration so no orphaned session leaks resources.
            del self._sessions[new_session_id]
            raise

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
        return {
            "session_id": new_session_id,
            "scene": scene,
            "is_gaussian": is_gs,
            "available_actions": self._get_available_actions(session),
            "observation": self._serialize_observation(
                session.last_sensor_obs,
                include_data=include_data,
                max_elements=max_elements,
            ),
        }

    def _get_scene_info(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        refresh_observation = self._coerce_bool(
            payload.get("refresh_observation", False),
            field_name="payload.refresh_observation",
        )
        if refresh_observation or session.last_sensor_obs is None:
            session.last_sensor_obs = session.simulator.get_sensor_observations(
                agent_ids=session.agent_id
            )

        pathfinder = self._get_pathfinder(session)
        scene_dataset_cfg = session.settings.get("scene_dataset_config_file")
        return {
            "session_id": session.session_id,
            "scene": session.scene,
            "scene_dataset_config_file": (
                scene_dataset_cfg if isinstance(scene_dataset_cfg, str) else None
            ),
            "step_count": session.step_count,
            "sensor_keys": sorted(
                list(session.last_sensor_obs.keys()) if session.last_sensor_obs else []
            ),
            "available_actions": self._get_available_actions(session),
            "agent_state": self._build_metrics(session)["agent_state"],
            "navmesh_loaded": bool(
                getattr(pathfinder, "is_loaded", False)
                if pathfinder is not None
                else False
            ),
            "navmesh_bounds": self._get_navmesh_bounds(pathfinder),
            "navigable_area": self._get_navigable_area(pathfinder),
            "current_goal": session.last_goal,
            "trajectory_length": len(session.trajectory),
            "collision_count": len(session.collision_points),
            "state_summary": self._build_state_summary(session),
        }

    _SCENE_GRAPH_QUERY_TYPES = frozenset({"all", "room", "object"})

    def _get_scene_graph(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Query the precomputed scene graph for rooms or objects by type/label."""
        session = self._require_session(session_id)
        if session.mapless:
            raise HabitatAdapterError(
                "get_scene_graph is not available in mapless nav mode — "
                "the scene graph provides absolute object positions, which is "
                "equivalent to having a map and violates the mapless constraint."
            )
        sg = session.scene_graph
        if sg is None:
            return {
                "session_id": session.session_id,
                "scene_id": session.scene,
                "nodes": [],
                "rooms": [],
                "total_matched": 0,
                "scene_graph_available": False,
            }

        query_type = payload.get("query_type")
        if query_type not in self._SCENE_GRAPH_QUERY_TYPES:
            raise HabitatAdapterError(
                f'Field "payload.query_type" must be one of {sorted(self._SCENE_GRAPH_QUERY_TYPES)}'
            )
        room_type = payload.get("room_type")
        object_label = payload.get("object_label")
        max_results = self._coerce_int(
            payload.get("max_results", 10), field_name="payload.max_results"
        )
        if max_results <= 0:
            max_results = 10

        all_nodes = sg.get("nodes", [])
        # SG also has a top-level "rooms" list and "objects" list with richer data
        sg_rooms = sg.get("rooms", [])  # room dicts with centroid, area, adjacency
        sg_objects = sg.get("objects", [])  # object dicts with label, position, bbox, room_id
        obj_index_by_label = sg.get("object_index_by_label", {})  # label -> list of indices

        matched_nodes: list[Dict[str, Any]] = []
        matched_room_ids: set[str] = set()

        if query_type == "room":
            # Room nodes have no semantic label (only room_index).
            # room_type is accepted as a parameter but cannot be used for label
            # filtering — callers should use query_type="object" with an object
            # label (e.g. "sink" to find the kitchen) and check the room_id field.
            room_nodes = [n for n in all_nodes if n.get("type") == "room"]
            matched_nodes = room_nodes[:max_results]
            matched_room_ids = {str(n["id"]) for n in matched_nodes}

        elif query_type == "object":
            # Build id→full-object lookup for bbox enrichment
            obj_by_id: Dict[str, Any] = {o.get("id", ""): o for o in sg_objects}

            if object_label:
                # object_index_by_label values are already lists of object mini-records
                # (id, ins_id, room_id, position_xyz, room_assignment) — not integer indices
                needle = object_label.lower()
                for key, obj_list in obj_index_by_label.items():
                    if needle in key.lower():
                        for obj_mini in obj_list:
                            obj_id = obj_mini.get("id", "")
                            full = obj_by_id.get(obj_id, obj_mini)
                            matched_nodes.append({
                                "id": obj_id,
                                "type": "object",
                                "label": key,
                                "position_xyz": obj_mini.get("position_xyz"),
                                "bbox_min_xyz": full.get("bbox_min_xyz"),
                                "bbox_max_xyz": full.get("bbox_max_xyz"),
                                "room_id": obj_mini.get("room_id"),
                            })
                            if room_id := obj_mini.get("room_id"):
                                matched_room_ids.add(str(room_id))
                            if len(matched_nodes) >= max_results:
                                break
                    if len(matched_nodes) >= max_results:
                        break
            else:
                # No label filter: return first max_results object nodes
                for n in all_nodes:
                    if n.get("type") == "object":
                        matched_nodes.append(n)
                        if rid := n.get("room_id"):
                            matched_room_ids.add(str(rid))
                        if len(matched_nodes) >= max_results:
                            break

        else:  # "all"
            # When object_label is given, return only matching object nodes —
            # room nodes have no labels and would consume the result budget
            # before any matching objects are reached.
            # When no filter is given, return rooms first then objects.
            obj_needle = object_label.lower() if object_label else None
            for n in all_nodes:
                node_type = n.get("type", "")
                if node_type == "scene":
                    continue
                if obj_needle:
                    # With a label filter: skip rooms entirely, filter objects by label
                    if node_type != "object":
                        continue
                    if obj_needle not in str(n.get("label", "")).lower():
                        continue
                matched_nodes.append(n)
                if rid := n.get("room_id"):
                    matched_room_ids.add(str(rid))
                if len(matched_nodes) >= max_results:
                    break

        # Include the room detail records for matched rooms
        matched_rooms = [r for r in sg_rooms if str(r.get("id", "")) in matched_room_ids]

        result: Dict[str, Any] = {
            "session_id": session.session_id,
            "scene_id": session.scene,
            "nodes": matched_nodes,
            "rooms": matched_rooms,
            "total_matched": len(matched_nodes),
            "scene_graph_available": True,
        }
        # Room nodes carry no semantic label (only room_index). Warn when the
        # caller asked for room_type filtering so they can adjust their query.
        if query_type == "room" and room_type:
            result["note"] = (
                "room nodes in this scene graph have no semantic labels; "
                "room_type filter was not applied. To locate a room by function, "
                "query objects with a characteristic label (e.g. 'sink' for kitchen) "
                "and use the room_id field to identify the containing room."
            )
        return result

    def _close_session(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        del payload
        session = self._require_session(session_id)
        self._sessions.pop(session.session_id, None)
        self._stop_nav_loops_for_session(session.session_id, reason="close_session")
        self._dispose_session(session, reason="close_session")
        return {
            "session_id": session.session_id,
            "closed": True,
        }

    def _default_simulator_factory(
        self, settings: Mapping[str, Any]
    ) -> habitat_sim.Simulator:
        config = habitat_sim.utils.settings.make_cfg(dict(settings))
        return habitat_sim.Simulator(config)

    def _require_session(self, session_id: Optional[str]) -> _Session:
        if not isinstance(session_id, str) or not session_id:
            raise HabitatAdapterError('Field "session_id" must be provided')

        session = self._sessions.get(session_id)
        if session is None:
            raise HabitatAdapterError(f"Unknown session_id: {session_id}")
        session.last_activity_s = time.monotonic()
        return session

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        value = raw.strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off", ""):
            return False
        return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _capture_sensor_observations(self, session: _Session) -> Mapping[str, Any]:
        """Capture observations while keeping first-person views free of the visual robot.

        When visual robot rendering is enabled, we hide the robot for a primary
        capture pass, then optionally restore only the dedicated third-person
        sensor from a second capture with the robot visible.
        """
        simulator = session.simulator
        if (
            session.visual_robot_obj is None
            or not self._env_bool("HAB_VISUAL_ROBOT_HIDE_FROM_FIRST_PERSON", True)
        ):
            return simulator.get_sensor_observations(agent_ids=session.agent_id)

        third_person_enabled = bool(session.settings.get("third_person_color_sensor", False))
        third_person_uuid: Optional[str] = None
        if third_person_enabled:
            raw_uuid = session.settings.get("third_person_sensor_uuid", "third_rgb_sensor")
            if isinstance(raw_uuid, str) and raw_uuid:
                third_person_uuid = raw_uuid

        # Hide visual robot below the map for first-person/depth capture.
        try:
            import magnum as mn
            rotation_wxyz = self._current_rotation(session)
            hidden_rot = mn.Quaternion(
                mn.Vector3(
                    float(rotation_wxyz[1]),
                    float(rotation_wxyz[2]),
                    float(rotation_wxyz[3]),
                ),
                float(rotation_wxyz[0]),
            )
            hidden_pos = mn.Vector3(0.0, -1.0e4, 0.0)
            self._apply_visual_robot_transform(session.visual_robot_obj, hidden_pos, hidden_rot)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to hide visual robot for observation capture; fallback to direct capture. session=%s error=%s",
                session.session_id,
                exc,
            )
            return simulator.get_sensor_observations(agent_ids=session.agent_id)

        hidden_obs: Mapping[str, Any]
        try:
            hidden_obs = simulator.get_sensor_observations(agent_ids=session.agent_id)
        finally:
            self._sync_visual_robot_pose(session)

        if not third_person_uuid:
            return hidden_obs

        visible_obs = simulator.get_sensor_observations(agent_ids=session.agent_id)
        if third_person_uuid not in visible_obs:
            return hidden_obs
        merged_obs = dict(hidden_obs)
        merged_obs[third_person_uuid] = visible_obs[third_person_uuid]
        return merged_obs

    def _dispose_session(self, session: _Session, reason: str) -> None:
        self._detach_visual_robot(session)
        try:
            # Sessions are terminal here, so fully destroy the simulator instead of
            # keeping GL resources alive for a future reconfigure.
            session.simulator.close(destroy=True)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Error closing simulator id=%s reason=%s: %s",
                session.session_id,
                reason,
                exc,
            )
        finally:
            self._logger.info(
                "Session closed id=%s scene=%s reason=%s step_count=%s",
                session.session_id,
                session.scene,
                reason,
                session.step_count,
            )

    def _apply_sensor_overrides(
        self, settings: MutableMapping[str, Any], sensor_cfg: Mapping[str, Any]
    ) -> None:
        for key in ("width", "height"):
            self._maybe_update_int(settings, key, sensor_cfg)
        for key in ("sensor_height", "hfov", "zfar"):
            self._maybe_update_float(settings, key, sensor_cfg)
        for key in ("color_sensor", "depth_sensor", "semantic_sensor", "third_person_color_sensor"):
            self._maybe_update_bool(settings, key, sensor_cfg)
        if "third_person_sensor_uuid" in sensor_cfg:
            value = sensor_cfg["third_person_sensor_uuid"]
            if not isinstance(value, str) or not value:
                raise HabitatAdapterError(
                    'Field "payload.sensor.third_person_sensor_uuid" must be a non-empty str'
                )
            if value == "color_sensor":
                raise HabitatAdapterError(
                    'Field "payload.sensor.third_person_sensor_uuid" must not be "color_sensor"'
                )
            settings["third_person_sensor_uuid"] = value

    def _simulator_factory_with_third_person(
        self, settings: Mapping[str, Any]
    ) -> habitat_sim.Simulator:
        """Create simulator, injecting a third-person RGB sensor if requested.

        ``make_cfg`` is from the installed habitat_sim package and does not know
        about ``third_person_color_sensor``.  We build the config normally, then
        append the extra sensor spec before constructing the Simulator, so we
        don't need to modify the installed package.

        Camera framing (over-the-shoulder third-person):
        - Position: 1.5 m behind and ~1.2 m above the agent's eye level
        - Orientation: pitched ~20° down so the agent body (when visual
          robot is loaded) falls in the lower half of the frame

        The visual robot model is only rendered when HAB_VISUAL_ROBOT_ENABLE=1
        is set in the environment. Without it, the third-person view sees
        only the scene around the (invisible) agent position.
        """
        if not settings.get("third_person_color_sensor"):
            return self._simulator_factory(settings)

        import math
        import magnum as mn

        config = habitat_sim.utils.settings.make_cfg(dict(settings))

        uuid = settings.get("third_person_sensor_uuid", "third_rgb_sensor")
        tp_spec = habitat_sim.CameraSensorSpec()
        tp_spec.uuid = uuid
        tp_spec.sensor_type = habitat_sim.SensorType.COLOR
        tp_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # Half resolution to save bandwidth (still visually useful for rerun).
        h = max(1, int(settings.get("height", 480)) // 2)
        w = max(1, int(settings.get("width", 640)) // 2)
        tp_spec.resolution = mn.Vector2i([h, w])
        tp_spec.hfov = settings.get("hfov", 90)
        tp_spec.far = settings.get("zfar", 1000.0)

        # Position in agent-local frame: agent faces -Z, so +Z is behind.
        # ~1.5m behind, ~1.2m above eye level → over-the-shoulder framing.
        sensor_height = float(settings.get("sensor_height", 1.5))
        tp_spec.position = mn.Vector3(0.0, sensor_height + 1.2, 1.5)

        # Pitch ~20° downward so the agent body (when visible) sits in the
        # lower half of the frame. Negative pitch = nose tilts down in
        # habitat_sim's coordinate convention (sensor orientation is Euler
        # [pitch, yaw, roll] in radians on the agent-local frame).
        pitch_down_rad = -math.radians(20.0)
        tp_spec.orientation = mn.Vector3(pitch_down_rad, 0.0, 0.0)

        agent_id = int(settings.get("default_agent", 0))
        config.agents[agent_id].sensor_specifications.append(tp_spec)

        sim = habitat_sim.Simulator(config)
        self._logger.info(
            "Third-person RGB sensor registered uuid=%s resolution=%dx%d "
            "pitch=-20°",
            uuid, w, h,
        )
        return sim

    def _maybe_load_dataset_navmesh(
        self,
        simulator: habitat_sim.Simulator,
        scene: str,
        dataset_cfg_path: Optional[str],
    ) -> None:
        if dataset_cfg_path is None or self._looks_like_path(scene):
            return

        pathfinder = getattr(simulator, "pathfinder", None)
        if pathfinder is None or getattr(pathfinder, "is_loaded", False):
            return

        navmesh_path = self._resolve_dataset_navmesh_path(dataset_cfg_path, scene)
        if navmesh_path is None:
            return

        load_nav_mesh = getattr(pathfinder, "load_nav_mesh", None)
        if not callable(load_nav_mesh):
            self._logger.warning(
                "Dataset navmesh path resolved but pathfinder cannot load navmesh: scene=%s navmesh=%s",
                scene,
                navmesh_path,
            )
            return

        load_nav_mesh(navmesh_path)
        if getattr(pathfinder, "is_loaded", False):
            self._logger.info(
                "Loaded dataset navmesh scene=%s navmesh=%s",
                scene,
                navmesh_path,
            )
        else:
            self._logger.warning(
                "Failed to load dataset navmesh scene=%s navmesh=%s",
                scene,
                navmesh_path,
            )

    def _get_available_actions(self, session: _Session) -> Dict[str, Dict[str, Any]]:
        action_space = (
            session.simulator.get_agent(session.agent_id).agent_config.action_space
        )
        return {
            str(action_name): {"name": action_spec.name}
            for action_name, action_spec in action_space.items()
        }

    @staticmethod
    def _get_pathfinder(session: _Session) -> Any:
        return getattr(session.simulator, "pathfinder", None)

    def _require_pathfinder_loaded(self, session: _Session) -> Any:
        pathfinder = self._get_pathfinder(session)
        if pathfinder is None or not getattr(pathfinder, "is_loaded", False):
            raise HabitatAdapterError("NavMesh is not loaded for this scene")
        return pathfinder

    @staticmethod
    def _get_navmesh_bounds(pathfinder: Any) -> Optional[list[list[float]]]:
        if pathfinder is None or not getattr(pathfinder, "is_loaded", False):
            return None
        bounds = pathfinder.get_bounds()
        return [
            np.asarray(bounds[0], dtype=np.float32).tolist(),
            np.asarray(bounds[1], dtype=np.float32).tolist(),
        ]

    @staticmethod
    def _get_navigable_area(pathfinder: Any) -> Optional[float]:
        if pathfinder is None or not getattr(pathfinder, "is_loaded", False):
            return None
        area = getattr(pathfinder, "navigable_area", None)
        return float(area) if area is not None else None

    def _read_dataset_config(
        self, dataset_cfg_path: str
    ) -> Optional[Mapping[str, Any]]:
        try:
            with open(dataset_cfg_path, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
        except (OSError, json.JSONDecodeError) as exc:
            self._logger.warning(
                "Failed to parse scene dataset config %s: %s",
                dataset_cfg_path,
                exc,
            )
            return None

        if isinstance(data, Mapping):
            return data
        self._logger.warning(
            "Scene dataset config is not a JSON object: %s", dataset_cfg_path
        )
        return None

    def _is_gaussian_dataset_config(self, dataset_cfg_path: str) -> bool:
        dataset_cfg = self._read_dataset_config(dataset_cfg_path)
        if dataset_cfg is None:
            return False

        stages = dataset_cfg.get("stages")
        if not isinstance(stages, Mapping):
            return False

        default_attributes = stages.get("default_attributes")
        if isinstance(default_attributes, Mapping):
            render_asset_type = default_attributes.get("render_asset_type")
            if (
                isinstance(render_asset_type, str)
                and render_asset_type.lower() == "gaussian_splatting"
            ):
                return True

        paths = stages.get("paths")
        if not isinstance(paths, Mapping):
            return False
        return any(
            isinstance(key, str) and key.lower() in (".gs.ply", ".3dgs.ply")
            for key in paths.keys()
        )

    def _resolve_dataset_navmesh_path(
        self, dataset_cfg_path: str, scene: str
    ) -> Optional[str]:
        dataset_cfg = self._read_dataset_config(dataset_cfg_path)
        if dataset_cfg is None:
            return None

        navmesh_instances = dataset_cfg.get("navmesh_instances")
        if not isinstance(navmesh_instances, Mapping):
            return None

        relative_path = navmesh_instances.get(scene)
        if not isinstance(relative_path, str) or not relative_path:
            return None

        base_dir = os.path.dirname(os.path.abspath(dataset_cfg_path))
        navmesh_path = os.path.abspath(os.path.join(base_dir, relative_path))
        if not os.path.exists(navmesh_path):
            self._logger.warning(
                "Dataset navmesh path does not exist: scene=%s navmesh=%s",
                scene,
                navmesh_path,
            )
            return None
        return navmesh_path

    @staticmethod
    def _session_physics_enabled(session: _Session) -> bool:
        """Best-effort check for whether this session runs with physics enabled."""
        try:
            sim_cfg = getattr(getattr(session.simulator, "config", None), "sim_cfg", None)
            enabled = getattr(sim_cfg, "enable_physics", None)
            if isinstance(enabled, bool):
                return enabled
        except Exception:
            pass
        value = session.settings.get("enable_physics")
        if isinstance(value, bool):
            return value
        return False

    def _maybe_attach_visual_robot(self, session: _Session) -> None:
        session.visual_robot_enabled = self._env_bool("HAB_VISUAL_ROBOT_ENABLE", False)
        if not session.visual_robot_enabled:
            return

        robot_root = os.environ.get("HAB_VISUAL_ROBOT_ROOT", "").strip()
        urdf_path = os.environ.get("HAB_VISUAL_ROBOT_URDF", "").strip()
        mesh_path = os.environ.get("HAB_VISUAL_ROBOT_MESH", "").strip()
        if not mesh_path and robot_root:
            for candidate in (
                os.path.join(robot_root, "meshesColored", "base.glb"),
                os.path.join(robot_root, "meshes", "base.glb"),
            ):
                if os.path.isfile(candidate):
                    mesh_path = candidate
                    break

        # First try articulated URDF path (requires Bullet). If unavailable, fall
        # back to a rigid visual proxy mesh so third-person still shows a robot body.
        physics_enabled = self._session_physics_enabled(session)
        if urdf_path and os.path.isfile(urdf_path) and physics_enabled:
            try:
                aom = session.simulator.get_articulated_object_manager()
                ao = aom.add_articulated_object_from_urdf(
                    urdf_path,
                    True,   # fixed_base
                    1.0,    # global_scale
                    1.0,    # mass_scale
                    False,  # force_reload
                    False,  # maintain_link_order
                    False,  # inertia_from_urdf
                    "",     # light_setup_key
                )
                if ao is not None:
                    session.visual_robot_obj = ao
                    session.visual_robot_kind = "articulated"
                    self._sync_visual_robot_pose(session)
                    self._logger.info(
                        "Attached articulated visual robot session=%s urdf=%s",
                        session.session_id,
                        urdf_path,
                    )
                    return
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Failed to attach articulated robot; fallback to rigid proxy. session=%s urdf=%s error=%s",
                    session.session_id,
                    urdf_path,
                    exc,
                )
        elif urdf_path and os.path.isfile(urdf_path):
            self._logger.info(
                "Physics disabled for session=%s; skip articulated robot and use rigid proxy.",
                session.session_id,
            )

        if not mesh_path or not os.path.isfile(mesh_path):
            self._logger.warning(
                "Visual robot enabled but mesh path is unavailable. session=%s mesh=%s",
                session.session_id,
                mesh_path,
            )
            return

        try:
            otm = session.simulator.metadata_mediator.object_template_manager
            rom = session.simulator.get_rigid_object_manager()
            scale = self._env_float("HAB_VISUAL_ROBOT_SCALE", 1.0)
            template = otm.create_new_template(mesh_path)
            template.scale = [scale, scale, scale]
            handle = f"visual_robot_{session.session_id}"
            otm.register_template(template, handle, True)
            robot_obj = rom.add_object_by_template_handle(handle)
            if robot_obj is None:
                self._logger.warning(
                    "Failed to instantiate rigid visual robot session=%s mesh=%s",
                    session.session_id,
                    mesh_path,
                )
                return
            session.visual_robot_obj = robot_obj
            session.visual_robot_kind = "rigid"
            session.visual_robot_template_handle = handle
            self._sync_visual_robot_pose(session)
            self._logger.info(
                "Attached rigid visual robot session=%s mesh=%s handle=%s scale=%s",
                session.session_id,
                mesh_path,
                handle,
                scale,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to attach rigid visual robot session=%s mesh=%s error=%s",
                session.session_id,
                mesh_path,
                exc,
            )

    def _sync_visual_robot_pose(self, session: _Session) -> None:
        obj = session.visual_robot_obj
        if obj is None:
            return

        try:
            import magnum as mn
            position = self._current_position(session)
            rotation_wxyz = self._current_rotation(session)
            offset_y = self._env_float("HAB_VISUAL_ROBOT_OFFSET_Y", 0.0)
            yaw_offset_deg = self._env_float("HAB_VISUAL_ROBOT_YAW_OFFSET_DEG", 0.0)
            auto_ground_align = self._env_bool(
                "HAB_VISUAL_ROBOT_AUTO_GROUND_ALIGN", True
            )
            ground_target_y = float(position[1]) + float(offset_y)

            target_pos = mn.Vector3(
                float(position[0]),
                ground_target_y,
                float(position[2]),
            )
            target_rot = mn.Quaternion(
                mn.Vector3(
                    float(rotation_wxyz[1]),
                    float(rotation_wxyz[2]),
                    float(rotation_wxyz[3]),
                ),
                float(rotation_wxyz[0]),
            )
            if abs(yaw_offset_deg) > 1e-6:
                yaw_q = mn.Quaternion.rotation(
                    mn.Rad(float(np.deg2rad(yaw_offset_deg))),
                    mn.Vector3(0.0, 1.0, 0.0),
                )
                target_rot = target_rot * yaw_q

            self._apply_visual_robot_transform(obj, target_pos, target_rot)

            # Optional automatic ground alignment:
            # shift robot vertically so world-space AABB bottom sits on ground_target_y.
            if auto_ground_align:
                bottom_y = self._visual_robot_world_bottom_y(obj)
                if (
                    bottom_y is not None
                    and np.isfinite(bottom_y)
                ):
                    delta_y = ground_target_y - float(bottom_y)
                    if abs(delta_y) > 1e-6:
                        corrected_pos = mn.Vector3(
                            float(target_pos[0]),
                            float(target_pos[1]) + float(delta_y),
                            float(target_pos[2]),
                        )
                        self._apply_visual_robot_transform(
                            obj, corrected_pos, target_rot
                        )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to sync visual robot pose session=%s: %s",
                session.session_id,
                exc,
            )

    @staticmethod
    def _range3d_corners(range3d: Any) -> list:
        """Return 8 corners of a Magnum Range3D-like object."""
        return [
            range3d.back_bottom_left,
            range3d.back_bottom_right,
            range3d.back_top_right,
            range3d.back_top_left,
            range3d.front_top_left,
            range3d.front_top_right,
            range3d.front_bottom_right,
            range3d.front_bottom_left,
        ]

    def _visual_robot_world_bottom_y(self, obj: Any) -> Optional[float]:
        """Best-effort world-space AABB bottom Y for rigid/articulated robot."""
        try:
            import magnum as mn
        except ImportError:
            return None

        def _update_min_from_range3d(
            current_min: Optional[float], range3d: Any, world_transform: Any
        ) -> Optional[float]:
            try:
                for corner in self._range3d_corners(range3d):
                    world_point = world_transform.transform_point(corner)
                    y_value = float(world_point[1])
                    current_min = (
                        y_value if current_min is None else min(current_min, y_value)
                    )
            except Exception:
                pass
            return current_min

        min_world_y: Optional[float] = None

        # Articulated object: aggregate all link cumulative bounding boxes.
        try:
            num_links = getattr(obj, "num_links", None)
            get_link_scene_node = getattr(obj, "get_link_scene_node", None)
            if isinstance(num_links, int) and callable(get_link_scene_node):
                for link_ix in range(-1, num_links):
                    try:
                        link_node = get_link_scene_node(link_ix)
                        link_bb = getattr(link_node, "cumulative_bb", None)
                        abs_tf_fn = getattr(link_node, "absolute_transformation", None)
                        if link_bb is None or not callable(abs_tf_fn):
                            continue
                        min_world_y = _update_min_from_range3d(
                            min_world_y,
                            link_bb,
                            abs_tf_fn(),
                        )
                    except Exception:
                        continue
                if min_world_y is not None:
                    return min_world_y
        except Exception:
            pass

        # Rigid fallback: use local collision_shape_aabb transformed by object transform.
        try:
            local_bb = getattr(obj, "collision_shape_aabb", None)
            world_tf = getattr(obj, "transformation", None)
            if local_bb is not None and world_tf is not None:
                min_world_y = _update_min_from_range3d(min_world_y, local_bb, world_tf)
                if min_world_y is not None:
                    return min_world_y
        except Exception:
            pass

        # Last fallback: root scene node cumulative_bb if available.
        try:
            root_node = getattr(obj, "root_scene_node", None)
            node_bb = getattr(root_node, "cumulative_bb", None)
            abs_tf_fn = getattr(root_node, "absolute_transformation", None)
            if node_bb is not None and callable(abs_tf_fn):
                min_world_y = _update_min_from_range3d(min_world_y, node_bb, abs_tf_fn())
        except Exception:
            pass

        return min_world_y

    @staticmethod
    def _apply_visual_robot_transform(
        obj: Any, target_pos: Any, target_rot: Any
    ) -> None:
        # Prefer direct translation+rotation assignment. Fall back to matrix.
        if hasattr(obj, "translation"):
            obj.translation = target_pos
        if hasattr(obj, "rotation"):
            obj.rotation = target_rot
        elif hasattr(obj, "transformation"):
            try:
                import magnum as mn
                obj.transformation = mn.Matrix4.from_(
                    target_rot.to_matrix(), target_pos
                )
            except Exception:
                pass

    def _detach_visual_robot(self, session: _Session) -> None:
        if session.visual_robot_obj is None:
            return
        try:
            if session.visual_robot_kind == "rigid":
                rom = session.simulator.get_rigid_object_manager()
                object_id = getattr(session.visual_robot_obj, "object_id", None)
                if isinstance(object_id, int) and object_id >= 0:
                    rom.remove_object_by_id(object_id)
        except Exception:
            pass
        session.visual_robot_obj = None
