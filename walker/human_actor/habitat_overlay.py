import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import BakedMeshClip, HumanTrajectory, MeshHumanActor, MeshHumanManager
from .rendering import SimpleMeshRenderer, build_camera_from_habitat_sensor, composite_rgbd


def _sensor_type_name(sensor: Any) -> str:
    sensor_type = getattr(getattr(sensor, "spec", None), "sensor_type", None)
    return str(getattr(sensor_type, "name", sensor_type))


def _resolve_scene_instance_path(sim) -> str:
    scene_id = getattr(sim.config.sim_cfg, "scene_id", "")
    if isinstance(scene_id, str) and os.path.exists(scene_id):
        return scene_id

    dataset_cfg = getattr(sim.config.sim_cfg, "scene_dataset_config_file", "")
    if not dataset_cfg or dataset_cfg == "default" or not os.path.exists(dataset_cfg):
        return ""

    with open(dataset_cfg, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    scene_instances = dataset.get("scene_instances", {})
    paths_cfg = scene_instances.get("paths", {}) if isinstance(scene_instances, dict) else {}
    base_dir = os.path.dirname(os.path.abspath(dataset_cfg))

    candidates = [f"{scene_id}.scene_instance.json", f"{scene_id}.json"]
    for rel_list in paths_cfg.values():
        if not isinstance(rel_list, list):
            continue
        for rel in rel_list:
            search_dir = os.path.join(base_dir, rel)
            for name in candidates:
                candidate = os.path.join(search_dir, name)
                if os.path.exists(candidate):
                    return candidate
    return ""


def _resolve_path(path_value: str, base_dir: str) -> str:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return str(path)

    repo_root = Path(__file__).resolve().parents[2]
    walker_root = repo_root / "walker"
    candidates = []
    if base_dir:
        candidates.append(Path(base_dir) / path)
    candidates.append(Path.cwd() / path)
    candidates.append(walker_root / path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Mesh human asset not found: {path_value}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _load_config(sim) -> tuple[Dict[str, Any], str]:
    scene_instance_path = _resolve_scene_instance_path(sim)
    if not scene_instance_path:
        return {}, ""
    with open(scene_instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = data.get("mesh_humans", {})
    if not isinstance(cfg, dict):
        raise ValueError(f"mesh_humans in {scene_instance_path} must be an object.")
    return cfg, scene_instance_path


class HabitatMeshHumanOverlay:
    def __init__(self, sim, cfg: Dict[str, Any], scene_instance_path: str):
        self.enabled = bool(cfg.get("enabled", False))
        self.debug = bool(cfg.get("debug", False))
        self.fps = int(cfg.get("fps", 30))
        self.rgb_uuid = cfg.get("rgb_uuid")
        self.depth_uuid = cfg.get("depth_uuid")
        self._renderers: Dict[tuple[int, int], SimpleMeshRenderer] = {}
        self.manager = self._load_manager(cfg, os.path.dirname(scene_instance_path))

    def _load_manager(self, cfg: Dict[str, Any], base_dir: str) -> MeshHumanManager:
        actors_cfg = cfg.get("actors", [])
        if not isinstance(actors_cfg, list):
            raise ValueError("mesh_humans.actors must be a list.")

        actors = []
        for idx, actor_cfg in enumerate(actors_cfg):
            clips_cfg = actor_cfg.get("clips", {})
            if not isinstance(clips_cfg, dict) or not clips_cfg:
                raise ValueError(f"mesh_humans.actors[{idx}].clips must be a non-empty object.")

            clips = {
                str(name): BakedMeshClip(_resolve_path(str(path), base_dir), name=str(name))
                for name, path in clips_cfg.items()
            }
            trajectory = HumanTrajectory(
                _resolve_path(str(actor_cfg["trajectory"]), base_dir),
                fps=self.fps,
            )
            actors.append(
                MeshHumanActor(
                    actor_id=int(actor_cfg.get("actor_id", idx + 1)),
                    name=str(actor_cfg.get("name", f"mesh_human_{idx + 1}")),
                    clips=clips,
                    trajectory=trajectory,
                    fallback_clip=str(actor_cfg.get("fallback_clip", "walk")),
                    capsule_radius=float(actor_cfg.get("capsule_radius", 0.35)),
                    capsule_height=float(actor_cfg.get("capsule_height", 1.70)),
                )
            )
        return MeshHumanManager(actors)

    def _sim_time(self, sim) -> float:
        for name in ("gaussian_time", "world_time"):
            try:
                return float(getattr(sim, name))
            except Exception:
                pass
        return float(getattr(sim, "_num_total_frames", 0)) / max(1, self.fps)

    def apply(self, sim, observations: Dict[str, Any], sensors: List[Any]) -> None:
        if not self.enabled:
            return

        color_sensors = [
            s
            for s in sensors
            if _sensor_type_name(s) == "COLOR"
            and s.uuid in observations
        ]
        depth_sensors = [
            s
            for s in sensors
            if _sensor_type_name(s) == "DEPTH"
            and s.uuid in observations
        ]
        if self.rgb_uuid:
            color_sensors = [s for s in color_sensors if s.uuid == self.rgb_uuid]
        if self.depth_uuid:
            depth_sensors = [s for s in depth_sensors if s.uuid == self.depth_uuid]

        sim_time = self._sim_time(sim)
        meshes = self.manager.meshes_at(sim_time)
        for color_sensor in color_sensors:
            color_shape = np.asarray(observations[color_sensor.uuid]).shape[:2]
            depth_sensor = next(
                (
                    s
                    for s in depth_sensors
                    if np.asarray(observations[s.uuid]).shape == color_shape
                ),
                None,
            )
            if depth_sensor is None:
                continue

            camera = build_camera_from_habitat_sensor(color_sensor, debug=self.debug)
            key = (camera.width, camera.height)
            renderer = self._renderers.get(key)
            if renderer is None:
                renderer = SimpleMeshRenderer(camera.width, camera.height)
                self._renderers[key] = renderer

            human = renderer.render(meshes, camera)
            composed = composite_rgbd(
                observations[color_sensor.uuid],
                observations[depth_sensor.uuid],
                human["rgb"],
                human["depth"],
                human["id_mask"],
                debug=self.debug,
            )
            observations[color_sensor.uuid] = composed["rgb"]
            observations[depth_sensor.uuid] = composed["depth"]
            observations["human_id_mask"] = composed["id_mask"]


def load_mesh_human_overlay(sim) -> Optional[HabitatMeshHumanOverlay]:
    cfg, scene_instance_path = _load_config(sim)
    if not cfg or not bool(cfg.get("enabled", False)):
        return None
    return HabitatMeshHumanOverlay(sim, cfg, scene_instance_path)
