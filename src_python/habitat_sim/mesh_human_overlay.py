import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from habitat_sim.logging import logger
from habitat_sim.sensor import SensorType


def _ensure_walker_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    walker_dir = repo_root / "walker"
    if str(walker_dir) not in sys.path:
        sys.path.insert(0, str(walker_dir))
    return walker_dir


def _resolve_scene_instance_path(sim) -> str:
    try:
        from habitat_sim.gaussian_avatar import _resolve_scene_instance_path as resolve

        return resolve(sim)
    except Exception:
        scene_id = getattr(sim.config.sim_cfg, "scene_id", "")
        if isinstance(scene_id, str) and os.path.exists(scene_id):
            return scene_id
    return ""


def _load_mesh_human_config(sim) -> tuple[Dict[str, Any], str]:
    scene_instance_path = _resolve_scene_instance_path(sim)
    if not scene_instance_path or not os.path.exists(scene_instance_path):
        return {}, ""
    with open(scene_instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = data.get("mesh_humans", {})
    if not isinstance(cfg, dict):
        raise ValueError(f"mesh_humans in {scene_instance_path} must be an object.")
    return cfg, scene_instance_path


def _resolve_path(path_value: str, base_dir: str) -> str:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return str(path)

    candidates = []
    if base_dir:
        candidates.append(Path(base_dir) / path)
    candidates.append(Path.cwd() / path)
    walker_dir = _ensure_walker_on_path()
    candidates.append(walker_dir / path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Mesh human asset not found: {path_value}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def _load_manager_from_config(cfg: Dict[str, Any], base_dir: str):
    _ensure_walker_on_path()
    from human_actor import BakedMeshClip, HumanTrajectory, MeshHumanActor, MeshHumanManager

    actors_cfg = cfg.get("actors", [])
    if not isinstance(actors_cfg, list):
        raise ValueError("mesh_humans.actors must be a list.")

    actors = []
    fps = int(cfg.get("fps", 30))
    for idx, actor_cfg in enumerate(actors_cfg):
        if not isinstance(actor_cfg, dict):
            raise ValueError(f"mesh_humans.actors[{idx}] must be an object.")
        clips_cfg = actor_cfg.get("clips", {})
        if not isinstance(clips_cfg, dict) or not clips_cfg:
            raise ValueError(f"mesh_humans.actors[{idx}].clips must be a non-empty object.")

        clips = {}
        for clip_name, clip_path in clips_cfg.items():
            clips[str(clip_name)] = BakedMeshClip(
                _resolve_path(str(clip_path), base_dir),
                name=str(clip_name),
            )

        trajectory_path = _resolve_path(str(actor_cfg["trajectory"]), base_dir)
        trajectory = HumanTrajectory(trajectory_path, fps=fps)
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


class MeshHumanOverlay:
    def __init__(self, sim, cfg: Dict[str, Any], scene_instance_path: str):
        _ensure_walker_on_path()
        from human_actor.rendering import SimpleMeshRenderer

        self.enabled = bool(cfg.get("enabled", False))
        self.debug = bool(cfg.get("debug", False))
        self.fps = int(cfg.get("fps", 30))
        self.rgb_uuid = cfg.get("rgb_uuid")
        self.depth_uuid = cfg.get("depth_uuid")
        self.manager = _load_manager_from_config(
            cfg,
            os.path.dirname(scene_instance_path) if scene_instance_path else "",
        )
        self._renderers: Dict[tuple[int, int], SimpleMeshRenderer] = {}

    def _sim_time(self, sim) -> float:
        for name in ("gaussian_time", "world_time"):
            try:
                return float(getattr(sim, name))
            except Exception:
                pass
        return float(getattr(sim, "_num_total_frames", 0)) / max(1, self.fps)

    def _find_pairs(self, observations: Dict[str, Any], sensors: List[Any]):
        color_sensors = [
            s
            for s in sensors
            if getattr(s.spec, "sensor_type", None) == SensorType.COLOR and s.uuid in observations
        ]
        depth_sensors = [
            s
            for s in sensors
            if getattr(s.spec, "sensor_type", None) == SensorType.DEPTH and s.uuid in observations
        ]
        if self.rgb_uuid:
            color_sensors = [s for s in color_sensors if s.uuid == self.rgb_uuid]
        if self.depth_uuid:
            depth_sensors = [s for s in depth_sensors if s.uuid == self.depth_uuid]

        pairs = []
        for color in color_sensors:
            color_shape = np.asarray(observations[color.uuid]).shape[:2]
            for depth in depth_sensors:
                if np.asarray(observations[depth.uuid]).shape == color_shape:
                    pairs.append((color, depth))
                    break
        return pairs

    def apply(self, sim, observations: Dict[str, Any], sensors: List[Any]) -> None:
        if not self.enabled:
            return

        _ensure_walker_on_path()
        from human_actor.rendering import (
            SimpleMeshRenderer,
            build_camera_from_habitat_sensor,
            composite_rgbd,
        )

        pairs = self._find_pairs(observations, sensors)
        if not pairs and self.debug:
            logger.warning("mesh_humans enabled but no COLOR/DEPTH sensor pair was found.")
            return

        sim_time = self._sim_time(sim)
        meshes = self.manager.meshes_at(sim_time)

        for color_sensor, depth_sensor in pairs:
            rgb_gs = observations[color_sensor.uuid]
            depth_gs = observations[depth_sensor.uuid]
            camera = build_camera_from_habitat_sensor(
                color_sensor,
                debug=self.debug,
            )
            key = (camera.width, camera.height)
            renderer = self._renderers.get(key)
            if renderer is None:
                renderer = SimpleMeshRenderer(camera.width, camera.height)
                self._renderers[key] = renderer

            human = renderer.render(meshes, camera)
            composed = composite_rgbd(
                rgb_gs=rgb_gs,
                depth_gs=depth_gs,
                rgb_human=human["rgb"],
                depth_human=human["depth"],
                id_mask=human["id_mask"],
                debug=self.debug,
            )

            observations[color_sensor.uuid] = composed["rgb"]
            observations[depth_sensor.uuid] = composed["depth"]
            observations["human_id_mask"] = composed["id_mask"]
            observations[f"{color_sensor.uuid}_human_id_mask"] = composed["id_mask"]


def load_mesh_human_overlay(sim) -> Optional[MeshHumanOverlay]:
    cfg, scene_instance_path = _load_mesh_human_config(sim)
    if not cfg or not bool(cfg.get("enabled", False)):
        return None
    overlay = MeshHumanOverlay(sim, cfg, scene_instance_path)
    logger.info("Mesh human overlay enabled from %s", scene_instance_path or "config")
    return overlay
