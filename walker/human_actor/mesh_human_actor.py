from typing import Dict, Optional, Tuple

import numpy as np

from .clip import BakedMeshClip
from .trajectory import HumanTrajectory


STATE_TO_CLIP = {
    0: "idle",
    1: "walk",
    2: "slow_walk",
    3: "turn_left",
    4: "turn_right",
    5: "stop",
}


class MeshHumanActor:
    def __init__(
        self,
        actor_id: int,
        name: str,
        clips: Dict[str, BakedMeshClip],
        trajectory: HumanTrajectory,
        fallback_clip: str = "walk",
        capsule_radius: float = 0.35,
        capsule_height: float = 1.70,
    ):
        if not clips:
            raise ValueError("MeshHumanActor requires at least one clip.")
        if fallback_clip not in clips:
            fallback_clip = "idle" if "idle" in clips else next(iter(clips.keys()))

        self.actor_id = int(actor_id)
        self.name = str(name)
        self.clips = clips
        self.trajectory = trajectory
        self.fallback_clip = fallback_clip
        self.capsule_radius = float(capsule_radius)
        self.capsule_height = float(capsule_height)

    def state_to_clip_name(self, state_id: int) -> str:
        return STATE_TO_CLIP.get(int(state_id), self.fallback_clip)

    def _active_clip(self, sim_time: float) -> Tuple[BakedMeshClip, str, float]:
        sample = self.trajectory.sample(sim_time)
        desired_name = self.state_to_clip_name(sample["state_id"])

        if desired_name in self.clips:
            return self.clips[desired_name], desired_name, sim_time

        if desired_name in ("idle", "stop") and self.fallback_clip in self.clips:
            return self.clips[self.fallback_clip], desired_name, 0.0

        return self.clips[self.fallback_clip], self.fallback_clip, sim_time

    def root_pose_at(self, sim_time: float) -> np.ndarray:
        sample = self.trajectory.sample(sim_time)
        return np.array(
            [sample["x"], sample["y"], sample["z"], sample["heading"]],
            dtype=np.float32,
        )

    def _rotation_at(self, sim_time: float) -> np.ndarray:
        heading = float(self.root_pose_at(sim_time)[3])
        c = np.cos(heading)
        s = np.sin(heading)
        return np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=np.float32,
        )

    def world_vertices_at(self, sim_time: float) -> np.ndarray:
        clip, _, clip_time = self._active_clip(sim_time)
        verts_local = clip.vertices_at(clip_time)
        x, y, z, _ = self.root_pose_at(sim_time)
        rotation = self._rotation_at(sim_time)
        return (verts_local @ rotation.T + np.array([x, y, z], dtype=np.float32)).astype(
            np.float32
        )

    def world_normals_at(self, sim_time: float) -> Optional[np.ndarray]:
        clip, _, clip_time = self._active_clip(sim_time)
        normals_local = clip.normals_at(clip_time)
        if normals_local is None:
            return None
        rotation = self._rotation_at(sim_time)
        return (normals_local @ rotation.T).astype(np.float32)

    def mesh_at(self, sim_time: float) -> dict:
        clip, state_name, _ = self._active_clip(sim_time)
        normals = self.world_normals_at(sim_time)
        return {
            "vertices": self.world_vertices_at(sim_time),
            "faces": clip.faces,
            "face_uvs": clip.face_uvs,
            "normals": normals,
            "actor_id": self.actor_id,
            "name": self.name,
            "state_name": state_name,
            "root_pose": self.root_pose_at(sim_time),
        }

    def capsule_at(self, sim_time: float) -> dict:
        x, y, z, heading = self.root_pose_at(sim_time)
        return {
            "actor_id": self.actor_id,
            "name": self.name,
            "center": np.array(
                [x, y + self.capsule_height * 0.5, z],
                dtype=np.float32,
            ),
            "radius": self.capsule_radius,
            "height": self.capsule_height,
            "heading": float(heading),
        }
