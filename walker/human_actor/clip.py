from pathlib import Path
from typing import Optional

import numpy as np


class BakedMeshClip:
    """A baked, in-place mesh animation clip stored in npz format."""

    REQUIRED_FIELDS = ("vertices", "faces", "face_uvs", "fps", "bbox_min", "bbox_max")

    def __init__(self, npz_path: str, name: str = None):
        self.path = Path(npz_path)
        self.name = name or self.path.stem

        if not self.path.exists():
            raise FileNotFoundError(f"Baked mesh clip not found: {self.path}")

        data = np.load(self.path, allow_pickle=True)
        missing = [field for field in self.REQUIRED_FIELDS if field not in data.files]
        if missing:
            raise ValueError(
                f"Clip '{self.path}' is missing required field(s): {missing}. "
                f"Available fields: {data.files}"
            )

        self.vertices = np.asarray(data["vertices"], dtype=np.float32)
        self.faces = np.asarray(data["faces"], dtype=np.int32)
        self.face_uvs = np.asarray(data["face_uvs"], dtype=np.float32)
        self.normals = (
            np.asarray(data["normals"], dtype=np.float32)
            if "normals" in data.files
            else None
        )
        self.fps = int(np.asarray(data["fps"]).reshape(-1)[0])
        self.bbox_min = np.asarray(data["bbox_min"], dtype=np.float32)
        self.bbox_max = np.asarray(data["bbox_max"], dtype=np.float32)

        self._validate()

    @property
    def num_frames(self) -> int:
        return int(self.vertices.shape[0])

    def _validate(self) -> None:
        if self.vertices.ndim != 3 or self.vertices.shape[-1] != 3:
            raise ValueError(
                f"Clip '{self.path}' field 'vertices' must have shape [T, N, 3], "
                f"got {self.vertices.shape}."
            )
        if self.faces.ndim != 2 or self.faces.shape[-1] != 3:
            raise ValueError(
                f"Clip '{self.path}' field 'faces' must have shape [F, 3], "
                f"got {self.faces.shape}."
            )
        if self.face_uvs.ndim != 3 or self.face_uvs.shape[0] != self.faces.shape[0]:
            raise ValueError(
                f"Clip '{self.path}' field 'face_uvs' must start with one entry per face, "
                f"got face_uvs {self.face_uvs.shape} and faces {self.faces.shape}."
            )
        if self.normals is not None and self.normals.shape != self.vertices.shape:
            raise ValueError(
                f"Clip '{self.path}' field 'normals' must match vertices shape "
                f"{self.vertices.shape}, got {self.normals.shape}."
            )
        if self.fps <= 0:
            raise ValueError(f"Clip '{self.path}' fps must be positive, got {self.fps}.")
        if self.num_frames <= 0:
            raise ValueError(f"Clip '{self.path}' has no animation frames.")
        if self.bbox_min.shape != (3,) or self.bbox_max.shape != (3,):
            raise ValueError(
                f"Clip '{self.path}' bbox_min/bbox_max must have shape [3], "
                f"got {self.bbox_min.shape} and {self.bbox_max.shape}."
            )

    def frame_id(self, sim_time: float) -> int:
        if sim_time < 0:
            sim_time = 0.0
        return int(sim_time * self.fps) % self.num_frames

    def vertices_at(self, sim_time: float) -> np.ndarray:
        return self.vertices[self.frame_id(sim_time)]

    def normals_at(self, sim_time: float) -> Optional[np.ndarray]:
        if self.normals is None:
            return None
        return self.normals[self.frame_id(sim_time)]
