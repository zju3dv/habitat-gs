from pathlib import Path

import numpy as np


class HumanTrajectory:
    """Trajectory sampler for [x, y, z, heading, speed, state_id, time] rows."""

    def __init__(self, npy_path: str, fps: int = 30):
        self.path = Path(npy_path)
        self.fps = int(fps)

        if self.fps <= 0:
            raise ValueError(f"Trajectory fps must be positive, got {fps}.")
        if not self.path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self.path}")

        self.data = np.asarray(np.load(self.path), dtype=np.float32)
        if self.data.ndim != 2 or self.data.shape[1] != 7:
            raise ValueError(
                f"Trajectory '{self.path}' must have shape [T, 7], got {self.data.shape}."
            )
        if self.data.shape[0] == 0:
            raise ValueError(f"Trajectory '{self.path}' has no frames.")

    @property
    def num_frames(self) -> int:
        return int(self.data.shape[0])

    def frame_id(self, sim_time: float) -> int:
        idx = int(max(0.0, sim_time) * self.fps)
        return max(0, min(idx, self.num_frames - 1))

    def sample(self, sim_time: float) -> dict:
        row = self.data[self.frame_id(sim_time)]
        return {
            "x": float(row[0]),
            "y": float(row[1]),
            "z": float(row[2]),
            "heading": float(row[3]),
            "speed": float(row[4]),
            "state_id": int(row[5]),
            "time": float(row[6]),
        }
