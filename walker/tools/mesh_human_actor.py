import numpy as np


class MeshHumanActor:
    """
    Baked mesh human + root trajectory.

    baked mesh:
    - X/Z: horizontal plane
    - Y: up axis
    - unit: meter
    - animation: in-place

    trajectory:
    - shape [T, 7]
    - columns: x, y, z, heading, speed, state_id, time
    """

    def __init__(self, clip_path, trajectory_path=None):
        data = np.load(clip_path, allow_pickle=True)

        self.vertices = data["vertices"].astype(np.float32)   # [T, N, 3]
        self.faces = data["faces"].astype(np.int32)           # [F, 3]
        self.face_uvs = data["face_uvs"].astype(np.float32)   # [F, 3, 2]

        self.normals = None
        if "normals" in data.files:
            self.normals = data["normals"].astype(np.float32)

        self.fps = int(data["fps"][0])
        self.num_anim_frames = self.vertices.shape[0]

        self.bbox_min = data["bbox_min"].astype(np.float32)
        self.bbox_max = data["bbox_max"].astype(np.float32)

        self.trajectory = None
        self.traj_fps = self.fps

        if trajectory_path is not None:
            self.trajectory = np.load(trajectory_path).astype(np.float32)

        print("Loaded MeshHumanActor")
        print("  clip:", clip_path)
        print("  vertices:", self.vertices.shape)
        print("  faces:", self.faces.shape)
        print("  fps:", self.fps)
        print("  bbox_min:", self.bbox_min)
        print("  bbox_max:", self.bbox_max)

        if self.trajectory is not None:
            print("  trajectory:", trajectory_path)
            print("  trajectory shape:", self.trajectory.shape)

    def anim_frame_id(self, sim_time):
        return int(sim_time * self.fps) % self.num_anim_frames

    def local_vertices_at(self, sim_time):
        fid = self.anim_frame_id(sim_time)
        return self.vertices[fid]

    def root_pose_at(self, sim_time):
        if self.trajectory is None:
            return np.array([0.0, 0.0, 2.0, 0.0], dtype=np.float32)

        frame_id = int(sim_time * self.traj_fps)
        frame_id = max(0, min(frame_id, len(self.trajectory) - 1))

        x, y, z, heading, speed, state_id, t = self.trajectory[frame_id]

        return np.array([x, y, z, heading], dtype=np.float32)

    def world_vertices_at(self, sim_time):
        verts = self.local_vertices_at(sim_time)
        x, y, z, heading = self.root_pose_at(sim_time)

        c = np.cos(heading)
        s = np.sin(heading)

        # Y-up yaw rotation
        R = np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=np.float32,
        )

        verts_world = verts @ R.T
        verts_world += np.array([x, y, z], dtype=np.float32)

        return verts_world

    def capsule_at(self, sim_time, radius=0.35, height=1.70):
        x, y, z, heading = self.root_pose_at(sim_time)

        return {
            "center": np.array([x, y + height * 0.5, z], dtype=np.float32),
            "radius": radius,
            "height": height,
            "heading": heading,
        }
