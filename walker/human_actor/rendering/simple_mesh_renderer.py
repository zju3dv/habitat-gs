from typing import Iterable

import numpy as np

from .camera import CameraParams


def _actor_color(actor_id: int) -> np.ndarray:
    palette = {
        1: (64, 220, 128),
        2: (235, 92, 80),
        3: (96, 150, 240),
        4: (190, 190, 190),
    }
    return np.array(palette.get(int(actor_id), (120, 180, 210)), dtype=np.uint8)


class SimpleMeshRenderer:
    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Renderer size must be positive, got {width}x{height}.")

    def _project(self, vertices: np.ndarray, camera: CameraParams):
        verts = np.asarray(vertices, dtype=np.float32)
        ones = np.ones((verts.shape[0], 1), dtype=np.float32)
        verts_h = np.concatenate([verts, ones], axis=1)
        cam = verts_h @ camera.camera_T_world.T

        # Habitat pinhole camera coordinates use local -Z as forward.
        depth = -cam[:, 2]
        valid = np.isfinite(depth) & (depth > camera.near) & (depth < camera.far)
        u = camera.fx * (cam[:, 0] / np.maximum(depth, 1e-8)) + camera.cx
        v = camera.cy - camera.fy * (cam[:, 1] / np.maximum(depth, 1e-8))
        return np.stack([u, v], axis=1).astype(np.float32), depth.astype(np.float32), valid

    def _print_bbox(self, actor_id: int, pixels: np.ndarray, valid: np.ndarray) -> None:
        valid_pixels = pixels[valid]
        if valid_pixels.size == 0:
            print(f"[MeshHumanRenderer] actor_id={actor_id} projected bbox: none valid_vertices=0")
            return
        xmin, ymin = valid_pixels.min(axis=0)
        xmax, ymax = valid_pixels.max(axis=0)
        print(
            "[MeshHumanRenderer] "
            f"actor_id={actor_id} projected bbox: "
            f"{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}, "
            f"valid_vertices={int(valid.sum())}"
        )

    def render(self, meshes: list[dict], camera: CameraParams) -> dict:
        if camera.width != self.width or camera.height != self.height:
            raise ValueError(
                f"Renderer size {self.width}x{self.height} does not match camera "
                f"{camera.width}x{camera.height}."
            )

        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth_buffer = np.full((self.height, self.width), np.inf, dtype=np.float32)
        id_mask = np.zeros((self.height, self.width), dtype=np.int32)

        for mesh in meshes:
            vertices = np.asarray(mesh["vertices"], dtype=np.float32)
            faces = np.asarray(mesh["faces"], dtype=np.int32)
            actor_id = int(mesh.get("actor_id", 1))
            color = _actor_color(actor_id)

            pixels, depths, valid_vertices = self._project(vertices, camera)
            self._print_bbox(actor_id, pixels, valid_vertices)

            for face in faces:
                i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
                if not (valid_vertices[i0] and valid_vertices[i1] and valid_vertices[i2]):
                    continue

                p0, p1, p2 = pixels[i0], pixels[i1], pixels[i2]
                z0, z1, z2 = depths[i0], depths[i1], depths[i2]
                xmin = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
                xmax = min(self.width - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
                ymin = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
                ymax = min(self.height - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))
                if xmin > xmax or ymin > ymax:
                    continue

                area = (
                    (p1[0] - p0[0]) * (p2[1] - p0[1])
                    - (p1[1] - p0[1]) * (p2[0] - p0[0])
                )
                if abs(float(area)) < 1e-8:
                    continue

                xs = np.arange(xmin, xmax + 1, dtype=np.float32) + 0.5
                ys = np.arange(ymin, ymax + 1, dtype=np.float32) + 0.5
                grid_x, grid_y = np.meshgrid(xs, ys)

                w0 = (
                    (p1[0] - grid_x) * (p2[1] - grid_y)
                    - (p1[1] - grid_y) * (p2[0] - grid_x)
                ) / area
                w1 = (
                    (p2[0] - grid_x) * (p0[1] - grid_y)
                    - (p2[1] - grid_y) * (p0[0] - grid_x)
                ) / area
                w2 = 1.0 - w0 - w1
                inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
                if not inside.any():
                    continue

                tri_depth = w0 * z0 + w1 * z1 + w2 * z2
                region = depth_buffer[ymin : ymax + 1, xmin : xmax + 1]
                update = inside & (tri_depth < region)
                if not update.any():
                    continue

                region[update] = tri_depth[update]
                rgb_region = rgb[ymin : ymax + 1, xmin : xmax + 1]
                mask_region = id_mask[ymin : ymax + 1, xmin : xmax + 1]
                rgb_region[update] = color
                mask_region[update] = actor_id

        depth = depth_buffer.copy()
        depth[~np.isfinite(depth)] = 0.0
        return {
            "rgb": rgb,
            "depth": depth.astype(np.float32),
            "id_mask": id_mask,
        }
