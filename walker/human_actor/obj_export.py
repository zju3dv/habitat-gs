from pathlib import Path
from typing import List

import numpy as np


def write_obj(out_path: str, vertices: np.ndarray, faces: np.ndarray):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# mesh human actor obj\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            a, b, c = face + 1
            f.write(f"f {a} {b} {c}\n")


def write_multi_actor_obj(out_path: str, meshes: List[dict]):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vertex_offset = 0
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# multi actor mesh human obj\n")
        for mesh in meshes:
            name = str(mesh.get("name", "actor"))
            vertices = np.asarray(mesh["vertices"], dtype=np.float32)
            faces = np.asarray(mesh["faces"], dtype=np.int32)

            f.write(f"o {name}\n")
            for v in vertices:
                f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
            for face in faces:
                a, b, c = face + 1 + vertex_offset
                f.write(f"f {a} {b} {c}\n")

            vertex_offset += vertices.shape[0]
