import argparse
from pathlib import Path
import numpy as np


def write_obj(out_path, vertices, faces):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("# exported baked human frame\n")

        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for face in faces:
            a, b, c = face + 1
            f.write(f"f {a} {b} {c}\n")

    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.npz)

    vertices = data["vertices"]
    faces = data["faces"]

    frame_id = args.frame % vertices.shape[0]
    v = vertices[frame_id]

    print("vertices:", vertices.shape)
    print("faces:", faces.shape)
    print("frame:", frame_id)
    print("bbox_min:", v.min(axis=0))
    print("bbox_max:", v.max(axis=0))

    write_obj(args.out, v, faces)


if __name__ == "__main__":
    main()
