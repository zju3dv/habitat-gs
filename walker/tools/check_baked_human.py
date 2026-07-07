import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--up_axis", type=str, default="y", choices=["x", "y", "z"])
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=True)

    vertices = d["vertices"]
    faces = d["faces"]
    bbox_min = d["bbox_min"]
    bbox_max = d["bbox_max"]

    axis_map = {"x": 0, "y": 1, "z": 2}
    up_axis = axis_map[args.up_axis]
    horizontal_axes = [i for i in range(3) if i != up_axis]

    height = bbox_max[up_axis] - bbox_min[up_axis]

    centers = vertices.mean(axis=1)
    disp = centers[-1] - centers[0]
    horizontal_disp = disp[horizontal_axes]

    print("files:", d.files)
    print("vertices:", vertices.shape)
    print("faces:", faces.shape)
    print("fps:", d["fps"])
    print("bbox_min:", bbox_min)
    print("bbox_max:", bbox_max)
    print("height:", height)
    print("center first:", centers[0])
    print("center last :", centers[-1])
    print("center displacement:", disp)
    print("horizontal displacement:", horizontal_disp)
    print("horizontal displacement norm:", np.linalg.norm(horizontal_disp))

    if height > 3.0:
        print("\nWARNING: height too large. Maybe still in centimeters.")
    elif height < 1.0:
        print("\nWARNING: height too small. Check scale.")
    else:
        print("\nOK: height looks reasonable.")

    if np.linalg.norm(horizontal_disp) > 0.05:
        print("WARNING: root motion still exists.")
    else:
        print("OK: root motion looks removed / in-place.")


if __name__ == "__main__":
    main()
