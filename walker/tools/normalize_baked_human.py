import argparse
from pathlib import Path

import numpy as np


def recompute_bbox(vertices):
    bbox_min = vertices.reshape(-1, 3).min(axis=0).astype(np.float32)
    bbox_max = vertices.reshape(-1, 3).max(axis=0).astype(np.float32)
    return bbox_min, bbox_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_npz", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--up_axis", type=str, default="y", choices=["x", "y", "z"])
    parser.add_argument("--remove_root_motion", action="store_true")
    parser.add_argument("--ground_align", action="store_true")
    args = parser.parse_args()

    data = np.load(args.in_npz, allow_pickle=True)

    vertices = data["vertices"].astype(np.float32)
    normals = data["normals"].astype(np.float32) if "normals" in data.files else None
    faces = data["faces"].astype(np.int32)
    face_uvs = data["face_uvs"].astype(np.float32)

    print("Input:")
    print("  vertices:", vertices.shape)
    print("  faces:", faces.shape)

    # cm -> m
    vertices = vertices * args.scale

    axis_map = {"x": 0, "y": 1, "z": 2}
    up_axis = axis_map[args.up_axis]
    horizontal_axes = [i for i in range(3) if i != up_axis]

    if args.remove_root_motion:
        centers = vertices.mean(axis=1)
        original_disp = centers[-1, horizontal_axes] - centers[0, horizontal_axes]
        root_offsets = centers[:, horizontal_axes] - centers[0, horizontal_axes]

        for k, axis in enumerate(horizontal_axes):
            vertices[:, :, axis] -= root_offsets[:, k][:, None]

        print("Removed root motion on horizontal axes:", horizontal_axes)
        print("Original horizontal center displacement:", original_disp)

    if args.ground_align:
        min_up = vertices[:, :, up_axis].min()
        vertices[:, :, up_axis] -= min_up
        print(f"Ground aligned. Shifted up-axis by {-min_up:.6f}")

    bbox_min, bbox_max = recompute_bbox(vertices)
    height = bbox_max[up_axis] - bbox_min[up_axis]

    print("Output:")
    print("  bbox_min:", bbox_min)
    print("  bbox_max:", bbox_max)
    print("  height:", height)

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "vertices": vertices,
        "faces": faces,
        "face_uvs": face_uvs,
        "fps": data["fps"],
        "start_frame": data["start_frame"],
        "end_frame": data["end_frame"],
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "up_axis": np.array([args.up_axis]),
        "scale_applied": np.array([args.scale], dtype=np.float32),
        "root_motion_removed": np.array([args.remove_root_motion]),
    }

    if normals is not None:
        save_dict["normals"] = normals

    if "mesh_name" in data.files:
        save_dict["mesh_name"] = data["mesh_name"]

    if "texture_paths" in data.files:
        save_dict["texture_paths"] = data["texture_paths"]

    np.savez_compressed(out_path, **save_dict)

    print("")
    print("Saved normalized human:", out_path)


if __name__ == "__main__":
    main()
