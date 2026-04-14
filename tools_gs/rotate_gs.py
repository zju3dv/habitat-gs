#!/usr/bin/env python3
"""Rotate a 3DGS PLY asset into a new PLY file."""

import argparse
import math
import sys
from typing import Dict, Tuple

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


# SH constants must match renderer (src/esp/gfx/gaussian_cuda_rasterizer/auxiliary.h)
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = np.array(
    [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    ]
)
SH_C3 = np.array(
    [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    ]
)


def rotation_matrix_from_euler(rx: float, ry: float, rz: float, order: str = "xyz") -> np.ndarray:
    """Create a rotation matrix (3x3) from Euler angles in degrees."""
    return R.from_euler(order, [rx, ry, rz], degrees=True).as_matrix()


def matrix_to_quaternion_wxyz(rot_matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a quaternion in [w, x, y, z] order."""
    quat_xyzw = R.from_matrix(rot_matrix).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication for arrays in [w, x, y, z] order."""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def deduce_sh_degree_from_coeff_count(coeff_count: int) -> int:
    """Infer SH degree from the number of coefficients per channel (excluding DC)."""
    if coeff_count <= 0:
        return 0
    return max(0, int(round(math.sqrt(coeff_count + 1.0) - 1.0)))


def fibonacci_sphere(samples: int) -> np.ndarray:
    """Evenly distribute sample directions on a unit sphere."""
    i = np.arange(samples, dtype=np.float64)
    phi = (1 + math.sqrt(5)) / 2
    z = 1.0 - 2.0 * (i + 0.5) / samples
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = 2.0 * math.pi * i / phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def sh_basis(directions: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate real SH basis with the ordering used by the CUDA renderer."""
    if degree < 0:
        raise ValueError("SH degree must be non-negative")
    if degree > 3:
        raise ValueError("SH degree > 3 is not supported by this script")

    dirs = np.asarray(directions, dtype=np.float64)
    coeff_count = (degree + 1) ** 2
    basis = np.zeros((dirs.shape[0], coeff_count), dtype=np.float64)

    x = dirs[:, 0]
    y = dirs[:, 1]
    z = dirs[:, 2]
    basis[:, 0] = SH_C0

    if degree >= 1:
        basis[:, 1] = -SH_C1 * y
        basis[:, 2] = SH_C1 * z
        basis[:, 3] = -SH_C1 * x

    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        basis[:, 4] = SH_C2[0] * xy
        basis[:, 5] = SH_C2[1] * yz
        basis[:, 6] = SH_C2[2] * (2.0 * zz - xx - yy)
        basis[:, 7] = SH_C2[3] * xz
        basis[:, 8] = SH_C2[4] * (xx - yy)

    if degree >= 3:
        xx, yy, zz = x * x, y * y, z * z
        basis[:, 9] = SH_C3[0] * y * (3.0 * xx - yy)
        basis[:, 10] = SH_C3[1] * xy * z
        basis[:, 11] = SH_C3[2] * y * (4.0 * zz - xx - yy)
        basis[:, 12] = SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
        basis[:, 13] = SH_C3[4] * x * (4.0 * zz - xx - yy)
        basis[:, 14] = SH_C3[5] * z * (xx - yy)
        basis[:, 15] = SH_C3[6] * x * (xx - 3.0 * yy)

    return basis


def build_sh_rotation_matrix(rot_matrix: np.ndarray, degree: int) -> np.ndarray:
    """Compute the SH rotation matrix (including DC) for the given degree."""
    degree = int(degree)
    if degree <= 0:
        return np.eye(1, dtype=np.float32)

    samples = max(200, (degree + 1) ** 2 * 8)
    dirs = fibonacci_sphere(samples)
    source_dirs = dirs @ rot_matrix
    target_basis = sh_basis(dirs, degree)
    source_basis = sh_basis(source_dirs, degree)

    rot = np.linalg.pinv(target_basis) @ source_basis
    rot[0, :] = 0.0
    rot[:, 0] = 0.0
    rot[0, 0] = 1.0
    return rot.astype(np.float32)


def rotate_sh_coeffs(dc: np.ndarray, rest_coeff: np.ndarray, sh_rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate SH coefficients."""
    if rest_coeff.size == 0:
        return dc, rest_coeff

    coeffs = np.concatenate([dc[:, None, :], rest_coeff], axis=1)
    coeffs = coeffs.transpose(0, 2, 1)
    rotated = coeffs @ sh_rot.T
    rotated = rotated.transpose(0, 2, 1)
    rotated_dc = rotated[:, 0, :].astype(dc.dtype, copy=False)
    rotated_rest = rotated[:, 1:, :].astype(rest_coeff.dtype, copy=False)
    return rotated_dc, rotated_rest


def rotate_vectors(vectors: np.ndarray, rot_matrix: np.ndarray) -> np.ndarray:
    """Rotate Nx3 vectors with the provided matrix."""
    return (vectors @ rot_matrix.T).astype(vectors.dtype, copy=False)


def rotate_ply(
    input_path: str,
    output_path: str,
    rot_matrix: np.ndarray,
    rot_quat_wxyz: np.ndarray,
    sh_rot_cache: Dict[int, np.ndarray],
) -> None:
    """Rotate a 3DGS PLY asset."""
    print(f"Reading PLY file: {input_path}")
    try:
        plydata = PlyData.read(input_path)
    except Exception as exc:
        print(f"Error: failed to read PLY file: {exc}")
        sys.exit(1)

    if "vertex" not in plydata:
        print("Error: PLY file does not contain vertex element")
        sys.exit(1)

    vertex = plydata["vertex"]
    names = vertex.data.dtype.names or []
    if any(name.startswith("motion_") for name in names):
        print("Error: rotate_gs.py now supports only 3DGS PLY assets.")
        sys.exit(1)

    new_vertex = vertex.data.copy()

    positions = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])
    rotated_positions = rotate_vectors(positions, rot_matrix)
    new_vertex["x"] = rotated_positions[:, 0]
    new_vertex["y"] = rotated_positions[:, 1]
    new_vertex["z"] = rotated_positions[:, 2]

    if all(field in names for field in ("nx", "ny", "nz")):
        normals = np.column_stack([vertex["nx"], vertex["ny"], vertex["nz"]])
        rotated_normals = rotate_vectors(normals, rot_matrix)
        new_vertex["nx"] = rotated_normals[:, 0]
        new_vertex["ny"] = rotated_normals[:, 1]
        new_vertex["nz"] = rotated_normals[:, 2]

    if all(f"rot_{idx}" in names for idx in range(4)):
        quats = np.column_stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]])
        rotated_quats = quaternion_multiply(rot_quat_wxyz[None, :], quats)
        rotated_quats /= np.linalg.norm(rotated_quats, axis=1, keepdims=True) + 1e-8
        for idx in range(4):
            new_vertex[f"rot_{idx}"] = rotated_quats[:, idx].astype(new_vertex[f"rot_{idx}"].dtype, copy=False)

    has_dc = all(f"f_dc_{idx}" in names for idx in range(3))
    rest_fields = sorted(
        [name for name in names if name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[2]),
    )
    if has_dc and rest_fields:
        rest_count = len(rest_fields)
        if rest_count % 3 != 0:
            print(f"Warning: SH rest count {rest_count} is not divisible by 3, skipping SH rotation")
        else:
            dc = np.column_stack([vertex[f"f_dc_{idx}"] for idx in range(3)])
            rest_flat = np.column_stack([vertex[field] for field in rest_fields])
            coeff_count = rest_count // 3
            degree = deduce_sh_degree_from_coeff_count(coeff_count)
            sh_rot = sh_rot_cache.setdefault(degree, build_sh_rotation_matrix(rot_matrix, degree))

            rest_coeff = rest_flat.reshape(len(vertex), 3, coeff_count).transpose(0, 2, 1)
            rotated_dc, rotated_rest = rotate_sh_coeffs(dc, rest_coeff, sh_rot)
            rotated_rest_flat = rotated_rest.transpose(0, 2, 1).reshape(len(vertex), rest_count)

            for idx in range(3):
                new_vertex[f"f_dc_{idx}"] = rotated_dc[:, idx].astype(new_vertex[f"f_dc_{idx}"].dtype, copy=False)
            for idx, field in enumerate(rest_fields):
                new_vertex[field] = rotated_rest_flat[:, idx].astype(new_vertex[field].dtype, copy=False)

    elements = [PlyElement.describe(new_vertex, "vertex")]
    for elem in getattr(plydata, "elements", []):
        if getattr(elem, "name", None) != "vertex":
            elements.append(elem)

    try:
        PlyData(elements, text=plydata.text).write(output_path)
    except Exception as exc:
        print(f"Error: failed to save PLY file: {exc}")
        sys.exit(1)
    print(f"Saved rotated PLY to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rotate a 3DGS PLY asset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rotate_gs.py --input input.ply --output output.ply --rx 45 --rz 30
        """,
    )
    parser.add_argument("--input", required=True, help="Input 3DGS PLY file path")
    parser.add_argument("--output", default="output.ply", help="Output file path")
    parser.add_argument(
        "--rx",
        type=float,
        default=0.0,
        help="Rotation angle around the X axis (degrees), default 0",
    )
    parser.add_argument(
        "--ry",
        type=float,
        default=0.0,
        help="Rotation angle around the Y axis (degrees), default 0",
    )
    parser.add_argument(
        "--rz",
        type=float,
        default=0.0,
        help="Rotation angle around the Z axis (degrees), default 0",
    )
    parser.add_argument(
        "--rotation-order",
        type=str,
        default="xyz",
        choices=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"],
        help="Euler rotation order, default xyz",
    )
    args = parser.parse_args()

    input_lower = args.input.lower()
    if not input_lower.endswith(".ply"):
        print("Error: rotate_gs.py supports only 3DGS .ply inputs.")
        sys.exit(1)

    if args.rx == 0 and args.ry == 0 and args.rz == 0:
        print("Warning: all rotation angles are zero; output will match input.")
        proceed = input("Continue without rotation? (y/n): ")
        if proceed.lower() != "y":
            print("Cancelled.")
            sys.exit(0)

    rot_matrix = rotation_matrix_from_euler(args.rx, args.ry, args.rz, args.rotation_order)
    rot_quat_wxyz = matrix_to_quaternion_wxyz(rot_matrix)
    sh_rot_cache: Dict[int, np.ndarray] = {}

    print(
        f"Applying rotation (order={args.rotation_order}): "
        f"rx={args.rx}°, ry={args.ry}°, rz={args.rz}°"
    )
    rotate_ply(args.input, args.output, rot_matrix, rot_quat_wxyz, sh_rot_cache)


if __name__ == "__main__":
    main()
