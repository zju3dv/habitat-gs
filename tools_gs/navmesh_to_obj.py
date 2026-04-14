#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

try:
    import habitat_sim
except ImportError as exc:
    raise SystemExit(
        "navmesh_to_obj.py requires habitat-sim (Python module `habitat_sim`)."
    ) from exc


Vec3 = Tuple[float, float, float]


def _load_pathfinder(navmesh_path: str):
    if not os.path.isfile(navmesh_path):
        raise FileNotFoundError(f"NavMesh file not found: {navmesh_path}")

    pf = habitat_sim.nav.PathFinder()
    loaded = pf.load_nav_mesh(navmesh_path)
    if not loaded or not pf.is_loaded:
        raise RuntimeError(f"Failed to load navmesh: {navmesh_path}")
    return pf


def _get_largest_island_index(pathfinder) -> int:
    if pathfinder.num_islands <= 0:
        raise RuntimeError("Loaded navmesh has no islands.")
    return max(
        range(pathfinder.num_islands),
        key=lambda island_ix: pathfinder.island_area(island_index=island_ix),
    )


def _extract_triangle_vertices(pathfinder, island_index: int) -> List[Vec3]:
    points = pathfinder.build_navmesh_vertices(island_index)
    if len(points) == 0:
        raise RuntimeError(
            "No triangle vertices were returned from navmesh. "
            f"island_index={island_index}"
        )
    if len(points) % 3 != 0:
        raise RuntimeError(
            f"Unexpected navmesh triangle vertex count: {len(points)} (not divisible by 3)"
        )
    return [
        (float(p[0]), float(p[1]), float(p[2]))
        for p in points
    ]


def _deduplicate_vertices(
    tri_vertices: Sequence[Vec3],
    precision: int,
) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    vertex_map: Dict[Tuple[float, float, float], int] = {}
    unique_vertices: List[Vec3] = []
    faces: List[Tuple[int, int, int]] = []

    face: List[int] = []
    for v in tri_vertices:
        key = (round(v[0], precision), round(v[1], precision), round(v[2], precision))
        obj_index = vertex_map.get(key)
        if obj_index is None:
            unique_vertices.append(v)
            obj_index = len(unique_vertices)
            vertex_map[key] = obj_index
        face.append(obj_index)
        if len(face) == 3:
            faces.append((face[0], face[1], face[2]))
            face.clear()

    return unique_vertices, faces


def _build_non_dedup_mesh(
    tri_vertices: Sequence[Vec3],
) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    vertices = list(tri_vertices)
    faces: List[Tuple[int, int, int]] = []
    for i in range(0, len(vertices), 3):
        # OBJ index is 1-based.
        faces.append((i + 1, i + 2, i + 3))
    return vertices, faces


def _write_obj(
    output_path: str,
    vertices: Sequence[Vec3],
    faces: Sequence[Tuple[int, int, int]],
    source_navmesh_path: str,
    island_index: int,
) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Exported from Habitat-Sim navmesh\n")
        f.write(f"# source: {source_navmesh_path}\n")
        f.write(f"# island_index: {island_index}\n")
        f.write(f"# vertices: {len(vertices)}\n")
        f.write(f"# faces: {len(faces)}\n")

        for v in vertices:
            f.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        for a, b, c in faces:
            f.write(f"f {a} {b} {c}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Habitat-Sim .navmesh file to an OBJ mesh for visualization "
            "(e.g., in MeshLab)."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input Habitat-Sim .navmesh file path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output OBJ path. Default: same folder/name as input with .obj suffix "
            "(e.g., scene.navmesh -> scene.obj)."
        ),
    )
    parser.add_argument(
        "--island-index",
        type=int,
        default=-1,
        help=(
            "Navmesh island index to export. Use -1 for all islands "
            "(default: -1). Ignored when --largest-island is set."
        ),
    )
    parser.add_argument(
        "--largest-island",
        action="store_true",
        help="Export only the largest navmesh island by area.",
    )
    parser.add_argument(
        "--y-offset",
        type=float,
        default=0.0,
        help="Optional Y offset applied to all exported vertices (default: 0.0).",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable vertex deduplication (write non-indexed triangle vertices).",
    )
    parser.add_argument(
        "--dedup-precision",
        type=int,
        default=6,
        help=(
            "Decimal precision used when deduplicating vertices "
            "(default: 6)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.island_index < -1:
        print("Error: --island-index must be -1 or a non-negative integer.", file=sys.stderr)
        return 1
    if args.dedup_precision < 0:
        print("Error: --dedup-precision must be >= 0.", file=sys.stderr)
        return 1

    input_path = os.path.abspath(args.input)
    output_path = (
        os.path.abspath(args.output)
        if args.output is not None
        else os.path.splitext(input_path)[0] + ".obj"
    )

    try:
        pathfinder = _load_pathfinder(input_path)

        if args.largest_island:
            island_index = _get_largest_island_index(pathfinder)
        else:
            island_index = args.island_index
            if island_index >= pathfinder.num_islands:
                raise ValueError(
                    f"island-index {island_index} out of range. "
                    f"This navmesh has {pathfinder.num_islands} islands."
                )

        tri_vertices = _extract_triangle_vertices(pathfinder, island_index)
        if args.y_offset != 0.0:
            tri_vertices = [
                (v[0], v[1] + args.y_offset, v[2]) for v in tri_vertices
            ]

        if args.no_deduplicate:
            vertices, faces = _build_non_dedup_mesh(tri_vertices)
        else:
            vertices, faces = _deduplicate_vertices(
                tri_vertices, precision=args.dedup_precision
            )

        _write_obj(
            output_path=output_path,
            vertices=vertices,
            faces=faces,
            source_navmesh_path=input_path,
            island_index=island_index,
        )

        print("NavMesh -> OBJ conversion finished.")
        print(f"  input:  {input_path}")
        print(f"  output: {output_path}")
        print(f"  islands in navmesh: {pathfinder.num_islands}")
        print(f"  exported island: {island_index}")
        print(f"  triangles: {len(tri_vertices) // 3}")
        print(f"  vertices in OBJ: {len(vertices)}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
