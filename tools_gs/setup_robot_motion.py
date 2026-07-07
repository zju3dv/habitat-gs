#!/usr/bin/env python3
"""Create a simple NavMesh-following trajectory for a rigid object robot."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT / "data" / "scene_datasets" / "gs_scenes"
TRAIN_ROOT = GS_ROOT / "train"


def _fail(message: str) -> None:
    raise SystemExit(message)


def _scene_paths(scene_id: str) -> dict[str, Path]:
    scene_dir = TRAIN_ROOT / scene_id
    if not scene_dir.exists():
        _fail(f"Scene directory not found: {scene_dir}")
    scene_file = scene_dir / "configs" / "scenes" / f"{scene_id}.scene_instance.json"
    navmesh = scene_dir / f"{scene_id}.navmesh"
    if not scene_file.exists():
        _fail(f"Scene instance not found: {scene_file}")
    if not navmesh.exists():
        matches = sorted(scene_dir.glob("*.navmesh"))
        if not matches:
            _fail(f"NavMesh not found under {scene_dir}")
        navmesh = matches[0]
    return {"scene": scene_file, "navmesh": navmesh}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _find_object_instance(scene_cfg: dict[str, Any], object_name: str) -> dict[str, Any]:
    objects = scene_cfg.setdefault("object_instances", [])
    for obj in objects:
        name = str(obj.get("name", obj.get("template_name", "")))
        template = str(obj.get("template_name", ""))
        if object_name in (name, template) or object_name in name or object_name in template:
            return obj
    _fail(f"Object instance matching {object_name!r} not found in scene.")


def _load_pathfinder(navmesh: Path):
    try:
        from habitat_sim.nav import PathFinder
    except Exception as exc:
        _fail(f"Failed to import habitat_sim.nav.PathFinder: {exc}")
    pathfinder = PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh)):
        _fail(f"Failed to load navmesh: {navmesh}")
    return pathfinder


def _shortest_path(pathfinder, start: np.ndarray, end: np.ndarray) -> Optional[np.ndarray]:
    from habitat_sim.nav import ShortestPath

    path = ShortestPath()
    path.requested_start = start
    path.requested_end = end
    if not pathfinder.find_path(path):
        return None
    points = np.asarray(path.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        return None
    return points


def _path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _dense_path(points: np.ndarray, spacing: float) -> np.ndarray:
    dense = [points[0]]
    spacing = max(float(spacing), 1.0e-3)
    for p0, p1 in zip(points[:-1], points[1:]):
        segment = p1 - p0
        length = float(np.linalg.norm(segment))
        steps = max(1, int(math.ceil(length / spacing)))
        for step in range(1, steps + 1):
            dense.append((p0 + segment * (step / steps)).astype(np.float32))
    return np.asarray(dense, dtype=np.float32)


def _sample_navmesh_path(
    pathfinder,
    start: np.ndarray,
    length: float,
    random_seed: int,
    attempts: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    start = np.asarray(pathfinder.snap_point(start), dtype=np.float32)
    best = None
    best_score = float("inf")
    longest = None
    longest_len = -1.0
    for _ in range(max(1, attempts)):
        end = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        points = _shortest_path(pathfinder, start, end)
        if points is None:
            continue
        dist = _path_length(points)
        if dist > longest_len:
            longest = points
            longest_len = dist
        score = abs(dist - length)
        if score < best_score:
            best = points
            best_score = score
        # Lightly perturb the RNG state used by Habitat's internal sampler timing.
        _ = rng.random()
    if best is None:
        best = longest
    if best is None:
        _fail("Failed to sample any valid robot path.")
    return best


def _yaw_from_delta(delta: np.ndarray, fallback: float) -> float:
    dx = float(delta[0])
    dz = float(delta[2])
    if abs(dx) + abs(dz) < 1.0e-6:
        return fallback
    # Habitat object front is configured as -Z; rotate it toward the horizontal path tangent.
    return math.atan2(-dx, -dz)


def _make_keyframes(points: np.ndarray, speed: float) -> list[dict[str, Any]]:
    speed = max(float(speed), 1.0e-3)
    keyframes = []
    elapsed = 0.0
    yaw = 0.0
    for idx, point in enumerate(points):
        if idx < points.shape[0] - 1:
            yaw = _yaw_from_delta(points[idx + 1] - point, yaw)
        elif idx > 0:
            yaw = _yaw_from_delta(point - points[idx - 1], yaw)
        if idx > 0:
            elapsed += float(np.linalg.norm(points[idx] - points[idx - 1])) / speed
        keyframes.append(
            {
                "time": round(elapsed, 4),
                "translation": [round(float(v), 5) for v in point],
                "yaw": round(float(yaw), 6),
            }
        )
    return keyframes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write object_trajectories into a scene instance for a robot OBJ."
    )
    parser.add_argument("scene", help="Scene id, e.g. scene08.")
    parser.add_argument("--object", default="scout", help="Object/template substring to animate.")
    parser.add_argument("--path-length", type=float, default=6.0)
    parser.add_argument("--speed", type=float, default=0.8, help="Meters per second.")
    parser.add_argument("--spacing", type=float, default=0.25, help="Waypoint spacing in meters.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attempts", type=int, default=128)
    parser.add_argument("--loop", action="store_true", default=True)
    args = parser.parse_args()

    paths = _scene_paths(args.scene)
    scene_cfg = _load_json(paths["scene"])
    obj = _find_object_instance(scene_cfg, args.object)
    obj["motion_type"] = "kinematic"

    start = np.asarray(obj.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
    pathfinder = _load_pathfinder(paths["navmesh"])
    coarse = _sample_navmesh_path(
        pathfinder,
        start,
        length=args.path_length,
        random_seed=args.seed,
        attempts=args.attempts,
    )
    dense = _dense_path(coarse, args.spacing)
    keyframes = _make_keyframes(dense, args.speed)
    obj["translation"] = keyframes[0]["translation"]

    trajectories = scene_cfg.setdefault("object_trajectories", [])
    trajectories[:] = [
        traj for traj in trajectories if str(traj.get("object", "")) != args.object
    ]
    trajectories.append(
        {
            "object": args.object,
            "loop": bool(args.loop),
            "keyframes": keyframes,
        }
    )

    avatar_time_max = max(
        (
            float(avatar.get("time_end", 0.0))
            for avatar in scene_cfg.get("gaussian_avatars", [])
            if isinstance(avatar, dict)
        ),
        default=0.0,
    )
    time_max = max(float(keyframes[-1]["time"]), avatar_time_max)
    scene_cfg["time_max"] = round(time_max, 4)
    scene_cfg["time_loop"] = True
    _write_json(paths["scene"], scene_cfg)

    print(f"[INFO] Scene: {paths['scene']}")
    print(f"[INFO] Object: {args.object}")
    print(f"[INFO] Path points: coarse={len(coarse)}, dense={len(dense)}")
    print(f"[INFO] Path length: {_path_length(dense):.3f}m")
    print(f"[INFO] Duration: {keyframes[-1]['time']:.3f}s at {args.speed:.3f}m/s")
    print(f"[INFO] Preview: python tools_gs/preview_scene.py {args.scene} --physics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
