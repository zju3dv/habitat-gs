#!/usr/bin/env python3
"""Plot avatar and robot paths from a scene/capture as a top-down PNG."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = (
    REPO_ROOT
    / "data"
    / "scene_datasets"
    / "Custom_Assets"
    / "lab528"
    / "configs"
    / "scenes"
    / "lab528.scene_instance.json"
)
DEFAULT_POSES = REPO_ROOT / "outputs" / "lab528_scout_test" / "poses.json"
DEFAULT_OUT = REPO_ROOT / "outputs" / "lab528_scout_test" / "paths.png"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_poses(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        poses = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    poses.append(json.loads(line))
        return poses
    data = _load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("frames"), list):
        return data["frames"]
    raise ValueError(f"Unsupported poses format: {path}")


def _resolve_scene_path(scene_file: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (scene_file.parent / path).resolve()


def _load_avatar_tracks(scene_file: Path) -> list[dict[str, Any]]:
    scene = _load_json(scene_file)
    tracks = []
    for idx, avatar in enumerate(scene.get("gaussian_avatars", [])):
        if not isinstance(avatar, dict) or not avatar.get("driver"):
            continue
        driver_path = _resolve_scene_path(scene_file, str(avatar["driver"]))
        with driver_path.open("rb") as f:
            driver = pickle.load(f)
        transl = np.asarray(driver.get("transl"), dtype=np.float32)
        if transl.ndim != 2 or transl.shape[1] != 3 or transl.shape[0] == 0:
            continue
        offset_y = float(avatar.get("offset_y", 0.0))
        tracks.append(
            {
                "name": str(avatar.get("name") or f"avatar{idx + 1}"),
                "points": transl + np.array([0.0, offset_y, 0.0], dtype=np.float32),
                "driver": driver_path,
            }
        )
    return tracks


def _load_robot_track(poses: list[dict[str, Any]]) -> np.ndarray | None:
    points = []
    for frame in poses:
        if isinstance(frame, dict) and frame.get("robot_position") is not None:
            points.append(frame["robot_position"])
    if not points:
        return None
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr


def _load_target_track(poses: list[dict[str, Any]]) -> tuple[str | None, np.ndarray | None]:
    points = []
    target_id = None
    for frame in poses:
        if not isinstance(frame, dict):
            continue
        if frame.get("target_id") is not None:
            target_id = str(frame.get("target_id"))
        position = frame.get("target_position")
        if position is not None:
            points.append(position)
    if not points:
        return target_id, None
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return target_id, None
    return target_id, arr


def _load_pose_avatar_tracks(poses: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    by_name: dict[str, list[list[float]]] = {}
    for frame in poses:
        if not isinstance(frame, dict):
            continue
        for avatar in frame.get("avatars", []) or []:
            if not isinstance(avatar, dict) or not avatar.get("active", True):
                continue
            position = avatar.get("position")
            if position is None:
                continue
            name = str(avatar.get("name") or f"avatar_{avatar.get('index', len(by_name))}")
            by_name.setdefault(name, []).append(position)
    return {
        name: np.asarray(points, dtype=np.float32)
        for name, points in by_name.items()
        if len(points) >= 2
    }


def _load_navmesh_triangles(navmesh: Path | None) -> np.ndarray | None:
    if navmesh is None or not navmesh.exists():
        return None
    try:
        import habitat_sim
    except Exception:
        return None
    pathfinder = habitat_sim.nav.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh)) or not pathfinder.is_loaded:
        return None
    verts = np.asarray(pathfinder.build_navmesh_vertices(-1), dtype=np.float32)
    if verts.size == 0 or verts.shape[0] % 3 != 0:
        return None
    return verts.reshape(-1, 3, 3)


def _apply_plot_orientation(ax, flip_x: bool, flip_z: bool) -> None:
    if flip_x:
        ax.invert_xaxis()
    if flip_z:
        ax.invert_yaxis()


def _plot_path(ax, points: np.ndarray, label: str, color: str, linestyle: str = "-") -> None:
    if points.shape[0] < 2:
        ax.scatter(points[:, 0], points[:, 2], label=label, color=color, s=28)
        return
    ax.plot(points[:, 0], points[:, 2], linestyle=linestyle, color=color, linewidth=2.0, label=label)
    ax.scatter(points[0, 0], points[0, 2], color=color, s=36, marker="o")
    ax.scatter(points[-1, 0], points[-1, 2], color=color, s=48, marker="x")


def _add_arrows(ax, points: np.ndarray, color: str, max_arrows: int = 8) -> None:
    if points.shape[0] < 2:
        return
    step = max(1, int(math.ceil(points.shape[0] / max_arrows)))
    starts = points[:-1:step]
    deltas = points[1::step] - starts
    if starts.shape[0] == 0:
        return
    ax.quiver(
        starts[:, 0],
        starts[:, 2],
        deltas[:, 0],
        deltas[:, 2],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        width=0.004,
        alpha=0.8,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw top-down paths for Gaussian avatars and the scout robot."
    )
    parser.add_argument("--scene", default=str(DEFAULT_SCENE), help="Scene instance JSON.")
    parser.add_argument("--poses", default=str(DEFAULT_POSES), help="Capture poses.json or poses.jsonl.")
    parser.add_argument("--navmesh", default=None, help="Optional .navmesh to draw as background.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path.")
    parser.add_argument(
        "--prefer-poses-avatars",
        action="store_true",
        help="Plot avatar positions from poses instead of full driver trajectories.",
    )
    parser.add_argument(
        "--episode-from-poses",
        action="store_true",
        help="Prefer robot/target/pedestrian tracks from poses.json as one episode.",
    )
    parser.add_argument("--flip-x", action="store_true", help="Mirror the map horizontally.")
    parser.add_argument("--flip-z", action="store_true", help="Mirror the map vertically.")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene_file = Path(args.scene).expanduser().resolve()
    poses_file = Path(args.poses).expanduser().resolve()
    out_file = Path(args.out).expanduser().resolve()
    navmesh = Path(args.navmesh).expanduser().resolve() if args.navmesh else None

    poses = _load_poses(poses_file)
    robot_points = _load_robot_track(poses)
    target_id, target_points = _load_target_track(poses)
    pose_avatar_tracks = _load_pose_avatar_tracks(poses)
    scene_avatar_tracks = _load_avatar_tracks(scene_file)
    navmesh_tris = _load_navmesh_triangles(navmesh)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
    if navmesh_tris is not None:
        for tri in navmesh_tris:
            poly = plt.Polygon(
                tri[:, [0, 2]],
                closed=True,
                facecolor="#d7d7d7",
                edgecolor="#b8b8b8",
                linewidth=0.25,
                alpha=0.5,
            )
            ax.add_patch(poly)

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
    ]
    avatar_source = "poses" if (args.prefer_poses_avatars or args.episode_from_poses) and pose_avatar_tracks else "drivers"
    if args.episode_from_poses and target_points is not None:
        _plot_path(ax, target_points, f"target: {target_id or 'target'}", "#1f77b4")
        _add_arrows(ax, target_points, "#1f77b4")
        ped_idx = 0
        for name, points in sorted(pose_avatar_tracks.items()):
            if target_id is not None and name == target_id:
                continue
            color = colors[(ped_idx + 1) % len(colors)]
            _plot_path(ax, points, f"pedestrian: {name}", color)
            _add_arrows(ax, points, color)
            ped_idx += 1
    elif avatar_source == "poses":
        for idx, (name, points) in enumerate(sorted(pose_avatar_tracks.items())):
            color = colors[idx % len(colors)]
            _plot_path(ax, points, name, color)
            _add_arrows(ax, points, color)
    else:
        for idx, track in enumerate(scene_avatar_tracks):
            color = colors[idx % len(colors)]
            _plot_path(ax, track["points"], track["name"], color)
            _add_arrows(ax, track["points"], color)

    if robot_points is not None:
        _plot_path(ax, robot_points, "scout_robot", "#111111", linestyle="--")
        _add_arrows(ax, robot_points, "#111111")

    ax.set_title("Top-down paths (X-Z)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#eeeeee", linewidth=0.8)
    ax.legend(loc="best")
    _apply_plot_orientation(ax, flip_x=bool(args.flip_x), flip_z=bool(args.flip_z))
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    print(f"[INFO] scene: {scene_file}")
    print(f"[INFO] poses: {poses_file if poses_file.exists() else 'not found'}")
    print(f"[INFO] avatar_source: {avatar_source}")
    print(f"[INFO] target: {target_id if target_id is not None else 'none'}")
    print(f"[INFO] avatars: {len(pose_avatar_tracks) if avatar_source == 'poses' else len(scene_avatar_tracks)}")
    print(f"[INFO] robot_points: {0 if robot_points is None else robot_points.shape[0]}")
    print(f"[INFO] output: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
