#!/usr/bin/env python3
"""Configure lab528 with the minimal RPF target/oncoming pedestrian routes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_FILE = (
    REPO_ROOT
    / "data"
    / "scene_datasets"
    / "Custom_Assets"
    / "lab528"
    / "configs"
    / "scenes"
    / "lab528.scene_instance.json"
)
DATASET = (
    REPO_ROOT
    / "data"
    / "scene_datasets"
    / "Custom_Assets"
    / "lab528"
    / "lab528.scene_dataset_config.json"
)
NAVMESH = REPO_ROOT / "data" / "scene_datasets" / "Custom_Assets" / "lab528_1m_r.navmesh"
SETUP_ROUTES = REPO_ROOT / "tools_gs" / "setup_avatar_routes.py"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _fmt_point(x: float, y: float, z: float) -> str:
    return f"{x:.4f},{y:.4f},{z:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set up lab528 toy RPF routes and scout start pose."
    )
    parser.add_argument("--center-x", type=float, default=1.0, help="Target route X center line.")
    parser.add_argument(
        "--z-start",
        type=float,
        default=-2.2,
        help="Target route start Z before optional --center-z-offset.",
    )
    parser.add_argument(
        "--z-end",
        type=float,
        default=5.0,
        help="Target route end Z before optional --center-z-offset.",
    )
    parser.add_argument(
        "--center-z-offset",
        type=float,
        default=0.0,
        help="Shift both pedestrian routes and scout start along Z.",
    )
    parser.add_argument(
        "--oncoming-side-offset",
        type=float,
        default=-1.0,
        help="Oncoming pedestrian X offset relative to target. Negative is target-left for +Z target motion.",
    )
    parser.add_argument(
        "--robot-lateral-offset",
        type=float,
        default=-0.8,
        help="Scout initial X offset relative to target. Negative is target-left for +Z target motion.",
    )
    parser.add_argument("--robot-behind-distance", type=float, default=2.0)
    parser.add_argument("--ground-y", type=float, default=0.097)
    parser.add_argument("--length", type=int, default=220)
    parser.add_argument("--time-max", type=float, default=24.0)
    parser.add_argument("--label", default="toy_z_follow")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=52)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    z_start = args.z_start + args.center_z_offset
    z_end = args.z_end + args.center_z_offset
    target_x = args.center_x
    oncoming_x = target_x + args.oncoming_side_offset
    robot_x = target_x + args.robot_lateral_offset
    robot_z = z_start - args.robot_behind_distance

    target_route = (
        "avatar1:"
        f"{_fmt_point(target_x, args.ground_y, z_start)}:"
        f"{_fmt_point(target_x, args.ground_y, z_end)}"
    )
    oncoming_route = (
        "avatar2:"
        f"{_fmt_point(oncoming_x, args.ground_y, z_end)}:"
        f"{_fmt_point(oncoming_x, args.ground_y, z_start)}"
    )

    cmd = [
        args.python,
        str(SETUP_ROUTES),
        str(SCENE_FILE),
        "--navmesh",
        str(NAVMESH),
        "--label",
        args.label,
        "--length",
        str(args.length),
        "--gpu-index",
        str(args.gpu_index),
        "--random-seed",
        str(args.random_seed),
        "--time-max",
        str(args.time_max),
        "--route",
        target_route,
        "--route",
        oncoming_route,
    ]
    if args.force:
        cmd.append("--force")

    print("[INFO] target route:", target_route)
    print("[INFO] oncoming route:", oncoming_route)
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))

    scene = _load_json(SCENE_FILE)
    objects = scene.setdefault("object_instances", [])
    if not objects:
        objects.append({"template_name": "scout"})
    scout = objects[0]
    for obj in objects:
        if str(obj.get("template_name", "")).lower() == "scout":
            scout = obj
            break
    scout["template_name"] = "scout"
    scout["translation"] = [round(robot_x, 4), round(args.ground_y, 4), round(robot_z, 4)]
    scout["translation_origin"] = "asset_local"
    scout["motion_type"] = "kinematic"
    _write_json(SCENE_FILE, scene)

    print(f"[INFO] scout start: {scout['translation']}")
    print("[INFO] Preview/capture:")
    print(
        "  python tools_gs/preview_scene.py "
        f"{DATASET.relative_to(REPO_ROOT)} --capture --rpf-expert "
        "--rpf-target avatar1_actor --capture-out outputs/lab528_rpf_expert"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
