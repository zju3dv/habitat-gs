#!/usr/bin/env python3
"""Generate gaussian avatar drivers from user-specified NavMesh routes."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
AVATAR_ROOT = REPO_ROOT / "data" / "scene_datasets" / "gs_scenes" / "avatars"
GAMMA_ROOT = REPO_ROOT / "gamma" / "GAMMA-release"
GEN_TRAJ = REPO_ROOT / "tools_gs" / "generate_trajectory.py"
ALIGN_AVATAR = REPO_ROOT / "tools_gs" / "align_avatar_ground.py"
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


def _fail(message: str) -> None:
    raise SystemExit(message)


def _ensure(path: Path, label: str) -> Path:
    if not path.exists():
        _fail(f"{label} not found: {path}")
    return path


def _rel_json_path(from_file: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_file.parent.resolve())


def _parse_xyz(text: str) -> list[float]:
    parts = text.split(",")
    if len(parts) != 3:
        _fail(f"Expected xyz as x,y,z, got: {text}")
    try:
        return [float(v) for v in parts]
    except ValueError:
        _fail(f"Invalid xyz value: {text}")


def _parse_route(text: str) -> dict[str, Any]:
    parts = text.split(":")
    if len(parts) < 3:
        _fail(
            "Route format is avatar:start:end[:via...], "
            f"where xyz is x,y,z. Got: {text}"
        )
    avatar_id = parts[0].strip()
    if not avatar_id:
        _fail(f"Route is missing avatar id: {text}")
    return {
        "avatar": avatar_id,
        "start": _parse_xyz(parts[1]),
        "end": _parse_xyz(parts[2]),
        "via": [_parse_xyz(v) for v in parts[3:] if v.strip()],
    }


def _load_routes(args: argparse.Namespace) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for route_text in args.route or []:
        routes.append(_parse_route(route_text))
    if args.routes_json:
        with Path(args.routes_json).expanduser().open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            _fail("--routes-json must contain a list of route objects.")
        for item in data:
            if not isinstance(item, dict):
                _fail("Each route JSON item must be an object.")
            routes.append(
                {
                    "avatar": str(item["avatar"]),
                    "start": [float(v) for v in item["start"]],
                    "end": [float(v) for v in item["end"]],
                    "via": [[float(v) for v in p] for p in item.get("via", [])],
                }
            )
    if not routes:
        _fail("No routes specified. Use --route or --routes-json.")
    return routes


def _avatar_paths(avatar_id: str) -> dict[str, Path]:
    avatar_dir = _ensure(AVATAR_ROOT / avatar_id, "Avatar directory")
    return {
        "avatar_dir": avatar_dir,
        "canonical": _ensure(avatar_dir / "canonical_gs.npz", "Avatar canonical_gs.npz"),
        "smplx": _ensure(AVATAR_ROOT / "smplx", "SMPL-X model directory"),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _infer_navmesh(scene_file: Path, explicit: str | None) -> Path:
    if explicit:
        return _ensure(Path(explicit).expanduser().resolve(), "NavMesh")

    scene_id = scene_file.name.replace(".scene_instance.json", "")
    custom_root = REPO_ROOT / "data" / "scene_datasets" / "Custom_Assets"
    candidates = [
        scene_file.parents[2] / f"{scene_id}.navmesh",
        scene_file.parents[3] / f"{scene_id}.navmesh" if len(scene_file.parents) > 3 else None,
        custom_root / f"{scene_id}.navmesh",
        custom_root / f"{scene_id}_1m_r.navmesh",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    _fail("Could not infer navmesh. Pass --navmesh explicitly.")


def _infer_ground_y(navmesh: Path, routes: list[dict[str, Any]], explicit: str) -> float:
    if explicit.lower() != "auto":
        return float(explicit)
    try:
        from habitat_sim.nav import PathFinder

        pathfinder = PathFinder()
        if not pathfinder.load_nav_mesh(str(navmesh)):
            raise RuntimeError("failed to load navmesh")
        ys = []
        for route in routes:
            for point in [route["start"], *route.get("via", []), route["end"]]:
                snapped = pathfinder.snap_point(point)
                if np.all(np.isfinite(snapped)):
                    ys.append(float(snapped[1]))
        if ys:
            return float(np.median(ys))
    except Exception:
        pass
    ys = []
    for route in routes:
        for point in [route["start"], *route.get("via", []), route["end"]]:
            ys.append(float(point[1]))
    return float(np.median(ys))


def _route_args(route: dict[str, Any]) -> list[str]:
    cmd = ["--start", *[str(v) for v in route["start"]]]
    for via in route.get("via", []):
        cmd.extend(["--via", *[str(v) for v in via]])
    cmd.extend(["--end", *[str(v) for v in route["end"]]])
    return cmd


def _make_avatar_config(
    *,
    scene_file: Path,
    avatar_id: str,
    avatar: dict[str, Path],
    driver: Path,
    scale: float,
    offset_y: float,
    time_max: float,
) -> dict[str, Any]:
    return {
        "name": f"{avatar_id}_actor",
        "canonical_gaussians": _rel_json_path(scene_file, avatar["canonical"]),
        "driver": _rel_json_path(scene_file, driver),
        "smpl_model_path": _rel_json_path(scene_file, avatar["smplx"]),
        "smpl_type": "smplx",
        "scale": scale,
        "offset_y": offset_y,
        "time_begin": 0.0,
        "time_end": time_max,
    }


def _generate_driver(
    *,
    python: str,
    navmesh: Path,
    route: dict[str, Any],
    driver: Path,
    length: int,
    gpu_index: int,
    random_seed: int,
    force: bool,
) -> None:
    if driver.exists() and not force:
        print(f"[INFO] Reusing existing driver: {driver}")
        return
    cmd = [
        python,
        str(GEN_TRAJ),
        "--navmesh",
        str(navmesh),
        "--output",
        str(driver),
        *_route_args(route),
        "--smpl-model-path",
        str(AVATAR_ROOT / "smplx"),
        "--gamma-root",
        str(GAMMA_ROOT),
        "--body-model-path",
        str(AVATAR_ROOT / "smplx"),
        "--marker-path",
        str(GAMMA_ROOT / "exp_GAMMAPrimitive" / "data" / "Mosh"),
        "--length",
        str(length),
        "--include-proxy",
        "false",
        "--gpu-index",
        str(gpu_index),
        "--random-seed",
        str(random_seed),
    ]
    print(f"[INFO] Generating route driver for {route['avatar']}")
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate gaussian avatar drivers from explicit start/end/via routes."
    )
    parser.add_argument(
        "scene",
        nargs="?",
        default=str(DEFAULT_SCENE),
        help="Path to .scene_instance.json. Defaults to lab528.",
    )
    parser.add_argument("--navmesh", default=None, help="Path to .navmesh.")
    parser.add_argument(
        "--route",
        action="append",
        help="Route as avatar:start:end[:via...], xyz as x,y,z. Repeat for multiple avatars.",
    )
    parser.add_argument("--routes-json", default=None, help="JSON list of route objects.")
    parser.add_argument("--label", default="manual", help="Driver filename label.")
    parser.add_argument("--length", type=int, default=70)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--ground-y", default="auto")
    parser.add_argument("--foot-clearance", type=float, default=0.015)
    parser.add_argument("--time-max", type=float, default=20.0)
    parser.add_argument("--append", action="store_true", help="Append instead of replacing gaussian_avatars.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    scene_file = _ensure(Path(args.scene).expanduser().resolve(), "Scene instance")
    navmesh = _infer_navmesh(scene_file, args.navmesh)
    routes = _load_routes(args)
    ground_y = _infer_ground_y(navmesh, routes, str(args.ground_y))

    scene_config = _load_json(scene_file)
    avatars = list(scene_config.get("gaussian_avatars", [])) if args.append else []

    generated_indices = []
    scene_id = scene_file.name.replace(".scene_instance.json", "")
    for route_index, route in enumerate(routes):
        avatar_id = route["avatar"]
        avatar = _avatar_paths(avatar_id)
        seed = int(args.random_seed) + route_index
        driver = avatar["avatar_dir"] / f"driver_{scene_id}_{args.label}{route_index}_gamma.pkl"
        _generate_driver(
            python=args.python,
            navmesh=navmesh,
            route=route,
            driver=driver,
            length=args.length,
            gpu_index=args.gpu_index,
            random_seed=seed,
            force=args.force,
        )
        generated_indices.append(len(avatars))
        avatars.append(
            _make_avatar_config(
                scene_file=scene_file,
                avatar_id=avatar_id,
                avatar=avatar,
                driver=driver,
                scale=args.scale,
                offset_y=0.0,
                time_max=args.time_max,
            )
        )

    scene_config["time_max"] = args.time_max
    scene_config["time_loop"] = True
    scene_config["gaussian_avatars"] = avatars
    _write_json(scene_file, scene_config)

    for avatar_index in generated_indices:
        cmd = [
            args.python,
            str(ALIGN_AVATAR),
            str(scene_file),
            "--avatar-index",
            str(avatar_index),
            "--target-y",
            str(ground_y),
            "--clearance",
            str(args.foot_clearance),
        ]
        print(f"[INFO] Auto-aligning avatar {avatar_index} to ground_y={ground_y:.6f}")
        print(" ".join(cmd))
        subprocess.check_call(cmd, cwd=str(REPO_ROOT))

    print(f"[INFO] Scene config updated: {scene_file}")
    print("[INFO] Preview:")
    print(
        "  python tools_gs/preview_scene.py "
        "data/scene_datasets/Custom_Assets/lab528/lab528.scene_dataset_config.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
