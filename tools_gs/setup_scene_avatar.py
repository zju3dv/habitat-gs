#!/usr/bin/env python3
"""Create a Habitat-GS scene instance with a walking gaussian avatar."""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT / "data" / "scene_datasets" / "gs_scenes"
TRAIN_ROOT = GS_ROOT / "train"
AVATAR_ROOT = GS_ROOT / "avatars"
GAMMA_ROOT = REPO_ROOT / "gamma" / "GAMMA-release"
GEN_TRAJ = REPO_ROOT / "tools_gs" / "generate_trajectory.py"
ALIGN_AVATAR = REPO_ROOT / "tools_gs" / "align_avatar_ground.py"


def _fail(message: str) -> None:
    raise SystemExit(message)


def _ensure(path: Path, label: str) -> Path:
    if not path.exists():
        _fail(f"{label} not found: {path}")
    return path


def _rel_json_path(from_file: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_file.parent.resolve())


def _scene_paths(scene_id: str) -> dict[str, Path]:
    scene_dir = TRAIN_ROOT / scene_id
    _ensure(scene_dir, "Scene directory")

    gs_ply = scene_dir / f"{scene_id}.gs.ply"
    if not gs_ply.exists():
        candidates = sorted(scene_dir.glob("*.gs.ply"))
        if not candidates:
            _fail(f"No .gs.ply found in {scene_dir}")
        gs_ply = candidates[0]

    navmesh = scene_dir / f"{scene_id}.navmesh"
    if not navmesh.exists():
        candidates = sorted(scene_dir.glob("*.navmesh"))
        if not candidates:
            _fail(f"No .navmesh found in {scene_dir}")
        navmesh = candidates[0]

    return {
        "scene_dir": scene_dir,
        "gs_ply": gs_ply,
        "navmesh": navmesh,
        "stage": scene_dir / "configs" / "stages" / f"{scene_id}_stage.stage_config.json",
        "scene": scene_dir / "configs" / "scenes" / f"{scene_id}.scene_instance.json",
        "dataset": GS_ROOT / f"{scene_id}.scene_dataset_config.json",
    }


def _avatar_paths(avatar_id: str) -> dict[str, Path]:
    avatar_dir = AVATAR_ROOT / avatar_id
    canonical = avatar_dir / "canonical_gs.npz"
    return {
        "avatar_dir": _ensure(avatar_dir, "Avatar directory"),
        "canonical": _ensure(canonical, "Avatar canonical_gs.npz"),
        "smplx": _ensure(AVATAR_ROOT / "smplx", "SMPL-X model directory"),
    }


def _run_gamma(
    *,
    python: str,
    scene_id: str,
    avatar_id: str,
    navmesh: Path,
    output: Path,
    path_length: float,
    length: int,
    gpu_index: int,
    random_seed: int,
    force: bool,
) -> None:
    if output.exists() and not force:
        print(f"[INFO] Reusing existing driver: {output}")
        return

    cmd = [
        python,
        str(GEN_TRAJ),
        "--navmesh",
        str(navmesh),
        "--output",
        str(output),
        "--path-length",
        str(path_length),
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
    print(f"[INFO] Generating GAMMA driver for {scene_id} / {avatar_id}")
    print(" ".join(cmd))
    try:
        subprocess.check_call(cmd, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as exc:
        if gpu_index < 0:
            _fail(
                "GAMMA failed with a negative GPU index. "
                "Retry with --gpu-index 0, or omit --gpu-index to use the default."
            )
        raise exc


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


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


def _write_configs(
    *,
    scene_id: str,
    paths: dict[str, Path],
    avatar_configs: list[dict[str, Any]],
    time_max: float,
) -> None:
    _write_json(paths["stage"], {"render_asset": _rel_json_path(paths["stage"], paths["gs_ply"])})

    scene_config = {
        "stage_instance": {"template_name": f"{scene_id}_stage"},
        "navmesh_instance": f"{scene_id}_navmesh",
        "time_max": time_max,
        "time_loop": True,
        "gaussian_avatars": avatar_configs,
    }
    _write_json(paths["scene"], scene_config)

    dataset_config = {
        "stages": {"paths": {".json": [os.path.relpath(paths["stage"].parent, GS_ROOT)]}},
        "scene_instances": {"paths": {".json": [os.path.relpath(paths["scene"].parent, GS_ROOT)]}},
        "navmesh_instances": {f"{scene_id}_navmesh": os.path.relpath(paths["navmesh"], GS_ROOT)},
    }
    _write_json(paths["dataset"], dataset_config)


def _print_driver_summary(driver: Path) -> None:
    with driver.open("rb") as f:
        data = pickle.load(f)
    joint_mats = data.get("joint_mats")
    frames = joint_mats.shape[0] if joint_mats is not None else "?"
    print(f"[INFO] Driver ready: {driver}")
    print(f"[INFO] frames={frames}, fps={data.get('fps')}, smpl_type={data.get('smpl_type')}")


def _list_avatars() -> None:
    for path in sorted(AVATAR_ROOT.glob("avatar*")):
        if (path / "canonical_gs.npz").exists():
            print(path.name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a walking gaussian avatar setup for data/scene_datasets/gs_scenes/train/<scene>."
    )
    parser.add_argument("scene", nargs="?", help="Scene id under data/scene_datasets/gs_scenes/train, e.g. scene06.")
    parser.add_argument("--avatar", default="avatar1", help="Avatar id, e.g. avatar1/avatar2/avatar3.")
    parser.add_argument(
        "--avatars",
        nargs="+",
        default=None,
        help="Multiple avatar ids to place in the scene.",
    )
    parser.add_argument("--list-avatars", action="store_true", help="List available gaussian avatar appearances.")
    parser.add_argument("--path-length", type=float, default=3.0, help="Requested path length passed to GAMMA.")
    parser.add_argument("--length", type=int, default=70, help="Max GAMMA primitive depth.")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="CUDA device index for GAMMA when CUDA is available. Use 0 by default.",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset-y", type=float, default=0.0)
    parser.add_argument(
        "--no-auto-align",
        action="store_true",
        help="Keep --offset-y instead of automatically aligning skinned gaussian feet to the ground.",
    )
    parser.add_argument(
        "--ground-y",
        default="auto",
        help="Ground height used by auto alignment, or 'auto' to infer from the scene navmesh.",
    )
    parser.add_argument("--foot-clearance", type=float, default=0.015)
    parser.add_argument("--time-max", type=float, default=20.0)
    parser.add_argument("--force", action="store_true", help="Regenerate driver even if it already exists.")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if args.list_avatars:
        _list_avatars()
        return 0
    if not args.scene:
        parser.error("scene is required unless --list-avatars is used")

    paths = _scene_paths(args.scene)
    avatar_ids = args.avatars if args.avatars else [args.avatar]
    _ensure(GAMMA_ROOT, "GAMMA root")
    _ensure(GEN_TRAJ, "Trajectory generator")

    avatar_configs = []
    drivers = []
    for avatar_index, avatar_id in enumerate(avatar_ids):
        avatar = _avatar_paths(avatar_id)
        seed = int(args.random_seed) + avatar_index
        seed_suffix = "" if seed == 0 and len(avatar_ids) == 1 else f"_seed{seed}"
        driver = avatar["avatar_dir"] / f"driver_{args.scene}{seed_suffix}_gamma.pkl"
        _run_gamma(
            python=args.python,
            scene_id=args.scene,
            avatar_id=avatar_id,
            navmesh=paths["navmesh"],
            output=driver,
            path_length=args.path_length,
            length=args.length,
            gpu_index=args.gpu_index,
            random_seed=seed,
            force=args.force,
        )
        avatar_configs.append(
            _make_avatar_config(
                scene_file=paths["scene"],
                avatar_id=avatar_id,
                avatar=avatar,
                driver=driver,
                scale=args.scale,
                offset_y=args.offset_y,
                time_max=args.time_max,
            )
        )
        drivers.append(driver)

    _write_configs(
        scene_id=args.scene,
        paths=paths,
        avatar_configs=avatar_configs,
        time_max=args.time_max,
    )
    if not args.no_auto_align:
        _ensure(ALIGN_AVATAR, "Avatar alignment tool")
        for avatar_index, _avatar_id in enumerate(avatar_ids):
            align_cmd = [
                args.python,
                str(ALIGN_AVATAR),
                str(paths["scene"]),
                "--avatar-index",
                str(avatar_index),
                "--target-y",
                str(args.ground_y),
                "--clearance",
                str(args.foot_clearance),
            ]
            print(f"[INFO] Auto-aligning avatar {avatar_index} feet to ground")
            print(" ".join(align_cmd))
            subprocess.check_call(align_cmd, cwd=str(REPO_ROOT))
    for driver in drivers:
        _print_driver_summary(driver)
    print(f"[INFO] Scene config: {paths['scene']}")
    print(f"[INFO] Dataset config: {paths['dataset']}")
    print(f"[INFO] Preview: python tools_gs/preview_scene.py {args.scene}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
