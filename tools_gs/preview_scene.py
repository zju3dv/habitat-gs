#!/usr/bin/env python3
"""Small launcher for quickly previewing Habitat-GS gaussian scenes."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT / "data" / "scene_datasets" / "gs_scenes"
DEFAULT_SCENE = "scene08"


def _existing(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"Path not found: {path}")
    return path


def _iter_dataset_configs() -> Iterable[Path]:
    if not GS_ROOT.exists():
        return []
    return sorted(GS_ROOT.glob("*scene_dataset_config.json"))


def _scene_ids_from_dataset(dataset: Path) -> list[str]:
    try:
        with dataset.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    scene_ids: set[str] = set()
    scene_instances = data.get("scene_instances") or {}
    paths = scene_instances.get("paths") if isinstance(scene_instances, dict) else None
    if isinstance(paths, dict):
        for rel_paths in paths.values():
            for rel in rel_paths:
                scene_dir = (dataset.parent / rel).resolve()
                for scene_file in scene_dir.glob("*.scene_instance.json"):
                    scene_ids.add(scene_file.name.replace(".scene_instance.json", ""))

    navmesh_instances = data.get("navmesh_instances") or {}
    if isinstance(navmesh_instances, dict):
        scene_ids.update(str(key) for key in navmesh_instances.keys() if not str(key).endswith("_navmesh"))

    return sorted(scene_ids)


def _print_available_scenes() -> None:
    configs = list(_iter_dataset_configs())
    if not configs:
        print(f"No scene_dataset_config.json found under {GS_ROOT}")
        return
    for cfg in configs:
        print(os.path.relpath(cfg, REPO_ROOT))
        scenes = _scene_ids_from_dataset(cfg)
        for scene in scenes[:80]:
            print(f"  {scene}")
        if len(scenes) > 80:
            print(f"  ... {len(scenes) - 80} more")


def _find_dataset_for_scene(scene: str) -> Optional[Path]:
    direct = GS_ROOT / f"{scene}.scene_dataset_config.json"
    if direct.exists():
        return direct

    for cfg in _iter_dataset_configs():
        if scene in _scene_ids_from_dataset(cfg):
            return cfg
    return None


def _resolve_dataset_and_scene(target: str, dataset: Optional[str]) -> Tuple[Path, str]:
    target_path = Path(target).expanduser()
    if target_path.exists():
        target_path = target_path.resolve()
        if target_path.name.endswith(".scene_instance.json"):
            scene = target_path.name.replace(".scene_instance.json", "")
            dataset_path = Path(dataset).expanduser().resolve() if dataset else _find_dataset_for_scene(scene)
            if dataset_path is None:
                raise SystemExit(f"Could not find dataset config for scene {scene!r}. Pass --dataset.")
            return _existing(dataset_path), scene
        if target_path.name.endswith(".scene_dataset_config.json"):
            scenes = _scene_ids_from_dataset(target_path)
            if not scenes:
                raise SystemExit(f"No scenes found in dataset config: {target_path}")
            return target_path, scenes[0]
        if target_path.is_dir():
            scene = target_path.name
            dataset_path = Path(dataset).expanduser().resolve() if dataset else _find_dataset_for_scene(scene)
            if dataset_path is None:
                raise SystemExit(f"Could not find dataset config for directory {target_path}. Pass --dataset.")
            return _existing(dataset_path), scene

    scene = target
    dataset_path = Path(dataset).expanduser().resolve() if dataset else _find_dataset_for_scene(scene)
    if dataset_path is None:
        raise SystemExit(f"Could not find dataset config for scene {scene!r}. Try --list.")
    return _existing(dataset_path), scene


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quickly launch examples/gaussian_viewer.py for a GS scene."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=DEFAULT_SCENE,
        help="Scene id, scene directory, scene_instance.json, scene_dataset_config.json, or .gs.ply.",
    )
    parser.add_argument("--dataset", help="Override scene_dataset_config.json.")
    parser.add_argument("--list", action="store_true", help="List discovered dataset configs and scenes.")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--time-rate", type=float, default=1.0)
    parser.add_argument("--no-playback", action="store_true", help="Open paused instead of playing animation.")
    parser.add_argument("--physics", action="store_true", help="Enable physics/NavMesh in the viewer.")
    parser.add_argument("--capture", action="store_true", help="Capture scout front-camera frames and exit.")
    parser.add_argument("--capture-out", default="outputs/scout_capture", help="Capture output directory.")
    parser.add_argument(
        "--capture-route",
        default=None,
        help="Scout route as x,y,z:x,y,z[:x,y,z...].",
    )
    parser.add_argument("--capture-fps", type=float, default=10.0)
    parser.add_argument("--capture-speed", type=float, default=0.6)
    parser.add_argument("--capture-max-frames", type=int, default=None)
    parser.add_argument("--capture-depth", action="store_true", help="Also save depth .npy frames.")
    parser.add_argument(
        "--rpf-expert",
        action="store_true",
        help="In capture mode, control scout online with the RPF expert.",
    )
    parser.add_argument("--rpf-target", default="avatar1_actor", help="Target avatar name or index.")
    parser.add_argument("--rpf-preferred-side", type=int, default=1, choices=[-1, 1])
    parser.add_argument(
        "--viewer",
        default=str(REPO_ROOT / "examples" / "gaussian_viewer.py"),
        help="Path to gaussian_viewer.py.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch the viewer.",
    )
    args = parser.parse_args()

    if args.list:
        _print_available_scenes()
        return 0

    viewer = _existing(Path(args.viewer).expanduser().resolve())
    target_path = Path(args.target).expanduser()

    cmd = [args.python, str(viewer), "--width", str(args.width), "--height", str(args.height)]
    if target_path.exists() and target_path.suffix == ".ply":
        cmd.extend(["--input", str(target_path.resolve())])
    else:
        dataset, scene = _resolve_dataset_and_scene(args.target, args.dataset)
        cmd.extend(["--dataset", str(dataset), "--scene", scene])

    if args.time is not None:
        cmd.extend(["--time", str(args.time)])
    if not args.no_playback:
        cmd.append("--playback")
    cmd.extend(["--time-rate", str(args.time_rate)])
    if args.physics:
        cmd.append("--enable-physics")
    if args.capture:
        cmd.append("--capture")
        cmd.extend(["--capture-out", args.capture_out])
        cmd.extend(["--capture-fps", str(args.capture_fps)])
        cmd.extend(["--capture-speed", str(args.capture_speed)])
        if args.capture_route:
            cmd.extend(["--capture-route", args.capture_route])
        if args.capture_max_frames is not None:
            cmd.extend(["--capture-max-frames", str(args.capture_max_frames)])
        if args.capture_depth:
            cmd.append("--capture-depth")
        if args.rpf_expert:
            cmd.append("--rpf-expert")
            cmd.extend(["--rpf-target", args.rpf_target])
            cmd.extend(["--rpf-preferred-side", str(args.rpf_preferred_side)])

    print("Launching:")
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
