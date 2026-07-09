#!/usr/bin/env python3
"""Automatically set gaussian avatar offset_y so the animated feet touch ground."""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT / "data" / "scene_datasets" / "gs_scenes"


def _fail(message: str) -> None:
    raise SystemExit(message)


def _resolve_scene_instance(target: str) -> Path:
    path = Path(target).expanduser()
    if path.exists():
        if path.is_dir():
            scene_id = path.name
            candidate = path / "configs" / "scenes" / f"{scene_id}.scene_instance.json"
            if candidate.exists():
                return candidate.resolve()
        if path.name.endswith(".scene_instance.json"):
            return path.resolve()
        _fail(f"Unsupported target path: {path}")

    scene_id = target
    candidate = GS_ROOT / "train" / scene_id / "configs" / "scenes" / f"{scene_id}.scene_instance.json"
    if candidate.exists():
        return candidate.resolve()
    _fail(f"Scene instance not found for {target!r}: {candidate}")


def _resolve_from_scene(scene_file: Path, rel_path: str) -> Path:
    path = Path(rel_path).expanduser()
    if path.is_absolute():
        return path
    return (scene_file.parent / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _sample_frame_indices(frame_count: int, sample_frames: int) -> np.ndarray:
    if frame_count <= 0:
        _fail("Driver joint_mats has no frames.")
    if sample_frames <= 0 or sample_frames >= frame_count:
        return np.arange(frame_count, dtype=np.int64)
    return np.unique(np.linspace(0, frame_count - 1, sample_frames, dtype=np.int64))


def _compute_min_skinned_y(
    *,
    canonical_path: Path,
    driver_path: Path,
    scale: float,
    sample_frames: int,
    candidate_points: int,
    bottom_percent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(canonical_path) as data:
        means = np.asarray(data["means"], dtype=np.float32)
        weights = np.asarray(data["lbs_weights"], dtype=np.float32)
        inv_bind = np.asarray(data["joints_inv_bind_matrix"], dtype=np.float32)

    with driver_path.open("rb") as f:
        driver = pickle.load(f)
    if "joint_mats" not in driver:
        _fail(f"Driver missing joint_mats: {driver_path}")
    joint_mats = np.asarray(driver["joint_mats"], dtype=np.float32)
    if joint_mats.ndim != 4 or joint_mats.shape[2:] != (4, 4):
        _fail(f"Unexpected joint_mats shape in {driver_path}: {joint_mats.shape}")

    joint_count = joint_mats.shape[1]
    if weights.shape[1] != joint_count or inv_bind.shape[0] != joint_count:
        _fail(
            "Joint count mismatch: "
            f"weights={weights.shape}, inv_bind={inv_bind.shape}, joint_mats={joint_mats.shape}"
        )

    n_points = means.shape[0]
    bottom_count = n_points
    if bottom_percent > 0.0:
        bottom_count = min(bottom_count, max(1, int(round(n_points * bottom_percent / 100.0))))
    if candidate_points > 0:
        bottom_count = min(bottom_count, candidate_points)

    candidate_idx = np.argpartition(means[:, 1], bottom_count - 1)[:bottom_count]
    points = np.concatenate(
        [means[candidate_idx], np.ones((bottom_count, 1), dtype=np.float32)], axis=1
    )
    candidate_weights = weights[candidate_idx]

    frame_indices = _sample_frame_indices(joint_mats.shape[0], sample_frames)
    min_ys = []
    for frame_idx in frame_indices:
        mats = joint_mats[frame_idx].copy()
        if scale != 1.0:
            root_t = mats[0, :3, 3].copy()
            mats[:, :3, :3] *= scale
            mats[:, :3, 3] = mats[:, :3, 3] * scale + (1.0 - scale) * root_t

        # Match avatar_lbs.cu: y = sum_j w_j * row_y(M_j) * inv_bind_j * [p, 1].
        y_rows = np.einsum("jk,jkl->jl", mats[:, 1, :], inv_bind)
        ys = (points @ y_rows.T * candidate_weights).sum(axis=1)
        min_ys.append(float(np.min(ys)))

    min_ys = np.asarray(min_ys, dtype=np.float32)
    if not np.all(np.isfinite(min_ys)):
        _fail("Failed to compute finite skinned y minimum.")
    transl = np.asarray(driver.get("transl", joint_mats[:, 0, :3, 3]), dtype=np.float32)
    return frame_indices, min_ys, transl


def _infer_scene_navmesh(scene_file: Path) -> Optional[Path]:
    scene_id = scene_file.name.replace(".scene_instance.json", "")
    scene_dir = scene_file.parents[2] if len(scene_file.parents) >= 3 else scene_file.parent
    candidate = scene_dir / f"{scene_id}.navmesh"
    if candidate.exists():
        return candidate
    matches = sorted(scene_dir.glob("*.navmesh"))
    return matches[0] if matches else None


def _ground_y_for_frames(
    scene_file: Path,
    transl: np.ndarray,
    frame_indices: np.ndarray,
    target_y: Optional[float],
) -> np.ndarray:
    if target_y is not None:
        return np.full(frame_indices.shape, float(target_y), dtype=np.float32)

    navmesh = _infer_scene_navmesh(scene_file)
    if navmesh is None:
        return transl[frame_indices, 1].astype(np.float32)

    try:
        from habitat_sim.nav import PathFinder

        pathfinder = PathFinder()
        if not pathfinder.load_nav_mesh(str(navmesh)):
            return transl[frame_indices, 1].astype(np.float32)
        ground = []
        for idx in frame_indices:
            point = transl[int(idx)]
            snapped = pathfinder.snap_point(point)
            if np.all(np.isfinite(snapped)):
                ground.append(float(snapped[1]))
            else:
                ground.append(float(point[1]))
        return np.asarray(ground, dtype=np.float32)
    except Exception:
        return transl[frame_indices, 1].astype(np.float32)


def align_scene_avatar(
    scene_file: Path,
    *,
    avatar_index: int,
    target_y: Optional[float],
    clearance: float,
    sample_frames: int,
    candidate_points: int,
    bottom_percent: float,
    dry_run: bool,
) -> float:
    config = _load_json(scene_file)
    avatars = config.get("gaussian_avatars")
    if not isinstance(avatars, list) or not avatars:
        _fail(f"No gaussian_avatars found in {scene_file}")
    if avatar_index < 0 or avatar_index >= len(avatars):
        _fail(f"avatar_index {avatar_index} out of range; scene has {len(avatars)} avatars.")

    avatar = avatars[avatar_index]
    canonical_path = _resolve_from_scene(scene_file, avatar["canonical_gaussians"])
    driver_path = _resolve_from_scene(scene_file, avatar["driver"])
    scale = float(avatar.get("scale", 1.0))

    frame_indices, min_ys, transl = _compute_min_skinned_y(
        canonical_path=canonical_path,
        driver_path=driver_path,
        scale=scale,
        sample_frames=sample_frames,
        candidate_points=candidate_points,
        bottom_percent=bottom_percent,
    )
    ground_ys = _ground_y_for_frames(scene_file, transl, frame_indices, target_y)
    required_offsets = ground_ys + float(clearance) - min_ys
    offset_y = float(np.median(required_offsets))

    print(f"[INFO] scene: {scene_file}")
    print(f"[INFO] avatar: {avatar.get('name', avatar_index)}")
    print(
        "[INFO] skinned_min_y_without_offset: "
        f"min={float(np.min(min_ys)):.6f}, median={float(np.median(min_ys)):.6f}, "
        f"max={float(np.max(min_ys)):.6f}"
    )
    print(
        "[INFO] ground_y: "
        f"min={float(np.min(ground_ys)):.6f}, median={float(np.median(ground_ys)):.6f}, "
        f"max={float(np.max(ground_ys)):.6f}"
    )
    target_label = "auto-navmesh" if target_y is None else f"{target_y:.6f}"
    print(f"[INFO] target_y: {target_label}, clearance: {clearance:.6f}")
    print(
        "[INFO] required_offset_y: "
        f"min={float(np.min(required_offsets)):.6f}, "
        f"median={float(np.median(required_offsets)):.6f}, "
        f"max={float(np.max(required_offsets)):.6f}"
    )
    print(f"[INFO] offset_y: {avatar.get('offset_y', 0.0)} -> {offset_y:.6f}")

    if not dry_run:
        avatar["offset_y"] = round(offset_y, 6)
        _write_json(scene_file, config)
    return offset_y


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set gaussian avatar offset_y from skinned gaussian points instead of manual tuning."
    )
    parser.add_argument("scene", help="Scene id, scene directory, or .scene_instance.json path.")
    parser.add_argument("--avatar-index", type=int, default=0)
    parser.add_argument(
        "--target-y",
        default="auto",
        help="Ground height in Habitat world y, or 'auto' to snap driver frames to the scene navmesh.",
    )
    parser.add_argument("--clearance", type=float, default=0.015, help="Small foot clearance above ground.")
    parser.add_argument("--sample-frames", type=int, default=64)
    parser.add_argument("--candidate-points", type=int, default=30000)
    parser.add_argument("--bottom-percent", type=float, default=8.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scene_file = _resolve_scene_instance(args.scene)
    target_y = None if str(args.target_y).lower() == "auto" else float(args.target_y)
    align_scene_avatar(
        scene_file,
        avatar_index=args.avatar_index,
        target_y=target_y,
        clearance=args.clearance,
        sample_frames=args.sample_frames,
        candidate_points=args.candidate_points,
        bottom_percent=args.bottom_percent,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
