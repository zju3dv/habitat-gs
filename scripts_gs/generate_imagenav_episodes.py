#!/usr/bin/env python3
"""
Generate ImageNav episodes for Gaussian Splatting scenes WITH goal-image
quality control.

For each candidate episode the script renders the goal image using the same
deterministic rotation that ImageGoalSensor will use at training time,
and rejects episodes whose goal image lacks sufficient visual information
(e.g. pointing at sky / floor / blank wall).

Usage:
    python scripts_gs/generate_imagenav_episodes.py          # all scenes
    python scripts_gs/generate_imagenav_episodes.py --scene scene01  # one scene
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCENES_ROOT = PROJECT_ROOT / "data" / "scene_datasets" / "gs_scenes"

# ── Episode sampling parameters ──────────────────────────────────────
ISLAND_RADIUS_LIMIT = 1.5
CLOSEST_DIST = 1.0
FURTHEST_DIST = 30.0
GEO_TO_EUCLID_RATIO = 1.1
RETRIES_PER_TARGET = 10
GOAL_RADIUS = 0.2
NUM_TRAIN_EPISODES = 1000
NUM_VAL_EPISODES = 100

# ── Goal-image quality thresholds ────────────────────────────────────
# An image is rejected when BOTH metrics are below their thresholds.
# Laplacian std measures edge/texture density; RGB std measures colour variety.
MIN_LAPLACIAN_STD = 8.0      # reject featureless images (sky, floor)
MIN_RGB_STD = 15.0            # reject near-uniform-colour images


# =====================================================================
# Image quality helpers
# =====================================================================

def _laplacian_std(img: np.ndarray) -> float:
    """Standard deviation of 3x3 Laplacian on a grayscale image."""
    gray = np.mean(img[:, :, :3].astype(np.float32), axis=2)
    # 3x3 Laplacian kernel via finite differences
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.std(lap))


def _rgb_std(img: np.ndarray) -> float:
    """Mean per-channel standard deviation."""
    return float(np.mean([np.std(img[:, :, c].astype(np.float32)) for c in range(3)]))


def image_quality_ok(img: np.ndarray) -> bool:
    """Return True if the goal image carries enough visual information."""
    return _laplacian_std(img) >= MIN_LAPLACIAN_STD or _rgb_std(img) >= MIN_RGB_STD


# =====================================================================
# Goal-image rendering (matches ImageGoalSensor logic)
# =====================================================================

def render_goal_image(sim, goal_position: List[float], episode_id: str, rgb_sensor_uuid: str) -> np.ndarray:
    """Render the goal image exactly as ImageGoalSensor would."""
    import habitat_sim

    seed = abs(hash(episode_id)) % (2**32)
    rng = np.random.RandomState(seed)
    angle = rng.uniform(0, 2 * np.pi)
    quat = habitat_sim.utils.common.quat_from_angle_axis(
        float(angle), np.array([0.0, 1.0, 0.0])
    )

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = np.array(goal_position, dtype=np.float32)
    state.rotation = quat
    agent.set_state(state, reset_sensors=True)
    obs = sim.get_sensor_observations()
    return obs[rgb_sensor_uuid]


# =====================================================================
# Navmesh helpers (identical to generate_episodes.py)
# =====================================================================

def _geodesic_distance(pathfinder, source, target):
    import habitat_sim
    path = habitat_sim.ShortestPath()
    path.requested_start = source
    path.requested_end = target
    pathfinder.find_path(path)
    return path.geodesic_distance


def _ratio_sample_rate(ratio, threshold):
    assert ratio < threshold
    return 20 * (ratio - 0.98) ** 2


def is_compatible_episode(source, target, pathfinder, near, far, ratio_thresh):
    if abs(source[1] - target[1]) > 0.5:
        return False, 0.0
    geo = _geodesic_distance(pathfinder, source, target)
    if np.isinf(geo):
        return False, 0.0
    if not (near <= geo <= far):
        return False, 0.0
    euclid = float(np.linalg.norm(source - target))
    if euclid < 1e-6:
        return False, 0.0
    ratio = geo / euclid
    if ratio < ratio_thresh and np.random.rand() > _ratio_sample_rate(ratio, ratio_thresh):
        return False, 0.0
    if pathfinder.island_radius(source) < ISLAND_RADIUS_LIMIT:
        return False, 0.0
    return True, geo


# =====================================================================
# Core generation
# =====================================================================

def generate_episodes(sim, pathfinder, scene_id, num_episodes, seed, rgb_uuid):
    np.random.seed(seed)
    pathfinder.seed(seed)

    episodes = []
    ep_count = 0
    attempts = 0
    max_attempts = num_episodes * 1000
    quality_rejects = 0

    while ep_count < num_episodes and attempts < max_attempts:
        attempts += 1

        target = np.array(pathfinder.get_random_navigable_point(), dtype=np.float64)
        if np.isnan(target).any():
            continue
        if pathfinder.island_radius(target) < ISLAND_RADIUS_LIMIT:
            continue

        # Find compatible source
        ok = False
        for _ in range(RETRIES_PER_TARGET):
            source = np.array(pathfinder.get_random_navigable_point(), dtype=np.float64)
            if np.isnan(source).any():
                continue
            ok, geo = is_compatible_episode(
                source, target, pathfinder,
                CLOSEST_DIST, FURTHEST_DIST, GEO_TO_EUCLID_RATIO,
            )
            if ok:
                break
        if not ok:
            continue

        # Candidate episode id
        episode_id = str(ep_count)

        # ── Quality gate: render goal image and check ────────────
        goal_img = render_goal_image(
            sim,
            [float(target[0]), float(target[1]), float(target[2])],
            episode_id,
            rgb_uuid,
        )
        if not image_quality_ok(goal_img):
            quality_rejects += 1
            continue

        angle = np.random.uniform(0, 2 * np.pi)
        rotation = [0.0, float(np.sin(angle / 2)), 0.0, float(np.cos(angle / 2))]

        episodes.append({
            "episode_id": episode_id,
            "scene_id": scene_id,
            "start_position": [float(source[0]), float(source[1]), float(source[2])],
            "start_rotation": rotation,
            "goals": [{
                "position": [float(target[0]), float(target[1]), float(target[2])],
                "radius": GOAL_RADIUS,
            }],
            "info": {"geodesic_distance": float(geo)},
        })
        ep_count += 1

    if ep_count < num_episodes:
        print(
            f"\n  WARNING: only {ep_count}/{num_episodes} episodes "
            f"(quality_rejects={quality_rejects})"
        )

    return episodes, quality_rejects


# =====================================================================
# Scene discovery
# =====================================================================

def discover_scenes(root: Path):
    splits = {}
    for split in ("train", "val"):
        d = root / split
        if not d.is_dir():
            continue
        scenes = []
        for sd in sorted(d.iterdir()):
            if not sd.is_dir():
                continue
            nav = sd / f"{sd.name}.navmesh"
            gs = sd / f"{sd.name}.gs.ply"
            if nav.exists() and gs.exists():
                scenes.append({
                    "name": sd.name,
                    "navmesh": str(nav),
                    "gs_ply": str(gs),
                    "scene_id": f"gs_scenes/{split}/{sd.name}/{sd.name}.gs.ply",
                })
        splits[split] = scenes
    return splits


def save_episodes(episodes, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump({"episodes": episodes}, f)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate ImageNav episodes with goal-image quality control")
    parser.add_argument("--scenes-root", default=str(SCENES_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--train-episodes", type=int, default=NUM_TRAIN_EPISODES)
    parser.add_argument("--val-episodes", type=int, default=NUM_VAL_EPISODES)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--scene", type=str, default=None, help="Generate for a single scene only")
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    output_root = Path(args.output_root) if args.output_root else scenes_root / "episodes" / "imagenav"

    import habitat_sim
    from habitat_sim.utils.settings import make_cfg, default_sim_settings

    print(f"Scenes root : {scenes_root}")
    print(f"Output root : {output_root}")
    print()

    splits = discover_scenes(scenes_root)
    total_quality_rejects = 0

    for split, scenes in splits.items():
        num_ep = args.train_episodes if split == "train" else args.val_episodes
        print(f"=== {split} ({len(scenes)} scenes, {num_ep} ep each) ===")

        for idx, info in enumerate(scenes):
            if args.scene and info["name"] != args.scene:
                continue

            name = info["name"]
            seed = args.seed + idx * 10000
            print(f"  [{idx+1}/{len(scenes)}] {name} ... ", end="", flush=True)

            # Build simulator WITH rendering for this scene
            sim_settings = dict(default_sim_settings)
            sim_settings["scene"] = info["gs_ply"]
            sim_settings["scene_dataset_config_file"] = str(
                scenes_root / f"{split}.scene_dataset_config.json"
            )
            sim_settings["default_agent_navmesh"] = False
            sim_settings["width"] = 256
            sim_settings["height"] = 256
            sim_settings["enable_physics"] = False

            cfg = make_cfg(sim_settings)
            sim = habitat_sim.Simulator(cfg)

            # Verify navmesh
            if not sim.pathfinder.is_loaded:
                print("navmesh NOT loaded, skipping")
                sim.close()
                continue

            area = sim.pathfinder.navigable_area
            print(f"area={area:.0f}m² ", end="", flush=True)

            # Detect the RGB sensor uuid
            rgb_uuid = None
            for s_uuid, s_obj in sim._sensors.items():
                if "color" in s_uuid or "rgb" in s_uuid:
                    rgb_uuid = s_uuid
                    break
            if rgb_uuid is None:
                print("no RGB sensor, skipping")
                sim.close()
                continue

            episodes, qr = generate_episodes(
                sim, sim.pathfinder, info["scene_id"], num_ep, seed, rgb_uuid,
            )
            total_quality_rejects += qr
            sim.close()

            out = str(output_root / split / "content" / f"{name}.json.gz")
            save_episodes(episodes, out)

            if episodes:
                dists = [e["info"]["geodesic_distance"] for e in episodes]
                print(
                    f"-> {len(episodes)} ep "
                    f"(dist {min(dists):.1f}-{max(dists):.1f}m, "
                    f"mean {np.mean(dists):.1f}m, "
                    f"img_reject {qr})"
                )
            else:
                print(f"-> 0 ep (img_reject {qr})")

        index = str(output_root / split / f"{split}.json.gz")
        save_episodes([], index)
        print(f"  Index: {index}\n")

    print(f"Done. Total image-quality rejects: {total_quality_rejects}")


if __name__ == "__main__":
    main()
