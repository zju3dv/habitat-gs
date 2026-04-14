#!/usr/bin/env python3
"""
Generate PointNav episodes for Gaussian Splatting scenes.

Produces per-scene .json.gz episode files and a master index for each split,
compatible with habitat-lab's PointNav-v1 dataset format.

Usage:
    python scripts_gs/generate_pointnav_episodes.py             # all scenes
    python scripts_gs/generate_pointnav_episodes.py --scene scene28  # one scene

The script auto-discovers scenes from the gs_scenes directory structure and
generates episodes using navmesh sampling + geodesic distance filtering.
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root (habitat-gs/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # habitat-gs/
SCENES_ROOT = PROJECT_ROOT / "data" / "scene_datasets" / "gs_scenes"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ISLAND_RADIUS_LIMIT = 1.5       # min navigable island radius (metres)
CLOSEST_DIST = 1.0              # min geodesic distance (metres)
FURTHEST_DIST = 30.0            # max geodesic distance (metres)
GEO_TO_EUCLID_RATIO = 1.1      # ratio threshold for straight-line rejection
RETRIES_PER_TARGET = 10         # source samples per target
GOAL_RADIUS = 0.2               # success radius (metres)
NUM_TRAIN_EPISODES = 1000       # episodes per training scene
NUM_VAL_EPISODES = 100          # episodes per validation scene

# Scene dataset config path relative to CWD (habitat-gs/)
SCENE_DATASET_CONFIG = "data/scene_datasets/gs_scenes/gs_scenes.scene_dataset_config.json"


def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    """Aggressive rejection sampling for near-straight-line episodes."""
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2


def _geodesic_distance(pathfinder, source: np.ndarray, target: np.ndarray) -> float:
    """Compute geodesic distance between two points via pathfinder."""
    import habitat_sim
    path = habitat_sim.ShortestPath()
    path.requested_start = source
    path.requested_end = target
    pathfinder.find_path(path)
    return path.geodesic_distance


def is_compatible_episode(
    source: np.ndarray,
    target: np.ndarray,
    pathfinder,
    near_dist: float,
    far_dist: float,
    geo_to_euclid_ratio: float,
) -> Tuple[bool, float]:
    """Check whether (source, target) forms a valid navigation episode."""
    # Height difference check (same floor)
    if abs(source[1] - target[1]) > 0.5:
        return False, 0.0

    geo_dist = _geodesic_distance(pathfinder, source, target)
    if np.isinf(geo_dist):
        return False, 0.0

    if not (near_dist <= geo_dist <= far_dist):
        return False, 0.0

    euclid_dist = float(np.linalg.norm(source - target))
    if euclid_dist < 1e-6:
        return False, 0.0

    ratio = geo_dist / euclid_dist
    if ratio < geo_to_euclid_ratio:
        if np.random.rand() > _ratio_sample_rate(ratio, geo_to_euclid_ratio):
            return False, 0.0

    if pathfinder.island_radius(source) < ISLAND_RADIUS_LIMIT:
        return False, 0.0

    return True, geo_dist


def generate_episodes_for_scene(
    pathfinder,
    scene_id: str,
    num_episodes: int,
    seed: int = 0,
) -> List[dict]:
    """Generate PointNav episodes for a single scene using its navmesh."""
    np.random.seed(seed)
    pathfinder.seed(seed)

    episodes = []
    episode_count = 0
    attempts = 0
    max_attempts = num_episodes * 500  # safety limit

    while episode_count < num_episodes and attempts < max_attempts:
        attempts += 1

        target = np.array(pathfinder.get_random_navigable_point(), dtype=np.float64)
        if np.isnan(target).any():
            continue
        if pathfinder.island_radius(target) < ISLAND_RADIUS_LIMIT:
            continue

        for _ in range(RETRIES_PER_TARGET):
            source = np.array(pathfinder.get_random_navigable_point(), dtype=np.float64)
            if np.isnan(source).any():
                continue

            ok, geo_dist = is_compatible_episode(
                source, target, pathfinder,
                near_dist=CLOSEST_DIST,
                far_dist=FURTHEST_DIST,
                geo_to_euclid_ratio=GEO_TO_EUCLID_RATIO,
            )
            if ok:
                break
        else:
            continue

        if not ok:
            continue

        # Random agent heading
        angle = np.random.uniform(0, 2 * np.pi)
        rotation = [0.0, float(np.sin(angle / 2)), 0.0, float(np.cos(angle / 2))]

        episode = {
            "episode_id": str(episode_count),
            "scene_id": scene_id,
            "start_position": [float(source[0]), float(source[1]), float(source[2])],
            "start_rotation": rotation,
            "goals": [
                {
                    "position": [float(target[0]), float(target[1]), float(target[2])],
                    "radius": GOAL_RADIUS,
                }
            ],
            "info": {"geodesic_distance": float(geo_dist)},
        }
        episodes.append(episode)
        episode_count += 1

    if episode_count < num_episodes:
        print(
            f"  WARNING: only generated {episode_count}/{num_episodes} episodes "
            f"(navmesh may be too small or fragmented)"
        )

    return episodes


def discover_scenes(scenes_root: Path) -> Dict[str, List[dict]]:
    """Discover all scenes grouped by split (train / val)."""
    splits: Dict[str, List[dict]] = {}
    for split_name in ("train", "val"):
        split_dir = scenes_root / split_name
        if not split_dir.is_dir():
            continue
        scenes = []
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            navmesh = scene_dir / f"{scene_dir.name}.navmesh"
            gs_ply = scene_dir / f"{scene_dir.name}.gs.ply"
            if navmesh.exists() and gs_ply.exists():
                # scene_id relative to scenes_dir ("data/scene_datasets")
                scene_id = f"gs_scenes/{split_name}/{scene_dir.name}/{scene_dir.name}.gs.ply"
                scenes.append({
                    "name": scene_dir.name,
                    "navmesh": str(navmesh),
                    "scene_id": scene_id,
                })
        splits[split_name] = scenes
    return splits


def save_episodes(
    episodes: List[dict],
    output_path: str,
) -> None:
    """Save episodes list to a gzipped JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {"episodes": episodes}
    with gzip.open(output_path, "wt") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate PointNav episodes for GS scenes"
    )
    parser.add_argument(
        "--scenes-root", type=str, default=str(SCENES_ROOT),
        help="Root directory containing train/ and val/ scene folders",
    )
    parser.add_argument(
        "--output-root", type=str, default=None,
        help="Output root for episode files. Default: <scenes_root>/episodes",
    )
    parser.add_argument(
        "--train-episodes", type=int, default=NUM_TRAIN_EPISODES,
        help=f"Episodes per training scene (default: {NUM_TRAIN_EPISODES})",
    )
    parser.add_argument(
        "--val-episodes", type=int, default=NUM_VAL_EPISODES,
        help=f"Episodes per validation scene (default: {NUM_VAL_EPISODES})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--scene", type=str, default=None,
        help="Generate episodes for a single scene only (matches the scene "
             "directory name, e.g. 'scene28')",
    )
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    output_root = (Path(args.output_root) if args.output_root
                   else scenes_root / "episodes" / "pointnav")

    # Must import habitat_sim after argument parsing to get fast --help
    import habitat_sim

    print(f"Scenes root : {scenes_root}")
    print(f"Output root : {output_root}")
    print()

    splits = discover_scenes(scenes_root)
    if not splits:
        print("ERROR: No scenes found!", file=sys.stderr)
        sys.exit(1)

    for split_name, scenes in splits.items():
        num_ep = args.train_episodes if split_name == "train" else args.val_episodes
        print(f"=== Split: {split_name} ({len(scenes)} scenes, {num_ep} episodes each) ===")

        for idx, scene_info in enumerate(scenes):
            scene_name = scene_info["name"]
            if args.scene and scene_name != args.scene:
                continue
            navmesh_path = scene_info["navmesh"]
            scene_id = scene_info["scene_id"]
            seed = args.seed + idx * 10000

            print(f"  [{idx+1}/{len(scenes)}] {scene_name} ... ", end="", flush=True)

            # Create a minimal simulator just for pathfinding
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = "NONE"
            sim_cfg.create_renderer = False
            sim_cfg.enable_physics = False

            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = []

            cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
            sim = habitat_sim.Simulator(cfg)

            # Load the navmesh
            loaded = sim.pathfinder.load_nav_mesh(navmesh_path)
            if not loaded:
                print(f"FAILED to load navmesh: {navmesh_path}")
                sim.close()
                continue

            # Print navmesh stats
            area = sim.pathfinder.navigable_area
            bounds = sim.pathfinder.get_bounds()
            print(f"navmesh area={area:.1f}m^2 ", end="", flush=True)

            # Generate episodes
            episodes = generate_episodes_for_scene(
                sim.pathfinder, scene_id, num_ep, seed=seed,
            )
            sim.close()

            # Save per-scene episodes
            content_path = str(
                output_root / split_name / "content" / f"{scene_name}.json.gz"
            )
            save_episodes(episodes, content_path)

            # Print stats
            if episodes:
                dists = [ep["info"]["geodesic_distance"] for ep in episodes]
                print(
                    f"-> {len(episodes)} episodes "
                    f"(dist: {min(dists):.1f}-{max(dists):.1f}m, "
                    f"mean={np.mean(dists):.1f}m)"
                )
            else:
                print("-> 0 episodes")

        # Save master index (empty episodes list; per-scene files are loaded via content/)
        index_path = str(output_root / split_name / f"{split_name}.json.gz")
        save_episodes([], index_path)
        print(f"  Master index: {index_path}")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
