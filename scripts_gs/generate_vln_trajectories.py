#!/usr/bin/env python3
"""
Generate StreamVLN-format trajectory data from VLN episodes on GS scenes.

Loads VLN episodes, runs a greedy path follower in habitat_sim with GS
rendering, and saves RGB frames + action annotations compatible with
StreamVLN's VLNActionDataset.

Usage:
    python scripts_gs/generate_vln_trajectories.py
    python scripts_gs/generate_vln_trajectories.py --split train --resume

Output structure:
    gs_scenes/trajectory_data/vln/
        annotations.json          # [{id, video, instructions, actions}, ...]
        images/
            scene01_gs_000001/rgb/001.jpg, 002.jpg, ...
            ...
"""

import argparse
import gzip
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # habitat-gs/
SCENES_ROOT = PROJECT_ROOT / "data" / "scene_datasets" / "gs_scenes"
EPISODES_ROOT = SCENES_ROOT / "episodes" / "vln"
OUTPUT_ROOT = SCENES_ROOT / "trajectory_data" / "vln"

# Action codes (StreamVLN convention)
ACT_STOP = 0
ACT_FORWARD = 1
ACT_LEFT = 2
ACT_RIGHT = 3

# Navigation parameters (must match StreamVLN config)
FORWARD_STEP = 0.25     # metres
TURN_ANGLE = 15.0        # degrees
TURN_THRESHOLD = 7.5     # degrees – if heading error < this, move forward
WAYPOINT_RADIUS = 0.5    # metres – advance to next waypoint when closer
LAST_WP_RADIUS = 0.25   # tighter radius for the final waypoint
MAX_STEPS = 498          # max steps per episode (matches StreamVLN)

# Rendering
RENDER_W = 640
RENDER_H = 480
RENDER_HFOV = 79

# ═══════════════════════════════════════════════════════════════════════
_shutdown = False
def _sig(s, f):
    global _shutdown; _shutdown = True
    print("\nInterrupt – saving progress after current scene …")
signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)


# =====================================================================
#  Helpers
# =====================================================================

def load_episodes(split: str) -> Tuple[List[dict], dict]:
    """Load VLN episodes from .json.gz, return (episodes, full_data)."""
    path = EPISODES_ROOT / split / f"{split}.json.gz"
    with gzip.open(str(path), "rt") as f:
        data = json.load(f)
    return data["episodes"], data


def heading_from_quat(q):
    """Extract Y-axis heading from numpy quaternion (w,x,y,z)."""
    return 2 * math.atan2(q.y, q.w)


def desired_heading(src, tgt):
    """Heading angle from src to tgt (habitat: -Z forward, Y up)."""
    dx = tgt[0] - src[0]
    dz = tgt[2] - src[2]
    return math.atan2(dx, -dz)


def angle_diff(a, b):
    """Signed angular difference a-b, normalised to [-pi, pi]."""
    d = a - b
    while d > math.pi:  d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


def xz_dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[2] - b[2])**2)


# =====================================================================
#  Simulator setup
# =====================================================================

def create_sim(gs_ply: str, navmesh: str, scene_cfg: str, gpu: int = 0):
    """Create habitat_sim Simulator with GS scene, RGB sensor, and navmesh."""
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = gs_ply
    sim_cfg.scene_dataset_config_file = scene_cfg
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = gpu

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [RENDER_H, RENDER_W]
    rgb_spec.hfov = RENDER_HFOV

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_STEP)),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_ANGLE)),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_ANGLE)),
    }

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    sim.pathfinder.load_nav_mesh(navmesh)
    return sim


# =====================================================================
#  Greedy path follower
# =====================================================================

ACTION_NAME = {ACT_FORWARD: "move_forward", ACT_LEFT: "turn_left", ACT_RIGHT: "turn_right"}


def follow_path(sim, episode: dict, output_dir: str) -> Optional[dict]:
    """
    Follow the reference_path using a greedy heading-based controller.
    Saves RGB frames and returns annotation dict, or None on failure.
    """
    import quaternion as _quat  # noqa

    agent = sim.get_agent(0)
    state = agent.get_state()

    # Set initial state
    state.position = np.array(episode["start_position"], dtype=np.float32)
    q = episode["start_rotation"]  # [x, y, z, w]
    state.rotation = np.quaternion(q[3], q[0], q[1], q[2])
    agent.set_state(state)

    ref_path = episode["reference_path"]
    scene_name = episode["scene_id"].split("/")[-2]
    ep_id = int(episode["episode_id"])

    # Create output directory
    ep_name = f"{scene_name}_gs_{ep_id:06d}"
    rgb_dir = os.path.join(output_dir, "images", ep_name, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    actions = [-1]  # StreamVLN convention: initial marker
    frame_count = 0
    next_wp = 1  # start navigating towards waypoint 1

    # Save initial frame
    obs = sim.get_sensor_observations()
    rgb = obs["color_sensor"]
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    Image.fromarray(rgb).save(os.path.join(rgb_dir, f"{frame_count + 1:03d}.jpg"))
    frame_count += 1

    stuck_count = 0
    prev_pos = None

    while next_wp < len(ref_path) and frame_count < MAX_STEPS:
        state = agent.get_state()
        pos = state.position
        cur_heading = heading_from_quat(state.rotation)

        # Determine radius for current waypoint
        radius = LAST_WP_RADIUS if next_wp == len(ref_path) - 1 else WAYPOINT_RADIUS

        # Check if we've reached the current waypoint
        if xz_dist(pos, ref_path[next_wp]) < radius:
            next_wp += 1
            stuck_count = 0
            if next_wp >= len(ref_path):
                break
            continue

        # Compute desired heading towards waypoint
        target = np.array(ref_path[next_wp])
        dh = angle_diff(desired_heading(pos, target), cur_heading)

        # Decide action
        # In habitat_sim: turn_left increases heading, turn_right decreases
        if abs(dh) > math.radians(TURN_THRESHOLD):
            action = ACT_LEFT if dh > 0 else ACT_RIGHT
        else:
            action = ACT_FORWARD

        # Execute action
        sim.step(ACTION_NAME[action])

        # Stuck detection (before recording, so we can skip cleanly)
        new_state = agent.get_state()
        new_pos = new_state.position
        if prev_pos is not None and np.linalg.norm(new_pos - prev_pos) < 0.001 and action == ACT_FORWARD:
            stuck_count += 1
            if stuck_count > 30:
                next_wp += 1
                stuck_count = 0
                if next_wp >= len(ref_path):
                    break
                continue
        else:
            stuck_count = 0
        prev_pos = new_pos.copy()

        # Record action and save frame (kept in sync)
        actions.append(action)
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"]
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        Image.fromarray(rgb).save(os.path.join(rgb_dir, f"{frame_count + 1:03d}.jpg"))
        frame_count += 1

    # Validate trajectory
    if len(actions) < 4:
        return None  # too short

    assert len(actions) == frame_count, \
        f"Mismatch: {len(actions)} actions vs {frame_count} frames"

    instructions = episode["instruction"]
    if isinstance(instructions, dict):
        instructions = [instructions["instruction_text"]]
    elif isinstance(instructions, str):
        instructions = [instructions]

    return {
        "id": ep_id,
        "video": f"images/{ep_name}",
        "instructions": instructions,
        "actions": actions,
    }


# =====================================================================
#  Scene discovery
# =====================================================================

def discover_scenes(episodes: List[dict]) -> Dict[str, List[dict]]:
    """Group episodes by scene_id."""
    scenes: Dict[str, List[dict]] = {}
    for ep in episodes:
        sid = ep["scene_id"]
        scenes.setdefault(sid, []).append(ep)
    return dict(sorted(scenes.items()))


def scene_paths(scene_id: str, split: str) -> Tuple[str, str, str]:
    """Derive gs_ply, navmesh, scene_dataset_cfg paths from scene_id."""
    # scene_id: "gs_scenes/train/scene01/scene01.gs.ply"
    parts = scene_id.split("/")
    split_dir = parts[1]  # "train" or "val"
    scene_name = parts[2]  # "scene01"
    gs_ply = str(SCENES_ROOT / split_dir / scene_name / f"{scene_name}.gs.ply")
    navmesh = str(SCENES_ROOT / split_dir / scene_name / f"{scene_name}.navmesh")
    scene_cfg = str(SCENES_ROOT / f"{split_dir}.scene_dataset_config.json")
    return gs_ply, navmesh, scene_cfg


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Generate StreamVLN trajectory data for GS scenes")
    ap.add_argument("--split", choices=["train", "val", "all"], default="all")
    ap.add_argument("--output", type=str, default=str(OUTPUT_ROOT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="Skip episodes whose image directory already exists")
    ap.add_argument("--scenes", type=str, default="",
                    help="Comma-separated scene names to process (e.g. 'scene28,scene29'). "
                         "Empty = all scenes in the split.")
    ap.add_argument("--output-suffix", type=str, default="",
                    help="Suffix appended to annotation output files for parallel-safe writes "
                         "(e.g. 'p1' → annotations_train_p1.json, annotations_p1.json). "
                         "Image dirs are scene_eps-keyed and naturally non-conflicting.")
    args = ap.parse_args()

    import habitat_sim

    output_dir = args.output
    splits = ["train", "val"] if args.split == "all" else [args.split]
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    scene_filter = set(s.strip() for s in args.scenes.split(",") if s.strip()) if args.scenes else None
    global_t0 = time.time()

    for split in splits:
        if _shutdown:
            break

        ep_path = EPISODES_ROOT / split / f"{split}.json.gz"
        if not ep_path.exists():
            print(f"Skipping {split}: {ep_path} not found")
            continue

        episodes, _ = load_episodes(split)
        scenes = discover_scenes(episodes)

        # Apply scene filter
        if scene_filter is not None:
            scenes = {sid: eps for sid, eps in scenes.items()
                      if sid.split("/")[-2] in scene_filter}
            if not scenes:
                print(f"  {split}: no scenes match filter {sorted(scene_filter)}")
                continue

        print(f"\n{'=' * 60}")
        print(f"  {split.upper()} — {sum(len(e) for e in scenes.values())} episodes, "
              f"{len(scenes)} scenes (suffix={args.output_suffix or 'none'})")
        print(f"{'=' * 60}")

        annotations = []
        # Load existing annotations if resuming
        anno_path = os.path.join(output_dir, f"annotations_{split}{suffix}.json")
        done_ids = set()
        if args.resume and os.path.exists(anno_path):
            with open(anno_path) as f:
                annotations = json.load(f)
            done_ids = {a["id"] for a in annotations}
            print(f"  Loaded {len(annotations)} existing annotations from {anno_path}")

        for si, (scene_id, scene_eps) in enumerate(scenes.items()):
            if _shutdown:
                break

            scene_name = scene_id.split("/")[-2]
            gs_ply, navmesh, scene_cfg = scene_paths(scene_id, split)

            if not os.path.exists(gs_ply):
                print(f"  [{si+1}/{len(scenes)}] {scene_name}: SKIP (gs.ply not found)")
                continue

            print(f"\n  [{si+1}/{len(scenes)}] {scene_name} ({len(scene_eps)} episodes)")

            # Filter already-done episodes
            if args.resume:
                scene_eps = [e for e in scene_eps if int(e["episode_id"]) not in done_ids]
                if not scene_eps:
                    print(f"    All episodes already done")
                    continue
                print(f"    {len(scene_eps)} remaining")

            t0 = time.time()
            sim = create_sim(gs_ply, navmesh, scene_cfg, args.gpu)

            done = 0
            skipped = 0
            for ei, ep in enumerate(scene_eps):
                if _shutdown:
                    break

                result = follow_path(sim, ep, output_dir)
                if result is not None:
                    annotations.append(result)
                    done += 1
                else:
                    skipped += 1

                if (ei + 1) % 100 == 0:
                    print(f"      {ei+1}/{len(scene_eps)} "
                          f"(ok={done}, skip={skipped})", flush=True)

            sim.close()
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    Done: {done} trajectories, {skipped} skipped "
                  f"({elapsed:.1f}s, {rate:.1f} ep/s)")

            # Save checkpoint after each scene
            with open(anno_path, "w") as f:
                json.dump(annotations, f)

        # Save final annotations.json (main file for VLNActionDataset)
        final_anno = os.path.join(output_dir, f"annotations{suffix}.json")
        with open(final_anno, "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"\n  Saved {len(annotations)} annotations -> {final_anno}")

    elapsed_total = time.time() - global_t0
    print(f"\nAll done! Total time: {elapsed_total / 60:.1f} min")


if __name__ == "__main__":
    main()
