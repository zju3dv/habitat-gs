#!/usr/bin/env python3
"""
Generate Uni-NaVid-format trajectory data from VLN episodes on GS scenes.

Loads VLN episodes, runs a greedy path follower in habitat_sim with GS
rendering, and saves .mp4 trajectory videos + Uni-NaVid conversation JSON.

Usage:
    python scripts_gs/generate_uninavid_trajectories.py
    python scripts_gs/generate_uninavid_trajectories.py --split train --resume

Output structure:
    gs_scenes/trajectory_data/uninavid/
        nav_gs_train.json             # Uni-NaVid conversation format
        nav_gs_val.json
        nav_videos/
            scene01_gs_000000.mp4
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

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # habitat-gs/
SCENES_ROOT = PROJECT_ROOT / "data" / "scene_datasets" / "gs_scenes"
EPISODES_ROOT = SCENES_ROOT / "episodes" / "vln"
OUTPUT_ROOT = SCENES_ROOT / "trajectory_data" / "uninavid"

# Navigation parameters (match standard Uni-NaVid)
FORWARD_STEP = 0.25     # metres
TURN_ANGLE = 30.0       # degrees
TURN_THRESHOLD = 15.0   # degrees – half of turn angle
WAYPOINT_RADIUS = 0.5   # metres – advance to next waypoint when closer
LAST_WP_RADIUS = 0.25   # tighter radius for the final waypoint
MAX_STEPS = 498          # max steps per episode

# Rendering (match standard Uni-NaVid evaluation)
RENDER_W = 640
RENDER_H = 480
RENDER_HFOV = 120

# Video encoding
VIDEO_FPS = 10  # encode at 10fps; set video_fps=10 in training config

# Uni-NaVid prompt template (must contain NAVIGATION_IDENTIFIER)
NAVIGATION_IDENTIFIER = "a video of historical observations and an image of the current observation"
PROMPT_TEMPLATE = (
    "<image>\n"
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given {nav_id}. "
    "Your assigned task is: '{{instruction}}'. "
    "Analyze this series of images to determine your next four actions. "
    "The predicted action should be one of the following: forward, left, right, or stop."
).format(nav_id=NAVIGATION_IDENTIFIER)

# Action names
ACT_FORWARD = 1
ACT_LEFT = 2
ACT_RIGHT = 3
ACTION_WORDS = {ACT_FORWARD: "forward", ACT_LEFT: "left", ACT_RIGHT: "right"}

# ═══════════════════════════════════════════════════════════════════════
_shutdown = False
def _sig(s, f):
    global _shutdown; _shutdown = True
    print("\nInterrupt – saving progress after current scene ...")
signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)


# =====================================================================
#  Helpers
# =====================================================================

def load_episodes(split: str) -> List[dict]:
    """Load VLN episodes from .json.gz."""
    path = EPISODES_ROOT / split / f"{split}.json.gz"
    with gzip.open(str(path), "rt") as f:
        data = json.load(f)
    return data["episodes"]


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


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))


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
    Saves an .mp4 video and returns Uni-NaVid annotation dict, or None on failure.
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

    # Collect frames and actions
    frames = []
    actions = []
    next_wp = 1  # start navigating towards waypoint 1
    stuck_count = 0
    prev_pos = None

    # Initial observation
    obs = sim.get_sensor_observations()
    rgb = obs["color_sensor"]
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    frames.append(rgb)

    while next_wp < len(ref_path) and len(actions) < MAX_STEPS:
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
        if abs(dh) > math.radians(TURN_THRESHOLD):
            action = ACT_LEFT if dh > 0 else ACT_RIGHT
        else:
            action = ACT_FORWARD

        # Execute action
        sim.step(ACTION_NAME[action])

        # Stuck detection
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

        # Record action and frame
        actions.append(action)
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"]
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        frames.append(rgb)

    # Append stop action
    actions.append("stop")

    # Validate trajectory (need at least a few actions)
    if len(actions) < 4:
        return None

    # Convert action codes to words (except the final "stop" which is already a string)
    action_words = []
    for a in actions:
        if isinstance(a, str):
            action_words.append(a)
        else:
            action_words.append(ACTION_WORDS[a])

    # Save video as .mp4
    ep_name = f"{scene_name}_gs_{ep_id:06d}"
    video_dir = os.path.join(output_dir, "nav_videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{ep_name}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (RENDER_W, RENDER_H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    # Build instruction text
    instruction_data = episode["instruction"]
    if isinstance(instruction_data, dict):
        instruction_text = instruction_data["instruction_text"]
    elif isinstance(instruction_data, str):
        instruction_text = instruction_data
    elif isinstance(instruction_data, list):
        instruction_text = instruction_data[0] if instruction_data else ""
    else:
        instruction_text = str(instruction_data)

    # Build Uni-NaVid conversation format
    human_msg = PROMPT_TEMPLATE.format(instruction=instruction_text)
    gpt_msg = " ".join(action_words)

    return {
        "id": f"NAV_ID_gs_{scene_name}_{ep_id:06d}",
        "video": f"nav_videos/{ep_name}.mp4",
        "conversations": [
            {"from": "human", "value": human_msg},
            {"from": "gpt", "value": gpt_msg},
        ],
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


def scene_paths(scene_id: str) -> Tuple[str, str, str]:
    """Derive gs_ply, navmesh, scene_dataset_cfg paths from scene_id."""
    parts = scene_id.split("/")
    split_dir = parts[1]   # "train" or "val"
    scene_name = parts[2]  # "scene01"
    gs_ply = str(SCENES_ROOT / split_dir / scene_name / f"{scene_name}.gs.ply")
    navmesh = str(SCENES_ROOT / split_dir / scene_name / f"{scene_name}.navmesh")
    scene_cfg = str(SCENES_ROOT / f"{split_dir}.scene_dataset_config.json")
    return gs_ply, navmesh, scene_cfg


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Generate Uni-NaVid trajectory data for GS scenes")
    ap.add_argument("--split", choices=["train", "val", "all"], default="all")
    ap.add_argument("--output", type=str, default=str(OUTPUT_ROOT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="Skip episodes whose video file already exists")
    ap.add_argument("--scenes", type=str, default="",
                    help="Comma-separated scene names to process (e.g. 'scene28,scene29')")
    args = ap.parse_args()

    import habitat_sim  # noqa: delayed import

    output_dir = args.output
    splits = ["train", "val"] if args.split == "all" else [args.split]
    scene_filter = set(s.strip() for s in args.scenes.split(",") if s.strip()) if args.scenes else None
    global_t0 = time.time()

    for split in splits:
        if _shutdown:
            break

        ep_path = EPISODES_ROOT / split / f"{split}.json.gz"
        if not ep_path.exists():
            print(f"Skipping {split}: {ep_path} not found")
            continue

        episodes = load_episodes(split)
        scenes = discover_scenes(episodes)

        # Apply scene filter
        if scene_filter is not None:
            scenes = {sid: eps for sid, eps in scenes.items()
                      if sid.split("/")[-2] in scene_filter}
            if not scenes:
                print(f"  {split}: no scenes match filter {sorted(scene_filter)}")
                continue

        print(f"\n{'=' * 60}")
        print(f"  {split.upper()} - {sum(len(e) for e in scenes.values())} episodes, "
              f"{len(scenes)} scenes")
        print(f"{'=' * 60}")

        # Load existing annotations if resuming
        anno_path = os.path.join(output_dir, f"nav_gs_{split}.json")
        annotations = []
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
            gs_ply, navmesh, scene_cfg = scene_paths(scene_id)

            if not os.path.exists(gs_ply):
                print(f"  [{si+1}/{len(scenes)}] {scene_name}: SKIP (gs.ply not found)")
                continue

            print(f"\n  [{si+1}/{len(scenes)}] {scene_name} ({len(scene_eps)} episodes)")

            # Filter already-done episodes
            if args.resume:
                scene_eps = [e for e in scene_eps
                             if f"NAV_ID_gs_{scene_name}_{int(e['episode_id']):06d}" not in done_ids]
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

                if (ei + 1) % 50 == 0:
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

        # Save final annotations
        os.makedirs(output_dir, exist_ok=True)
        with open(anno_path, "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"\n  Saved {len(annotations)} annotations -> {anno_path}")

    elapsed_total = time.time() - global_t0
    print(f"\nAll done! Total time: {elapsed_total / 60:.1f} min")


if __name__ == "__main__":
    main()
