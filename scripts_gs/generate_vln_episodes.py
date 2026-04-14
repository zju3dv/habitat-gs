#!/usr/bin/env python3
"""
Generate VLN episodes for Gaussian Splatting scenes.

Pipeline:
  1. Sample start/goal from navmesh, compute shortest path → reference_path
  2. (Optional) Render key RGB views along the path via habitat_sim + GS
  3. Generate navigation instructions via an OpenAI-compatible VLM
  4. Build instruction_vocab, output R2RVLN-v1 format .json.gz

Usage:
    # Full pipeline (render + VLM, configure via env vars)
    export OPENAI_BASE_URL=https://api.openai.com/v1
    export OPENAI_API_KEY=sk-...
    python scripts_gs/generate_vln_episodes.py

    # Text-only (skip rendering, use geometric descriptions)
    python scripts_gs/generate_vln_episodes.py --text-only

    # Resume from checkpoint
    python scripts_gs/generate_vln_episodes.py --resume

    # Quick test (2 episodes per scene)
    python scripts_gs/generate_vln_episodes.py --train-episodes 2 --val-episodes 2

Output:
    data/scene_datasets/gs_scenes/episodes/vln/{split}/{split}.json.gz
"""

import argparse
import base64
import gzip
import io
import json
import math
import os
import re
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent                          # habitat-gs/
SCENES_ROOT = PROJECT_ROOT / "data" / "scene_datasets" / "gs_scenes"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "scene_datasets" / "gs_scenes" / "episodes" / "vln"

# ═══════════════════════════════════════════════════════════════════════
#  Episode quality parameters
# ═══════════════════════════════════════════════════════════════════════
ISLAND_RADIUS_LIMIT = 2.0       # min navigable-island radius (m) – avoids tiny fragments
MIN_GEO_DIST = 5.0              # min geodesic distance (m)
MAX_GEO_DIST = 80.0             # max geodesic distance (m)
MIN_WAYPOINTS = 4               # min reference_path waypoints after subsampling
GEO_EUCLID_MIN_RATIO = 1.05    # reject near-straight-line paths
WAYPOINT_SPACING = 2.0          # subsample reference_path to ~2 m spacing
HEIGHT_DIFF_LIMIT = 0.5         # max |Δy| between start and goal (m)

NUM_TRAIN_EPISODES = 200
NUM_VAL_EPISODES = 50

# ═══════════════════════════════════════════════════════════════════════
#  Rendering
# ═══════════════════════════════════════════════════════════════════════
RENDER_W = 640
RENDER_H = 480
RENDER_HFOV = 79
NUM_KEY_VIEWS = 4               # key viewpoint images sent to VLM per episode

# ═══════════════════════════════════════════════════════════════════════
#  LLM / VLM
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_MODEL = "gpt-5.4-mini"
API_CONCURRENT = 5              # max parallel API requests
API_RETRY = 3
API_BACKOFF_BASE = 2            # exponential backoff base (seconds)

# ═══════════════════════════════════════════════════════════════════════
#  Global state (for graceful shutdown)
# ═══════════════════════════════════════════════════════════════════════
_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n⚠  Interrupt received – finishing current scene then saving checkpoint …")

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# =====================================================================
#  Utility helpers
# =====================================================================

def load_api_config(path: Optional[str]) -> dict:
    """Load API config from a JSON file, or fall back to environment variables.

    Supported JSON formats:
      - {"provider": {"openai": {"options": {"baseURL": ..., "apiKey": ...}}}}
      - {"base_url": ..., "api_key": ...}

    If *path* is None or the file does not exist, the function reads
    OPENAI_BASE_URL and OPENAI_API_KEY from the environment instead.
    """
    if path and os.path.isfile(path):
        with open(path) as f:
            cfg = json.load(f)
        # Support nested format (codex_api.json style)
        if "provider" in cfg:
            o = cfg["provider"]["openai"]["options"]
            return {"base_url": o["baseURL"], "api_key": o["apiKey"]}
        return {"base_url": cfg["base_url"], "api_key": cfg["api_key"]}
    # Fall back to standard OpenAI env vars
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit(
            "ERROR: No API config found. Either:\n"
            "  - Pass --api-config /path/to/config.json, or\n"
            "  - Set OPENAI_BASE_URL and OPENAI_API_KEY environment variables."
        )
    return {"base_url": base_url, "api_key": api_key}


def discover_scenes(root: Path) -> Dict[str, List[dict]]:
    """Discover GS scenes grouped by split."""
    splits: Dict[str, List[dict]] = {}
    for split in ("train", "val"):
        d = root / split
        if not d.is_dir():
            continue
        scenes = []
        for sd in sorted(d.iterdir()):
            if not sd.is_dir():
                continue
            nm = sd / f"{sd.name}.navmesh"
            gs = sd / f"{sd.name}.gs.ply"
            if nm.exists():
                scenes.append({
                    "name": sd.name,
                    "navmesh": str(nm),
                    "gs_ply": str(gs) if gs.exists() else None,
                    "scene_id": f"gs_scenes/{split}/{sd.name}/{sd.name}.gs.ply",
                    "scene_dataset_cfg": str(root / f"{split}.scene_dataset_config.json"),
                })
        splits[split] = scenes
    return splits


def tokenize_simple(text: str) -> List[str]:
    """Word-level tokenisation (lowercase, alphanumeric tokens)."""
    return re.findall(r"\w+", text.lower())


def build_vocab(texts: List[str]) -> List[str]:
    words: set = set()
    for t in texts:
        words.update(tokenize_simple(t))
    return sorted(words)


# =====================================================================
#  Phase 1 – Path sampling (pathfinder only, no renderer)
# =====================================================================

def _shortest_path(pf, a, b):
    import habitat_sim
    p = habitat_sim.ShortestPath()
    p.requested_start = a
    p.requested_end = b
    ok = pf.find_path(p)
    if not ok:
        return float("inf"), []
    return p.geodesic_distance, [list(x) for x in p.points]


def _subsample_waypoints(wps: list, spacing: float = WAYPOINT_SPACING) -> list:
    """Keep start, then a point every ~spacing metres, then end."""
    if len(wps) <= 2:
        return list(wps)
    out = [wps[0]]
    acc = 0.0
    for i in range(1, len(wps)):
        acc += float(np.linalg.norm(np.array(wps[i]) - np.array(wps[i - 1])))
        if acc >= spacing:
            out.append(wps[i])
            acc = 0.0
    if out[-1] != wps[-1]:
        out.append(wps[-1])
    return out


def _heading(a, b):
    """Heading angle (rad) from position a looking towards b.  habitat: -Z forward, Y up."""
    return math.atan2(b[0] - a[0], -(b[2] - a[2]))


def _heading_to_quat(h):
    """Pure-Y rotation → [x, y, z, w] quaternion (episode JSON convention)."""
    return [0.0, float(math.sin(h / 2)), 0.0, float(math.cos(h / 2))]


def sample_paths(pf, num: int, seed: int) -> List[dict]:
    """Sample start/goal pairs from the navmesh and compute shortest paths."""
    np.random.seed(seed)
    pf.seed(seed)
    episodes: List[dict] = []
    attempts = 0
    max_attempts = num * 1000

    while len(episodes) < num and attempts < max_attempts:
        if _shutdown:
            break
        attempts += 1

        # --- sample goal ---
        tgt = np.array(pf.get_random_navigable_point(), dtype=np.float64)
        if np.isnan(tgt).any() or pf.island_radius(tgt) < ISLAND_RADIUS_LIMIT:
            continue

        # --- try several sources for this goal ---
        for _ in range(20):
            src = np.array(pf.get_random_navigable_point(), dtype=np.float64)
            if np.isnan(src).any() or pf.island_radius(src) < ISLAND_RADIUS_LIMIT:
                continue
            if abs(float(src[1] - tgt[1])) > HEIGHT_DIFF_LIMIT:
                continue

            geo, wps = _shortest_path(pf, src, tgt)
            if np.isinf(geo) or not (MIN_GEO_DIST <= geo <= MAX_GEO_DIST):
                continue
            if len(wps) < MIN_WAYPOINTS:
                continue

            euclid = float(np.linalg.norm(src - tgt))
            if euclid < 1e-6 or geo / euclid < GEO_EUCLID_MIN_RATIO:
                continue

            ref = _subsample_waypoints(wps)
            if len(ref) < 3:
                ref = list(wps)

            h = _heading(ref[0], ref[1]) if len(ref) > 1 else float(np.random.uniform(0, 2 * np.pi))

            episodes.append({
                "start": [float(v) for v in src],
                "rot":   _heading_to_quat(h),
                "goal":  [float(v) for v in tgt],
                "ref":   [[float(v) for v in p] for p in ref],
                "geo":   float(geo),
            })
            break  # got a valid episode

    return episodes


# =====================================================================
#  Phase 2 – Rendering (optional – requires GS-capable habitat_sim)
# =====================================================================

def _key_view_indices(n_wps: int, n_views: int = NUM_KEY_VIEWS) -> List[int]:
    """Evenly-spaced indices including first and last."""
    if n_wps <= n_views:
        return list(range(n_wps))
    return [int(round(i * (n_wps - 1) / (n_views - 1))) for i in range(n_views)]


def create_render_sim(gs_path: str, nm_path: str, scene_dataset_cfg: str, gpu: int = 0):
    """Create a habitat_sim Simulator configured for GS rendering."""
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = gs_path
    sim_cfg.scene_dataset_config_file = scene_dataset_cfg
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = gpu

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color_sensor"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [RENDER_H, RENDER_W]
    sensor.hfov = RENDER_HFOV

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    # Also load navmesh so agent snapping works
    sim.pathfinder.load_nav_mesh(nm_path)
    return sim


def render_key_views(sim, episode: dict, n_views: int = NUM_KEY_VIEWS) -> List:
    """Render RGB images at key waypoints along the episode's reference path."""
    from PIL import Image
    import quaternion as _quat  # noqa – registers np.quaternion

    wps = episode["ref"]
    idxs = _key_view_indices(len(wps), n_views)
    images = []
    agent = sim.get_agent(0)

    for i in idxs:
        pos = wps[i]
        # Face towards next waypoint (or from previous for last point)
        if i < len(wps) - 1:
            h = _heading(pos, wps[i + 1])
        else:
            h = _heading(wps[max(i - 1, 0)], pos)

        state = agent.get_state()
        state.position = np.array(pos, dtype=np.float32)
        state.rotation = np.quaternion(math.cos(h / 2), 0, math.sin(h / 2), 0)
        agent.set_state(state)

        obs = sim.get_sensor_observations()
        rgb = obs.get("color_sensor")
        if rgb is None:
            continue
        if rgb.ndim == 3 and rgb.shape[-1] == 4:  # RGBA → RGB
            rgb = rgb[:, :, :3]
        images.append(Image.fromarray(rgb))

    return images


# =====================================================================
#  Phase 3 – Instruction generation via GPT-5.4
# =====================================================================

def _path_description(ref: list, geo: float) -> str:
    """Geometric text description of a path (used as LLM prompt context)."""
    parts: List[str] = []
    for i in range(len(ref) - 1):
        seg_dist = float(np.linalg.norm(np.array(ref[i + 1]) - np.array(ref[i])))
        h_deg = math.degrees(_heading(ref[i], ref[i + 1]))
        if i == 0:
            parts.append(f"Start facing {h_deg:.0f} deg, walk {seg_dist:.1f}m")
        else:
            prev_h = math.degrees(_heading(ref[i - 1], ref[i]))
            turn = h_deg - prev_h
            while turn > 180:
                turn -= 360
            while turn < -180:
                turn += 360
            if abs(turn) < 10:
                td = "continue straight"
            elif turn > 0:
                td = f"turn right {abs(turn):.0f} deg"
            else:
                td = f"turn left {abs(turn):.0f} deg"
            parts.append(f"{td}, walk {seg_dist:.1f}m")
    return f"Total distance {geo:.0f}m. Segments: " + " -> ".join(parts)


def _img_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


_INSTR_SYSTEM = (
    "You are a navigation instructor. Given a description (and optionally images) "
    "of a walking path through an outdoor environment, generate a clear, natural "
    "navigation instruction in 2-4 English sentences. Reference visible landmarks "
    "(trees, buildings, roads, fences, terrain), turn directions, and approximate "
    "distances. Output ONLY the instruction text, nothing else."
)


def _call_api(client, model: str, images: List, desc: str) -> Optional[str]:
    """Single API call – VLM (with images) or text-only."""
    if images:
        user_content = [
            {"type": "text", "text": f"Path info: {desc}\n\nHere are views at key waypoints:"},
        ]
        for img in images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_img_to_b64(img)}"},
            })
    else:
        user_content = f"Path info: {desc}"

    for attempt in range(API_RETRY):
        try:
            # Use streaming: the internal API gateway strips `message.content` from
            # non-streaming responses (returns empty content with non-zero token count),
            # but streams deltas correctly. Same total tokens, just incremental.
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _INSTR_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=256,
                temperature=0.7,
                stream=True,
            )
            chunks = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            text = "".join(chunks).strip()
            if text:
                return text
        except Exception as e:
            if attempt < API_RETRY - 1:
                time.sleep(API_BACKOFF_BASE ** attempt)
            else:
                print(f"      API error (attempt {attempt+1}/{API_RETRY}): {e}")
    return None


# Flag: if VLM image mode fails once, fall back to text-only for the rest
_vlm_mode_available = True


def generate_instruction(client, model: str, images: List, desc: str) -> str:
    global _vlm_mode_available

    # Try VLM with images first
    if images and _vlm_mode_available:
        result = _call_api(client, model, images, desc)
        if result is not None:
            return result
        # If failed, maybe images not supported – try text-only
        print("      VLM call failed, trying text-only fallback …")
        _vlm_mode_available = False

    # Text-only
    result = _call_api(client, model, [], desc)
    if result is not None:
        return result

    # Last resort: return raw description
    return desc


# =====================================================================
#  Assembly & I/O
# =====================================================================

def save_dataset(episodes: List[dict], vocab: List[str], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "instruction_vocab": {"word_list": vocab},
        "episodes": episodes,
    }
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Generate VLN episodes for GS scenes")
    ap.add_argument("--scenes-root", type=str, default=str(SCENES_ROOT))
    ap.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    ap.add_argument("--train-episodes", type=int, default=NUM_TRAIN_EPISODES)
    ap.add_argument("--val-episodes", type=int, default=NUM_VAL_EPISODES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-only", action="store_true",
                    help="Skip rendering; generate instructions from geometric descriptions only")
    ap.add_argument("--api-config", type=str, default=None,
                    help="Path to API config JSON (or set OPENAI_BASE_URL + OPENAI_API_KEY env vars)")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=API_CONCURRENT,
                    help="Max concurrent API requests")
    ap.add_argument("--gpu", type=int, default=0, help="GPU device for rendering")
    ap.add_argument("--resume", action="store_true",
                    help="Skip scenes that already have a checkpoint")
    args = ap.parse_args()

    # --- Lazy imports (keep --help fast) ---
    import habitat_sim
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("ERROR: pip install openai  (needed for instruction generation)")

    api_cfg = load_api_config(args.api_config)
    client = OpenAI(base_url=api_cfg["base_url"], api_key=api_cfg["api_key"])

    scenes_root = Path(args.scenes_root)
    output_root = Path(args.output_root)

    splits = discover_scenes(scenes_root)
    if not splits:
        sys.exit(f"ERROR: no scenes found under {scenes_root}")

    all_texts: List[str] = []       # accumulate for vocab building
    global_t0 = time.time()

    for split_name, scenes in splits.items():
        n_ep = args.train_episodes if split_name == "train" else args.val_episodes
        print(f"\n{'=' * 64}")
        print(f"  {split_name.upper()} — {len(scenes)} scenes x {n_ep} episodes")
        print(f"{'=' * 64}")

        split_episodes: List[dict] = []

        for si, sc in enumerate(scenes):
            if _shutdown:
                print("Shutting down (interrupt).")
                break

            ckpt_path = output_root / split_name / "checkpoints" / f"{sc['name']}.json"

            # --- Resume support ---
            if args.resume and ckpt_path.exists():
                cached = json.loads(ckpt_path.read_text())
                split_episodes.extend(cached)
                all_texts.extend(e["instruction"]["instruction_text"] for e in cached)
                print(f"  [{si+1}/{len(scenes)}] {sc['name']}: loaded {len(cached)} cached episodes")
                continue

            print(f"\n  [{si+1}/{len(scenes)}] {sc['name']}")

            # ── Phase 1: Sample paths ────────────────────────────────
            t0 = time.time()
            pf_cfg = habitat_sim.SimulatorConfiguration()
            pf_cfg.scene_id = "NONE"
            pf_cfg.create_renderer = False
            pf_cfg.enable_physics = False
            pf_agent = habitat_sim.agent.AgentConfiguration()
            pf_agent.sensor_specifications = []
            pf_sim = habitat_sim.Simulator(habitat_sim.Configuration(pf_cfg, [pf_agent]))
            if not pf_sim.pathfinder.load_nav_mesh(sc["navmesh"]):
                print(f"    SKIP – cannot load navmesh {sc['navmesh']}")
                pf_sim.close()
                continue
            nav_area = pf_sim.pathfinder.navigable_area

            paths = sample_paths(pf_sim.pathfinder, n_ep, args.seed + si * 10000)
            pf_sim.close()

            if not paths:
                print(f"    SKIP – 0 valid paths (area={nav_area:.0f}m²)")
                continue
            dists = [p["geo"] for p in paths]
            print(f"    Paths: {len(paths)}/{n_ep}  area={nav_area:.0f}m²  "
                  f"dist={min(dists):.0f}-{max(dists):.0f}m  ({time.time()-t0:.1f}s)")

            # ── Phase 2: Render key views (optional) ─────────────────
            ep_images: Dict[int, list] = {}
            if not args.text_only and sc["gs_ply"]:
                t0 = time.time()
                try:
                    rsim = create_render_sim(
                        sc["gs_ply"], sc["navmesh"],
                        sc["scene_dataset_cfg"], args.gpu,
                    )
                    for ei, ep in enumerate(paths):
                        if _shutdown:
                            break
                        ep_images[ei] = render_key_views(rsim, ep)
                        if (ei + 1) % 200 == 0:
                            print(f"      rendered {ei+1}/{len(paths)}", flush=True)
                    rsim.close()
                    print(f"    Rendered {len(ep_images)} episodes ({time.time()-t0:.1f}s)")
                except Exception as exc:
                    print(f"    Render failed: {exc}")
                    traceback.print_exc()
                    print("    Falling back to text-only for this scene.")

            # ── Phase 3: Generate instructions (concurrent) ──────────
            t0 = time.time()
            tid_offset = len(split_episodes)
            results: Dict[int, str] = {}

            def _gen_one(idx):
                ep = paths[idx]
                desc = _path_description(ep["ref"], ep["geo"])
                imgs = ep_images.get(idx, [])
                return idx, generate_instruction(client, args.model, imgs, desc)

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_gen_one, i): i for i in range(len(paths))}
                done_count = 0
                for fut in as_completed(futures):
                    idx, txt = fut.result()
                    results[idx] = txt
                    done_count += 1
                    if done_count % 100 == 0:
                        elapsed = time.time() - t0
                        rate = done_count / elapsed if elapsed > 0 else 0
                        eta = (len(paths) - done_count) / rate if rate > 0 else 0
                        print(f"      instructions {done_count}/{len(paths)}  "
                              f"({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)

            # ── Phase 4: Assemble episodes ───────────────────────────
            scene_episodes: List[dict] = []
            for ei in range(len(paths)):
                ep = paths[ei]
                txt = results.get(ei, _path_description(ep["ref"], ep["geo"]))
                toks = tokenize_simple(txt)
                scene_episodes.append({
                    "episode_id": str(tid_offset + ei),
                    "scene_id": sc["scene_id"],
                    "start_position": ep["start"],
                    "start_rotation": ep["rot"],
                    "goals": [{"position": ep["goal"], "radius": 0.2}],
                    "reference_path": ep["ref"],
                    "instruction": {
                        "instruction_text": txt,
                        "instruction_tokens": toks,
                    },
                    "trajectory_id": tid_offset + ei,
                })
                all_texts.append(txt)

            # ── Save per-scene checkpoint ────────────────────────────
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path.write_text(json.dumps(scene_episodes, ensure_ascii=False))

            split_episodes.extend(scene_episodes)
            print(f"    Done: {len(scene_episodes)} episodes ({time.time()-t0:.1f}s)")

        # ── Save split dataset ───────────────────────────────────────
        if split_episodes:
            vocab = build_vocab(all_texts)
            out_path = str(output_root / split_name / f"{split_name}.json.gz")
            save_dataset(split_episodes, vocab, out_path)
            print(f"\n  Saved {len(split_episodes)} episodes -> {out_path}")
            print(f"  Vocabulary size: {len(vocab)} words")

    elapsed_total = time.time() - global_t0
    print(f"\nAll done! Total time: {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
