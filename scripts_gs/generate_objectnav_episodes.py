#!/usr/bin/env python3
"""
Generate ObjectNav episodes for Gaussian Splatting scenes using SAM + CLIP.

Pipeline:
  1. Sample navigable points, render RGB+Depth at each
  2. SAM segments objects, CLIP classifies each mask
  3. Back-project to 3D, cluster into object instances (DBSCAN)
  4. Generate ObjectNav episodes with view_points

Usage:
    python scripts_gs/generate_objectnav_episodes.py --visualize
    python scripts_gs/generate_objectnav_episodes.py --scene scene01 --visualize
"""

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCENES_ROOT = PROJECT_ROOT / "data" / "scene_datasets" / "gs_scenes"

# ---------------------------------------------------------------------------
# Episode generation constants (same as PointNav / ImageNav)
# ---------------------------------------------------------------------------
ISLAND_RADIUS_LIMIT = 1.5
CLOSEST_DIST = 1.0
FURTHEST_DIST = 30.0
GEO_TO_EUCLID_RATIO = 1.1
RETRIES_PER_TARGET = 10
NUM_TRAIN_EPISODES = 1000
NUM_VAL_EPISODES = 100

# ---------------------------------------------------------------------------
# Object detection constants
# ---------------------------------------------------------------------------
NUM_SAMPLE_POINTS = 200        # navigable points per scene for scanning
NUM_ROTATIONS = 4              # views per point (0, 90, 180, 270 degrees)
IMG_SIZE = 256                 # rendering resolution
HFOV = 90                     # horizontal field of view (degrees)
SAM_POINTS_PER_SIDE = 32      # SAM grid density
SAM_IOU_THRESH = 0.86
SAM_STABILITY_THRESH = 0.85
SAM_MIN_MASK_AREA = 500
CLIP_MIN_SCORE = 0.15         # minimum CLIP category score
MASK_MIN_AREA_RATIO = 0.005   # 0.5% of image
MASK_MAX_AREA_RATIO = 0.30    # 30% of image
DBSCAN_EPS = 2.0              # clustering radius (metres)
DBSCAN_MIN_SAMPLES = 2
MIN_VIEWPOINTS_PER_OBJECT = 3
MAX_INSTANCES_PER_CATEGORY = 15
MIN_INSTANCES_PER_CATEGORY = 2
INSTANCE_MERGE_DIST = 2.0     # merge instances closer than this
UBIQUITY_SAMPLE_POINTS = 50   # points to test for ubiquity filter
UBIQUITY_THRESHOLD = 0.70     # if >70% points are <5m from nearest instance, skip
UBIQUITY_DIST = 5.0           # distance threshold for ubiquity

# ---------------------------------------------------------------------------
# CLIP categories
# ---------------------------------------------------------------------------
CATEGORY_PROMPTS = {
    "car":          "a car",
    "bench":        "a bench",
    "tree":         "a tree",
    "street lamp":  "a street lamp",
    "traffic sign": "a traffic sign",
    "fire hydrant": "a fire hydrant",
    "trash can":    "a trash can",
    "bicycle":      "a bicycle",
    "potted plant": "a potted plant",
    "barrier":      "a traffic barrier",
    "statue":       "a statue",
    "chair":        "a chair",
}
BACKGROUND_PROMPTS = [
    "a building facade", "a wall", "a road", "a sidewalk",
    "the sky", "grass", "the ground",
]

SAM_CHECKPOINT = Path.home() / ".cache" / "sam_checkpoints" / "sam_vit_b_01ec64.pth"
CLIP_CHECKPOINT = Path.home() / ".cache" / "clip_models" / "vit_b_32_laion400m.pt"


# =====================================================================
# Navmesh helpers (from generate_episodes.py)
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
# Scene discovery (from generate_imagenav_episodes.py)
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


# =====================================================================
# Model loading
# =====================================================================

def load_sam(device: str):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device)
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=SAM_POINTS_PER_SIDE,
        pred_iou_thresh=SAM_IOU_THRESH,
        stability_score_thresh=SAM_STABILITY_THRESH,
        min_mask_region_area=SAM_MIN_MASK_AREA,
    )
    return mask_gen


def load_clip(device: str):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained=str(CLIP_CHECKPOINT),
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    model.eval().to(device)

    # Pre-encode text features
    all_prompts = list(CATEGORY_PROMPTS.values()) + BACKGROUND_PROMPTS
    text_tokens = tokenizer(all_prompts).to(device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    n_cat = len(CATEGORY_PROMPTS)
    cat_names = list(CATEGORY_PROMPTS.keys())
    return model, preprocess, text_features, cat_names, n_cat


# =====================================================================
# 3D back-projection
# =====================================================================

def quat_to_rotation_matrix(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def backproject_mask_to_3d(mask, depth, agent_pos, agent_rot_quat,
                           hfov=HFOV, img_size=IMG_SIZE):
    """Back-project masked depth pixels to world coordinates.

    Args:
        mask: (H, W) bool array
        depth: (H, W) float array (metric depth from habitat-sim)
        agent_pos: (3,) agent position in world frame
        agent_rot_quat: (4,) quaternion [x, y, z, w]
        hfov: horizontal field of view in degrees
        img_size: image width/height

    Returns:
        (N, 3) array of 3D world points
    """
    fx = fy = img_size / (2.0 * math.tan(math.radians(hfov) / 2.0))
    cx = cy = img_size / 2.0

    vs, us = np.where(mask)
    ds = depth[vs, us]
    valid = ds > 0
    us, vs, ds = us[valid], vs[valid], ds[valid]

    if len(ds) == 0:
        return np.zeros((0, 3))

    # Camera frame (habitat: -z forward, y down)
    x_cam = (us - cx) * ds / fx
    y_cam = (vs - cy) * ds / fy
    z_cam = -ds

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    R = quat_to_rotation_matrix(agent_rot_quat)
    points_world = (R @ points_cam.T).T + agent_pos

    return points_world


def compute_center_iou(mask, img_size=IMG_SIZE, center_frac=0.5):
    """IoU between mask and a centered rectangle."""
    h, w = mask.shape
    ch, cw = int(h * center_frac), int(w * center_frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    center_rect = np.zeros_like(mask)
    center_rect[y0:y0+ch, x0:x0+cw] = True
    intersection = np.sum(mask & center_rect)
    union = np.sum(mask | center_rect)
    return float(intersection / max(union, 1))


# =====================================================================
# Object scanning
# =====================================================================

def sample_observation_points(pathfinder, num_points, num_rotations):
    """Sample navigable points with multiple rotations."""
    import habitat_sim
    points = []
    attempts = 0
    max_attempts = num_points * 10
    while len(points) < num_points and attempts < max_attempts:
        attempts += 1
        pos = np.array(pathfinder.get_random_navigable_point(), dtype=np.float64)
        if np.isnan(pos).any():
            continue
        if pathfinder.island_radius(pos) < ISLAND_RADIUS_LIMIT:
            continue
        for i in range(num_rotations):
            angle = 2.0 * math.pi * i / num_rotations
            quat = habitat_sim.utils.common.quat_from_angle_axis(
                float(angle), np.array([0.0, 1.0, 0.0])
            )
            # Convert numpy quaternion to [x, y, z, w] list
            q = [float(quat.x), float(quat.y),
                 float(quat.z), float(quat.w)]
            points.append((pos.copy(), q))
    return points


def scan_scene_for_objects(sim, pathfinder, mask_gen, clip_model,
                           clip_preprocess, text_features, cat_names,
                           n_cat, device, num_points, num_rotations,
                           visualize_dir=None):
    """Scan a scene with SAM + CLIP and return raw detections."""
    obs_points = sample_observation_points(pathfinder, num_points, num_rotations)

    # Detect RGB and depth sensor uuids
    rgb_uuid = depth_uuid = None
    for s_uuid in sim._sensors:
        if "color" in s_uuid or "rgb" in s_uuid:
            rgb_uuid = s_uuid
        if "depth" in s_uuid:
            depth_uuid = s_uuid
    assert rgb_uuid and depth_uuid, "Need both RGB and depth sensors"

    detections = []
    vis_count = 0
    max_vis = 20 if visualize_dir else 0

    for idx, (pos, quat) in enumerate(obs_points):
        # Set agent state and render
        agent = sim.get_agent(0)
        state = agent.get_state()
        state.position = np.array(pos, dtype=np.float32)

        import habitat_sim
        q = habitat_sim.utils.common.quat_from_coeffs(np.array(quat, dtype=np.float32))
        state.rotation = q
        agent.set_state(state, reset_sensors=True)
        obs = sim.get_sensor_observations()

        rgb = obs[rgb_uuid][:, :, :3].copy()  # (H, W, 3) uint8
        depth = obs[depth_uuid].copy()         # (H, W) or (H, W, 1)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        # Run SAM
        masks = mask_gen.generate(rgb)

        # Classify each mask
        vis_annotations = []
        for m in masks:
            seg = m["segmentation"]
            area_ratio = m["area"] / (rgb.shape[0] * rgb.shape[1])
            if area_ratio < MASK_MIN_AREA_RATIO or area_ratio > MASK_MAX_AREA_RATIO:
                continue

            # Crop for CLIP
            x, y, w, h = [int(v) for v in m["bbox"]]
            pad = int(max(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(rgb.shape[1], x + w + pad)
            y2 = min(rgb.shape[0], y + h + pad)
            crop = Image.fromarray(rgb[y1:y2, x1:x2])
            crop_tensor = clip_preprocess(crop).unsqueeze(0).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda"):
                img_feat = clip_model.encode_image(crop_tensor)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                sim_scores = (100.0 * img_feat @ text_features.T).softmax(dim=-1)[0]

            # Split into category vs background scores
            cat_scores = sim_scores[:n_cat].cpu().numpy()
            best_cat_idx = int(np.argmax(cat_scores))
            best_cat_score = cat_scores[best_cat_idx]

            if best_cat_score < CLIP_MIN_SCORE:
                continue

            category = cat_names[best_cat_idx]

            # Back-project to 3D
            pts_3d = backproject_mask_to_3d(seg, depth, pos, quat)
            if len(pts_3d) < 10:
                continue

            centroid = np.median(pts_3d, axis=0)
            iou = compute_center_iou(seg)

            detections.append({
                "category": category,
                "centroid": centroid.tolist(),
                "score": float(best_cat_score),
                "agent_position": pos.tolist(),
                "agent_rotation": quat,
                "iou": float(iou),
                "num_pixels": int(np.sum(seg)),
            })

            if vis_count < max_vis:
                vis_annotations.append({
                    "bbox": [x1, y1, x2, y2],
                    "category": category,
                    "score": float(best_cat_score),
                    "mask": seg,
                })

        # Save visualization
        if vis_annotations and vis_count < max_vis and visualize_dir:
            save_visualization(rgb, vis_annotations, visualize_dir, idx)
            vis_count += 1

        if (idx + 1) % 100 == 0:
            print(f"    scanned {idx+1}/{len(obs_points)} views, "
                  f"{len(detections)} detections so far", flush=True)

    return detections


# =====================================================================
# Visualization
# =====================================================================

def save_visualization(rgb, annotations, out_dir, idx):
    """Save annotated image with mask outlines and labels."""
    os.makedirs(out_dir, exist_ok=True)
    img = rgb.copy()

    # Simple overlay: darken non-mask areas, colour mask borders
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    ]
    for i, ann in enumerate(annotations):
        color = colors[i % len(colors)]
        mask = ann["mask"]

        # Draw mask border
        from scipy import ndimage
        eroded = ndimage.binary_erosion(mask, iterations=2)
        border = mask & ~eroded
        img[border] = color

        # Add label text via simple pixel drawing (avoid matplotlib dependency)
        x1, y1, x2, y2 = ann["bbox"]
        label = f"{ann['category']} {ann['score']:.2f}"
        # Put a colored bar at top of bbox
        bar_h = min(15, y2 - y1)
        img[y1:y1+bar_h, x1:min(x1+len(label)*7, x2)] = color

    Image.fromarray(img).save(os.path.join(out_dir, f"view_{idx:04d}.png"))


# =====================================================================
# 3D clustering
# =====================================================================

def cluster_detections(detections):
    """Cluster raw detections into object instances per category."""
    from sklearn.cluster import DBSCAN

    # Group by category
    by_cat = {}
    for d in detections:
        by_cat.setdefault(d["category"], []).append(d)

    objects_per_category = {}
    for cat, dets in by_cat.items():
        centroids = np.array([d["centroid"] for d in dets])
        clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(centroids)

        objects = []
        for label in sorted(set(clustering.labels_)):
            if label == -1:  # noise
                continue
            cluster_mask = clustering.labels_ == label
            cluster_dets = [d for d, m in zip(dets, cluster_mask) if m]
            cluster_centroids = np.array([d["centroid"] for d in cluster_dets])
            obj_pos = np.median(cluster_centroids, axis=0)

            # Deduplicate view_points (same agent position → keep best score)
            vp_map = {}
            for d in cluster_dets:
                key = tuple(np.round(d["agent_position"], 2))
                if key not in vp_map or d["score"] > vp_map[key]["score"]:
                    vp_map[key] = d
            view_points = [
                {
                    "position": v["agent_position"],
                    "rotation": v["agent_rotation"],
                    "iou": v["iou"],
                }
                for v in vp_map.values()
            ]

            if len(view_points) < MIN_VIEWPOINTS_PER_OBJECT:
                continue

            objects.append({
                "position": obj_pos.tolist(),
                "view_points": view_points,
                "num_detections": len(cluster_dets),
                "mean_score": float(np.mean([d["score"] for d in cluster_dets])),
            })

        if objects:
            objects_per_category[cat] = objects

    return objects_per_category


# =====================================================================
# Quality filters
# =====================================================================

def merge_close_instances(objects, min_dist=INSTANCE_MERGE_DIST):
    """Merge object instances that are within min_dist of each other."""
    if len(objects) <= 1:
        return objects
    positions = np.array([o["position"] for o in objects])
    merged = []
    used = set()
    for i in range(len(objects)):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(objects)):
            if j in used:
                continue
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                group.append(j)
                used.add(j)
        used.add(i)
        # Merge: pick the one with most detections, combine view_points
        best = max(group, key=lambda k: objects[k]["num_detections"])
        merged_obj = dict(objects[best])
        all_vps = []
        for k in group:
            all_vps.extend(objects[k]["view_points"])
        # Deduplicate
        vp_unique = {}
        for vp in all_vps:
            key = tuple(np.round(vp["position"], 2))
            if key not in vp_unique or vp["iou"] > vp_unique[key]["iou"]:
                vp_unique[key] = vp
        merged_obj["view_points"] = list(vp_unique.values())
        merged.append(merged_obj)
    return merged


def cap_instances(objects, max_instances=MAX_INSTANCES_PER_CATEGORY):
    """Keep at most max_instances, selecting the most spread out ones."""
    if len(objects) <= max_instances:
        return objects
    # Greedy farthest-point selection
    positions = np.array([o["position"] for o in objects])
    selected = [0]
    for _ in range(max_instances - 1):
        sel_pos = positions[selected]
        dists = np.min(
            np.linalg.norm(positions[:, None, :] - sel_pos[None, :, :], axis=2),
            axis=1,
        )
        dists[selected] = -1  # exclude already selected
        selected.append(int(np.argmax(dists)))
    return [objects[i] for i in selected]


def ubiquity_filter(objects, pathfinder, cat_name):
    """Check if objects are too ubiquitous (everywhere in the scene)."""
    if len(objects) < 3:
        return True  # too few to be ubiquitous

    obj_positions = np.array([o["position"] for o in objects])

    # Sample random navigable points
    close_count = 0
    tested = 0
    for _ in range(UBIQUITY_SAMPLE_POINTS * 3):  # oversample
        pt = np.array(pathfinder.get_random_navigable_point(), dtype=np.float64)
        if np.isnan(pt).any():
            continue
        if pathfinder.island_radius(pt) < ISLAND_RADIUS_LIMIT:
            continue
        min_dist = np.min(np.linalg.norm(obj_positions - pt, axis=1))
        if min_dist < UBIQUITY_DIST:
            close_count += 1
        tested += 1
        if tested >= UBIQUITY_SAMPLE_POINTS:
            break

    if tested == 0:
        return True

    ratio = close_count / tested
    if ratio > UBIQUITY_THRESHOLD:
        print(f"    SKIP '{cat_name}': ubiquitous ({ratio:.0%} of points within {UBIQUITY_DIST}m)")
        return False
    return True


def apply_quality_filters(objects_per_category, pathfinder):
    """Apply all quality filters to detected objects."""
    filtered = {}
    for cat, objects in objects_per_category.items():
        # Merge close instances
        objects = merge_close_instances(objects)

        # Cap max instances
        objects = cap_instances(objects)

        # Min instances check
        if len(objects) < MIN_INSTANCES_PER_CATEGORY:
            continue

        # Ubiquity filter
        if not ubiquity_filter(objects, pathfinder, cat):
            continue

        filtered[cat] = objects

    return filtered


# =====================================================================
# Episode generation
# =====================================================================

def build_goals_by_category(objects_per_category, scene_basename, category_to_id):
    """Convert clustered objects to ObjectNavDatasetV1 goals format."""
    goals_by_category = {}
    for cat, objects in objects_per_category.items():
        goals_key = f"{scene_basename}_{cat}"
        goals = []
        for i, obj in enumerate(objects):
            goals.append({
                "object_id": f"{cat}_{i}",
                "object_name": f"{cat}_{i}",
                "object_name_id": i,
                "object_category": cat,
                "position": obj["position"],
                "view_points": [
                    {
                        "agent_state": {
                            "position": vp["position"],
                            "rotation": vp["rotation"],
                        },
                        "iou": vp["iou"],
                    }
                    for vp in obj["view_points"]
                ],
            })
        goals_by_category[goals_key] = goals
    return goals_by_category


def generate_objectnav_episodes(pathfinder, scene_id, goals_by_category,
                                 category_to_id, num_episodes, seed):
    """Generate ObjectNav episodes for a single scene."""
    np.random.seed(seed)
    pathfinder.seed(seed)

    scene_basename = os.path.basename(scene_id)

    # Collect available categories and their view_point positions
    cat_viewpoints = {}  # cat -> list of viewpoint positions
    for cat in category_to_id:
        goals_key = f"{scene_basename}_{cat}"
        if goals_key not in goals_by_category:
            continue
        vps = []
        for goal in goals_by_category[goals_key]:
            for vp in goal["view_points"]:
                vps.append(np.array(vp["agent_state"]["position"]))
        if vps:
            cat_viewpoints[cat] = vps

    if not cat_viewpoints:
        return []

    available_cats = list(cat_viewpoints.keys())
    episodes = []
    ep_count = 0
    attempts = 0
    max_attempts = num_episodes * 500

    while ep_count < num_episodes and attempts < max_attempts:
        attempts += 1

        # Pick random category
        cat = available_cats[np.random.randint(len(available_cats))]
        vps = cat_viewpoints[cat]

        # Pick random viewpoint as target
        vp_pos = vps[np.random.randint(len(vps))]
        target = np.array(vp_pos, dtype=np.float64)

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

        # Random agent heading
        angle = np.random.uniform(0, 2 * math.pi)
        rotation = [0.0, float(np.sin(angle / 2)), 0.0, float(np.cos(angle / 2))]

        episodes.append({
            "episode_id": str(ep_count),
            "scene_id": scene_id,
            "start_position": [float(source[0]), float(source[1]), float(source[2])],
            "start_rotation": rotation,
            "object_category": cat,
            "goals": [],
            "info": {"geodesic_distance": float(geo)},
        })
        ep_count += 1

    if ep_count < num_episodes:
        print(f"  WARNING: only {ep_count}/{num_episodes} episodes "
              f"(available cats: {available_cats})")

    return episodes


# =====================================================================
# Save functions
# =====================================================================

def save_objectnav_dataset(episodes, goals_by_category, category_to_id, path):
    """Save per-scene ObjectNav dataset."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "category_to_task_category_id": category_to_id,
        "category_to_scene_annotation_category_id": category_to_id,
        "goals_by_category": goals_by_category,
        "episodes": episodes,
    }
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


def save_master_index(category_to_id, path):
    """Save master index with category mappings and empty episodes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "category_to_task_category_id": category_to_id,
        "category_to_scene_annotation_category_id": category_to_id,
        "episodes": [],
    }
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate ObjectNav episodes for GS scenes using SAM + CLIP"
    )
    parser.add_argument("--scenes-root", default=str(SCENES_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--train-episodes", type=int, default=NUM_TRAIN_EPISODES)
    parser.add_argument("--val-episodes", type=int, default=NUM_VAL_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene", type=str, default=None,
                        help="Process a single scene only")
    parser.add_argument("--num-sample-points", type=int, default=NUM_SAMPLE_POINTS)
    parser.add_argument("--visualize", action="store_true",
                        help="Save annotated detection images")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    output_root = (Path(args.output_root) if args.output_root
                   else scenes_root / "episodes" / "objectnav")
    visualize_root = output_root / "visualize"

    import habitat_sim
    from habitat_sim.utils.settings import make_cfg, default_sim_settings

    print(f"Scenes root : {scenes_root}")
    print(f"Output root : {output_root}")
    print(f"Device      : {args.device}")
    print()

    # Load models
    print("Loading SAM...", flush=True)
    mask_gen = load_sam(args.device)
    print("Loading CLIP...", flush=True)
    clip_model, clip_preprocess, text_features, cat_names, n_cat = load_clip(args.device)
    print("Models loaded!\n")

    # Build global category mapping
    category_to_id = {name: i for i, name in enumerate(CATEGORY_PROMPTS.keys())}

    splits = discover_scenes(scenes_root)
    if not splits:
        print("ERROR: No scenes found!", file=sys.stderr)
        sys.exit(1)

    # Track which categories actually appear
    categories_found = set()

    for split, scenes in splits.items():
        num_ep = args.train_episodes if split == "train" else args.val_episodes
        print(f"=== {split} ({len(scenes)} scenes, {num_ep} ep each) ===")

        for idx, info in enumerate(scenes):
            if args.scene and info["name"] != args.scene:
                continue

            name = info["name"]
            seed = args.seed + idx * 10000
            print(f"\n  [{idx+1}/{len(scenes)}] {name}", flush=True)

            # Build simulator
            sim_settings = dict(default_sim_settings)
            sim_settings["scene"] = info["gs_ply"]
            sim_settings["scene_dataset_config_file"] = str(
                scenes_root / f"{split}.scene_dataset_config.json"
            )
            sim_settings["default_agent_navmesh"] = False
            sim_settings["width"] = IMG_SIZE
            sim_settings["height"] = IMG_SIZE
            sim_settings["enable_physics"] = False
            sim_settings["depth_sensor"] = True

            cfg = make_cfg(sim_settings)
            sim = habitat_sim.Simulator(cfg)

            if not sim.pathfinder.is_loaded:
                print("    navmesh NOT loaded, skipping")
                sim.close()
                continue

            area = sim.pathfinder.navigable_area
            print(f"    navmesh area={area:.0f}m²", flush=True)

            # Scan for objects
            vis_dir = str(visualize_root / name) if args.visualize else None
            print(f"    scanning ({args.num_sample_points} pts × {NUM_ROTATIONS} rots)...",
                  flush=True)
            detections = scan_scene_for_objects(
                sim, sim.pathfinder, mask_gen,
                clip_model, clip_preprocess, text_features,
                cat_names, n_cat, args.device,
                args.num_sample_points, NUM_ROTATIONS,
                visualize_dir=vis_dir,
            )
            print(f"    raw detections: {len(detections)}")

            if not detections:
                print("    NO objects detected, skipping")
                sim.close()
                continue

            # Cluster and filter
            objects_per_category = cluster_detections(detections)
            print(f"    clustered: {', '.join(f'{c}={len(o)}' for c, o in objects_per_category.items())}")

            objects_per_category = apply_quality_filters(
                objects_per_category, sim.pathfinder
            )
            print(f"    after filters: {', '.join(f'{c}={len(o)}' for c, o in objects_per_category.items())}")

            if not objects_per_category:
                print("    NO valid objects after filtering, skipping")
                sim.close()
                continue

            categories_found.update(objects_per_category.keys())

            # Build goals
            scene_basename = os.path.basename(info["scene_id"])
            goals_by_category = build_goals_by_category(
                objects_per_category, scene_basename, category_to_id
            )

            # Generate episodes
            episodes = generate_objectnav_episodes(
                sim.pathfinder, info["scene_id"], goals_by_category,
                category_to_id, num_ep, seed,
            )
            sim.close()

            # Save per-scene file
            out_path = str(output_root / split / "content" / f"{name}.json.gz")
            save_objectnav_dataset(episodes, goals_by_category, category_to_id, out_path)

            if episodes:
                dists = [e["info"]["geodesic_distance"] for e in episodes]
                cat_counts = {}
                for e in episodes:
                    cat_counts[e["object_category"]] = cat_counts.get(e["object_category"], 0) + 1
                print(f"    -> {len(episodes)} episodes "
                      f"(dist {min(dists):.1f}-{max(dists):.1f}m, mean {np.mean(dists):.1f}m)")
                print(f"       categories: {dict(cat_counts)}")
            else:
                print("    -> 0 episodes")

        # Save master index
        index_path = str(output_root / split / f"{split}.json.gz")
        save_master_index(category_to_id, index_path)
        print(f"\n  Master index: {index_path}")

    print(f"\nCategories found across all scenes: {sorted(categories_found)}")
    print(f"Category mapping: {category_to_id}")
    print("Done!")


if __name__ == "__main__":
    main()
