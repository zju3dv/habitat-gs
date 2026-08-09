#!/usr/bin/env python3
"""Generate ObjectNav episodes for the InteriorGS scenes from InteriorGS's OWN
ground-truth object annotations (labels.json), instead of the SAM+CLIP vision
pipeline used by generate_objectnav_episodes.py.

Why: the SAM+CLIP pipeline over-detects and mis-localizes objects (e.g. 15
"refrigerators" scattered across a scene, objects placed below the floor). Using
InteriorGS's instance-level semantic 3D bounding boxes gives exact object
positions and categories.

Pipeline per scene:
  1. load <scene>.navmesh -> habitat PathFinder
  2. load InteriorGS labels.json (8-corner oriented boxes, native Z-up frame)
  3. transform boxes native (Right,Back,Up) -> habitat (Y-up): (x,y,z)->(x, z, -y)
  4. map InteriorGS fine labels -> the 12 ObjectNav indoor categories
  5. per object: sample navigable view_points near it on the navmesh, facing it
  6. drop objects that sit below the local floor, float entirely above reach
     (MAX_GOAL_BOTTOM_ABOVE_FLOOR), or are too small to identify (MIN_GOAL_MAX_DIM)
  7. reuse the SAME episode sampler + output schema as the original generator

Each emitted goal carries `src_label` (the raw InteriorGS label) and
`semantic_ins_id` so the many-to-one label collapse stays auditable downstream.

Output schema is byte-for-byte structurally identical to the original
objectnav/<split>/content/<scene>.json.gz files.
"""
import argparse, glob, gzip, json, math, os
import numpy as np

# ------------------------------------------------------------------ constants
# Unified 22-class taxonomy (identical to generate_objectnav_episodes.py output)
UNIFIED_CATEGORY_TO_ID = {
    "car": 0, "bench": 1, "tree": 2, "street lamp": 3, "traffic sign": 4,
    "fire hydrant": 5, "trash can": 6, "bicycle": 7, "potted plant": 8,
    "barrier": 9, "statue": 10, "chair": 11, "sofa": 12, "bed": 13,
    "dining table": 14, "toilet": 15, "sink": 16, "tv": 17, "refrigerator": 18,
    "bookshelf": 19, "cabinet": 20, "lamp": 21,
}

# Episode sampling constants (same as the original generator)
ISLAND_RADIUS_LIMIT = 1.5
CLOSEST_DIST = 1.0
FURTHEST_DIST = 15.0          # indoor
GEO_TO_EUCLID_RATIO = 1.1
RETRIES_PER_TARGET = 10
NUM_TRAIN_EPISODES = 1000
NUM_VAL_EPISODES = 100

# View-point sampling
VP_MIN_DIST = 0.5            # min horizontal stance->object distance (m)
VP_EXTRA_RADIUS = 2.0       # sample up to footprint_radius + this (m)
VP_MAX_PER_OBJECT = 25
VP_SAMPLE_ATTEMPTS = 120
MIN_VIEWPOINTS_PER_OBJECT = 1   # object must have >=1 reachable stance to be a goal
MAX_GOAL_BOTTOM_ABOVE_FLOOR = 2.0
MIN_GOAL_MAX_DIM = 0.25

# InteriorGS fine label -> ObjectNav category. Conservative, furniture-scale only.
# Deliberately EXCLUDED: ceiling/wall lights (downlights, chandelier, ceiling lamp,
# spotlight, track light...), store fixtures beyond a plain shelf, food/decor/small
# items, windows/doors/curtains, carpets, architectural elements.
LABEL_TO_CAT = {
    # chair (seats you walk up to)
    "chair": "chair", "armchair": "chair", "high chair": "chair",
    "stool": "chair", "massage chair": "chair",
    # sofa
    "sofa": "sofa", "multi person sofa": "sofa", "sofa combination": "sofa",
    # bed
    "bed": "bed", "functional bed": "bed", "bed combination": "bed",
    # dining table (generic tables / desks — the schema has one generic-table class)
    "dining table": "dining table", "table": "dining table",
    "teatable": "dining table", "conference table": "dining table",
    "side table": "dining table", "desk": "dining table",
    # toilet
    "toilet": "toilet",
    # sink / washbasin
    "basin": "sink", "hand sink combination": "sink", "basin cabinet": "sink",
    # tv (real flatscreens only; "monitor" = computer monitor, deliberately NOT mapped)
    "tv": "tv",
    # refrigerator / freezer
    "refrigerator": "refrigerator", "freezer": "refrigerator",
    "refrigerator combination": "refrigerator",
    # bookshelf (plain shelf)
    "shelf": "bookshelf",
    # cabinet / wardrobe / nightstand
    "cabinet": "cabinet", "wardrobe": "cabinet", "storage cabinet": "cabinet",
    "display cabinet": "cabinet", "tv cabinet": "cabinet", "bedside table": "cabinet",
    "trash_can": "trash can", "trash can": "trash can",
    # lamp (floor/table lamps only, never ceiling fixtures)
    "floor lamp": "lamp", "table lamp": "lamp",
    # potted plant
    "plant": "potted plant", "flowerpot": "potted plant",
    "landscape plants": "potted plant", "green plant combination": "potted plant",
    "potted plant": "potted plant",
}


# ------------------------------------------------------------------ geometry
def native_to_habitat(P):
    """InteriorGS native (Right, Back, Up / Z-up) -> habitat (Y-up).
    (x, y, z) -> (x, z, -y).  Verified rigorously against the on-disk gs.ply."""
    P = np.asarray(P, dtype=np.float64)
    return np.column_stack([P[:, 0], P[:, 2], -P[:, 1]])


def load_gt_objects(labels_path):
    """Return list of (category, center_xyz_habitat, footprint_radius,
    box_bottom_y, box_top_y, box_max_dim, ins_id, label)."""
    raw = json.load(open(labels_path))
    out = []
    for o in raw:
        if "bounding_box" not in o:           # skip the scene 'room' entry
            continue
        lab = o["label"].strip().lower()
        cat = LABEL_TO_CAT.get(lab)
        if cat is None:
            continue
        corners = np.array([[c["x"], c["y"], c["z"]] for c in o["bounding_box"]], float)
        H = native_to_habitat(corners)                       # 8x3 habitat
        center = H.mean(0)
        xz = H[:, [0, 2]]
        foot_r = float(np.linalg.norm(xz.max(0) - xz.min(0)) / 2.0)
        box_bottom_y = float(H[:, 1].min())
        box_top_y = float(H[:, 1].max())
        box_max_dim = float((H.max(0) - H.min(0)).max())
        out.append((cat, center, foot_r, box_bottom_y, box_top_y, box_max_dim,
                    o.get("ins_id", ""), lab))
    return out


def sample_view_points(pf, center, foot_r, floor_y):
    """Navigable stances near the object on the navmesh, each facing it."""
    import numpy as _np
    cx, cz = float(center[0]), float(center[2])
    radius = foot_r + VP_EXTRA_RADIUS
    vps, seen = [], set()
    for _ in range(VP_SAMPLE_ATTEMPTS):
        p = pf.get_random_navigable_point_near([cx, floor_y, cz], radius, max_tries=10)
        if p is None:
            continue
        p = _np.asarray(p, float)
        if _np.isnan(p).any():
            continue
        d = math.hypot(p[0] - cx, p[2] - cz)
        if d < VP_MIN_DIST or d > radius + 0.5:
            continue
        if pf.island_radius(p) < ISLAND_RADIUS_LIMIT:
            continue
        key = (round(float(p[0]), 1), round(float(p[2]), 1))
        if key in seen:
            continue
        seen.add(key)
        # face the object: habitat forward is -Z; yaw theta about +Y
        theta = math.atan2(-(cx - p[0]), -(cz - p[2]))
        rot = [0.0, float(math.sin(theta / 2)), 0.0, float(math.cos(theta / 2))]
        iou = round(max(0.1, min(1.0, 1.2 / max(d, VP_MIN_DIST))), 4)
        vps.append({"position": [float(p[0]), float(p[1]), float(p[2])],
                    "rotation": rot, "iou": iou})
        if len(vps) >= VP_MAX_PER_OBJECT:
            break
    return vps


# ------------------------------------------------------------------ episode machinery (ported verbatim)
def _geodesic_distance(pf, source, target):
    import habitat_sim
    path = habitat_sim.ShortestPath()
    path.requested_start = source
    path.requested_end = target
    pf.find_path(path)
    return path.geodesic_distance


def _ratio_sample_rate(ratio, threshold):
    return 20 * (ratio - 0.98) ** 2


def is_compatible_episode(source, target, pf, near, far, ratio_thresh):
    if abs(source[1] - target[1]) > 0.5:
        return False, 0.0
    geo = _geodesic_distance(pf, source, target)
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
    if pf.island_radius(source) < ISLAND_RADIUS_LIMIT:
        return False, 0.0
    return True, geo


def build_goals_by_category(objects_per_category, scene_basename):
    goals_by_category = {}
    for cat, objects in objects_per_category.items():
        goals = []
        for i, obj in enumerate(objects):
            goal = {
                "object_id": f"{cat}_{i}",
                "object_name": f"{cat}_{i}",
                "object_name_id": i,
                "object_category": cat,
                "position": obj["position"],
                "view_points": [
                    {"agent_state": {"position": vp["position"], "rotation": vp["rotation"]},
                     "iou": vp["iou"]}
                    for vp in obj["view_points"]
                ],
            }
            goals.append(goal)
        goals_by_category[f"{scene_basename}_{cat}"] = goals
    return goals_by_category


def build_provenance(objects_per_category, scene_basename):
    """object_id -> {src_label, semantic_ins_id}, for a SIDECAR file.

    This deliberately does NOT live inside the goal dicts. habitat-lab builds
    goals with ``ObjectGoal(**serialized_goal)``, whose attrs schema is exactly
    {position, radius, object_id, object_name, object_name_id, object_category,
    room_id, room_name, view_points} — any extra key raises TypeError and the
    whole dataset fails to load. Provenance still matters (`object_category` is a
    many-to-one collapse of the InteriorGS label, so without src_label there is
    no way to tell what a goal actually is), so it is written alongside instead.
    """
    prov = {}
    for cat, objects in objects_per_category.items():
        for i, obj in enumerate(objects):
            entry = {}
            if obj.get("src_label") is not None:
                entry["src_label"] = obj["src_label"]
            if obj.get("ins_id") not in (None, ""):
                entry["semantic_ins_id"] = str(obj["ins_id"])
            if entry:
                entry["object_category"] = cat
                prov[f"{cat}_{i}"] = entry
    return {"scene": scene_basename, "goals": prov}


def generate_episodes(pf, scene_id, goals_by_category, num_episodes, seed):
    np.random.seed(seed)
    pf.seed(seed)
    scene_basename = os.path.basename(scene_id)
    # Categories come from the goal set itself, NOT from UNIFIED_CATEGORY_TO_ID.
    # Iterating the 22-class map silently skips every goal whose category is not
    # one of those 22 — so on a fine-grained relabeled goal set (armchair, teatable,
    # shelf, ...) it would return almost no episodes with no error. Sorted so the
    # category order, and therefore the RNG stream, is deterministic.
    prefix = f"{scene_basename}_"
    cat_viewpoints = {}
    for key in sorted(goals_by_category):
        if not key.startswith(prefix):
            continue
        cat = key[len(prefix):]
        vps = [np.array(vp["agent_state"]["position"])
               for goal in goals_by_category[key] for vp in goal["view_points"]]
        if vps:
            cat_viewpoints[cat] = vps
    if not cat_viewpoints:
        return []
    available = list(cat_viewpoints.keys())
    episodes, ep, attempts = [], 0, 0
    max_attempts = num_episodes * 500
    while ep < num_episodes and attempts < max_attempts:
        attempts += 1
        cat = available[np.random.randint(len(available))]
        vps = cat_viewpoints[cat]
        target = np.array(vps[np.random.randint(len(vps))], dtype=np.float64)
        if np.isnan(target).any() or pf.island_radius(target) < ISLAND_RADIUS_LIMIT:
            continue
        ok = False
        for _ in range(RETRIES_PER_TARGET):
            source = np.array(pf.get_random_navigable_point(), dtype=np.float64)
            if np.isnan(source).any():
                continue
            ok, geo = is_compatible_episode(source, target, pf,
                                            CLOSEST_DIST, FURTHEST_DIST, GEO_TO_EUCLID_RATIO)
            if ok:
                break
        if not ok:
            continue
        angle = np.random.uniform(0, 2 * math.pi)
        episodes.append({
            "episode_id": str(ep), "scene_id": scene_id,
            "start_position": [float(source[0]), float(source[1]), float(source[2])],
            "start_rotation": [0.0, float(np.sin(angle / 2)), 0.0, float(np.cos(angle / 2))],
            "object_category": cat, "goals": [],
            "info": {"geodesic_distance": float(geo)},
        })
        ep += 1
    if ep < num_episodes:
        print(f"  WARNING only {ep}/{num_episodes} episodes (cats {available})")
    return episodes


def _atomic_write_gz_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with gzip.open(tmp, "wt") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def save_dataset(episodes, goals_by_category, path, category_map=None):
    m = UNIFIED_CATEGORY_TO_ID if category_map is None else category_map
    _atomic_write_gz_json({
        "category_to_task_category_id": m,
        "category_to_scene_annotation_category_id": m,
        "goals_by_category": goals_by_category,
        "episodes": episodes,
    }, path)


# ------------------------------------------------------------------ per-scene driver
def process_scene(scene, split, scenes_root, labels_dir, out_root, seed,
                  max_goal_bottom=MAX_GOAL_BOTTOM_ABOVE_FLOOR,
                  min_goal_size=MIN_GOAL_MAX_DIM):
    import habitat_sim
    igs_id = scene[len("interior_"):]
    navmesh = os.path.join(scenes_root, split, scene, f"{scene}.navmesh")
    labels = os.path.join(labels_dir, f"{igs_id}_labels.json")
    if not os.path.exists(navmesh) or not os.path.exists(labels):
        print(f"  SKIP {scene}: missing navmesh/labels"); return None
    pf = habitat_sim.nav.PathFinder()
    if not pf.load_nav_mesh(navmesh) or not pf.is_loaded:
        print(f"  SKIP {scene}: navmesh load failed"); return None
    pf.seed(seed)
    # floor Y from navmesh samples (single-floor scenes)
    ys = [pf.get_random_navigable_point()[1] for _ in range(50)]
    floor_y = float(np.median([y for y in ys if not math.isnan(y)]))

    gt = load_gt_objects(labels)
    objects_per_category = {}
    n_obj, n_kept, n_subfloor, n_toohigh, n_toosmall = 0, 0, 0, 0, 0
    for cat, center, foot_r, box_bottom_y, box_top_y, box_max_dim, ins_id, lab in gt:
        n_obj += 1
        # too small to be visually identifiable from a stance — check before the
        # expensive view_point sampling
        if box_max_dim < min_goal_size:
            n_toosmall += 1
            continue
        vps = sample_view_points(pf, center, foot_r, floor_y)
        if len(vps) < MIN_VIEWPOINTS_PER_OBJECT:
            continue
        # drop objects whose whole box sits below the LOCAL navigable floor (the
        # nearby view_points): these are floor-embedded / mislabelled, not valid
        # standing-furniture goals (e.g. a "table" annotation at Y=-0.4 under a flat floor).
        local_floor = float(np.median([vp["position"][1] for vp in vps]))
        if box_top_y < local_floor - 0.1:
            n_subfloor += 1
            continue
        # ...and the mirror case: objects whose whole box floats above reach
        # (ceiling fixtures, hanging planters, high wall-mounted units).
        if box_bottom_y > local_floor + max_goal_bottom:
            n_toohigh += 1
            continue
        n_kept += 1
        objects_per_category.setdefault(cat, []).append({
            "position": [float(center[0]), float(center[1]), float(center[2])],
            "view_points": vps, "ins_id": ins_id, "src_label": lab,
        })
    scene_id = f"gs_scenes/{split}/{scene}/{scene}.gs.ply"
    gbc = build_goals_by_category(objects_per_category, f"{scene}.gs.ply")
    n_ep = NUM_TRAIN_EPISODES if split == "train" else NUM_VAL_EPISODES
    eps = generate_episodes(pf, scene_id, gbc, n_ep, seed)
    out = os.path.join(out_root, split, "content", f"{scene}.json.gz")
    save_dataset(eps, gbc, out)
    prov_path = os.path.join(out_root, split, "provenance", f"{scene}.json")
    os.makedirs(os.path.dirname(prov_path), exist_ok=True)
    with open(prov_path, "w") as fh:
        json.dump(build_provenance(objects_per_category, f"{scene}.gs.ply"), fh, indent=1)
    cats = {c: len(v) for c, v in objects_per_category.items()}
    print(f"  {scene} [{split}]: GT-objs(mapped)={n_obj} kept={n_kept} "
          f"(subfloor-dropped={n_subfloor} toohigh-dropped={n_toohigh} "
          f"toosmall-dropped={n_toosmall}) cats={cats} episodes={len(eps)}")
    return {"scene": scene, "split": split, "n_mapped": n_obj, "n_kept": n_kept,
            "n_subfloor": n_subfloor, "n_toohigh": n_toohigh, "n_toosmall": n_toosmall,
            "cats": cats, "n_episodes": len(eps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", default="data/scene_datasets/gs_scenes")
    ap.add_argument("--labels-dir", default="/tmp/igs")
    ap.add_argument("--out-root", default="data/scene_datasets/gs_scenes/episodes/objectnav_interiorgs_gt")
    ap.add_argument("--scene", default=None, help="only this scene")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-goal-bottom", type=float, default=MAX_GOAL_BOTTOM_ABOVE_FLOOR,
                    help="drop goals whose bounding box sits entirely more than this many "
                         "metres above the local navigable floor (default: %(default)s). "
                         "Use a large value to disable.")
    ap.add_argument("--min-goal-size", type=float, default=MIN_GOAL_MAX_DIM,
                    help="drop goals whose largest bounding-box dimension is below this "
                         "many metres (default: %(default)s). Use 0 to disable.")
    args = ap.parse_args()

    # discover interior scenes + split from the existing episode tree
    interior = []
    for f in sorted(glob.glob(f"{args.scenes_root}/episodes/objectnav/*/content/interior_*.json.gz")):
        split = f.split("/objectnav/")[1].split("/")[0]
        scn = os.path.basename(f)[:-len(".json.gz")]
        interior.append((scn, split))
    if args.scene:
        interior = [(s, sp) for s, sp in interior if s == args.scene]
    print(f"Processing {len(interior)} interior scenes -> {args.out_root}")

    summary = []
    for i, (scn, split) in enumerate(interior):
        print(f"[{i+1}/{len(interior)}] {scn}")
        r = process_scene(scn, split, args.scenes_root, args.labels_dir, args.out_root,
                          args.seed, args.max_goal_bottom, args.min_goal_size)
        if r:
            summary.append(r)

    # top-level split files (category map only, 0 episodes) for structural parity
    for split in ("train", "val"):
        if any(s["split"] == split for s in summary):
            save_dataset([], {}, os.path.join(args.out_root, split, f"{split}.json.gz"))
    json.dump(summary, open(os.path.join(args.out_root, "_summary.json"), "w"), indent=1)
    print(f"\nDONE. {len(summary)} scenes. summary -> {args.out_root}/_summary.json")


if __name__ == "__main__":
    main()
