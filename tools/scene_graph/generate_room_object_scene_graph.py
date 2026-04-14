#!/usr/bin/env python3
"""Generate room-object contains graph from semantic structure/label files.

Inputs:
- structure.json (rooms polygon profiles)
- labels.json (instance labels and optional 3D bounding boxes)

Outputs:
- room_object_scene_graph.json
- room_object_contains_graph.dot
- room_object_contains_compact.dot
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


def _round(v: float, n: int = 6) -> float:
    return round(float(v), n)


def polygon_area(poly: Sequence[Point2D]) -> float:
    area2 = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5


def polygon_centroid(poly: Sequence[Point2D]) -> Point2D:
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(area2) < 1e-12:
        mean_x = sum(p[0] for p in poly) / len(poly)
        mean_y = sum(p[1] for p in poly) / len(poly)
        return mean_x, mean_y
    inv = 1.0 / (3.0 * area2)
    return cx * inv, cy * inv


def point_on_segment(px: float, py: float, p1: Point2D, p2: Point2D, eps: float = 1e-9) -> bool:
    x1, y1 = p1
    x2, y2 = p2
    if (
        px < min(x1, x2) - eps
        or px > max(x1, x2) + eps
        or py < min(y1, y2) - eps
        or py > max(y1, y2) + eps
    ):
        return False
    cross = abs((px - x1) * (y2 - y1) - (py - y1) * (x2 - x1))
    scale = max(1.0, math.hypot(x2 - x1, y2 - y1))
    return cross <= eps * scale


def point_in_polygon(px: float, py: float, poly: Sequence[Point2D], eps: float = 1e-9) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        if point_on_segment(px, py, p1, p2, eps=eps):
            return True
        x1, y1 = p1
        x2, y2 = p2
        intersects = (y1 > py) != (y2 > py)
        if intersects:
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x_cross >= px - eps:
                inside = not inside
    return inside


def point_to_segment_distance(px: float, py: float, p1: Point2D, p2: Point2D) -> float:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / denom
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def point_to_polygon_distance(px: float, py: float, poly: Sequence[Point2D]) -> float:
    if point_in_polygon(px, py, poly):
        return 0.0
    dmin = float("inf")
    for i in range(len(poly)):
        d = point_to_segment_distance(px, py, poly[i], poly[(i + 1) % len(poly)])
        if d < dmin:
            dmin = d
    return dmin


def point_to_line_distance(px: float, py: float, p1: Point2D, p2: Point2D) -> float:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    denom = math.hypot(dx, dy)
    if denom == 0.0:
        return math.hypot(px - x1, py - y1)
    return abs((px - x1) * dy - (py - y1) * dx) / denom


def _orientation(a: Point2D, b: Point2D, c: Point2D) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point2D, b: Point2D, c: Point2D, d: Point2D, eps: float = 1e-9) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (
        o3 > eps and o4 < -eps or o3 < -eps and o4 > eps
    ):
        return True

    if abs(o1) <= eps and point_on_segment(c[0], c[1], a, b, eps=eps):
        return True
    if abs(o2) <= eps and point_on_segment(d[0], d[1], a, b, eps=eps):
        return True
    if abs(o3) <= eps and point_on_segment(a[0], a[1], c, d, eps=eps):
        return True
    if abs(o4) <= eps and point_on_segment(b[0], b[1], c, d, eps=eps):
        return True
    return False


def segment_to_segment_distance(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> float:
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_to_segment_distance(a[0], a[1], c, d),
        point_to_segment_distance(b[0], b[1], c, d),
        point_to_segment_distance(c[0], c[1], a, b),
        point_to_segment_distance(d[0], d[1], a, b),
    )


def segment_collinear_overlap_length(
    a: Point2D,
    b: Point2D,
    c: Point2D,
    d: Point2D,
    line_eps: float = 1e-3,
    angle_eps: float = 1e-3,
) -> float:
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    cdx = d[0] - c[0]
    cdy = d[1] - c[1]
    lab = math.hypot(abx, aby)
    lcd = math.hypot(cdx, cdy)
    if lab <= line_eps or lcd <= line_eps:
        return 0.0

    cross_norm = abs(abx * cdy - aby * cdx) / (lab * lcd)
    if cross_norm > angle_eps:
        return 0.0

    if point_to_line_distance(c[0], c[1], a, b) > line_eps:
        return 0.0
    if point_to_line_distance(d[0], d[1], a, b) > line_eps:
        return 0.0
    if point_to_line_distance(a[0], a[1], c, d) > line_eps:
        return 0.0
    if point_to_line_distance(b[0], b[1], c, d) > line_eps:
        return 0.0

    ux = abx / lab
    uy = aby / lab
    t1 = 0.0
    t2 = lab
    t3 = (c[0] - a[0]) * ux + (c[1] - a[1]) * uy
    t4 = (d[0] - a[0]) * ux + (d[1] - a[1]) * uy
    lo = max(min(t1, t2), min(t3, t4))
    hi = min(max(t1, t2), max(t3, t4))
    return max(0.0, hi - lo)


def polygon_min_boundary_distance(poly_a: Sequence[Point2D], poly_b: Sequence[Point2D]) -> float:
    dmin = float("inf")
    for i in range(len(poly_a)):
        a1 = poly_a[i]
        a2 = poly_a[(i + 1) % len(poly_a)]
        for j in range(len(poly_b)):
            b1 = poly_b[j]
            b2 = poly_b[(j + 1) % len(poly_b)]
            d = segment_to_segment_distance(a1, a2, b1, b2)
            if d < dmin:
                dmin = d
    return dmin


def polygon_shared_boundary_length(poly_a: Sequence[Point2D], poly_b: Sequence[Point2D]) -> float:
    overlap = 0.0
    for i in range(len(poly_a)):
        a1 = poly_a[i]
        a2 = poly_a[(i + 1) % len(poly_a)]
        for j in range(len(poly_b)):
            b1 = poly_b[j]
            b2 = poly_b[(j + 1) % len(poly_b)]
            overlap += segment_collinear_overlap_length(a1, a2, b1, b2)
    return overlap


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_").lower() or "unknown"


def _escape_dot_label(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def load_rooms(structure_data: Dict) -> List[Dict]:
    rooms = []
    raw_rooms = structure_data.get("rooms", [])
    for i, room in enumerate(raw_rooms, start=1):
        raw_profile = room.get("profile", [])
        profile_xy: List[Point2D] = []
        for p in raw_profile:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                profile_xy.append((float(p[0]), float(p[1])))
        if len(profile_xy) < 3:
            continue
        area = polygon_area(profile_xy)
        cx, cy = polygon_centroid(profile_xy)
        rooms.append(
            {
                "id": f"room_{i}",
                "room_index": i,
                "profile_xy": [[_round(x), _round(y)] for x, y in profile_xy],
                "area_xy": _round(area),
                "centroid_xy": {"x": _round(cx), "y": _round(cy)},
                "_profile_tuple": profile_xy,
                "_area_raw": area,
            }
        )
    return rooms


def load_openings(structure_data: Dict) -> List[Dict]:
    openings = []
    for idx, hole in enumerate(structure_data.get("holes", []), start=1):
        raw_profile = hole.get("profile", [])
        profile_xyz = []
        for p in raw_profile:
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                profile_xyz.append((_round(float(p[0])), _round(float(p[1])), _round(float(p[2]))))
        if not profile_xyz:
            continue
        cx = sum(p[0] for p in profile_xyz) / len(profile_xyz)
        cy = sum(p[1] for p in profile_xyz) / len(profile_xyz)
        opening_type = str(hole.get("type", "UNKNOWN")).strip().upper() or "UNKNOWN"
        openings.append(
            {
                "id": f"opening_{idx}",
                "opening_type": opening_type,
                "thickness": _round(float(hole.get("thickness", 0.0))),
                "center_xy": {"x": _round(cx), "y": _round(cy)},
                "profile_xyz": [
                    {"x": p[0], "y": p[1], "z": p[2]}
                    for p in profile_xyz
                ],
            }
        )
    return openings


def compute_room_adjacency(
    rooms: List[Dict],
    openings: List[Dict],
    distance_tolerance: float = 0.25,
    shared_boundary_min: float = 0.05,
    opening_room_distance_tolerance: float = 0.35,
) -> List[Dict]:
    room_profiles = {room["id"]: room["_profile_tuple"] for room in rooms}
    room_ids = [room["id"] for room in rooms]

    pair_openings: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for opening in openings:
        cx = opening["center_xy"]["x"]
        cy = opening["center_xy"]["y"]
        near_rooms = []
        for room_id in room_ids:
            d = point_to_polygon_distance(cx, cy, room_profiles[room_id])
            if d <= opening_room_distance_tolerance:
                near_rooms.append({"room_id": room_id, "distance_xy": _round(d)})
        near_rooms.sort(key=lambda x: (x["distance_xy"], x["room_id"]))
        opening["near_rooms"] = near_rooms
        opening["connects_room_ids"] = sorted([x["room_id"] for x in near_rooms])
        if len(near_rooms) >= 2:
            ids = [x["room_id"] for x in near_rooms]
            for a, b in combinations(sorted(ids), 2):
                pair_openings[(a, b)][opening["opening_type"]] += 1

    pair_metrics = []
    for a, b in combinations(room_ids, 2):
        poly_a = room_profiles[a]
        poly_b = room_profiles[b]
        min_dist = polygon_min_boundary_distance(poly_a, poly_b)
        shared_len = polygon_shared_boundary_length(poly_a, poly_b)
        opening_counter = pair_openings.get((a, b), Counter())
        opening_count = sum(opening_counter.values())

        if opening_count > 0:
            adjacency_type = "opening"
            adjacent = True
        elif shared_len >= shared_boundary_min:
            adjacency_type = "shared_boundary"
            adjacent = True
        elif min_dist <= distance_tolerance:
            adjacency_type = "near_boundary"
            adjacent = True
        else:
            adjacency_type = "none"
            adjacent = False

        pair_metrics.append(
            {
                "room_a": a,
                "room_b": b,
                "adjacent": adjacent,
                "adjacency_type": adjacency_type,
                "min_boundary_distance_xy": _round(min_dist),
                "shared_boundary_length_xy": _round(shared_len),
                "opening_count": opening_count,
                "openings_by_type": dict(sorted(opening_counter.items())),
            }
        )

    pair_metrics.sort(key=lambda x: (x["room_a"], x["room_b"]))
    return pair_metrics


def _bbox_xy_overlap_stats(a: Dict, b: Dict) -> Dict[str, float]:
    ax1, ay1 = a["bbox_min_xyz"]["x"], a["bbox_min_xyz"]["y"]
    ax2, ay2 = a["bbox_max_xyz"]["x"], a["bbox_max_xyz"]["y"]
    bx1, by1 = b["bbox_min_xyz"]["x"], b["bbox_min_xyz"]["y"]
    bx2, by2 = b["bbox_max_xyz"]["x"], b["bbox_max_xyz"]["y"]

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    iou = inter_area / union if union > 0.0 else 0.0
    overlap_min = inter_area / min(area_a, area_b) if min(area_a, area_b) > 0.0 else 0.0
    return {
        "inter_area": inter_area,
        "iou": iou,
        "overlap_min": overlap_min,
    }


def _bbox_3d_intersection_stats(a: Dict, b: Dict) -> Dict[str, float]:
    ax1, ay1, az1 = a["bbox_min_xyz"]["x"], a["bbox_min_xyz"]["y"], a["bbox_min_xyz"]["z"]
    ax2, ay2, az2 = a["bbox_max_xyz"]["x"], a["bbox_max_xyz"]["y"], a["bbox_max_xyz"]["z"]
    bx1, by1, bz1 = b["bbox_min_xyz"]["x"], b["bbox_min_xyz"]["y"], b["bbox_min_xyz"]["z"]
    bx2, by2, bz2 = b["bbox_max_xyz"]["x"], b["bbox_max_xyz"]["y"], b["bbox_max_xyz"]["z"]

    inter_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_y = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_z = max(0.0, min(az2, bz2) - max(az1, bz1))
    inter_vol = inter_x * inter_y * inter_z

    vol_a = max(0.0, (ax2 - ax1) * (ay2 - ay1) * (az2 - az1))
    vol_b = max(0.0, (bx2 - bx1) * (by2 - by1) * (bz2 - bz1))
    union = vol_a + vol_b - inter_vol
    iou = inter_vol / union if union > 0.0 else 0.0
    return {
        "inter_vol": inter_vol,
        "iou": iou,
        "vol_a": vol_a,
        "vol_b": vol_b,
    }


def compute_object_object_relations(
    objects: List[Dict],
    same_room_only: bool = True,
    near_distance_xy: float = 1.2,
    near_top_k: int = 8,
    directional_max_distance_xy: float = 2.5,
    directional_min_delta_xy: float = 0.2,
    vertical_min_delta: float = 0.15,
) -> Dict:
    object_edges: List[Dict] = []
    relation_type_counts: Counter = Counter()
    object_index: Dict[str, List[Dict]] = defaultdict(list)
    pair_records: List[Dict] = []

    room_to_object_indices: Dict[str, List[int]] = defaultdict(list)
    if same_room_only:
        for i, obj in enumerate(objects):
            if obj["room_id"] is None:
                continue
            room_to_object_indices[obj["room_id"]].append(i)
        object_groups: List[List[int]] = list(room_to_object_indices.values())
    else:
        object_groups = [list(range(len(objects)))]

    index_pairs_iter: List[Tuple[int, int]] = []
    for indices in object_groups:
        for i, j in combinations(indices, 2):
            index_pairs_iter.append((i, j))

    near_pair_keys: set = set()
    near_top_k = max(1, int(near_top_k))
    for indices in object_groups:
        if len(indices) < 2:
            continue
        neighbors: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        for i, j in combinations(indices, 2):
            ai, bi = objects[i], objects[j]
            dx = ai["position_xyz"]["x"] - bi["position_xyz"]["x"]
            dy = ai["position_xyz"]["y"] - bi["position_xyz"]["y"]
            dxy = math.hypot(dx, dy)
            if dxy <= near_distance_xy:
                neighbors[i].append((dxy, j))
                neighbors[j].append((dxy, i))
        for i, cand in neighbors.items():
            cand.sort(key=lambda x: (x[0], objects[x[1]]["id"]))
            for _, j in cand[:near_top_k]:
                near_pair_keys.add((min(i, j), max(i, j)))

    def add_edge(
        source: str,
        target: str,
        relation: str,
        room_id: str,
        distance_xy: float,
        distance_xyz: float,
    ) -> None:
        edge = {
            "source": source,
            "target": target,
            "relation": relation,
            "relation_group": "object_object",
            "room_id": room_id,
            "distance_xy": _round(distance_xy),
            "distance_xyz": _round(distance_xyz),
        }
        object_edges.append(edge)
        relation_type_counts[relation] += 1
        object_index[source].append(
            {
                "target": target,
                "relation": relation,
                "distance_xy": _round(distance_xy),
                "distance_xyz": _round(distance_xyz),
            }
        )

    for i, j in index_pairs_iter:
        a = objects[i]
        b = objects[j]
        room_id = a["room_id"] if a["room_id"] == b["room_id"] else None
        if same_room_only and room_id is None:
            continue

        ax, ay, az = a["position_xyz"]["x"], a["position_xyz"]["y"], a["position_xyz"]["z"]
        bx, by, bz = b["position_xyz"]["x"], b["position_xyz"]["y"], b["position_xyz"]["z"]
        dx, dy, dz = ax - bx, ay - by, az - bz
        dist_xy = math.hypot(dx, dy)
        dist_xyz = math.sqrt(dx * dx + dy * dy + dz * dz)
        pair_key = (i, j) if i < j else (j, i)
        near_flag = pair_key in near_pair_keys

        xy_stats = _bbox_xy_overlap_stats(a, b)
        box3d_stats = _bbox_3d_intersection_stats(a, b)

        # Skip obviously unrelated far-away pairs.
        if (
            not near_flag
            and xy_stats["inter_area"] <= 0.0
            and box3d_stats["inter_vol"] <= 0.0
        ):
            continue

        pair_rel_types: set = set()

        overlap_xy_flag = xy_stats["inter_area"] > 0.0
        intersect_3d_flag = box3d_stats["inter_vol"] > 0.0

        if near_flag:
            pair_rel_types.add("near")
            add_edge(a["id"], b["id"], "near", room_id, dist_xy, dist_xyz)
            add_edge(b["id"], a["id"], "near", room_id, dist_xy, dist_xyz)

        if near_flag and dist_xy <= directional_max_distance_xy:
            if abs(dx) >= directional_min_delta_xy:
                if dx > 0:
                    pair_rel_types.add("right")
                    add_edge(a["id"], b["id"], "right", room_id, dist_xy, dist_xyz)
                    add_edge(b["id"], a["id"], "left", room_id, dist_xy, dist_xyz)
                else:
                    pair_rel_types.add("left")
                    add_edge(a["id"], b["id"], "left", room_id, dist_xy, dist_xyz)
                    add_edge(b["id"], a["id"], "right", room_id, dist_xy, dist_xyz)
            if abs(dy) >= directional_min_delta_xy:
                if dy > 0:
                    pair_rel_types.add("front")
                    add_edge(a["id"], b["id"], "front", room_id, dist_xy, dist_xyz)
                    add_edge(b["id"], a["id"], "behind", room_id, dist_xy, dist_xyz)
                else:
                    pair_rel_types.add("behind")
                    add_edge(a["id"], b["id"], "behind", room_id, dist_xy, dist_xyz)
                    add_edge(b["id"], a["id"], "front", room_id, dist_xy, dist_xyz)

        if abs(dz) >= vertical_min_delta and xy_stats["overlap_min"] >= 0.05:
            if dz > 0:
                pair_rel_types.add("above")
                add_edge(a["id"], b["id"], "above", room_id, dist_xy, dist_xyz)
                add_edge(b["id"], a["id"], "below", room_id, dist_xy, dist_xyz)
            else:
                pair_rel_types.add("below")
                add_edge(a["id"], b["id"], "below", room_id, dist_xy, dist_xyz)
                add_edge(b["id"], a["id"], "above", room_id, dist_xy, dist_xyz)

        if pair_rel_types:
            pair_records.append(
                {
                    "object_a": a["id"],
                    "object_b": b["id"],
                    "room_id": room_id,
                    "distance_xy": _round(dist_xy),
                    "distance_xyz": _round(dist_xyz),
                    "delta_xyz": {"x": _round(dx), "y": _round(dy), "z": _round(dz)},
                    "internal_geom": {
                        "overlap_xy": overlap_xy_flag,
                        "intersect_3d": intersect_3d_flag,
                    },
                    "xy_overlap_ratio_min": _round(xy_stats["overlap_min"]),
                    "xy_iou": _round(xy_stats["iou"]),
                    "bbox_iou_3d": _round(box3d_stats["iou"]),
                    "relation_groups": sorted(pair_rel_types),
                }
            )

    for obj_id in object_index:
        object_index[obj_id].sort(key=lambda x: (x["relation"], x["target"]))
    pair_records.sort(key=lambda x: (x["object_a"], x["object_b"]))

    return {
        "parameters": {
            "same_room_only": same_room_only,
            "near_distance_xy": near_distance_xy,
            "near_top_k": near_top_k,
            "directional_max_distance_xy": directional_max_distance_xy,
            "directional_min_delta_xy": directional_min_delta_xy,
            "vertical_min_delta": vertical_min_delta,
        },
        "pair_count_evaluated": len(index_pairs_iter),
        "pair_count_with_relations": len(pair_records),
        "relation_type_counts": dict(sorted(relation_type_counts.items())),
        "pairs": pair_records,
        "index_by_object": dict(sorted(object_index.items())),
        "edges": object_edges,
    }


def _bbox_stats(points: Sequence[Point3D]) -> Dict:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    min_xyz = {"x": _round(min(xs)), "y": _round(min(ys)), "z": _round(min(zs))}
    max_xyz = {"x": _round(max(xs)), "y": _round(max(ys)), "z": _round(max(zs))}
    center = {
        "x": _round(sum(xs) / len(xs)),
        "y": _round(sum(ys) / len(ys)),
        "z": _round(sum(zs) / len(zs)),
    }
    size = {
        "x": _round(max_xyz["x"] - min_xyz["x"]),
        "y": _round(max_xyz["y"] - min_xyz["y"]),
        "z": _round(max_xyz["z"] - min_xyz["z"]),
    }
    return {"center_xyz": center, "min_xyz": min_xyz, "max_xyz": max_xyz, "size_xyz": size}


def load_objects(labels_path: Path) -> Tuple[List[Dict], List[Dict], Counter]:
    labels = json.loads(labels_path.read_text())
    objects = []
    without_bbox = []
    missing_label_counter: Counter = Counter()

    for idx, item in enumerate(labels):
        ins_id = str(item.get("ins_id", "")).strip()
        label = str(item.get("label", "unknown")).strip() or "unknown"
        bbox = item.get("bounding_box")
        if not isinstance(bbox, list):
            without_bbox.append({"ins_id": ins_id, "label": label})
            missing_label_counter[label] += 1
            continue
        points: List[Point3D] = []
        for p in bbox:
            if isinstance(p, dict) and {"x", "y", "z"}.issubset(p.keys()):
                points.append((float(p["x"]), float(p["y"]), float(p["z"])))
        if len(points) < 2:
            without_bbox.append({"ins_id": ins_id, "label": label})
            missing_label_counter[label] += 1
            continue
        stats = _bbox_stats(points)
        object_id = f"obj_{ins_id}" if ins_id else f"obj_idx_{idx}"
        objects.append(
            {
                "id": object_id,
                "ins_id": ins_id,
                "label": label,
                "position_xyz": stats["center_xyz"],
                "bbox_min_xyz": stats["min_xyz"],
                "bbox_max_xyz": stats["max_xyz"],
                "bbox_size_xyz": stats["size_xyz"],
                "room_id": None,
                "room_assignment": "unassigned",
                "room_distance_xy": None,
            }
        )
    return objects, without_bbox, missing_label_counter


def assign_objects_to_rooms(objects: List[Dict], rooms: List[Dict], tolerance: float = 0.25) -> None:
    room_profiles = [(r["id"], r["_profile_tuple"], r["_area_raw"]) for r in rooms]

    for obj in objects:
        px = obj["position_xyz"]["x"]
        py = obj["position_xyz"]["y"]
        containing = []
        for room_id, poly, area in room_profiles:
            if point_in_polygon(px, py, poly):
                containing.append((room_id, area))
        if containing:
            room_id = min(containing, key=lambda x: x[1])[0]
            obj["room_id"] = room_id
            obj["room_assignment"] = "inside_polygon"
            obj["room_distance_xy"] = 0.0
            continue

        if not rooms:
            obj["room_id"] = None
            obj["room_assignment"] = "unassigned_no_rooms"
            obj["room_distance_xy"] = None
            continue

        nearest_room_id = None
        nearest_dist = float("inf")
        for room_id, poly, _ in room_profiles:
            d = point_to_polygon_distance(px, py, poly)
            if d < nearest_dist:
                nearest_dist = d
                nearest_room_id = room_id
        if nearest_room_id is not None and nearest_dist <= tolerance:
            obj["room_id"] = nearest_room_id
            obj["room_assignment"] = "nearest_room_within_tolerance"
            obj["room_distance_xy"] = _round(nearest_dist)
        else:
            obj["room_id"] = None
            obj["room_assignment"] = "unassigned"
            obj["room_distance_xy"] = _round(nearest_dist) if math.isfinite(nearest_dist) else None

    # Keep room order stable and deterministic for downstream traversal.
    for room in rooms:
        room.pop("_profile_tuple", None)
        room.pop("_area_raw", None)
        room["object_ids"] = sorted([o["id"] for o in objects if o["room_id"] == room["id"]])
        room["object_count"] = len(room["object_ids"])


def build_scene_graph(
    scene_dir: Path,
    tolerance: float = 0.25,
    room_adj_distance_tolerance: float = 0.25,
    room_adj_shared_boundary_min: float = 0.05,
    opening_room_distance_tolerance: float = 0.35,
    obj_rel_same_room_only: bool = True,
    obj_rel_near_distance_xy: float = 1.2,
    obj_rel_near_top_k: int = 8,
    obj_rel_directional_max_distance_xy: float = 2.5,
    obj_rel_directional_min_delta_xy: float = 0.2,
    obj_rel_vertical_min_delta: float = 0.15,
) -> Dict:
    structure_path = scene_dir / "structure.json"
    labels_path = scene_dir / "labels.json"
    if not structure_path.exists():
        raise FileNotFoundError(f"Missing file: {structure_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing file: {labels_path}")

    structure_data = json.loads(structure_path.read_text())
    rooms = load_rooms(structure_data)
    openings = load_openings(structure_data)
    room_adjacency_pairs = compute_room_adjacency(
        rooms,
        openings,
        distance_tolerance=room_adj_distance_tolerance,
        shared_boundary_min=room_adj_shared_boundary_min,
        opening_room_distance_tolerance=opening_room_distance_tolerance,
    )
    objects, objects_without_bbox, missing_bbox_counter = load_objects(labels_path)
    assign_objects_to_rooms(objects, rooms, tolerance=tolerance)
    object_object_relations = compute_object_object_relations(
        objects,
        same_room_only=obj_rel_same_room_only,
        near_distance_xy=obj_rel_near_distance_xy,
        near_top_k=obj_rel_near_top_k,
        directional_max_distance_xy=obj_rel_directional_max_distance_xy,
        directional_min_delta_xy=obj_rel_directional_min_delta_xy,
        vertical_min_delta=obj_rel_vertical_min_delta,
    )

    adjacency_list: Dict[str, List[str]] = {room["id"]: [] for room in rooms}
    for p in room_adjacency_pairs:
        if p["adjacent"]:
            adjacency_list[p["room_a"]].append(p["room_b"])
            adjacency_list[p["room_b"]].append(p["room_a"])
    for rid in adjacency_list:
        adjacency_list[rid] = sorted(adjacency_list[rid])
    for room in rooms:
        room["adjacent_room_ids"] = adjacency_list.get(room["id"], [])

    scene_id = scene_dir.name
    nodes = [
        {
            "id": "scene",
            "type": "scene",
            "scene_id": scene_id,
        }
    ]
    edges = []

    for room in rooms:
        nodes.append(
            {
                "id": room["id"],
                "type": "room",
                "room_index": room["room_index"],
                "area_xy": room["area_xy"],
                "centroid_xy": room["centroid_xy"],
                "profile_xy": room["profile_xy"],
                "object_count": room["object_count"],
                "adjacent_room_ids": room["adjacent_room_ids"],
            }
        )
        edges.append(
            {
                "source": "scene",
                "target": room["id"],
                "relation": "contains",
            }
        )

    for pair in room_adjacency_pairs:
        if not pair["adjacent"]:
            continue
        for src, dst in ((pair["room_a"], pair["room_b"]), (pair["room_b"], pair["room_a"])):
            edges.append(
                {
                    "source": src,
                    "target": dst,
                    "relation": "adjacent_to",
                    "adjacency_type": pair["adjacency_type"],
                    "min_boundary_distance_xy": pair["min_boundary_distance_xy"],
                    "shared_boundary_length_xy": pair["shared_boundary_length_xy"],
                    "opening_count": pair["opening_count"],
                    "openings_by_type": pair["openings_by_type"],
                }
            )

    for obj in objects:
        nodes.append(
            {
                "id": obj["id"],
                "type": "object",
                "ins_id": obj["ins_id"],
                "label": obj["label"],
                "position_xyz": obj["position_xyz"],
                "bbox_min_xyz": obj["bbox_min_xyz"],
                "bbox_max_xyz": obj["bbox_max_xyz"],
                "bbox_size_xyz": obj["bbox_size_xyz"],
                "room_id": obj["room_id"],
                "room_assignment": obj["room_assignment"],
                "room_distance_xy": obj["room_distance_xy"],
            }
        )
        if obj["room_id"] is None:
            edges.append(
                {
                    "source": "scene",
                    "target": obj["id"],
                    "relation": "contains_unassigned_object",
                }
            )
        else:
            rel = "contains" if obj["room_assignment"] == "inside_polygon" else "contains_near_boundary"
            edges.append({"source": obj["room_id"], "target": obj["id"], "relation": rel})

    edges.extend(object_object_relations["edges"])

    room_to_objects: Dict[str, Dict] = {}
    for room in rooms:
        members = [o for o in objects if o["room_id"] == room["id"]]
        by_label: Dict[str, List[str]] = defaultdict(list)
        for o in members:
            by_label[o["label"]].append(o["id"])
        room_to_objects[room["id"]] = {
            "object_count": len(members),
            "object_ids": sorted([o["id"] for o in members]),
            "objects_by_label": {k: sorted(v) for k, v in sorted(by_label.items())},
        }

    unassigned = [o for o in objects if o["room_id"] is None]
    room_to_objects["unassigned"] = {
        "object_count": len(unassigned),
        "object_ids": sorted([o["id"] for o in unassigned]),
    }

    index_by_label: Dict[str, List[Dict]] = defaultdict(list)
    for o in objects:
        index_by_label[o["label"]].append(
            {
                "id": o["id"],
                "ins_id": o["ins_id"],
                "room_id": o["room_id"],
                "position_xyz": o["position_xyz"],
                "room_assignment": o["room_assignment"],
            }
        )
    for label in index_by_label:
        index_by_label[label].sort(key=lambda x: x["id"])

    graph = {
        "scene_id": scene_id,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "coordinate_system": {
            "horizontal_axes": ["x", "y"],
            "vertical_axis": "z",
            "unit": "meter",
        },
        "stats": {
            "room_count": len(rooms),
            "object_count_with_bbox": len(objects),
            "object_count_without_bbox": len(objects_without_bbox),
            "room_adjacency_undirected_count": sum(1 for p in room_adjacency_pairs if p["adjacent"]),
            "object_object_pair_count_with_relations": object_object_relations["pair_count_with_relations"],
            "object_object_edge_count": len(object_object_relations["edges"]),
            "edge_count": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
        "rooms": rooms,
        "openings": openings,
        "room_adjacency": {
            "pairs": room_adjacency_pairs,
            "adjacency_list": adjacency_list,
            "parameters": {
                "distance_tolerance": room_adj_distance_tolerance,
                "shared_boundary_min": room_adj_shared_boundary_min,
                "opening_room_distance_tolerance": opening_room_distance_tolerance,
            },
        },
        "object_object_relations": {
            "parameters": object_object_relations["parameters"],
            "pair_count_evaluated": object_object_relations["pair_count_evaluated"],
            "pair_count_with_relations": object_object_relations["pair_count_with_relations"],
            "relation_type_counts": object_object_relations["relation_type_counts"],
            "pairs": object_object_relations["pairs"],
            "index_by_object": object_object_relations["index_by_object"],
        },
        "objects": objects,
        "room_to_objects": room_to_objects,
        "object_index_by_label": dict(sorted(index_by_label.items())),
        "objects_without_bbox": {
            "count": len(objects_without_bbox),
            "counts_by_label": dict(sorted(missing_bbox_counter.items())),
            "instances": objects_without_bbox,
        },
    }
    return graph


def write_full_dot(graph: Dict, output_path: Path) -> None:
    room_ids = [r["id"] for r in graph["rooms"]]
    room_set = set(room_ids)
    object_map = {o["id"]: o for o in graph["objects"]}
    lines = [
        "digraph RoomObjectContains {",
        "  rankdir=LR;",
        "  graph [fontname=\"Helvetica\"];",
        "  node [fontname=\"Helvetica\"];",
        "  edge [fontname=\"Helvetica\"];",
        "  scene [shape=box, style=filled, fillcolor=\"#f2f2f2\", label=\"scene\"];",
    ]
    for room in graph["rooms"]:
        label = _escape_dot_label(
            f"{room['id']}\\nobjects={room['object_count']}\\narea={room['area_xy']:.2f}"
        )
        lines.append(
            f'  {room["id"]} [shape=box, style=filled, fillcolor="#d8ecff", label="{label}"];'
        )
    for obj in graph["objects"]:
        obj_label = _escape_dot_label(f"{obj['label']}\\n{obj['id']}")
        lines.append(
            f'  {obj["id"]} [shape=ellipse, style=filled, fillcolor="#f7f7f7", label="{obj_label}"];'
        )
    rendered_room_adjacency = set()
    for edge in graph["edges"]:
        src = edge["source"]
        dst = edge["target"]
        rel = edge["relation"]
        if src == "scene" and dst in room_set:
            lines.append(f'  scene -> {dst} [label="{rel}"];')
        elif src in room_set and dst in room_set and rel == "adjacent_to":
            pair = tuple(sorted((src, dst)))
            if pair in rendered_room_adjacency:
                continue
            rendered_room_adjacency.add(pair)
            adj_type = edge.get("adjacency_type", "adjacent")
            md = edge.get("min_boundary_distance_xy", "")
            label = _escape_dot_label(f"adjacent_to\\n{adj_type}\\nd={md}")
            lines.append(
                f'  {pair[0]} -> {pair[1]} [dir=both, color="#b35b2f", style=dashed, penwidth=2, label="{label}"];'
            )
        elif src in room_set and dst in object_map:
            lines.append(f'  {src} -> {dst} [label="{rel}"];')
        elif src == "scene" and dst in object_map:
            lines.append(f'  scene -> {dst} [label="{rel}"];')
    lines.append("}")
    output_path.write_text("\n".join(lines) + "\n")


def write_compact_dot(graph: Dict, output_path: Path) -> None:
    room_set = {r["id"] for r in graph["rooms"]}
    lines = [
        "digraph RoomObjectContainsCompact {",
        "  rankdir=LR;",
        "  graph [fontname=\"Helvetica\"];",
        "  node [fontname=\"Helvetica\"];",
        "  edge [fontname=\"Helvetica\"];",
        "  scene [shape=box, style=filled, fillcolor=\"#f2f2f2\", label=\"scene\"];",
    ]
    for room in graph["rooms"]:
        room_id = room["id"]
        lines.append(
            f'  {room_id} [shape=box, style=filled, fillcolor="#d8ecff", label="{room_id}\\nobjects={room["object_count"]}"];'
        )
        lines.append(f'  scene -> {room_id} [label="contains"];')

    rendered_room_adjacency = set()
    for edge in graph["edges"]:
        src = edge["source"]
        dst = edge["target"]
        if src in room_set and dst in room_set and edge["relation"] == "adjacent_to":
            pair = tuple(sorted((src, dst)))
            if pair in rendered_room_adjacency:
                continue
            rendered_room_adjacency.add(pair)
            adj_type = edge.get("adjacency_type", "adjacent")
            lines.append(
                f'  {pair[0]} -> {pair[1]} [dir=both, color="#b35b2f", style=dashed, penwidth=2, label="{adj_type}"];'
            )

    for room in graph["rooms"]:
        room_id = room["id"]
        members = [o for o in graph["objects"] if o["room_id"] == room_id]
        counts = Counter(o["label"] for o in members)
        for label, count in sorted(counts.items()):
            node_id = f"{room_id}__{_slug(label)}"
            node_label = _escape_dot_label(f"{label} x{count}")
            lines.append(f'  {node_id} [shape=ellipse, label="{node_label}"];')
            lines.append(f'  {room_id} -> {node_id} [label="contains"];')

    unassigned = [o for o in graph["objects"] if o["room_id"] is None]
    if unassigned:
        counts = Counter(o["label"] for o in unassigned)
        lines.append('  unassigned [shape=box, style=filled, fillcolor="#ffe6e6", label="unassigned"];')
        lines.append('  scene -> unassigned [label="contains"];')
        for label, count in sorted(counts.items()):
            node_id = f"unassigned__{_slug(label)}"
            node_label = _escape_dot_label(f"{label} x{count}")
            lines.append(f'  {node_id} [shape=ellipse, label="{node_label}"];')
            lines.append(f'  unassigned -> {node_id} [label="contains"];')

    lines.append("}")
    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build room-object contains graph from semantic scene files."
    )
    parser.add_argument(
        "scene_dir",
        type=Path,
        help="Scene folder path containing structure.json and labels.json",
    )
    parser.add_argument(
        "--assign-tolerance",
        type=float,
        default=0.25,
        help="Fallback nearest-room assignment tolerance in meters (XY plane).",
    )
    parser.add_argument(
        "--room-adj-distance-tolerance",
        type=float,
        default=0.25,
        help="Room adjacency threshold on polygon boundary distance (meters).",
    )
    parser.add_argument(
        "--room-adj-shared-boundary-min",
        type=float,
        default=0.05,
        help="Minimum shared boundary length (meters) to mark two rooms as adjacent.",
    )
    parser.add_argument(
        "--opening-room-distance-tolerance",
        type=float,
        default=0.35,
        help="Distance threshold to link an opening center to nearby room boundaries (meters).",
    )
    parser.add_argument(
        "--obj-rel-cross-room",
        action="store_true",
        help="If set, compute object-object relations across rooms (default is same-room only).",
    )
    parser.add_argument(
        "--obj-rel-near-distance-xy",
        type=float,
        default=1.2,
        help="Object-object near relation threshold in XY plane (meters).",
    )
    parser.add_argument(
        "--obj-rel-near-top-k",
        type=int,
        default=8,
        help="For each object, keep at most top-K nearest near neighbors within threshold.",
    )
    parser.add_argument(
        "--obj-rel-directional-max-distance-xy",
        type=float,
        default=2.5,
        help="Only emit directional object relations when XY distance is below this threshold.",
    )
    parser.add_argument(
        "--obj-rel-directional-min-delta-xy",
        type=float,
        default=0.2,
        help="Minimum XY delta to emit left/right or front/behind object relations.",
    )
    parser.add_argument(
        "--obj-rel-vertical-min-delta",
        type=float,
        default=0.15,
        help="Minimum Z delta to emit above/below relation when XY overlaps.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output path for graph JSON (default: <scene_dir>/room_object_scene_graph.json).",
    )
    parser.add_argument(
        "--output-dot",
        type=Path,
        default=None,
        help="Output path for full graph DOT (default: <scene_dir>/room_object_contains_graph.dot).",
    )
    parser.add_argument(
        "--output-compact-dot",
        type=Path,
        default=None,
        help=(
            "Output path for compact graph DOT "
            "(default: <scene_dir>/room_object_contains_compact.dot)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    output_json = args.output_json or (scene_dir / "room_object_scene_graph.json")
    output_dot = args.output_dot or (scene_dir / "room_object_contains_graph.dot")
    output_compact_dot = args.output_compact_dot or (
        scene_dir / "room_object_contains_compact.dot"
    )

    graph = build_scene_graph(
        scene_dir=scene_dir,
        tolerance=args.assign_tolerance,
        room_adj_distance_tolerance=args.room_adj_distance_tolerance,
        room_adj_shared_boundary_min=args.room_adj_shared_boundary_min,
        opening_room_distance_tolerance=args.opening_room_distance_tolerance,
        obj_rel_same_room_only=not args.obj_rel_cross_room,
        obj_rel_near_distance_xy=args.obj_rel_near_distance_xy,
        obj_rel_near_top_k=args.obj_rel_near_top_k,
        obj_rel_directional_max_distance_xy=args.obj_rel_directional_max_distance_xy,
        obj_rel_directional_min_delta_xy=args.obj_rel_directional_min_delta_xy,
        obj_rel_vertical_min_delta=args.obj_rel_vertical_min_delta,
    )
    output_json.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n")
    write_full_dot(graph, output_dot)
    write_compact_dot(graph, output_compact_dot)

    print(f"scene_id: {graph['scene_id']}")
    print(f"rooms: {graph['stats']['room_count']}")
    print(f"room_adjacency_undirected: {graph['stats']['room_adjacency_undirected_count']}")
    print(f"objects_with_bbox: {graph['stats']['object_count_with_bbox']}")
    print(f"objects_without_bbox: {graph['stats']['object_count_without_bbox']}")
    print(
        "object_object_pairs_with_relations: "
        f"{graph['stats']['object_object_pair_count_with_relations']}"
    )
    print(f"object_object_edges: {graph['stats']['object_object_edge_count']}")
    print(f"json: {output_json}")
    print(f"dot: {output_dot}")
    print(f"compact_dot: {output_compact_dot}")


if __name__ == "__main__":
    main()
