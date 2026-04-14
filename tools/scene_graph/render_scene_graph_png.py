#!/usr/bin/env python3
"""Render room-object scene graph JSON into PNG visualizations using Pillow.

Outputs:
- room_object_map.png: top-down XY map with room polygons and object centers.
- room_object_contains_compact.png: compact scene->room->(label x count) graph.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        path = Path(p)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


PALETTE = [
    (97, 163, 214),
    (130, 191, 131),
    (240, 186, 95),
    (216, 140, 158),
    (164, 148, 219),
    (90, 190, 193),
    (196, 160, 102),
    (120, 183, 151),
]

RELATION_ORDER = ["above", "below", "front", "behind", "left", "right", "near"]
RELATION_COLORS = {
    "near": (140, 140, 140),
    "left": (80, 130, 200),
    "right": (80, 130, 200),
    "front": (50, 150, 120),
    "behind": (50, 150, 120),
    "above": (200, 95, 70),
    "below": (200, 95, 70),
}


def room_color(idx: int) -> Tuple[int, int, int]:
    return PALETTE[idx % len(PALETTE)]


def color_light(rgb: Tuple[int, int, int], alpha: float = 0.27) -> Tuple[int, int, int]:
    return tuple(int(c * (1 - alpha) + 255 * alpha) for c in rgb)


def _nice_label(s: str, max_len: int = 24) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    fill=(179, 91, 47, 255),
    width: int = 2,
    dash: float = 14.0,
    gap: float = 10.0,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return
    ux = dx / length
    uy = dy / length
    step = dash + gap
    t = 0.0
    while t < length:
        t_end = min(t + dash, length)
        sx = x1 + ux * t
        sy = y1 + uy * t
        ex = x1 + ux * t_end
        ey = y1 + uy * t_end
        draw.line((sx, sy, ex, ey), fill=fill, width=width)
        t += step


def _pick_primary_relation(relations: List[str]) -> str:
    rel_set = set(relations)
    for rel in RELATION_ORDER:
        if rel in rel_set:
            return rel
    return sorted(rel_set)[0] if rel_set else "near"


def _selected_object_relation_pairs(graph: Dict, max_lines: int = 700) -> List[Dict]:
    obj_rel = graph.get("object_object_relations", {})
    pairs = obj_rel.get("pairs", [])
    enriched = []
    for p in pairs:
        rels = [r for r in p.get("relation_groups", []) if r in RELATION_COLORS]
        if not rels:
            continue
        primary = _pick_primary_relation(rels)
        enriched.append(
            {
                "object_a": p["object_a"],
                "object_b": p["object_b"],
                "room_id": p.get("room_id"),
                "distance_xy": float(p.get("distance_xy", 0.0)),
                "relations": rels,
                "primary_relation": primary,
            }
        )
    enriched.sort(
        key=lambda x: (
            1 if x["primary_relation"] == "near" else 0,
            x["distance_xy"],
            x["object_a"],
            x["object_b"],
        )
    )
    return enriched[: max(0, int(max_lines))]


def _world_bounds(graph: Dict) -> Tuple[float, float, float, float]:
    xs = []
    ys = []
    for room in graph.get("rooms", []):
        for p in room.get("profile_xy", []):
            xs.append(float(p[0]))
            ys.append(float(p[1]))
    for obj in graph.get("objects", []):
        pos = obj.get("position_xyz", {})
        if "x" in pos and "y" in pos:
            xs.append(float(pos["x"]))
            ys.append(float(pos["y"]))
    if not xs or not ys:
        return -1.0, 1.0, -1.0, 1.0
    return min(xs), max(xs), min(ys), max(ys)


def _make_mapper(
    xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int, margin: int
):
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    sx = (width - 2 * margin) / dx
    sy = (height - 2 * margin) / dy
    scale = min(sx, sy)
    used_w = dx * scale
    used_h = dy * scale
    x0 = (width - used_w) / 2.0
    y0 = (height - used_h) / 2.0

    def mapper(x: float, y: float) -> Tuple[float, float]:
        px = x0 + (x - xmin) * scale
        py = height - (y0 + (y - ymin) * scale)
        return px, py

    return mapper


def render_room_object_map(
    graph: Dict,
    output_path: Path,
    width: int = 1800,
    height: int = 1200,
    max_object_rel_lines: int = 700,
) -> None:
    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img, "RGBA")
    title_font = load_font(34)
    text_font = load_font(19)
    small_font = load_font(15)

    xmin, xmax, ymin, ymax = _world_bounds(graph)
    map_px = _make_mapper(xmin, xmax, ymin, ymax, width=width - 420, height=height - 100, margin=40)
    x_offset = 20
    y_offset = 60

    def map_xy(x: float, y: float) -> Tuple[float, float]:
        px, py = map_px(x, y)
        return px + x_offset, py + y_offset

    draw.text((20, 14), f"Scene {graph.get('scene_id', 'unknown')} - Room/Object XY Map", fill=(25, 25, 25), font=title_font)

    rooms = graph.get("rooms", [])
    room_id_to_color: Dict[str, Tuple[int, int, int]] = {}
    room_id_to_center_px: Dict[str, Tuple[float, float]] = {}
    object_pos_px: Dict[str, Tuple[float, float]] = {}
    for idx, room in enumerate(rooms):
        rid = room["id"]
        color = room_color(idx)
        room_id_to_color[rid] = color
        pts = [map_xy(float(p[0]), float(p[1])) for p in room.get("profile_xy", [])]
        if len(pts) >= 3:
            draw.polygon(pts, fill=color_light(color) + (210,), outline=color + (255,), width=3)
        c = room.get("centroid_xy", {})
        if "x" in c and "y" in c:
            cx, cy = map_xy(float(c["x"]), float(c["y"]))
            draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(0, 0, 0, 255))
            draw.text((cx + 6, cy + 4), rid, fill=(35, 35, 35), font=text_font)
            room_id_to_center_px[rid] = (cx, cy)

    # Draw room-room adjacency lines.
    seen_pairs = set()
    for pair in graph.get("room_adjacency", {}).get("pairs", []):
        if not pair.get("adjacent", False):
            continue
        a = pair.get("room_a")
        b = pair.get("room_b")
        if a not in room_id_to_center_px or b not in room_id_to_center_px:
            continue
        key = tuple(sorted((a, b)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        p1 = room_id_to_center_px[a]
        p2 = room_id_to_center_px[b]
        _draw_dashed_line(draw, p1, p2, fill=(179, 91, 47, 255), width=3)
        mx = (p1[0] + p2[0]) / 2.0
        my = (p1[1] + p2[1]) / 2.0
        adj_type = str(pair.get("adjacency_type", "adjacent"))
        md = pair.get("min_boundary_distance_xy", "")
        draw.text((mx + 6, my - 12), f"{adj_type} d={md}", fill=(140, 60, 30), font=small_font)

    # Draw object centers.
    for obj in graph.get("objects", []):
        pos = obj.get("position_xyz", {})
        if "x" not in pos or "y" not in pos:
            continue
        px, py = map_xy(float(pos["x"]), float(pos["y"]))
        object_pos_px[obj["id"]] = (px, py)
        rid = obj.get("room_id")
        color = room_id_to_color.get(rid, (150, 150, 150))
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color + (255,))

    # Draw selected object-object relations.
    selected_pairs = _selected_object_relation_pairs(graph, max_lines=max_object_rel_lines)
    drawn_obj_rel_counts: Counter = Counter()
    for pair in selected_pairs:
        a = pair["object_a"]
        b = pair["object_b"]
        if a not in object_pos_px or b not in object_pos_px:
            continue
        rel = pair["primary_relation"]
        color = RELATION_COLORS.get(rel, (140, 140, 140))
        p1 = object_pos_px[a]
        p2 = object_pos_px[b]
        if rel == "near":
            _draw_dashed_line(
                draw,
                p1,
                p2,
                fill=(color[0], color[1], color[2], 120),
                width=1,
                dash=8,
                gap=6,
            )
        else:
            draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(color[0], color[1], color[2], 130), width=2)
            draw.ellipse((p2[0] - 1.5, p2[1] - 1.5, p2[0] + 1.5, p2[1] + 1.5), fill=color + (190,))
        drawn_obj_rel_counts[rel] += 1

    # Draw legend panel.
    panel_x = width - 380
    panel_y = 80
    panel_w = 350
    panel_h = height - 120
    draw.rounded_rectangle(
        (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
        radius=18,
        fill=(255, 255, 255, 240),
        outline=(210, 210, 210, 255),
        width=2,
    )
    draw.text((panel_x + 18, panel_y + 14), "Legend / Stats", fill=(28, 28, 28), font=text_font)
    stats = graph.get("stats", {})
    lines = [
        f"rooms: {stats.get('room_count', len(rooms))}",
        f"room adjacency: {stats.get('room_adjacency_undirected_count', 0)}",
        f"objects(with bbox): {stats.get('object_count_with_bbox', 0)}",
        f"objects(without bbox): {stats.get('object_count_without_bbox', 0)}",
        f"obj-rel edges: {stats.get('object_object_edge_count', 0)}",
        f"obj-rel drawn: {sum(drawn_obj_rel_counts.values())}",
        "",
    ]
    y = panel_y + 48
    for line in lines:
        draw.text((panel_x + 18, y), line, fill=(50, 50, 50), font=small_font)
        y += 22

    for idx, room in enumerate(rooms):
        rid = room["id"]
        color = room_id_to_color[rid]
        draw.rectangle((panel_x + 18, y + 5, panel_x + 34, y + 21), fill=color + (255,))
        label = f"{rid}: {room.get('object_count', 0)} objects"
        draw.text((panel_x + 44, y), label, fill=(40, 40, 40), font=small_font)
        y += 26

    draw.line((panel_x + 18, y + 13, panel_x + 34, y + 13), fill=(179, 91, 47, 255), width=3)
    draw.text((panel_x + 44, y + 4), "room adjacency", fill=(140, 60, 30), font=small_font)
    y += 28

    draw.text((panel_x + 18, y), "Object relations", fill=(28, 28, 28), font=small_font)
    y += 20
    for rel in RELATION_ORDER:
        c = RELATION_COLORS[rel]
        draw.rectangle((panel_x + 18, y + 4, panel_x + 34, y + 18), fill=c + (255,))
        cnt = drawn_obj_rel_counts.get(rel, 0)
        draw.text((panel_x + 44, y), f"{rel}: {cnt}", fill=(60, 60, 60), font=small_font)
        y += 19
        if y > panel_y + panel_h - 250:
            break

    y += 10
    draw.text((panel_x + 18, y), "Top labels per room", fill=(28, 28, 28), font=small_font)
    y += 20
    room_to_objects = graph.get("room_to_objects", {})
    for room in rooms:
        rid = room["id"]
        by = room_to_objects.get(rid, {}).get("objects_by_label", {})
        top = sorted(((k, len(v)) for k, v in by.items()), key=lambda x: x[1], reverse=True)[:5]
        draw.text((panel_x + 18, y), f"{rid}:", fill=(35, 35, 35), font=small_font)
        y += 18
        for label, cnt in top:
            draw.text((panel_x + 30, y), f"- {_nice_label(label, 20)} x{cnt}", fill=(70, 70, 70), font=small_font)
            y += 17
        y += 8
        if y > panel_y + panel_h - 70:
            break

    img.save(output_path)


def _node_box(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], text: str, fill, outline, font):
    draw.rounded_rectangle(xy, radius=12, fill=fill, outline=outline, width=2)
    left, top, right, bottom = xy
    tw = draw.textlength(text, font=font)
    th = font.size if hasattr(font, "size") else 14
    tx = left + (right - left - tw) / 2
    ty = top + (bottom - top - th) / 2 - 2
    draw.text((tx, ty), text, fill=(24, 24, 24), font=font)


def render_compact_contains_graph(
    graph: Dict,
    output_path: Path,
    max_labels_per_room: int = 16,
    width: int = 2200,
    height: int = 1500,
) -> None:
    img = Image.new("RGB", (width, height), (248, 249, 250))
    draw = ImageDraw.Draw(img, "RGBA")
    title_font = load_font(34)
    text_font = load_font(18)
    small_font = load_font(15)

    scene_id = graph.get("scene_id", "unknown")
    draw.text((24, 14), f"Scene {scene_id} - Compact Contains Graph", fill=(25, 25, 25), font=title_font)

    rooms = graph.get("rooms", [])
    objects = graph.get("objects", [])
    room_to_members: Dict[str, List[Dict]] = {r["id"]: [] for r in rooms}
    unassigned = []
    for obj in objects:
        rid = obj.get("room_id")
        if rid in room_to_members:
            room_to_members[rid].append(obj)
        else:
            unassigned.append(obj)

    scene_box = (width // 2 - 110, 80, width // 2 + 110, 145)
    _node_box(draw, scene_box, "scene", fill=(240, 240, 240, 255), outline=(140, 140, 140, 255), font=text_font)
    sx = (scene_box[0] + scene_box[2]) // 2
    sy = scene_box[3]

    room_y_top = 250
    room_gap = width / max(1, len(rooms) + (1 if unassigned else 0))
    room_centers = []
    for i, room in enumerate(rooms):
        cx = int((i + 1) * room_gap)
        room_centers.append((room["id"], cx))

    if unassigned:
        room_centers.append(("unassigned", int((len(rooms) + 1) * room_gap)))

    room_boxes: Dict[str, Tuple[int, int, int, int]] = {}
    for idx, (rid, cx) in enumerate(room_centers):
        if rid == "unassigned":
            fill = (255, 231, 231, 255)
            outline = (191, 100, 100, 255)
            count = len(unassigned)
        else:
            fill = color_light(room_color(idx), alpha=0.35) + (255,)
            outline = room_color(idx) + (255,)
            count = len(room_to_members[rid])
        box = (cx - 130, room_y_top, cx + 130, room_y_top + 72)
        room_boxes[rid] = box
        _node_box(draw, box, f"{rid} ({count})", fill=fill, outline=outline, font=text_font)
        rx = (box[0] + box[2]) // 2
        ry = box[1]
        draw.line((sx, sy, rx, ry), fill=(120, 120, 120, 255), width=2)

    # Draw room-room adjacency edges.
    seen_pairs = set()
    for pair in graph.get("room_adjacency", {}).get("pairs", []):
        if not pair.get("adjacent", False):
            continue
        a = pair.get("room_a")
        b = pair.get("room_b")
        if a not in room_boxes or b not in room_boxes:
            continue
        key = tuple(sorted((a, b)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        box_a = room_boxes[a]
        box_b = room_boxes[b]
        ax = (box_a[0] + box_a[2]) / 2
        ay = (box_a[1] + box_a[3]) / 2
        bx = (box_b[0] + box_b[2]) / 2
        by = (box_b[1] + box_b[3]) / 2
        _draw_dashed_line(draw, (ax, ay), (bx, by), fill=(179, 91, 47, 255), width=3)
        mx = (ax + bx) / 2.0
        my = (ay + by) / 2.0
        draw.text(
            (mx + 6, my - 12),
            str(pair.get("adjacency_type", "adjacent")),
            fill=(140, 60, 30),
            font=small_font,
        )

    # Draw label-count nodes under each room.
    for rid, box in room_boxes.items():
        if rid == "unassigned":
            members = unassigned
        else:
            members = room_to_members[rid]
        label_counts = Counter(obj.get("label", "unknown") for obj in members)
        top = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:max_labels_per_room]

        if not top:
            continue
        cols = 2 if len(top) > 8 else 1
        rows = math.ceil(len(top) / cols)
        col_w = 250
        node_w = 220
        node_h = 40
        gap_x = 26
        gap_y = 14
        total_w = cols * col_w + (cols - 1) * gap_x
        start_x = int((box[0] + box[2]) / 2 - total_w / 2)
        start_y = box[3] + 50

        room_cx = (box[0] + box[2]) // 2
        room_bottom = box[3]
        for i, (label, cnt) in enumerate(top):
            c = i // rows
            r = i % rows
            x1 = start_x + c * (col_w + gap_x)
            y1 = start_y + r * (node_h + gap_y)
            x2 = x1 + node_w
            y2 = y1 + node_h
            draw.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=(255, 255, 255, 250), outline=(170, 170, 170, 255), width=2)
            txt = f"{_nice_label(label, 18)} x{cnt}"
            tw = draw.textlength(txt, font=small_font)
            draw.text((x1 + (node_w - tw) / 2, y1 + 10), txt, fill=(30, 30, 30), font=small_font)
            draw.line((room_cx, room_bottom, x1 + node_w / 2, y1), fill=(160, 160, 160, 255), width=1)

    # Object-object relation summary panel.
    panel_x = width - 390
    panel_y = 90
    panel_w = 360
    panel_h = 420
    draw.rounded_rectangle(
        (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
        radius=16,
        fill=(255, 255, 255, 242),
        outline=(210, 210, 210, 255),
        width=2,
    )
    draw.text((panel_x + 16, panel_y + 12), "Object-Object Relations", fill=(28, 28, 28), font=text_font)
    rel_counts = graph.get("object_object_relations", {}).get("relation_type_counts", {})
    y = panel_y + 44
    stats = graph.get("stats", {})
    draw.text(
        (panel_x + 16, y),
        f"edges={stats.get('object_object_edge_count', 0)}  pairs={stats.get('object_object_pair_count_with_relations', 0)}",
        fill=(65, 65, 65),
        font=small_font,
    )
    y += 24
    for rel in RELATION_ORDER:
        c = RELATION_COLORS[rel]
        draw.rectangle((panel_x + 16, y + 4, panel_x + 32, y + 18), fill=c + (255,))
        draw.text((panel_x + 42, y), f"{rel}: {rel_counts.get(rel, 0)}", fill=(52, 52, 52), font=small_font)
        y += 20

    y += 8
    draw.text((panel_x + 16, y), "Per-room relation counts", fill=(28, 28, 28), font=small_font)
    y += 20
    room_rel_counts: Dict[str, Counter] = defaultdict(Counter)
    for pair in graph.get("object_object_relations", {}).get("pairs", []):
        rid = pair.get("room_id")
        if not rid:
            continue
        for rel in pair.get("relation_groups", []):
            if rel in RELATION_COLORS:
                room_rel_counts[rid][rel] += 1
    for room in rooms:
        rid = room["id"]
        cnts = room_rel_counts.get(rid, Counter())
        top = sorted(cnts.items(), key=lambda x: x[1], reverse=True)[:3]
        summary = ", ".join([f"{k}:{v}" for k, v in top]) if top else "none"
        draw.text((panel_x + 16, y), f"{rid}: {summary}", fill=(60, 60, 60), font=small_font)
        y += 18
        if y > panel_y + panel_h - 24:
            break

    img.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render scene graph JSON to PNG visualizations.")
    parser.add_argument(
        "graph_json",
        type=Path,
        help="Path to room_object_scene_graph.json",
    )
    parser.add_argument(
        "--output-map",
        type=Path,
        default=None,
        help="Output path for room-object map PNG",
    )
    parser.add_argument(
        "--output-compact",
        type=Path,
        default=None,
        help="Output path for compact contains graph PNG",
    )
    parser.add_argument(
        "--max-labels-per-room",
        type=int,
        default=16,
        help="Max label-count nodes shown for each room in compact graph.",
    )
    parser.add_argument(
        "--max-object-rel-lines",
        type=int,
        default=700,
        help="Max object-object relation lines rendered in room_object_map.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = args.graph_json.resolve()
    graph = json.loads(graph_path.read_text())
    scene_dir = graph_path.parent
    output_map = args.output_map or (scene_dir / "room_object_map.png")
    output_compact = args.output_compact or (scene_dir / "room_object_contains_compact.png")

    render_room_object_map(
        graph,
        output_map,
        max_object_rel_lines=max(0, args.max_object_rel_lines),
    )
    render_compact_contains_graph(
        graph,
        output_compact,
        max_labels_per_room=max(1, args.max_labels_per_room),
    )

    print(f"map_png: {output_map}")
    print(f"compact_graph_png: {output_compact}")


if __name__ == "__main__":
    main()
