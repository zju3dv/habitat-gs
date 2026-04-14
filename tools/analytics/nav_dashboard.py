#!/usr/bin/env python3
"""Lightweight web dashboard for historical nav loop analysis.

Usage:
    python3 tools/analytics/nav_dashboard.py \\
        --artifacts-dir /path/to/artifacts/habitat-gs \\
        --port 8080
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, unquote

SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARD_HTML = SCRIPT_DIR / "dashboard.html"

def _safe_json_load(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        pass
    return entries


def scan_sessions(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """Scan artifacts directory for nav_status files and build session list.

    Searches both the flat layout (legacy ``nav_status_*.json``) and the
    new ``{session_id}/{loop_id}/nav_status.json`` subdirectory layout.
    """
    sessions = []
    # Collect candidate nav_status files from both flat and nested layouts
    nav_files: List[Path] = sorted(artifacts_dir.glob("nav_status_*.json"))
    nav_files.extend(sorted(artifacts_dir.glob("*/*/nav_status.json")))
    for nav_file in nav_files:
        # Skip auxiliary files
        if any(s in nav_file.name for s in [".events.", ".loop.", ".last-good.", ".hook-"]):
            continue

        nav = _safe_json_load(nav_file)
        if not isinstance(nav, dict):
            continue

        debug = nav.get("_debug", {}) or {}
        events_file = Path(str(nav_file) + ".events.jsonl")
        events = _load_jsonl(events_file)
        round_ends = [e for e in events if e.get("phase") == "round_end"]

        session_id = nav.get("session_id", "x")
        # Check for video in the nav_status parent dir (loop dir) and session dir
        nav_parent = nav_file.parent
        has_video = bool(
            list(nav_parent.glob("*color_sensor.mp4"))
            or list(artifacts_dir.glob(f"{session_id}/*color_sensor.mp4"))
            or list(artifacts_dir.glob(f"{session_id}*color_sensor.mp4"))
        )

        # Store relative path from artifacts_dir for URL construction
        try:
            rel_path = nav_file.relative_to(artifacts_dir)
        except ValueError:
            rel_path = Path(nav_file.name)

        # Distances are exposed as two SEPARATE fields so dashboard
        # consumers can label them explicitly and never merge heterogeneous
        # quantities. Both can be None independently; has_navmesh tells
        # the caller which one is authoritative for this session.
        sessions.append({
            "file": str(rel_path),
            "loop_id": nav.get("task_id", ""),
            "task_type": nav.get("task_type", "?"),
            "nav_mode": nav.get("nav_mode", "?"),
            "has_navmesh": bool(nav.get("has_navmesh", False)),
            "goal_description": nav.get("goal_description", ""),
            "outcome": nav.get("status", "?"),
            "rounds": len(round_ends),
            "total_steps": nav.get("total_steps", 0),
            "collisions": nav.get("collisions", 0),
            "gt_end_geodesic_distance": debug.get("gt_geodesic_distance"),
            "gt_end_euclidean_distance": debug.get("gt_euclidean_distance"),
            "finding": nav.get("finding"),
            "has_video": has_video,
            "timestamp": nav.get("updated_at", ""),
        })

    sessions.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
    return sessions


def load_session_detail(artifacts_dir: Path, file_name: str) -> Optional[Dict[str, Any]]:
    """Load full session detail for a specific nav_status file.

    ``file_name`` may be a bare filename (legacy) or a relative path
    like ``{session_id}/{loop_id}/nav_status.json`` (new layout).
    """
    nav_file = artifacts_dir / file_name
    if not nav_file.is_file():
        return None

    nav = _safe_json_load(nav_file)
    if not isinstance(nav, dict):
        return None

    # Events
    events = _load_jsonl(Path(str(nav_file) + ".events.jsonl"))

    # Spatial memory
    spatial_file_raw = nav.get("spatial_memory_file", "")
    spatial_path = spatial_file_raw if spatial_file_raw else ""
    spatial = _safe_json_load(Path(spatial_path)) if spatial_path and Path(spatial_path).is_file() else None

    # Match images to action_history entries by session_id + step number.
    # New layout: {session_id}/step{NNNNNN}_color_sensor.png
    #             {session_id}/pano_{dir}_step{NNNNNN}_color_sensor.png
    # Loop images: {session_id}/{loop_id}/step{NNNNNN}_color_sensor.png
    # Legacy:      {session_id}_step{NNNNNN}_color_sensor.png
    session_id = nav.get("session_id", "")
    nav_parent = nav_file.parent  # loop dir (new) or artifacts_dir (legacy)
    # Candidate directories to search for images (loop dir first, then session dir, then flat)
    image_search_dirs: List[Path] = [nav_parent]
    if session_id:
        session_dir = artifacts_dir / session_id
        if session_dir.is_dir() and session_dir != nav_parent:
            image_search_dirs.append(session_dir)
    if artifacts_dir not in image_search_dirs:
        image_search_dirs.append(artifacts_dir)

    def _find_image(name: str) -> Optional[str]:
        """Search candidate dirs for an image file, return artifacts-relative URL."""
        for d in image_search_dirs:
            candidate = d / name
            if candidate.is_file():
                try:
                    return "/artifacts/" + str(candidate.relative_to(artifacts_dir))
                except ValueError:
                    return "/artifacts/" + candidate.name
        return None

    action_history = nav.get("action_history", [])
    for entry in action_history:
        if not isinstance(entry, dict) or not session_id:
            continue
        step = entry.get("step")
        if step is None:
            continue
        step_str = f"{int(step):06d}"

        # Try visual_path first (if set and valid)
        vp = entry.get("visual_path", "")
        if vp:
            host_path = vp
            if os.path.isfile(host_path):
                try:
                    rel = str(Path(host_path).relative_to(artifacts_dir))
                except ValueError:
                    rel = os.path.basename(host_path)
                entry["visual_url"] = "/artifacts/" + rel
                continue

        # Match by step: regular look image (new + legacy patterns)
        color_url = (
            _find_image(f"step{step_str}_color_sensor.png")
            or _find_image(f"{session_id}_step{step_str}_color_sensor.png")
        )
        if color_url:
            entry["visual_url"] = color_url
        depth_url = (
            _find_image(f"step{step_str}_depth_sensor.png")
            or _find_image(f"{session_id}_step{step_str}_depth_sensor.png")
        )
        if depth_url:
            entry["depth_url"] = depth_url

        # Check for panorama images at this step.
        entry_action = str(entry.get("action", "")).lower()
        is_visual_action = any(k in entry_action for k in ("panorama", "look", "observe", "scan"))
        if is_visual_action:
            pano_urls = []
            step_int = int(step)
            for offset in range(-5, 6):
                candidate_step = f"{max(0, step_int + offset):06d}"
                found = []
                for direction in ("front", "right", "back", "left"):
                    # New pattern: pano_{dir}_step{N}_color_sensor.png
                    purl = (
                        _find_image(f"pano_{direction}_step{candidate_step}_color_sensor.png")
                        or _find_image(f"{session_id}_pano_{direction}_step{candidate_step}_color_sensor.png")
                    )
                    if purl:
                        found.append({"direction": direction, "url": purl})
                if len(found) == 4:
                    pano_urls = found
                    break
            if pano_urls:
                entry["pano_urls"] = pano_urls

    # Find associated video files in loop dir, session dir, and flat dir
    videos = []
    if session_id:
        for search_dir in image_search_dirs:
            for mp4 in search_dir.glob("*color_sensor.mp4"):
                try:
                    rel = str(mp4.relative_to(artifacts_dir))
                except ValueError:
                    rel = mp4.name
                videos.append({
                    "filename": mp4.name,
                    "url": "/artifacts/" + rel,
                    "size_kb": round(mp4.stat().st_size / 1024),
                })

    return {
        "nav_status": nav,
        "events": events,
        "spatial_memory": spatial,
        "videos": videos,
    }


def load_report(artifacts_dir: Path) -> Dict[str, Any]:
    """Load aggregate report from session_stats.jsonl."""
    stats_file = artifacts_dir / "session_stats.jsonl"
    entries = _load_jsonl(stats_file)

    if not entries:
        return {"total": 0, "entries": []}

    total = len(entries)
    reached = sum(1 for e in entries if e.get("outcome") == "reached")

    # Group by task_type x nav_mode
    groups: Dict[str, List[Dict]] = {}
    for e in entries:
        key = f"{e.get('task_type', '?')} x {e.get('nav_mode', '?')}"
        groups.setdefault(key, []).append(e)

    group_stats = []
    for key, group in sorted(groups.items()):
        n = len(group)
        g_reached = sum(1 for e in group if e.get("outcome") == "reached")
        group_stats.append({
            "group": key,
            "count": n,
            "reached": g_reached,
            "success_rate": round(100 * g_reached / n) if n else 0,
        })

    # Tool usage
    tool_totals: Dict[str, int] = {}
    for e in entries:
        for tool, count in (e.get("tool_usage") or {}).items():
            tool_totals[tool] = tool_totals.get(tool, 0) + count
    tool_list = sorted(tool_totals.items(), key=lambda x: -x[1])

    # Capability requests
    cap_requests = []
    for e in entries:
        for r in (e.get("capability_requests") or []):
            if r:
                cap_requests.append(r)

    return {
        "total": total,
        "reached": reached,
        "success_rate": round(100 * reached / total) if total else 0,
        "groups": group_stats,
        "tool_usage": tool_list,
        "capability_requests": cap_requests,
        "entries": entries,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    artifacts_dir: Path  # set by server setup

    def log_message(self, format, *args):
        pass  # suppress default logging

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path, content_type: str) -> None:
        try:
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            self.send_error(404)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)

        if path == "/":
            if DASHBOARD_HTML.is_file():
                self._send_file(DASHBOARD_HTML, "text/html; charset=utf-8")
            else:
                self.send_error(500, "dashboard.html not found")

        elif path == "/api/sessions":
            sessions = scan_sessions(self.artifacts_dir)
            self._send_json(sessions)

        elif path.startswith("/api/session/"):
            file_name = path[len("/api/session/"):]
            detail = load_session_detail(self.artifacts_dir, file_name)
            if detail:
                self._send_json(detail)
            else:
                self.send_error(404, "Session not found")

        elif path == "/api/report":
            report = load_report(self.artifacts_dir)
            self._send_json(report)

        elif path.startswith("/artifacts/"):
            rel_path = unquote(path[len("/artifacts/"):])
            # Security: prevent path traversal — resolve and verify inside artifacts
            file_path = (self.artifacts_dir / rel_path).resolve()
            if (
                file_path.is_file()
                and file_path.is_relative_to(self.artifacts_dir.resolve())
            ):
                ct = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
                self._send_file(file_path, ct)
            else:
                self.send_error(404)
        else:
            self.send_error(404)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nav loop analysis dashboard")
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Path to artifacts/habitat-gs directory",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    args = parser.parse_args()

    if args.artifacts_dir:
        artifacts = Path(args.artifacts_dir)
    else:
        # Auto-detect: NAV_ARTIFACTS_DIR env > project-local data/nav_artifacts > cwd
        env_dir = os.environ.get("NAV_ARTIFACTS_DIR")
        candidates = []
        if env_dir:
            candidates.append(Path(env_dir))
        candidates.extend([
            Path(__file__).resolve().parent.parent.parent / "data" / "nav_artifacts",
            Path.cwd() / "data" / "nav_artifacts",
            Path("/home/ssd/yyh/code/playground/clawdbot-tmp/clawdbot/workspace/artifacts/habitat-gs"),
            Path.cwd() / "artifacts" / "habitat-gs",
        ])
        artifacts = next((c for c in candidates if c.is_dir()), None)
        if artifacts is None:
            print("Cannot auto-detect artifacts directory. Use --artifacts-dir.", file=sys.stderr)
            raise SystemExit(1)

    print(f"Artifacts: {artifacts}")
    flat_count = len(list(artifacts.glob("nav_status_*.json")))
    nested_count = len(list(artifacts.glob("*/*/nav_status.json")))
    print(f"Sessions found: {flat_count + nested_count} (flat={flat_count}, nested={nested_count})")

    DashboardHandler.artifacts_dir = artifacts
    server = HTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
        server.server_close()


if __name__ == "__main__":
    main()
