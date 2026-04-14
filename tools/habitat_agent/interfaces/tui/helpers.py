"""Shared helpers for the TUI dashboard and ChatAgent.

Hosts the small, dependency-light utilities that were defined at the top
of ``tools/habitat_agent_tui.py`` (formatters, HTTP wrappers, polling,
path mapping, loop control, the regexes/constants used across multiple
parsers). Phase 1 PR 4 moved them here verbatim — no behaviour changes.

The split rationale:
  - ``helpers``    — pure utility functions, no dataclasses, no curses.
  - ``collectors`` — dataclasses + trace parsing + builder functions.
  - ``dashboard``  — curses-only rendering and the keyboard event loop.
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_LOOP_ID_RE = re.compile(r"\b(navloop-[0-9a-f]+)\b")
_TOOL_CALL_RE = re.compile(r"tool call:\s+([A-Za-z0-9_]+)\s+params=(\{.*\})")
_TOOL_DONE_RE = re.compile(r"tool done:\s+([A-Za-z0-9_]+)\s+ok\s+\((\d+)ms\)")
_TOOL_FAIL_RE = re.compile(r"tool fail:\s+([A-Za-z0-9_]+)\s+(.+?)\s+\((\d+)ms\)")
_TOKEN_KV_RE = re.compile(
    r'(?:"|\b)(prompt_tokens|completion_tokens|total_tokens)(?:"|\b)\s*[:=]\s*(\d+)'
)
# Bridge tools: actions that hit the bridge HTTP API (simulator control)
_BRIDGE_TOOLS = frozenset({
    "init", "close", "look", "forward", "turn", "step",
    "navigate", "find_path", "sample_point", "topdown",
    "panorama", "depth_analyze", "analyze_depth", "query_depth",
    "teleport", "export_video", "get_observation", "get_visuals",
    "get_metrics",
})


def _format_loop_distance(nav_status: Dict[str, Any]) -> str:
    """Format the distance column for the loop list / detail panel.

    The return value is ALWAYS labeled with a two/three-letter suffix
    identifying which quantity the number represents, so viewers can
    never confuse heterogeneous distances:

      "5.170 g"  — GT geodesic (authoritative; navmesh + GT run)
      "4.230 e"  — GT euclidean (no navmesh, or navmesh unreachable)
      "3.140 wp" — waypoint distance from the agent's most recent
                   find_path call; NOT a distance to the evaluation
                   goal. Only shown when no GT distance is available
                   so live monitoring still gets a progress signal.
      "-"        — no distance information of any kind

    Decision tree (strictly ordered, no silent merging):

        if has_navmesh and _debug.gt_geodesic_distance is numeric:
            → geodesic (g)
        elif _debug.gt_euclidean_distance is numeric:
            → euclidean (e)     # covers no-navmesh GT runs
        elif nav_status.geodesic_distance is numeric:
            → waypoint (wp)     # agent-queried, non-GT fallback
        else:
            → "-"

    The `has_navmesh` gate prevents a mis-populated gt_geodesic_distance
    from being presented as authoritative when the scene has no
    pathfinder. bool values are rejected explicitly because
    isinstance(True, int) is True in Python and would otherwise render
    as "1.000 g".
    """
    def _is_numeric(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    debug = nav_status.get("_debug") if isinstance(nav_status.get("_debug"), dict) else {}
    has_navmesh = bool(nav_status.get("has_navmesh", False))

    gt_geo = debug.get("gt_geodesic_distance")
    if has_navmesh and _is_numeric(gt_geo):
        return f"{float(gt_geo):.3f} g"

    gt_eu = debug.get("gt_euclidean_distance")
    if _is_numeric(gt_eu):
        return f"{float(gt_eu):.3f} e"

    live_geo = nav_status.get("geodesic_distance")
    if _is_numeric(live_geo):
        return f"{float(live_geo):.3f} wp"

    return "-"


def _is_bridge_tool(tool_name: str) -> bool:
    """Check if a tool name corresponds to a bridge API call."""
    return tool_name.lower().replace("-", "_") in _BRIDGE_TOOLS


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _parse_iso_to_epoch(value: str) -> Optional[float]:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


def _fmt_epoch_hms(epoch_s: Optional[float]) -> str:
    if epoch_s is None:
        return "--:--:--"
    return dt.datetime.fromtimestamp(epoch_s).strftime("%H:%M:%S")


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def _safe_json_load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _tail_lines(path: str, limit: int = 10) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    if limit <= 0:
        return []
    return [line.rstrip("\n") for line in lines[-limit:]]


def _truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _http_get_json(
    url: str, timeout_s: float
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        socket.timeout,
        TimeoutError,
        ConnectionError,
        OSError,
        http.client.HTTPException,
    ) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, f"unexpected payload type: {type(payload).__name__}"
    return payload, None


def _http_post_json(
    url: str, body: Dict[str, Any], timeout_s: float
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    data = json.dumps(body).encode("utf-8")
    try:
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        socket.timeout,
        TimeoutError,
        ConnectionError,
        OSError,
        http.client.HTTPException,
    ) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, f"unexpected payload type: {type(payload).__name__}"
    return payload, None


def _poll_bridge(
    base_url: str, timeout_s: float
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    healthz, err = _http_get_json(f"{base_url}/healthz", timeout_s=timeout_s)
    if err is not None:
        return None, None, err

    runtime_body = {
        "request_id": f"tui-runtime-{int(time.time())}",
        "action": "get_runtime_status",
        "payload": {"include_nav_status": True},
    }
    runtime_resp, runtime_err = _http_post_json(
        f"{base_url}/v1/request", body=runtime_body, timeout_s=timeout_s
    )
    if runtime_err is not None:
        return healthz, None, runtime_err
    if not runtime_resp.get("ok", False):
        return healthz, None, str(runtime_resp.get("error"))
    runtime = runtime_resp.get("result")
    if not isinstance(runtime, dict):
        return healthz, None, "runtime response missing result object"
    return healthz, runtime, None


def _all_loops(runtime: Dict[str, Any]) -> List[Dict[str, Any]]:
    loops: List[Dict[str, Any]] = []
    for key in ("nav_loops", "recently_closed_nav_loops"):
        items = runtime.get(key, [])
        if isinstance(items, list):
            loops.extend([item for item in items if isinstance(item, dict)])
    return loops


def _select_loop(
    runtime: Dict[str, Any], selected_loop_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    loops = _all_loops(runtime)
    if not loops:
        return None
    if selected_loop_id:
        for loop in loops:
            if loop.get("loop_id") == selected_loop_id:
                return loop
    return loops[0]


def _stop_single_nav_loop(
    base_url: str, loop_id: str, timeout_s: float
) -> Tuple[bool, str]:
    request = {
        "request_id": f"tui-stop-{int(time.time())}",
        "action": "stop_nav_loop",
        "payload": {"loop_id": loop_id},
    }
    resp, err = _http_post_json(f"{base_url}/v1/request", request, timeout_s)
    if err is not None:
        return False, f"stop {loop_id} failed: {err}"
    if not resp.get("ok", False):
        return False, f"stop {loop_id} failed: {resp.get('error')}"
    result = resp.get("result", {})
    returncode = result.get("returncode")
    return True, f"stopped {loop_id} returncode={returncode}"


def _stop_all_nav_loops(
    base_url: str, runtime: Optional[Dict[str, Any]], timeout_s: float
) -> Tuple[int, int, List[str]]:
    if not isinstance(runtime, dict):
        return 0, 0, ["runtime unavailable"]
    active = runtime.get("nav_loops", [])
    if not isinstance(active, list):
        return 0, 0, ["runtime nav_loops malformed"]
    attempted = 0
    succeeded = 0
    errors: List[str] = []
    for item in active:
        if not isinstance(item, dict):
            continue
        loop_id = item.get("loop_id")
        if not isinstance(loop_id, str) or not loop_id:
            continue
        attempted += 1
        ok, msg = _stop_single_nav_loop(base_url, loop_id, timeout_s)
        if ok:
            succeeded += 1
        else:
            errors.append(msg)
    return attempted, succeeded, errors


def _guess_workspace_host_from_nav_status_file(nav_status_file: str) -> Optional[str]:
    for marker in ("/data/nav_artifacts/", "/artifacts/habitat-gs/", "/artifacts/"):
        idx = nav_status_file.find(marker)
        if idx > 0:
            return nav_status_file[:idx]
    return None


__all__ = [
    "_ANSI_ESCAPE_RE",
    "_LOOP_ID_RE",
    "_TOOL_CALL_RE",
    "_TOOL_DONE_RE",
    "_TOOL_FAIL_RE",
    "_TOKEN_KV_RE",
    "_BRIDGE_TOOLS",
    "_format_loop_distance",
    "_is_bridge_tool",
    "_now_iso",
    "_strip_ansi",
    "_parse_iso_to_epoch",
    "_fmt_epoch_hms",
    "_fmt_duration",
    "_safe_json_load",
    "_tail_lines",
    "_truncate",
    "_http_get_json",
    "_http_post_json",
    "_poll_bridge",
    "_all_loops",
    "_select_loop",
    "_stop_single_nav_loop",
    "_stop_all_nav_loops",
    "_guess_workspace_host_from_nav_status_file",
]
