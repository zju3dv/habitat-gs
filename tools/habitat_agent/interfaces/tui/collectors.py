"""TUI dataclasses, trace collection, parsers, and panel builders.

Hosts everything that turns raw bridge / nav_agent output into the
structured data the curses dashboard renders. Phase 1 PR 4 moved this
out of ``tools/habitat_agent_tui.py`` verbatim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .helpers import (
    _LOOP_ID_RE,
    _TOKEN_KV_RE,
    _TOOL_CALL_RE,
    _TOOL_DONE_RE,
    _TOOL_FAIL_RE,
    _fmt_duration,
    _guess_workspace_host_from_nav_status_file,
    _now_iso,
    _parse_iso_to_epoch,
    _safe_json_load,
    _strip_ansi,
    _tail_lines,
    _truncate,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TraceSourceStatus:
    requested_mode: str = "auto"
    resolved_mode: str = "none"
    available: bool = False
    error: Optional[str] = None


@dataclass
class ToolTraceEvent:
    ts_s: Optional[float]
    ts_text: str
    kind: str  # call|done|fail
    tool: str
    command: Optional[str]
    result: Optional[str]
    latency_ms: Optional[int]
    loop_id_hint: Optional[str]
    raw: str


@dataclass
class RoundSnapshot:
    round: int
    start_ts_s: Optional[float] = None
    end_ts_s: Optional[float] = None
    status: Optional[str] = None
    nav_phase: Optional[str] = None
    total_steps: Optional[int] = None
    geodesic_distance: Optional[float] = None
    state_version: Optional[int] = None


@dataclass
class TokenUsage:
    available: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0  # number of LLM API calls
    reason: str = "gateway logs did not expose token usage fields"


@dataclass
class DashboardState:
    healthz: Optional[Dict[str, Any]] = None
    runtime: Optional[Dict[str, Any]] = None
    selected_loop_id: Optional[str] = None
    selected_loop: Optional[Dict[str, Any]] = None
    selected_nav_status: Optional[Dict[str, Any]] = None
    round_snapshots: List[RoundSnapshot] = field(default_factory=list)
    trace_source_status: TraceSourceStatus = field(default_factory=TraceSourceStatus)
    trace_events: List[ToolTraceEvent] = field(default_factory=list)
    filtered_trace_events: List[ToolTraceEvent] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    raw_log_tail: List[str] = field(default_factory=list)
    loop_lines: List[str] = field(default_factory=list)
    round_lines: List[str] = field(default_factory=list)
    tool_lines: List[str] = field(default_factory=list)
    memory_lines: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)
    current_round_id: Optional[int] = None
    current_round_call_count: int = 0
    last_poll_error: Optional[str] = None
    messages: List[str] = field(default_factory=list)
    show_raw_log: bool = False
    kill_all_confirm_until: float = 0.0

    def push_message(self, text: str) -> None:
        stamped = f"{_now_iso()} {text}"
        self.messages.append(stamped)
        if len(self.messages) > 10:
            self.messages = self.messages[-10:]


# ---------------------------------------------------------------------------
# Trace collector
# ---------------------------------------------------------------------------


class AgentTraceCollector:
    """Collect trace from multiple sources simultaneously: nav_agent trace + MCP log."""

    MCP_LOG_PATH = "/tmp/habitat_agent_mcp.log"

    def __init__(
        self,
        requested_mode: str,
        gateway_container: str,
        gateway_log_file: str,
        tail_lines: int,
    ) -> None:
        self.requested_mode = requested_mode
        self.gateway_container = gateway_container
        self.gateway_log_file = gateway_log_file
        self.tail_lines = max(100, tail_lines)
        self._resolved_mode: Optional[str] = None
        # Set by poll loop when a nav loop with .trace.jsonl is active
        self.nav_agent_trace_file: Optional[str] = None

    def _collect_trace(self) -> Tuple[List[str], Optional[str]]:
        """Collect from nav_agent.py .trace.jsonl file."""
        if not self.nav_agent_trace_file or not os.path.isfile(self.nav_agent_trace_file):
            return [], "nav_agent trace file not available"
        return _tail_lines(self.nav_agent_trace_file, limit=self.tail_lines), None

    def _collect_mcp_log(self) -> List[str]:
        """Collect from MCP server log."""
        if not os.path.isfile(self.MCP_LOG_PATH):
            return []
        return _tail_lines(self.MCP_LOG_PATH, limit=self.tail_lines // 2)

    def collect(self) -> Tuple[List[str], TraceSourceStatus]:
        """Collect from all available sources and merge."""
        sources: List[str] = []
        all_lines: List[str] = []

        # Source 1: nav_agent trace (JSONL)
        if self.nav_agent_trace_file and os.path.isfile(self.nav_agent_trace_file):
            trace_lines, _ = self._collect_trace()
            if trace_lines:
                all_lines.extend(trace_lines)
                sources.append("agent")

        # Source 2: MCP server log
        mcp_lines = self._collect_mcp_log()
        if mcp_lines:
            all_lines.extend(mcp_lines)
            sources.append("mcp")

        resolved = "+".join(sources) if sources else "none"
        status = TraceSourceStatus(
            requested_mode=self.requested_mode,
            resolved_mode=resolved,
            available=bool(sources),
            error=None if sources else "no trace sources available (no active navloop, no MCP log)",
        )
        return all_lines, status


# ---------------------------------------------------------------------------
# Trace parsers
# ---------------------------------------------------------------------------


def _extract_ts_and_message(line: str) -> Tuple[str, Optional[float], str]:
    parts = line.split(" ", 1)
    if len(parts) == 2:
        token, rest = parts
        if "T" in token and token.count("-") >= 2:
            ts_s = _parse_iso_to_epoch(token)
            if ts_s is not None:
                return token, ts_s, rest
    return "", None, line


def _extract_loop_hint(text: str) -> Optional[str]:
    match = _LOOP_ID_RE.search(text)
    return match.group(1) if match else None


def _parse_gateway_tool_event(
    message: str, ts_text: str, ts_s: Optional[float], raw_line: str
) -> Optional[ToolTraceEvent]:
    call_match = _TOOL_CALL_RE.search(message)
    if call_match:
        tool = call_match.group(1)
        params_text = call_match.group(2)
        command: Optional[str] = None
        try:
            params = json.loads(params_text)
            if isinstance(params, dict):
                cmd_val = params.get("command")
                if isinstance(cmd_val, str):
                    command = cmd_val
        except json.JSONDecodeError:
            command = None
        loop_hint = _extract_loop_hint(command or message)
        return ToolTraceEvent(
            ts_s=ts_s,
            ts_text=ts_text,
            kind="call",
            tool=tool,
            command=command,
            result=None,
            latency_ms=None,
            loop_id_hint=loop_hint,
            raw=raw_line,
        )

    done_match = _TOOL_DONE_RE.search(message)
    if done_match:
        tool = done_match.group(1)
        latency = int(done_match.group(2))
        return ToolTraceEvent(
            ts_s=ts_s,
            ts_text=ts_text,
            kind="done",
            tool=tool,
            command=None,
            result="ok",
            latency_ms=latency,
            loop_id_hint=_extract_loop_hint(message),
            raw=raw_line,
        )

    fail_match = _TOOL_FAIL_RE.search(message)
    if fail_match:
        tool = fail_match.group(1)
        reason = fail_match.group(2).strip()
        latency = int(fail_match.group(3))
        return ToolTraceEvent(
            ts_s=ts_s,
            ts_text=ts_text,
            kind="fail",
            tool=tool,
            command=None,
            result=reason,
            latency_ms=latency,
            loop_id_hint=_extract_loop_hint(message),
            raw=raw_line,
        )

    return None


def _parse_gateway_trace_lines(
    lines: Sequence[str],
) -> Tuple[List[ToolTraceEvent], List[str]]:
    events: List[ToolTraceEvent] = []
    diagnostics: List[str] = []
    for raw in lines:
        clean = _strip_ansi(raw).strip()
        if not clean:
            continue
        ts_text, ts_s, message = _extract_ts_and_message(clean)
        event = _parse_gateway_tool_event(message, ts_text, ts_s, clean)
        if event is not None:
            events.append(event)
        if "lane wait exceeded" in message:
            diagnostics.append(clean)
        elif "tool fail:" in message and event is None:
            diagnostics.append(clean)
    events.sort(key=lambda item: (item.ts_s or 0.0, item.ts_text))
    return events, diagnostics[-40:]


def _parse_nav_agent_trace_lines(
    lines: Sequence[str],
) -> Tuple[List[ToolTraceEvent], List[str]]:
    """Parse nav_agent.py .trace.jsonl format into ToolTraceEvents."""
    events: List[ToolTraceEvent] = []
    diagnostics: List[str] = []
    for raw in lines:
        text = raw.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        kind = obj.get("kind", "")
        ts_text = obj.get("ts", "")
        ts_s = _parse_iso_to_epoch(ts_text) if ts_text else None
        tool = obj.get("tool", "")

        if kind == "tool_call":
            args = obj.get("args", {})
            command = json.dumps(args, ensure_ascii=False) if args else ""
            events.append(ToolTraceEvent(
                ts_s=ts_s, ts_text=ts_text, kind="call", tool=tool,
                command=command, result=None, latency_ms=None,
                loop_id_hint=None, raw=text,
            ))
        elif kind == "tool_result":
            summary = obj.get("summary", "")
            events.append(ToolTraceEvent(
                ts_s=ts_s, ts_text=ts_text,
                kind="fail" if "error" in summary.lower() else "done",
                tool=tool, command=None, result=summary, latency_ms=None,
                loop_id_hint=None, raw=text,
            ))
        elif kind == "llm_call":
            input_t = obj.get("input_tokens", 0)
            output_t = obj.get("output_tokens", 0)
            model = obj.get("model", "")
            events.append(ToolTraceEvent(
                ts_s=ts_s, ts_text=ts_text, kind="call", tool="llm",
                command=f"{model} in={input_t} out={output_t}",
                result=None, latency_ms=None,
                loop_id_hint=None, raw=text,
            ))
    events.sort(key=lambda e: (e.ts_s or 0.0, e.ts_text))
    return events, diagnostics


def _scan_token_usage_from_lines(lines: Sequence[str]) -> TokenUsage:
    usage = TokenUsage()
    prompt = 0
    completion = 0
    total = 0
    found_any = False
    for raw in lines:
        clean = _strip_ansi(raw)
        for key, value_str in _TOKEN_KV_RE.findall(clean):
            value = int(value_str)
            found_any = True
            if key == "prompt_tokens":
                prompt += value
            elif key == "completion_tokens":
                completion += value
            elif key == "total_tokens":
                total += value
    if not found_any:
        return usage
    usage.available = True
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = total if total > 0 else (prompt + completion)
    usage.reason = ""
    return usage


def _scan_token_usage_from_trace_jsonl(lines: Sequence[str]) -> TokenUsage:
    """Extract token usage from nav_agent.py .trace.jsonl format."""
    usage = TokenUsage()
    prompt = 0
    completion = 0
    calls = 0
    for raw in lines:
        text = raw.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if obj.get("kind") == "llm_call":
            inp = obj.get("input_tokens", 0)
            out = obj.get("output_tokens", 0)
            if inp or out:
                prompt += inp or 0
                completion += out or 0
                calls += 1
    if calls == 0:
        usage.reason = "no llm_call events in trace"
        return usage
    usage.available = True
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = prompt + completion
    usage.llm_calls = calls
    usage.reason = ""
    return usage


# ---------------------------------------------------------------------------
# Round / loop / tool builders
# ---------------------------------------------------------------------------


def _build_round_snapshots(jsonl_lines: Sequence[str]) -> List[RoundSnapshot]:
    rounds: Dict[int, RoundSnapshot] = {}
    for line in jsonl_lines:
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        round_value = payload.get("round")
        if not isinstance(round_value, int):
            continue
        snap = rounds.get(round_value)
        if snap is None:
            snap = RoundSnapshot(round=round_value)
            rounds[round_value] = snap

        ts_text = payload.get("ts")
        ts_s = _parse_iso_to_epoch(ts_text) if isinstance(ts_text, str) else None
        phase = payload.get("phase")
        if phase == "round_start":
            snap.start_ts_s = ts_s
        elif phase == "round_end":
            snap.end_ts_s = ts_s

        if isinstance(payload.get("status"), str):
            snap.status = payload["status"]
        if isinstance(payload.get("nav_phase"), str):
            snap.nav_phase = payload["nav_phase"]
        if isinstance(payload.get("total_steps"), int):
            snap.total_steps = payload["total_steps"]
        if isinstance(payload.get("geodesic_distance"), (int, float)):
            snap.geodesic_distance = float(payload["geodesic_distance"])
        if isinstance(payload.get("state_version"), int):
            snap.state_version = payload["state_version"]

    return [rounds[k] for k in sorted(rounds.keys())]


def _load_round_snapshots(events_file: str, limit: int = 300) -> List[RoundSnapshot]:
    return _build_round_snapshots(_tail_lines(events_file, limit=limit))


def _build_round_lines(rounds: Sequence[RoundSnapshot], limit: int = 18) -> List[str]:
    if not rounds:
        return ["(no round events)"]
    out: List[str] = []
    for snap in list(rounds)[-limit:]:
        duration: Optional[float] = None
        if snap.start_ts_s is not None and snap.end_ts_s is not None:
            duration = max(0.0, snap.end_ts_s - snap.start_ts_s)
        status = snap.status or "?"
        nav_phase = snap.nav_phase or "?"
        steps = snap.total_steps if snap.total_steps is not None else "-"
        geo = (
            f"{snap.geodesic_distance:.3f}"
            if snap.geodesic_distance is not None
            else "-"
        )
        out.append(
            f"r{snap.round:<3} {status:<11} phase={nav_phase:<11} "
            f"steps={steps:<4} geo={geo:<8} dt={_fmt_duration(duration)}"
        )
    return out


def _round_window(rounds: Sequence[RoundSnapshot]) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    if not rounds:
        return None, None, None
    latest = rounds[-1]
    return latest.round, latest.start_ts_s, latest.end_ts_s


def _filter_tool_events_for_loop(
    events: Sequence[ToolTraceEvent],
    selected_loop_id: Optional[str],
    active_loop_count: int,
    round_start_s: Optional[float],
    round_end_s: Optional[float],
) -> List[ToolTraceEvent]:
    if not events:
        return []
    filtered: List[ToolTraceEvent] = []
    for event in events:
        raw = event.raw
        command = event.command or ""
        by_loop_hint = bool(
            selected_loop_id
            and (
                event.loop_id_hint == selected_loop_id
                or selected_loop_id in raw
                or selected_loop_id in command
            )
        )
        if by_loop_hint:
            filtered.append(event)
            continue

        if selected_loop_id and active_loop_count == 1:
            in_round = True
            if round_start_s is not None and event.ts_s is not None:
                in_round = event.ts_s >= round_start_s
            if in_round and round_end_s is not None and event.ts_s is not None:
                in_round = event.ts_s <= round_end_s
            if in_round:
                filtered.append(event)
    return filtered


def _build_tool_lines(events: Sequence[ToolTraceEvent], limit: int = 20) -> List[str]:
    if not events:
        return ["(no trace for selected loop/round)"]
    lines: List[str] = []
    for event in list(events)[-limit:]:
        from .helpers import _fmt_epoch_hms  # local to avoid cycle on collectors load
        ts = _fmt_epoch_hms(event.ts_s)
        if event.kind == "call":
            command = event.command or "(unknown command)"
            lines.append(f"{ts} CALL {event.tool:<8} {command}")
        elif event.kind == "done":
            latency = f"{event.latency_ms}ms" if event.latency_ms is not None else "-"
            lines.append(f"{ts} DONE {event.tool:<8} ok {latency}")
        else:
            latency = f"{event.latency_ms}ms" if event.latency_ms is not None else "-"
            reason = event.result or "failed"
            lines.append(f"{ts} FAIL {event.tool:<8} {reason} ({latency})")
    return lines


def _build_loop_lines(
    loops: Sequence[Dict[str, Any]], selected_loop_id: Optional[str], limit: int = 100
) -> List[str]:
    if not loops:
        return ["(none)"]
    lines: List[str] = []
    for loop in list(loops)[:limit]:
        loop_id = str(loop.get("loop_id", ""))
        marker = ">" if loop_id == selected_loop_id else " "
        proc_status = str(loop.get("proc_status", "?"))
        nav_mode = str(loop.get("nav_mode", "?"))
        nav = loop.get("nav_status") if isinstance(loop.get("nav_status"), dict) else {}
        steps = nav.get("total_steps", "-")
        geo = nav.get("geodesic_distance")
        geo_text = f"{float(geo):.3f}" if isinstance(geo, (int, float)) else "-"
        lines.append(
            f"{marker} {loop_id:<14} {proc_status:<12} mode={nav_mode:<7} steps={steps:<4} geo={geo_text}"
        )
    return lines


# ---------------------------------------------------------------------------
# Memory summary
# ---------------------------------------------------------------------------


def _summarize_memory_dir(path: str) -> Tuple[int, Optional[str], Optional[float]]:
    if not os.path.isdir(path):
        return 0, None, None
    count = 0
    latest_path: Optional[str] = None
    latest_mtime: Optional[float] = None
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        count += 1
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            continue
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = full
    return count, latest_path, latest_mtime


def _collect_memory_lines(
    selected_loop: Optional[Dict[str, Any]],
    selected_nav_status: Optional[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[str]:
    lines: List[str] = []
    if selected_loop is None:
        return ["spatial_memory: -", "agent_memory: -"]

    spatial_file = selected_loop.get("spatial_memory_file")
    if isinstance(spatial_file, str) and spatial_file:
        spatial_host = spatial_file
        mem = _safe_json_load(spatial_host)
        if isinstance(mem, dict):
            snapshots = mem.get("snapshots", [])
            rooms = mem.get("rooms", {})
            objects = mem.get("object_sightings", {})
            snap_count = len(snapshots) if isinstance(snapshots, list) else 0
            room_count = len(rooms) if isinstance(rooms, dict) else 0
            obj_count = len(objects) if isinstance(objects, dict) else 0
            try:
                mtime_s = os.path.getmtime(spatial_host)
            except OSError:
                mtime_s = None
            mtime_text = (
                dt.datetime.fromtimestamp(mtime_s).strftime("%H:%M:%S")
                if mtime_s is not None
                else "-"
            )
            lines.append(
                f"spatial_memory snapshots={snap_count} rooms={room_count} objects={obj_count} mtime={mtime_text}"
            )
        else:
            lines.append(f"spatial_memory unreadable: {spatial_host}")
    else:
        lines.append("spatial_memory: (not provided)")

    memory_dir = (args.agent_memory_dir or "").strip()
    if not memory_dir:
        workspace_host = (args.workspace_host or "").strip()
        if not workspace_host:
            nav_status_file = selected_loop.get("nav_status_file")
            if isinstance(nav_status_file, str):
                workspace_host = _guess_workspace_host_from_nav_status_file(nav_status_file) or ""
        if workspace_host:
            memory_dir = os.path.join(workspace_host, "memory")

    if memory_dir:
        count, latest_file, latest_mtime = _summarize_memory_dir(memory_dir)
        if latest_file:
            latest_name = os.path.basename(latest_file)
            latest_ts = dt.datetime.fromtimestamp(latest_mtime).strftime("%m-%d %H:%M:%S")
            lines.append(
                f"agent_memory files={count} latest={latest_name} @ {latest_ts}"
            )
        else:
            lines.append(f"agent_memory files=0 dir={memory_dir}")
    else:
        lines.append("agent_memory: (workspace host not configured)")

    if isinstance(selected_nav_status, dict):
        finding = selected_nav_status.get("finding")
        if isinstance(finding, str) and finding.strip():
            lines.append(f"finding: {_truncate(finding.strip(), 120)}")
    return lines


__all__ = [
    "TraceSourceStatus",
    "ToolTraceEvent",
    "RoundSnapshot",
    "TokenUsage",
    "DashboardState",
    "AgentTraceCollector",
    "_extract_ts_and_message",
    "_extract_loop_hint",
    "_parse_gateway_tool_event",
    "_parse_gateway_trace_lines",
    "_parse_nav_agent_trace_lines",
    "_scan_token_usage_from_lines",
    "_scan_token_usage_from_trace_jsonl",
    "_build_round_snapshots",
    "_load_round_snapshots",
    "_build_round_lines",
    "_round_window",
    "_filter_tool_events_for_loop",
    "_build_tool_lines",
    "_build_loop_lines",
    "_summarize_memory_dir",
    "_collect_memory_lines",
]
