"""Curses dashboard rendering and event loop.

Hosts the pure-rendering primitives (``_safe_addstr`` / ``_draw_box`` /
``_draw_panel_lines`` / ``_render``) plus ``_refresh_state`` and
``_run_dashboard``. Phase 1 PR 4 moved this verbatim out of
``tools/habitat_agent_tui.py``.

Everything here depends on:
  - ``habitat_agent.runtime.bridge_process.BridgeProcess`` for the
    bridge subprocess wrapper
  - ``.helpers``     for HTTP polling, formatters, loop selection
  - ``.collectors``  for trace parsing and panel-line builders
"""

from __future__ import annotations

import argparse
import curses
import datetime as dt
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from habitat_agent.runtime.bridge_process import BridgeProcess

from .collectors import (
    AgentTraceCollector,
    DashboardState,
    RoundSnapshot,
    _build_loop_lines,
    _build_round_lines,
    _build_tool_lines,
    _collect_memory_lines,
    _filter_tool_events_for_loop,
    _load_round_snapshots,
    _parse_gateway_trace_lines,
    _parse_nav_agent_trace_lines,
    _round_window,
    _scan_token_usage_from_lines,
    _scan_token_usage_from_trace_jsonl,
)
from .helpers import (
    _all_loops,
    _poll_bridge,
    _safe_json_load,
    _select_loop,
    _stop_all_nav_loops,
    _stop_single_nav_loop,
    _tail_lines,
    _truncate,
)


def _safe_addstr(
    stdscr: "curses._CursesWindow", y: int, x: int, text: str, attrs: int = 0
) -> None:
    try:
        stdscr.addstr(y, x, text, attrs)
    except curses.error:
        pass


def _safe_addch(
    stdscr: "curses._CursesWindow", y: int, x: int, ch: int, attrs: int = 0
) -> None:
    try:
        stdscr.addch(y, x, ch, attrs)
    except curses.error:
        pass


def _draw_box(
    stdscr: "curses._CursesWindow",
    top: int,
    left: int,
    height: int,
    width: int,
    title: str,
    color_pair: int = 0,
) -> None:
    if height < 3 or width < 8:
        return
    attrs = curses.color_pair(color_pair)
    bottom = top + height - 1
    right = left + width - 1
    for x in range(left + 1, right):
        _safe_addch(stdscr, top, x, curses.ACS_HLINE, attrs)
        _safe_addch(stdscr, bottom, x, curses.ACS_HLINE, attrs)
    for y in range(top + 1, bottom):
        _safe_addch(stdscr, y, left, curses.ACS_VLINE, attrs)
        _safe_addch(stdscr, y, right, curses.ACS_VLINE, attrs)
    _safe_addch(stdscr, top, left, curses.ACS_ULCORNER, attrs)
    _safe_addch(stdscr, top, right, curses.ACS_URCORNER, attrs)
    _safe_addch(stdscr, bottom, left, curses.ACS_LLCORNER, attrs)
    _safe_addch(stdscr, bottom, right, curses.ACS_LRCORNER, attrs)
    if title:
        title_text = f" {title} "
        _safe_addstr(
            stdscr,
            top,
            left + 2,
            _truncate(title_text, max(0, width - 4)),
            attrs | curses.A_BOLD,
        )


def _draw_panel_lines(
    stdscr: "curses._CursesWindow",
    top: int,
    left: int,
    height: int,
    width: int,
    title: str,
    lines: Sequence[str],
    color_pair: int = 0,
    highlight_index: Optional[int] = None,
) -> None:
    _draw_box(stdscr, top, left, height, width, title, color_pair=color_pair)
    if height < 3 or width < 4:
        return
    max_lines = height - 2
    max_width = width - 2
    for idx, line in enumerate(list(lines)[:max_lines]):
        attrs = curses.color_pair(color_pair)
        if highlight_index is not None and idx == highlight_index:
            attrs |= curses.A_REVERSE
        _safe_addstr(
            stdscr,
            top + 1 + idx,
            left + 1,
            _truncate(line, max_width),
            attrs,
        )


def _render(
    stdscr: "curses._CursesWindow",
    state: DashboardState,
    bridge: BridgeProcess,
    args: argparse.Namespace,
) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    width = max(0, w - 1)

    title = "HabitatAgent"
    now_local = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trace_mode = state.trace_source_status.resolved_mode
    header = f"{title} | {now_local} | trace={trace_mode}"
    _safe_addstr(stdscr, 0, 0, _truncate(header, width), curses.color_pair(1) | curses.A_BOLD)

    health_ok = state.healthz is not None and state.healthz.get("ok", False)
    health_text = "up" if health_ok else "down"
    health_color = 2 if health_ok else 3
    runtime = state.runtime or {}
    sys_line = (
        f"Bridge {bridge.status_text()} | endpoint={bridge.host}:{bridge.port}({health_text}) "
        f"| sessions={runtime.get('active_sessions', 0)} "
        f"| loops={runtime.get('active_nav_loops', 0)} "
        f"| closed={runtime.get('closed_nav_loops', 0)} "
        f"| uptime_s={runtime.get('uptime_s', '-')}"
    )
    _safe_addstr(stdscr, 1, 0, _truncate(sys_line, width), curses.color_pair(health_color))

    content_top = 2
    footer_h = 2
    bottom_h = 7
    main_h = h - content_top - footer_h - bottom_h
    if main_h < 6:
        bottom_h = max(4, h - content_top - footer_h - 6)
        main_h = h - content_top - footer_h - bottom_h
    if main_h < 4:
        compact = "terminal too small; resize window to at least 100x26"
        _safe_addstr(stdscr, content_top, 0, _truncate(compact, width), curses.color_pair(3))
        stdscr.refresh()
        return

    left_w = max(32, int(w * 0.34))
    mid_w = max(36, int(w * 0.34))
    right_w = w - left_w - mid_w
    if right_w < 32:
        deficit = 32 - right_w
        reduce_left = min(deficit // 2 + 1, max(0, left_w - 32))
        left_w -= reduce_left
        right_w += reduce_left
        if right_w < 32:
            reduce_mid = min(32 - right_w, max(0, mid_w - 32))
            mid_w -= reduce_mid
            right_w += reduce_mid
    x0 = 0
    x1 = left_w
    x2 = left_w + mid_w

    _draw_panel_lines(
        stdscr,
        content_top,
        x0,
        main_h,
        left_w,
        "Nav Loops (UP/DOWN select)",
        state.loop_lines,
        color_pair=4,
    )
    _draw_panel_lines(
        stdscr,
        content_top,
        x1,
        main_h,
        mid_w,
        "Round Timeline",
        state.round_lines,
        color_pair=0,
    )
    right_title = (
        "Raw Loop Log"
        if state.show_raw_log
        else f"Tool Trace (round={state.current_round_id or '-'} calls={state.current_round_call_count})"
    )
    right_lines = state.raw_log_tail if state.show_raw_log else state.tool_lines
    _draw_panel_lines(
        stdscr,
        content_top,
        x2,
        main_h,
        max(1, right_w),
        right_title,
        right_lines,
        color_pair=0,
    )

    bottom_top = content_top + main_h
    left_bottom_w = w // 2
    right_bottom_w = w - left_bottom_w

    selected = state.selected_loop or {}
    nav = state.selected_nav_status or {}
    nav_status = nav.get("status", "-")
    nav_phase = nav.get("nav_phase", "-")
    nav_steps = nav.get("total_steps", "-")
    nav_collisions = nav.get("collisions", "-")
    nav_geo = nav.get("geodesic_distance", "-")
    nav_pos = nav.get("current_position", "-")
    nav_lines = [
        f"selected={selected.get('loop_id', '-')}",
        f"task_status={nav_status} phase={nav_phase} steps={nav_steps} collisions={nav_collisions}",
        f"position={nav_pos}",
        f"geodesic_distance={nav_geo}",
    ] + state.memory_lines[:3]

    tu = state.token_usage
    token_line = (
        f"tokens in={tu.prompt_tokens:,} out={tu.completion_tokens:,} "
        f"total={tu.total_tokens:,} calls={tu.llm_calls} "
        f"avg={tu.prompt_tokens // max(tu.llm_calls, 1):,}/{tu.completion_tokens // max(tu.llm_calls, 1):,}"
        if tu.available
        else f"token_usage N/A ({tu.reason})"
    )
    diag_lines = [token_line]
    if state.trace_source_status.error:
        diag_lines.append(f"trace_error: {state.trace_source_status.error}")
    diag_lines.extend(state.diagnostics[-2:])
    if state.last_poll_error:
        diag_lines.append(f"poll_error: {state.last_poll_error}")
    if state.messages:
        diag_lines.append(state.messages[-1])

    _draw_panel_lines(
        stdscr,
        bottom_top,
        0,
        bottom_h,
        left_bottom_w,
        "Nav + Memory",
        nav_lines,
        color_pair=0,
    )
    _draw_panel_lines(
        stdscr,
        bottom_top,
        left_bottom_w,
        bottom_h,
        right_bottom_w,
        "Diagnostics",
        diag_lines,
        color_pair=3 if state.last_poll_error else 0,
    )

    footer = (
        "Keys: q quit | s start/stop bridge | r restart bridge | "
        "k stop selected loop | K stop all loops | l toggle raw/tool | ↑/↓ select loop"
    )
    _safe_addstr(stdscr, h - 1, 0, _truncate(footer, width), curses.color_pair(1))
    stdscr.refresh()


def _refresh_state(
    state: DashboardState,
    args: argparse.Namespace,
    trace_collector: AgentTraceCollector,
    base_url: str,
) -> None:
    healthz, runtime, poll_err = _poll_bridge(base_url=base_url, timeout_s=args.http_timeout)
    state.healthz = healthz
    state.runtime = runtime
    state.last_poll_error = poll_err

    runtime_obj = runtime or {}
    loops = _all_loops(runtime_obj)

    selected = _select_loop(runtime_obj, state.selected_loop_id) if runtime else None
    if selected is not None:
        state.selected_loop_id = str(selected.get("loop_id", ""))
        state.selected_loop = selected
    else:
        state.selected_loop_id = None
        state.selected_loop = None

    nav_status: Optional[Dict[str, Any]] = None
    raw_log_tail: List[str] = []
    rounds: List[RoundSnapshot] = []
    if selected is not None:
        nav_obj = selected.get("nav_status")
        if isinstance(nav_obj, dict):
            nav_status = nav_obj
        else:
            nav_file = selected.get("nav_status_file")
            if isinstance(nav_file, str) and nav_file:
                nav_status = _safe_json_load(nav_file)
        log_file = selected.get("log_file")
        if isinstance(log_file, str) and log_file:
            raw_log_tail = _tail_lines(log_file, limit=12)
        nav_file = selected.get("nav_status_file")
        if isinstance(nav_file, str) and nav_file:
            events_host = nav_file + ".events.jsonl"
            rounds = _load_round_snapshots(events_host, limit=500)
            # Discover nav_agent.py trace file for independent agent mode
            trace_host = nav_file + ".trace.jsonl"
            trace_collector.nav_agent_trace_file = trace_host if os.path.isfile(trace_host) else None
    state.selected_nav_status = nav_status
    state.raw_log_tail = raw_log_tail or ["(no log)"]
    state.round_snapshots = rounds

    trace_lines, source_status = trace_collector.collect()
    state.trace_source_status = source_status
    # Parse trace lines from all sources (agent JSONL + MCP plain text)
    # Try JSONL first for lines that look like JSON, gateway parser for the rest
    agent_lines = [l for l in trace_lines if l.strip().startswith("{")]
    other_lines = [l for l in trace_lines if not l.strip().startswith("{")]
    agent_events, agent_diags = _parse_nav_agent_trace_lines(agent_lines) if agent_lines else ([], [])
    other_events, other_diags = _parse_gateway_trace_lines(other_lines) if other_lines else ([], [])
    trace_events = agent_events + other_events
    trace_events.sort(key=lambda e: (e.ts_s or 0.0, e.ts_text))
    trace_diags = agent_diags + other_diags
    state.trace_events = trace_events[-500:]
    state.token_usage = _scan_token_usage_from_trace_jsonl(agent_lines) if agent_lines else _scan_token_usage_from_lines(other_lines)

    current_round_id, round_start_s, round_end_s = _round_window(rounds)
    state.current_round_id = current_round_id
    active_loop_count = int(runtime_obj.get("active_nav_loops", 0)) if runtime_obj else 0
    state.filtered_trace_events = _filter_tool_events_for_loop(
        state.trace_events,
        state.selected_loop_id,
        active_loop_count=active_loop_count,
        round_start_s=round_start_s,
        round_end_s=round_end_s,
    )[-200:]

    current_round_calls = 0
    for event in state.trace_events:
        if event.kind != "call":
            continue
        if event.ts_s is not None:
            if round_start_s is not None and event.ts_s < round_start_s:
                continue
            if round_end_s is not None and event.ts_s > round_end_s:
                continue
        current_round_calls += 1
    state.current_round_call_count = current_round_calls

    state.loop_lines = _build_loop_lines(loops, state.selected_loop_id)
    state.round_lines = _build_round_lines(rounds)
    state.tool_lines = _build_tool_lines(state.filtered_trace_events, limit=25)
    state.memory_lines = _collect_memory_lines(state.selected_loop, nav_status, args)
    state.diagnostics = trace_diags[-8:]


def _run_dashboard(stdscr: "curses._CursesWindow", args: argparse.Namespace) -> int:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(120)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, -1)

    bridge_script = args.bridge_script
    if not os.path.isabs(bridge_script):
        # Resolve relative to the legacy tools/ directory so existing
        # scripts that pass "habitat_agent_server.py" still find the
        # bridge entry point. From this file
        # (tools/habitat_agent/interfaces/tui/dashboard.py), tools/ is
        # parents[3].
        bridge_script = str((Path(__file__).resolve().parents[3] / bridge_script).resolve())

    bridge = BridgeProcess(
        host=args.host,
        port=args.port,
        python_bin=args.python_bin,
        bridge_script=bridge_script,
        log_path=args.bridge_log,
        log_format=args.log_format,
        session_idle_timeout_s=args.session_idle_timeout_s,
        access_log=args.access_log,
    )
    trace_collector = AgentTraceCollector(
        requested_mode=args.trace_source,
        gateway_container=args.gateway_container,
        gateway_log_file=args.gateway_log_file,
        tail_lines=args.trace_tail_lines,
    )
    state = DashboardState()

    base_url = f"http://{args.host}:{args.port}"
    health, _, _ = _poll_bridge(base_url=base_url, timeout_s=args.http_timeout)
    if health is None and not args.no_start_bridge:
        if bridge.start():
            state.push_message("bridge started by TUI")
            time.sleep(0.8)
        else:
            state.push_message(f"bridge start failed: {bridge.last_error}")

    stop_requested = False
    next_poll_ts = 0.0

    while not stop_requested:
        now = time.monotonic()
        if now >= next_poll_ts:
            next_poll_ts = now + max(args.poll_interval, 0.2)
            _refresh_state(
                state=state,
                args=args,
                trace_collector=trace_collector,
                base_url=base_url,
            )

        _render(stdscr, state, bridge, args)

        ch = stdscr.getch()
        if ch == -1:
            continue
        if ch in (ord("q"), ord("Q")):
            stop_requested = True
            continue
        if ch in (ord("s"), ord("S")):
            if bridge.process is not None and bridge.process.poll() is None:
                bridge.stop()
                state.push_message("bridge stopped by user")
            else:
                if bridge.start():
                    state.push_message("bridge started by user")
                else:
                    state.push_message(f"bridge start failed: {bridge.last_error}")
            next_poll_ts = 0.0
            continue
        if ch in (ord("r"), ord("R")):
            if bridge.restart():
                state.push_message("bridge restarted")
                time.sleep(0.3)
            else:
                state.push_message(f"bridge restart failed: {bridge.last_error}")
            next_poll_ts = 0.0
            continue
        if ch in (ord("l"), ord("L")):
            state.show_raw_log = not state.show_raw_log
            state.push_message(
                "switched to raw loop logs" if state.show_raw_log else "switched to structured tool trace"
            )
            continue
        if ch in (ord("k"),):
            if state.selected_loop_id:
                ok, msg = _stop_single_nav_loop(
                    base_url=base_url,
                    loop_id=state.selected_loop_id,
                    timeout_s=args.http_timeout,
                )
                state.push_message(msg)
                if not ok:
                    state.last_poll_error = msg
                next_poll_ts = 0.0
            else:
                state.push_message("no loop selected")
            continue
        if ch in (ord("K"),):
            now_mono = time.monotonic()
            if now_mono > state.kill_all_confirm_until:
                state.kill_all_confirm_until = now_mono + 3.0
                state.push_message("press K again within 3s to stop ALL active nav-loops")
                continue
            attempted, succeeded, errors = _stop_all_nav_loops(
                base_url=base_url,
                runtime=state.runtime,
                timeout_s=args.http_timeout,
            )
            state.kill_all_confirm_until = 0.0
            state.push_message(f"stop-all done: {succeeded}/{attempted} loops stopped")
            if errors:
                state.last_poll_error = errors[0]
            next_poll_ts = 0.0
            continue
        if ch in (curses.KEY_UP,):
            loops = _all_loops(state.runtime or {})
            if loops:
                ids = [str(item.get("loop_id", "")) for item in loops if item.get("loop_id")]
                if ids:
                    current = ids.index(state.selected_loop_id) if state.selected_loop_id in ids else 0
                    state.selected_loop_id = ids[(current - 1) % len(ids)]
                    next_poll_ts = 0.0
            continue
        if ch in (curses.KEY_DOWN,):
            loops = _all_loops(state.runtime or {})
            if loops:
                ids = [str(item.get("loop_id", "")) for item in loops if item.get("loop_id")]
                if ids:
                    current = ids.index(state.selected_loop_id) if state.selected_loop_id in ids else -1
                    state.selected_loop_id = ids[(current + 1) % len(ids)]
                    next_poll_ts = 0.0
            continue

    if bridge.started_by_tui and not args.keep_bridge:
        bridge.stop()
    return 0


__all__ = [
    "_safe_addstr",
    "_safe_addch",
    "_draw_box",
    "_draw_panel_lines",
    "_render",
    "_refresh_state",
    "_run_dashboard",
]
