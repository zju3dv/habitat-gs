"""HabitatDashboardApp — Textual-based dashboard + chat front-end.

Hosts the optional Textual UI engine. The whole module is wrapped in a
``_HAS_TEXTUAL`` guard so this file can be imported on systems that do
not have the textual / rich dependency installed; in that case
``HabitatDashboardApp`` is ``None`` and the curses dashboard is used as
the fallback. Phase 1 PR 5 moved this verbatim out of
``tools/habitat_agent_tui.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Textual / rich imports — optional, gated behind _HAS_TEXTUAL.
try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import (
        DataTable, Footer, Header, Input, RichLog, Static, TabbedContent, TabPane,
    )
    from textual import on, work
    from rich.markdown import Markdown as RichMarkdown
    from rich.text import Text

    _HAS_TEXTUAL = True
except ImportError:
    _HAS_TEXTUAL = False

from habitat_agent.agents.chat_agent import ChatAgent, _list_chat_sessions
from habitat_agent.runtime.bridge_process import BridgeProcess
from habitat_agent.interfaces.tui.collectors import (
    AgentTraceCollector,
    DashboardState,
    _build_loop_lines,
    _build_round_lines,
    _build_round_snapshots,
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
from habitat_agent.interfaces.tui.helpers import (
    _all_loops,
    _fmt_duration,
    _fmt_epoch_hms,
    _format_loop_distance,
    _is_bridge_tool,
    _poll_bridge,
    _safe_json_load,
    _select_loop,
    _stop_all_nav_loops,
    _stop_single_nav_loop,
    _tail_lines,
    _truncate,
)
from habitat_agent.interfaces.tui.dashboard import _refresh_state


# ---------------------------------------------------------------------------
# Pixel-art brand banner  (rendered in chat log on startup)
# Color: #7ec8e3  (light blue, retro pixel style)
# ---------------------------------------------------------------------------
_PIXEL_BANNER = (
    "\n"
    "[#5bc8f5]"
    "█  █  ██  ███  ███ █████  ██  █████    ██   ███ ████ █  █ █████\n"
    "████ █  █ ████  █    █   █  █   █     █  █ █    ███  ██ █   █  \n"
    "█  █ ████ █  █  █    █   ████   █     ████ █ ██ █    █ ██   █  \n"
    "█  █ █  █ ███  ███   █   █  █   █     █  █  ███ ████ █  █   █  \n"
    "[/#5bc8f5]"
    "[dim]──────────────────────────────────────────────────────────────\n"
    "  Navigation Agent for habitat-gs[/]\n"
    "\n"
)


# ---------------------------------------------------------------------------
# Textual engine
# ---------------------------------------------------------------------------

if _HAS_TEXTUAL:

    class HabitatDashboardApp(App):
        """HabitatAgent Textual dashboard."""

        TITLE = "HabitatAgent"

        DEFAULT_CSS = """
        Screen {
            layout: vertical;
            background: $surface-darken-3;
        }

        /* ── System status bar ── */
        #system-status {
            height: 3;
            padding: 0 2;
            margin: 0 1;
            background: $surface;
            border: heavy $primary-darken-2;
            border-title-color: $primary-lighten-2;
            border-title-style: bold;
        }
        #system-status.healthy {
            border: heavy $success-darken-2;
            border-title-color: $success;
        }
        #system-status.unhealthy {
            border: heavy $error;
            border-title-color: $error;
        }

        /* ── Main layout ── */
        #main-panels {
            height: 1fr;
            margin: 0 1;
        }
        #left-sidebar {
            width: 30%;
            min-width: 28;
        }
        #loop-table {
            height: 1fr;
            border: round $accent;
            border-title-color: $warning;
            border-title-style: bold;
            border-subtitle-color: $text-muted;
        }
        #nav-detail {
            height: auto;
            max-height: 14;
            border: round $secondary;
            border-title-color: $secondary-lighten-2;
            border-title-style: bold;
            padding: 0 1;
            overflow-y: auto;
        }

        /* ── Tabs area ── */
        #main-tabs {
            width: 70%;
        }
        #round-table {
            height: 1fr;
        }
        #trace-stats {
            height: 1;
            padding: 0 1;
            background: $surface;
        }
        #trace-table {
            height: 1fr;
        }
        #obs-table {
            height: 1fr;
        }
        #action-detail-text {
            height: auto;
            max-height: 8;
            border: round $primary-darken-2;
            padding: 0 1;
        }
        #last-obs-text {
            height: auto;
            max-height: 5;
            border: round $primary-darken-2;
            border-title-color: $text-muted;
            padding: 0 1;
            overflow-y: auto;
        }
        #memory-detail {
            padding: 1 2;
        }

        /* ── Bottom diagnostics ── */
        #diagnostics {
            height: 4;
            margin: 0 1;
            padding: 0 1;
            border: round $primary-darken-1;
            border-title-color: $text-muted;
            border-title-style: bold;
        }
        #diagnostics.has-error {
            border: round $error;
            border-title-color: $error;
        }

        /* ── Chat view (default main) ── */
        #chat-view {
            height: 1fr;
        }
        #chat-log {
            height: 1fr;
            margin: 0 1;
            border: round $primary-darken-2;
            border-title-color: $primary-lighten-2;
            border-title-style: bold;
            padding: 0 1;
        }
        #chat-streaming {
            height: auto;
            max-height: 12;
            margin: 0 2;
            padding: 0 1;
            overflow-y: auto;
        }
        #chat-input {
            margin: 0 1 1 1;
        }

        /* ── Monitor view (hidden by default, Ctrl+M toggle) ── */
        #monitor-view {
            display: none;
            height: 1fr;
        }
        #monitor-view.visible {
            display: block;
        }
        """

        BINDINGS = [
            Binding("ctrl+m", "toggle_monitor", "Monitor"),
            Binding("q", "quit", "Quit"),
            Binding("s", "toggle_bridge", "Bridge", show=False),
            Binding("r", "restart_bridge", "Restart", show=False),
            Binding("k", "kill_loop", "Kill Loop", show=False),
            Binding("K", "kill_all_loops", "Kill ALL", show=False),
            Binding("1", "switch_tab('tab-timeline')", "Timeline", show=False),
            Binding("2", "switch_tab('tab-trace')", "Trace", show=False),
            Binding("3", "switch_tab('tab-obs')", "Actions", show=False),
            Binding("4", "switch_tab('tab-memory')", "Memory", show=False),
        ]

        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__()
            self.dash_args = args
            self.base_url = f"http://{args.host}:{args.port}"
            self.state = DashboardState()
            self.bridge = self._create_bridge(args)
            self.trace_collector = AgentTraceCollector(
                requested_mode=args.trace_source,
                gateway_container=args.gateway_container,
                gateway_log_file=args.gateway_log_file,
                tail_lines=args.trace_tail_lines,
            )
            self._kill_all_confirm_until: float = 0.0
            self._in_monitor: bool = False
            try:
                self.chat_agent: Optional[ChatAgent] = ChatAgent(bridge_url=self.base_url)
            except Exception:
                self.chat_agent = None

        def _create_bridge(self, args: argparse.Namespace) -> BridgeProcess:
            bridge_script = args.bridge_script
            if not os.path.isabs(bridge_script):
                # tools/habitat_agent/interfaces/tui/textual_app.py →
                # parents[3] is the legacy tools/ dir where the bridge
                # script lives, so existing CLI invocations still work.
                bridge_script = str(
                    (Path(__file__).resolve().parents[3] / bridge_script).resolve()
                )
            return BridgeProcess(
                host=args.host,
                port=args.port,
                python_bin=args.python_bin,
                bridge_script=bridge_script,
                log_path=args.bridge_log,
                log_format=args.log_format,
                session_idle_timeout_s=args.session_idle_timeout_s,
                access_log=args.access_log,
            )

        # ── Layout ──────────────────────────────────────────────────

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            # ── Chat view (default main view) ──
            with Vertical(id="chat-view"):
                yield RichLog(id="chat-log", wrap=True, markup=True)
                yield Static("", id="chat-streaming")
                yield Input(placeholder="Type a message or /help...", id="chat-input")
            # ── Monitor view (hidden, toggle Ctrl+M) ──
            with Vertical(id="monitor-view"):
                yield Static("", id="system-status")
                with Horizontal(id="main-panels"):
                    with Vertical(id="left-sidebar"):
                        yield DataTable(id="loop-table", cursor_type="row", zebra_stripes=True)
                        yield Static("", id="nav-detail")
                    with TabbedContent(id="main-tabs"):
                        with TabPane("Timeline", id="tab-timeline"):
                            yield DataTable(id="round-table", cursor_type="none", zebra_stripes=True)
                        with TabPane("Trace", id="tab-trace"):
                            yield Static("", id="trace-stats")
                            yield DataTable(id="trace-table", cursor_type="none", zebra_stripes=True)
                        with TabPane("Action History", id="tab-obs"):
                            yield DataTable(id="obs-table", cursor_type="none", zebra_stripes=True)
                            yield Static("", id="action-detail-text")
                            yield Static("", id="last-obs-text")
                        with TabPane("Memory", id="tab-memory"):
                            yield Static("", id="memory-detail")
                yield Static("", id="diagnostics")
            yield Footer()

        def on_mount(self) -> None:
            # ── Chat view setup ──
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.border_title = "Chat"
            chat_log.write(_PIXEL_BANNER)
            chat_log.write(
                "Type a message to interact with the agent, or use commands:\n"
                "  [bold]/help[/]  — available commands\n"
                "  [bold]Ctrl+M[/] — switch to monitor dashboard\n"
            )
            if self.chat_agent and not self.chat_agent.configured:
                chat_log.write(
                    "\n[yellow]NAV_LLM_API_KEY not set.[/] "
                    "LLM features unavailable. Configure in [bold].env[/] file.\n"
                )
            elif self.chat_agent is None:
                chat_log.write(
                    "\n[yellow]Chat agent failed to initialize.[/] "
                    "Check habitat_agent_core imports.\n"
                )
            self.query_one("#chat-input", Input).focus()

            # ── Monitor view setup ──
            # Loop table
            lt = self.query_one("#loop-table", DataTable)
            lt.add_columns("Loop ID", "Task", "Status", "Mode", "Steps", "Geo")
            lt.border_title = "Nav Loops"

            # Round timeline
            rt = self.query_one("#round-table", DataTable)
            rt.add_columns("Rnd", "Status", "Phase", "Steps", "Geo", "Duration")

            # Trace table
            tt = self.query_one("#trace-table", DataTable)
            tt.add_columns("Time", "Kind", "Layer", "Tool", "Detail")

            # Action history table
            ot = self.query_one("#obs-table", DataTable)
            ot.add_columns("Step", "Action", "Pos / Heading", "Col", "Saw")

            # Bottom panel title
            self.query_one("#diagnostics", Static).border_title = "Diagnostics"
            self.query_one("#nav-detail", Static).border_title = "Nav Status"
            self.query_one("#action-detail-text", Static).border_title = "Action Detail (latest)"
            self.query_one("#last-obs-text", Static).border_title = "Last Action"

            # Auto-start bridge
            if not self.dash_args.no_start_bridge:
                health, _, _ = _poll_bridge(
                    base_url=self.base_url, timeout_s=self.dash_args.http_timeout
                )
                if health is None:
                    if self.bridge.start():
                        self.notify("Bridge started by TUI", severity="information")
                    else:
                        self.notify(
                            f"Bridge start failed: {self.bridge.last_error}",
                            severity="error",
                        )

            # Periodic polling
            self.set_interval(
                max(self.dash_args.poll_interval, 0.5), self._do_refresh
            )

        # ── Polling ─────────────────────────────────────────────────

        @work(exclusive=True, thread=True)
        def _do_refresh(self) -> None:
            _refresh_state(
                state=self.state,
                args=self.dash_args,
                trace_collector=self.trace_collector,
                base_url=self.base_url,
            )
            self.call_from_thread(self._update_widgets)

        # ── Helpers ─────────────────────────────────────────────────

        @staticmethod
        def _status_dot(ok: bool) -> str:
            return "[green bold]●[/]" if ok else "[red bold]●[/]"

        _STATUS_STYLES = {
            "running": "bold green", "navigating": "bold yellow",
            "in_progress": "bold yellow",
            "reached": "bold green", "success": "bold green",
            "completed": "bold green",
            "failed": "bold red", "error": "bold red", "stuck": "bold red",
            "blocked": "bold red", "timeout": "bold red",
            "exited": "dim", "stopped": "dim",
        }

        @classmethod
        def _colored_status(cls, text: str) -> Text:
            return Text(text, style=cls._STATUS_STYLES.get(text, ""))

        # ── Widget Update ───────────────────────────────────────────

        def _update_widgets(self) -> None:
            state = self.state
            runtime = state.runtime or {}

            self._update_status_bar(state, runtime)
            self._update_loop_table(state, runtime)
            self._update_nav_detail(state)
            self._update_round_table(state)
            self._update_trace_tab(state)
            self._update_action_history_tab(state)
            self._update_memory_tab(state)
            self._update_diagnostics(state)

            trace_mode = state.trace_source_status.resolved_mode
            self.sub_title = (
                f"trace={trace_mode}  "
                f"loops={runtime.get('active_nav_loops', 0)}"
            )

        def _update_status_bar(self, state: DashboardState, runtime: Dict[str, Any]) -> None:
            health_ok = state.healthz is not None and state.healthz.get("ok", False)
            dot = self._status_dot(health_ok)
            bar = self.query_one("#system-status", Static)
            bar.border_title = f"{dot} Bridge"
            bar.update(
                f"[bold]{self.bridge.status_text()}[/]  "
                f"[dim]|[/]  Sessions [bold cyan]{runtime.get('active_sessions', 0)}[/]  "
                f"[dim]|[/]  Active Loops [bold yellow]{runtime.get('active_nav_loops', 0)}[/]  "
                f"[dim]|[/]  Closed [dim]{runtime.get('closed_nav_loops', 0)}[/]  "
                f"[dim]|[/]  Uptime [dim]{runtime.get('uptime_s', '-')}s[/]"
            )
            bar.set_class(health_ok, "healthy")
            bar.set_class(not health_ok, "unhealthy")

        def _update_loop_table(self, state: DashboardState, runtime: Dict[str, Any]) -> None:
            lt = self.query_one("#loop-table", DataTable)
            loops = _all_loops(runtime)
            selected_id = state.selected_loop_id
            lt.clear()
            target_row = 0
            for idx, loop in enumerate(loops):
                loop_id = str(loop.get("loop_id", ""))
                task_type = str(loop.get("task_type", "?"))
                proc_status = str(loop.get("proc_status", "?"))
                nav_mode = str(loop.get("nav_mode", "?"))
                nav = loop.get("nav_status") if isinstance(loop.get("nav_status"), dict) else {}
                steps = str(nav.get("total_steps", "-"))
                geo_text = _format_loop_distance(nav)
                status_cell = self._colored_status(proc_status)
                mode_cell = (
                    Text(f" {nav_mode} ", style="bold white on dark_red")
                    if nav_mode == "mapless"
                    else Text(nav_mode)
                )
                lt.add_row(loop_id, task_type, status_cell, mode_cell, steps, geo_text)
                if loop_id == selected_id:
                    target_row = idx
            lt.border_subtitle = f"{len(loops)} loop(s)"
            if loops:
                self._updating_loop_table = True
                try:
                    lt.move_cursor(row=target_row)
                finally:
                    self._updating_loop_table = False

        def _update_nav_detail(self, state: DashboardState) -> None:
            nav = state.selected_nav_status or {}
            selected = state.selected_loop or {}
            sel_id = selected.get("loop_id", "-")
            widget = self.query_one("#nav-detail", Static)

            if not nav:
                widget.update("[dim]No loop selected[/]")
                widget.border_subtitle = ""
                return

            status_val = str(nav.get("status", "-"))
            ns_color = self._STATUS_STYLES.get(status_val, "white")
            phase_val = nav.get("nav_phase", "-")
            steps_val = nav.get("total_steps", "-")
            col_val = nav.get("collisions", 0)
            col_style = "bold red" if isinstance(col_val, int) and col_val > 0 else "dim"
            # Prefer _debug GT data (always available, even in mapless mode).
            # The bridge GT position is always authoritative; the
            # agent-written current_position is shown as a secondary
            # label so operators can see if the agent is tracking
            # correctly (they should match in non-mapless navmesh runs).
            debug = nav.get("_debug", {}) if isinstance(nav.get("_debug"), dict) else {}
            bridge_pos = debug.get("gt_position")
            agent_pos = nav.get("current_position")
            if isinstance(bridge_pos, list) and len(bridge_pos) >= 3:
                pos_text = (
                    f"[{bridge_pos[0]:.2f}, {bridge_pos[1]:.2f}, {bridge_pos[2]:.2f}] (bridge)"
                )
            elif isinstance(agent_pos, list) and len(agent_pos) >= 3:
                pos_text = (
                    f"[{agent_pos[0]:.2f}, {agent_pos[1]:.2f}, {agent_pos[2]:.2f}] (agent)"
                )
            else:
                pos_text = "-"

            # Distance is formatted through the shared helper so list and
            # detail views are guaranteed consistent and labeled with the
            # unit source (g=geodesic / e=euclidean / wp=waypoint). See
            # _format_loop_distance docstring for the decision tree.
            geo_text = _format_loop_distance(nav)
            gt_dir = debug.get("gt_goal_direction_deg")
            dir_text = f"{float(gt_dir):.1f}°" if isinstance(gt_dir, (int, float)) else "-"
            goal_type = nav.get("goal_type", "-")
            goal_desc = _truncate(str(nav.get("goal_description", "-")), 40)

            substeps = nav.get("substeps", [])
            sub_idx = nav.get("current_substep_index", 0)
            if isinstance(substeps, list) and substeps:
                current_sub = substeps[sub_idx] if sub_idx < len(substeps) else "-"
                # Extract readable text from substep (may be dict, str, or other)
                if isinstance(current_sub, dict):
                    sub_desc = (
                        current_sub.get("description")
                        or current_sub.get("action")
                        or current_sub.get("goal")
                        or current_sub.get("text")
                        or current_sub.get("name")
                    )
                    if sub_desc is None:
                        sub_desc = ", ".join(f"{k}={v}" for k, v in list(current_sub.items())[:3])
                    sub_label = _truncate(str(sub_desc), 50)
                elif isinstance(current_sub, str):
                    sub_label = _truncate(current_sub, 50)
                else:
                    sub_label = _truncate(str(current_sub), 50)
                sub_text = f"[bold]{sub_idx + 1}[/]/{len(substeps)}: {sub_label}"
            else:
                sub_text = "[dim]none[/]"

            version = nav.get("state_version", "-")
            updated = nav.get("updated_at", "")
            if isinstance(updated, str) and "T" in updated:
                updated = updated.split("T")[-1][:8]

            task_type_val = nav.get("task_type", "-")
            finding = nav.get("finding")
            if isinstance(finding, dict):
                finding_text = json.dumps(finding, ensure_ascii=False)
            elif isinstance(finding, str) and finding.strip():
                finding_text = finding.strip()
            else:
                finding_text = None

            lines = [
                f"[bold cyan]{sel_id}[/]  [dim]type[/] [bold]{task_type_val}[/]",
                f"[{ns_color}]{status_val}[/]  [dim]phase[/] {phase_val}",
                f"[dim]steps[/] [bold]{steps_val}[/]  [dim]col[/] [{col_style}]{col_val}[/]",
                f"[dim]pos[/] {pos_text}",
                f"[dim]dist[/] {geo_text}  [dim]dir[/] {dir_text}",
                f"[dim]goal[/] {goal_type} [bold]{goal_desc}[/]",
                f"[dim]substep[/] {sub_text}",
                f"[dim]v{version}  {updated}[/]",
            ]
            if finding_text:
                lines.append(f"[bold green]Finding:[/] {_truncate(finding_text, 200)}")
            widget.update("\n".join(lines))
            widget.border_subtitle = status_val

        def _update_round_table(self, state: DashboardState) -> None:
            rt = self.query_one("#round-table", DataTable)
            snapshots = list(state.round_snapshots)[-40:]
            # Fingerprint to avoid flicker
            round_fp = tuple(
                (s.round, s.status, s.total_steps, s.end_ts_s) for s in snapshots
            )
            if round_fp == getattr(self, "_round_fp", None):
                return
            self._round_fp = round_fp
            rt.clear()
            for snap in snapshots:
                duration: Optional[float] = None
                if snap.start_ts_s is not None and snap.end_ts_s is not None:
                    duration = max(0.0, snap.end_ts_s - snap.start_ts_s)
                status_text = snap.status or "?"
                phase_text = snap.nav_phase or "?"
                steps_text = str(snap.total_steps) if snap.total_steps is not None else "-"
                geo_text = f"{snap.geodesic_distance:.3f}" if snap.geodesic_distance is not None else "-"
                st_cell = self._colored_status(status_text)
                dur_cell = Text(_fmt_duration(duration))
                if duration is not None and duration > 30.0:
                    dur_cell = Text(_fmt_duration(duration), style="bold red")
                elif duration is not None and duration > 10.0:
                    dur_cell = Text(_fmt_duration(duration), style="yellow")
                rt.add_row(str(snap.round), st_cell, phase_text, steps_text, geo_text, dur_cell)
            if snapshots:
                rt.move_cursor(row=rt.row_count - 1)

        def _update_trace_tab(self, state: DashboardState) -> None:
            tt = self.query_one("#trace-table", DataTable)

            events = state.trace_events[-60:]

            # Fingerprint: skip rebuild if data unchanged (prevents flickering)
            fp = tuple(
                (e.ts_s, e.kind, e.tool, e.command, e.latency_ms)
                for e in events
            )
            if fp == getattr(self, "_trace_fp", None):
                return
            self._trace_fp = fp

            tt.clear()
            bridge_count = 0
            agent_count = 0
            total_latency = 0
            latency_count = 0

            for event in events:
                ts = _fmt_epoch_hms(event.ts_s)

                # Kind cell
                kind_styles = {"call": "bold yellow", "done": "bold green", "fail": "bold red reverse"}
                kind_cell = Text(event.kind.upper(), style=kind_styles.get(event.kind, ""))

                # Layer classification by tool name
                is_bridge = _is_bridge_tool(event.tool)
                if is_bridge:
                    layer_cell = Text(" Bridge ", style="bold white on dark_blue")
                    if event.kind == "call":
                        bridge_count += 1
                elif event.tool == "llm":
                    layer_cell = Text(" LLM ", style="bold white on dark_magenta")
                else:
                    layer_cell = Text(" Agent ", style="bold white on dark_green")
                    if event.kind == "call":
                        agent_count += 1

                tool_cell = Text(event.tool)
                if event.kind == "call":
                    detail = _truncate(event.command or "", 50)
                elif event.kind == "done":
                    lat = f"{event.latency_ms}ms" if event.latency_ms is not None else ""
                    detail = f"ok {lat}"
                else:
                    lat = f"{event.latency_ms}ms" if event.latency_ms is not None else ""
                    detail = f"{event.result or 'failed'} ({lat})"

                if event.latency_ms is not None:
                    total_latency += event.latency_ms
                    latency_count += 1

                tt.add_row(ts, kind_cell, layer_cell, tool_cell, detail)

            if events:
                tt.move_cursor(row=tt.row_count - 1)

            # Stats line
            total_calls = bridge_count + agent_count
            avg_lat = f"{total_latency // latency_count}ms" if latency_count else "-"
            round_id = state.current_round_id or "-"
            stats = self.query_one("#trace-stats", Static)
            stats.update(
                f"Round [bold]{round_id}[/]  [dim]|[/]  "
                f"Total [bold cyan]{total_calls}[/] calls  [dim]|[/]  "
                f"Bridge [bold blue]{bridge_count}[/]  "
                f"Agent [bold green]{agent_count}[/]  [dim]|[/]  "
                f"Avg latency [bold]{avg_lat}[/]"
            )

        def _update_action_history_tab(self, state: DashboardState) -> None:
            ot = self.query_one("#obs-table", DataTable)
            nav = state.selected_nav_status or {}
            history = nav.get("action_history")

            # Fingerprint to avoid flicker
            action_fp = len(history) if isinstance(history, list) else 0
            if action_fp == getattr(self, "_action_fp", None) and action_fp > 0:
                return
            self._action_fp = action_fp

            ot.clear()
            if isinstance(history, list) and history:
                recent = history[-50:]
                for i, entry in enumerate(recent):
                    if not isinstance(entry, dict):
                        ot.add_row(str(len(history) - len(recent) + i + 1), "-", "-", Text("?"), str(entry)[:60])
                        continue
                    # Flexible key extraction
                    step = str(entry.get("step", len(history) - len(recent) + i + 1))
                    action = str(entry.get("action", "-"))
                    pos = entry.get("pos") or entry.get("position") or entry.get("current_position")
                    heading = entry.get("heading_deg")
                    if isinstance(pos, list) and len(pos) >= 3:
                        pos_text = f"[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}]"
                    elif isinstance(pos, list) and len(pos) >= 2:
                        pos_text = f"[{', '.join(f'{v:.1f}' for v in pos)}]"
                    else:
                        pos_text = "-"
                    if isinstance(heading, (int, float)):
                        pos_text += f" h={heading:.0f}°"
                    collided = entry.get("collided", entry.get("collision", False))
                    col_cell = Text("YES", style="bold red") if collided else Text("no", style="dim")
                    saw = entry.get("perception") or entry.get("saw") or entry.get("observation") or entry.get("description") or entry.get("note")
                    if saw is None:
                        skip = {"step", "action", "pos", "position", "current_position", "collided", "collision", "visual_path", "heading_deg", "perception", "analysis", "decision"}
                        extras = {k: v for k, v in entry.items() if k not in skip and v is not None}
                        saw = _truncate(str(extras), 120) if extras else "-"
                    else:
                        saw = _truncate(str(saw), 120)
                    ot.add_row(step, action, pos_text, col_cell, saw)
                if ot.row_count:
                    ot.move_cursor(row=ot.row_count - 1)

            # Action detail panel: structured reasoning + visual_path of the last entry
            detail_widget = self.query_one("#action-detail-text", Static)
            if isinstance(history, list) and history:
                last_entry = history[-1] if isinstance(history[-1], dict) else {}
                visual_path = last_entry.get("visual_path", "")
                detail_parts = [f"[bold]Step {last_entry.get('step', '?')} — {last_entry.get('action', '?')}[/]"]
                # Structured reasoning fields
                perception = last_entry.get("perception")
                analysis = last_entry.get("analysis")
                decision = last_entry.get("decision")
                if perception or analysis or decision:
                    if perception:
                        detail_parts.append(f"[blue bold]Perception:[/] {perception}")
                    if analysis:
                        detail_parts.append(f"[yellow bold]Analysis:[/] {analysis}")
                    if decision:
                        detail_parts.append(f"[green bold]Decision:[/] {decision}")
                else:
                    # Legacy: fall back to saw
                    full_saw = str(last_entry.get("saw") or last_entry.get("observation") or "-")
                    detail_parts.append(full_saw)
                if visual_path:
                    detail_parts.append(f"[dim]Image: {visual_path}[/]")
                detail_widget.update("\n".join(detail_parts))
            else:
                detail_widget.update("[dim]No action history yet[/]")

            # Last observation text
            last_act = nav.get("last_action")
            lo_widget = self.query_one("#last-obs-text", Static)
            if last_act is not None:
                if isinstance(last_act, str) and last_act.strip():
                    lo_widget.update(f"{_truncate(last_act.strip(), 500)}")
                elif isinstance(last_act, dict):
                    lo_widget.update(f"{_truncate(json.dumps(last_act, ensure_ascii=False), 500)}")
                else:
                    lo_widget.update(f"{_truncate(str(last_act), 500)}")
            else:
                lo_widget.update("[dim]No action yet[/]")

        def _update_memory_tab(self, state: DashboardState) -> None:
            nav = state.selected_nav_status or {}
            selected = state.selected_loop or {}
            parts: List[str] = []

            # Spatial memory
            parts.append("[bold underline]Spatial Memory[/]")
            spatial_file = selected.get("spatial_memory_file")
            if isinstance(spatial_file, str) and spatial_file:
                host_path = spatial_file
                mem = _safe_json_load(host_path)
                if isinstance(mem, dict):
                    snaps = mem.get("snapshots", [])
                    rooms = mem.get("rooms", {})
                    objects = mem.get("object_sightings", {})
                    snap_n = len(snaps) if isinstance(snaps, list) else 0
                    room_n = len(rooms) if isinstance(rooms, dict) else 0
                    obj_n = len(objects) if isinstance(objects, dict) else 0
                    parts.append(
                        f"  Snapshots [bold cyan]{snap_n}[/]    "
                        f"Rooms [bold cyan]{room_n}[/]    "
                        f"Objects [bold cyan]{obj_n}[/]"
                    )
                    parts.append(f"  [dim]File: {host_path}[/]")
                else:
                    parts.append(f"  [dim]Unreadable: {host_path}[/]")
            else:
                parts.append("  [dim]Not configured[/]")

            parts.append("")

            # Agent memory
            parts.append("[bold underline]Agent Memory[/]")
            for ml in state.memory_lines:
                if ml.startswith("agent_memory"):
                    parts.append(f"  {ml}")
                    break
            else:
                parts.append("  [dim]Not configured[/]")

            parts.append("")

            # Finding
            finding = nav.get("finding")
            if isinstance(finding, str) and finding.strip():
                parts.append("[bold underline]Finding[/]")
                parts.append(f"  {_truncate(finding.strip(), 200)}")
            elif isinstance(finding, dict):
                parts.append("[bold underline]Finding[/]")
                parts.append(f"  {_truncate(json.dumps(finding, ensure_ascii=False), 200)}")

            # Action history stats
            history = nav.get("action_history")
            if isinstance(history, list):
                col_count = sum(1 for e in history if isinstance(e, dict) and e.get("collided"))
                parts.append("")
                parts.append("[bold underline]Action History[/]")
                parts.append(
                    f"  Total entries [bold cyan]{len(history)}[/]    "
                    f"Collisions [bold red]{col_count}[/]"
                )

            self.query_one("#memory-detail", Static).update("\n".join(parts))

        def _update_diagnostics(self, state: DashboardState) -> None:
            parts: List[str] = []
            if state.token_usage.available:
                tu = state.token_usage
                avg_in = tu.prompt_tokens // max(tu.llm_calls, 1)
                avg_out = tu.completion_tokens // max(tu.llm_calls, 1)
                parts.append(
                    f"[bold]Tokens[/] "
                    f"input=[cyan]{tu.prompt_tokens:,}[/]  "
                    f"output=[cyan]{tu.completion_tokens:,}[/]  "
                    f"total=[bold cyan]{tu.total_tokens:,}[/]  "
                    f"[dim]|[/]  calls=[bold]{tu.llm_calls}[/]  "
                    f"avg=[dim]{avg_in:,}/{avg_out:,}[/]"
                )
            else:
                parts.append(f"[dim]Tokens N/A ({state.token_usage.reason})[/]")
            trace_mode = state.trace_source_status.resolved_mode
            if state.trace_source_status.error:
                parts.append(f"[red]Trace ({trace_mode}): {_truncate(state.trace_source_status.error, 80)}[/]")
            if state.last_poll_error:
                parts.append(f"[red bold]Poll: {_truncate(state.last_poll_error, 80)}[/]")
            if state.messages:
                parts.append(f"[dim italic]{state.messages[-1]}[/]")
            diag = self.query_one("#diagnostics", Static)
            diag.update("\n".join(parts))
            diag.set_class(bool(state.last_poll_error), "has-error")

        # ── Event handlers ──────────────────────────────────────────

        @on(DataTable.RowHighlighted, "#loop-table")
        def _on_loop_selected(self, event: DataTable.RowHighlighted) -> None:
            # Guard: ignore events fired by move_cursor inside _update_loop_table
            if getattr(self, "_updating_loop_table", False):
                return
            if event.row_key is not None:
                lt = self.query_one("#loop-table", DataTable)
                try:
                    row_data = lt.get_row(event.row_key)
                    loop_id = str(row_data[0]) if row_data else None
                except Exception:
                    loop_id = None
                if loop_id and loop_id != self.state.selected_loop_id:
                    self.state.selected_loop_id = loop_id
                    self._do_refresh()

        # ── Actions ─────────────────────────────────────────────────

        def action_switch_tab(self, tab_id: str) -> None:
            tabs = self.query_one("#main-tabs", TabbedContent)
            tabs.active = tab_id

        # ── Chat / Monitor toggle ──────────────────────────────────

        def action_toggle_monitor(self) -> None:
            """Ctrl+M: toggle between Chat and Monitor views."""
            chat_view = self.query_one("#chat-view")
            monitor_view = self.query_one("#monitor-view")
            self._in_monitor = not self._in_monitor
            if self._in_monitor:
                chat_view.display = False
                monitor_view.display = True
                self.sub_title = "Monitor — Ctrl+M to return to Chat"
            else:
                monitor_view.display = False
                chat_view.display = True
                self.sub_title = "Chat"
                self.query_one("#chat-input", Input).focus()

        # ── Chat input handling ────────────────────────────────────

        @on(Input.Submitted, "#chat-input")
        def _on_chat_submit(self, event: Input.Submitted) -> None:
            user_text = event.value.strip()
            if not user_text:
                return
            event.input.clear()

            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.write(f"\n[bold green]> {user_text}[/]")

            # Slash commands
            if user_text.startswith("/"):
                self._handle_slash_command(user_text, chat_log)
                return

            # LLM message
            if self.chat_agent is None or not self.chat_agent.configured:
                chat_log.write(
                    "[red]LLM not configured.[/] Set [bold]NAV_LLM_API_KEY[/] in .env"
                )
                return

            event.input.disabled = True
            self.query_one("#chat-streaming", Static).update("[dim]Thinking...[/]")
            self._run_chat_agent(user_text)

        def _handle_slash_command(self, cmd: str, chat_log: RichLog) -> None:
            parts = cmd.split(None, 1)
            command = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if command == "/help":
                chat_log.write(
                    "\n[bold]Commands:[/]\n"
                    "  [cyan]/help[/]       Show this help\n"
                    "  [cyan]/status[/]     Bridge and session status\n"
                    "  [cyan]/copy[/]       Copy last agent response to clipboard\n"
                    "  [cyan]/sessions[/]   List recent chat sessions\n"
                    "  [cyan]/resume[/] ID  Resume a previous chat session\n"
                    "  [cyan]/monitor[/]    Switch to monitor dashboard (Ctrl+M)\n"
                    "  [cyan]/clear[/]      Clear chat and conversation history\n"
                    "  [cyan]/exit[/]       Exit HabitatAgent\n"
                    "\n[dim]Tip: Hold Shift + mouse drag to select text for copying.[/]"
                )
            elif command == "/status":
                runtime = self.state.runtime or {}
                health_ok = self.state.healthz is not None
                model = self.chat_agent.model if self.chat_agent else "-"
                configured = "yes" if (self.chat_agent and self.chat_agent.configured) else "no"
                sid = getattr(self.chat_agent, "session_id", "-") if self.chat_agent else "-"
                chat_log.write(
                    f"\n[bold]Status[/]\n"
                    f"  Bridge: {self.bridge.status_text()} ({'up' if health_ok else 'down'})\n"
                    f"  Sessions: {runtime.get('active_sessions', 0)}\n"
                    f"  Active loops: {runtime.get('active_nav_loops', 0)}\n"
                    f"  LLM model: {model}\n"
                    f"  LLM configured: {configured}\n"
                    f"  Chat session: {sid}"
                )
            elif command == "/copy":
                if self.chat_agent and self.chat_agent.last_response:
                    try:
                        self.copy_to_clipboard(self.chat_agent.last_response)
                        chat_log.write("[dim]Copied to clipboard.[/]")
                    except Exception:
                        # Fallback: try xclip
                        try:
                            proc = subprocess.Popen(
                                ["xclip", "-selection", "clipboard"],
                                stdin=subprocess.PIPE,
                            )
                            proc.communicate(self.chat_agent.last_response.encode("utf-8"))
                            chat_log.write("[dim]Copied to clipboard (xclip).[/]")
                        except Exception:
                            chat_log.write("[yellow]Clipboard not available. Use Shift+mouse to select.[/]")
                else:
                    chat_log.write("[dim]No agent response to copy.[/]")
            elif command == "/sessions":
                sessions = _list_chat_sessions(limit=10)
                if not sessions:
                    chat_log.write("[dim]No saved sessions.[/]")
                else:
                    lines = ["\n[bold]Recent sessions:[/]"]
                    for s in sessions:
                        ts = s["created_at"][:19] if s["created_at"] else "?"
                        lines.append(
                            f"  [cyan]{s['session_id']}[/]  {ts}  "
                            f"{s['messages']} msgs  [dim]{s['model']}[/]"
                        )
                    lines.append("\n[dim]Use /resume <session_id> to continue a session.[/]")
                    chat_log.write("\n".join(lines))
            elif command == "/resume":
                if not arg:
                    chat_log.write("[red]Usage: /resume <session_id>[/]")
                    return
                sessions = _list_chat_sessions(limit=50)
                match = [s for s in sessions if arg in s["session_id"]]
                if not match:
                    chat_log.write(f"[red]Session not found: {arg}[/]")
                    return
                target = match[0]
                try:
                    self.chat_agent = ChatAgent(
                        bridge_url=self.base_url,
                        resume_file=target["file"],
                    )
                    chat_log.clear()
                    chat_log.write(
                        f"[bold]Resumed session:[/] {target['session_id']}\n"
                        f"[dim]{target['messages']} messages loaded, model: {target['model']}[/]\n"
                    )
                    # Replay last few messages for context
                    recent = [m for m in self.chat_agent.conversation if m.get("role") in ("user", "assistant") and m.get("content")]
                    for m in recent[-6:]:
                        if m["role"] == "user":
                            chat_log.write(f"[bold green]>[/] {_truncate(m['content'], 200)}")
                        else:
                            chat_log.write(f"[bold purple]>[/] {_truncate(m['content'], 200)}")
                    chat_log.write("")
                except Exception as exc:
                    chat_log.write(f"[red]Resume failed: {exc}[/]")
            elif command == "/monitor":
                self.action_toggle_monitor()
            elif command == "/clear":
                chat_log.clear()
                if self.chat_agent:
                    self.chat_agent.conversation.clear()
                chat_log.write("[dim]Chat cleared.[/]")
            elif command == "/exit":
                self.exit()
            else:
                chat_log.write(
                    f"[red]Unknown command: {command}[/] — type [bold]/help[/]"
                )

        def _chat_write(self, text: str) -> None:
            """Write to chat log (called from main thread via call_from_thread)."""
            self.query_one("#chat-log", RichLog).write(text)

        def _chat_stream_update(self, text: str) -> None:
            """Update streaming display with accumulated text."""
            self.query_one("#chat-streaming", Static).update(f"[bold purple]>[/] {text}")

        def _chat_stream_done(self, text: str) -> None:
            """Finalize streamed message: render as Markdown, clear streaming display."""
            self.query_one("#chat-streaming", Static).update("")
            log = self.query_one("#chat-log", RichLog)
            log.write(Text("> ", style="bold purple"), expand=True)
            log.write(RichMarkdown(text))

        def _chat_stream_clear(self) -> None:
            """Clear streaming display."""
            self.query_one("#chat-streaming", Static).update("")

        def _chat_enable_input(self) -> None:
            """Re-enable chat input after agent finishes."""
            inp = self.query_one("#chat-input", Input)
            inp.disabled = False
            inp.focus()

        @work(thread=True, name="chat-agent")
        def _run_chat_agent(self, user_text: str) -> None:
            """Run LLM agent in background thread with streaming."""
            stream_buf = ""
            try:
                for event in self.chat_agent.process_message(user_text):
                    etype = event[0]
                    if etype == "token":
                        stream_buf += event[1]
                        self.call_from_thread(self._chat_stream_update, stream_buf)
                    elif etype == "message_done":
                        self.call_from_thread(self._chat_stream_done, stream_buf)
                        stream_buf = ""
                    elif etype == "tool_call":
                        _, name, args = event
                        args_s = json.dumps(args, ensure_ascii=False)
                        if len(args_s) > 80:
                            args_s = args_s[:77] + "..."
                        self.call_from_thread(
                            self._chat_write,
                            f"  [yellow]tool[/] [bold]{name}[/]({args_s})",
                        )
                    elif etype == "tool_result":
                        _, name, result_text, images = event
                        short = _truncate(result_text, 200)
                        self.call_from_thread(
                            self._chat_write, f"  [dim]{name} -> {short}[/]"
                        )
                        if images:
                            self.call_from_thread(
                                self._chat_write,
                                f"  [dim]({len(images)} image(s) captured)[/]",
                            )
                    elif etype == "error":
                        self.call_from_thread(
                            self._chat_write, f"[red]Error: {event[1]}[/]"
                        )
            except Exception as exc:
                self.call_from_thread(
                    self._chat_write, f"[red]Agent error: {exc}[/]"
                )
            finally:
                self.call_from_thread(self._chat_stream_clear)
                self.call_from_thread(self._chat_enable_input)

        # ── Bridge actions ─────────────────────────────────────────

        def action_toggle_bridge(self) -> None:
            if self.bridge.process is not None and self.bridge.process.poll() is None:
                self.bridge.stop()
                self.notify("Bridge stopped", severity="information")
            else:
                if self.bridge.start():
                    self.notify("Bridge started", severity="information")
                else:
                    self.notify(f"Bridge start failed: {self.bridge.last_error}", severity="error")
            self._do_refresh()

        def action_restart_bridge(self) -> None:
            if self.bridge.restart():
                self.notify("Bridge restarted", severity="information")
            else:
                self.notify(f"Restart failed: {self.bridge.last_error}", severity="error")
            self._do_refresh()

        def action_kill_loop(self) -> None:
            if not self.state.selected_loop_id:
                self.notify("No loop selected", severity="warning")
                return
            ok, msg = _stop_single_nav_loop(
                base_url=self.base_url,
                loop_id=self.state.selected_loop_id,
                timeout_s=max(self.dash_args.http_timeout, 10.0),
            )
            self.notify(msg, severity="information" if ok else "error")
            self._do_refresh()

        def action_kill_all_loops(self) -> None:
            now = time.monotonic()
            if now > self._kill_all_confirm_until:
                self._kill_all_confirm_until = now + 3.0
                self.notify("Press K again within 3s to stop ALL loops", severity="warning")
                return
            self._kill_all_confirm_until = 0.0
            attempted, succeeded, errors = _stop_all_nav_loops(
                base_url=self.base_url,
                runtime=self.state.runtime,
                timeout_s=max(self.dash_args.http_timeout, 10.0),
            )
            self.notify(f"Stopped {succeeded}/{attempted} loops", severity="information")
            if errors:
                self.notify(errors[0], severity="error")
            self._do_refresh()

        def on_unmount(self) -> None:
            # Close active session before stopping the bridge so the simulator
            # releases GPU/memory properly and leaves no zombie sessions.
            if self.chat_agent and getattr(self.chat_agent, "session_id", ""):
                try:
                    self.chat_agent.tool_ctx.bridge.call(
                        "close_session", {}, timeout=5.0
                    )
                except Exception:
                    pass  # Bridge may already be gone; ignore
            if self.bridge.started_by_tui and not self.dash_args.keep_bridge:
                self.bridge.stop()


else:
    HabitatDashboardApp = None  # type: ignore[assignment]


__all__ = ["HabitatDashboardApp", "_HAS_TEXTUAL"]
