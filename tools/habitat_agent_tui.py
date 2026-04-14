#!/usr/bin/env python3
"""Legacy entry shim — TUI code moved to habitat_agent.interfaces.tui.

This file is kept so the canonical invocation
    python tools/habitat_agent_tui.py
still works without any change. It also re-exports the public symbols
that other code paths may import (ChatAgent, HabitatDashboardApp,
BridgeProcess, the helpers/collectors/dashboard functions, etc.) so
backward-compat holds until **end of Phase 2**, when all legacy shims
are removed.

New code should import directly from:
    from habitat_agent.agents.chat_agent          import ChatAgent
    from habitat_agent.interfaces.tui.main        import main
    from habitat_agent.interfaces.tui.textual_app import HabitatDashboardApp
    from habitat_agent.interfaces.tui.dashboard   import _run_dashboard
    from habitat_agent.runtime.bridge_process     import BridgeProcess
"""

from __future__ import annotations

import sys

# Re-export the user-facing class + entry point.
from habitat_agent.agents.chat_agent import (
    ChatAgent,
    _CHAT_SYSTEM_PROMPT,
    _build_chat_tool_schemas,
    _chat_sessions_dir,
    _list_chat_sessions,
)
from habitat_agent.interfaces.tui.main import build_parser, main
from habitat_agent.interfaces.tui.textual_app import HabitatDashboardApp, _HAS_TEXTUAL
from habitat_agent.interfaces.tui.dashboard import (
    _draw_box,
    _draw_panel_lines,
    _refresh_state,
    _render,
    _run_dashboard,
    _safe_addch,
    _safe_addstr,
)
from habitat_agent.interfaces.tui.collectors import (
    AgentTraceCollector,
    DashboardState,
    RoundSnapshot,
    TokenUsage,
    ToolTraceEvent,
    TraceSourceStatus,
    _build_loop_lines,
    _build_round_lines,
    _build_round_snapshots,
    _build_tool_lines,
    _collect_memory_lines,
    _extract_loop_hint,
    _extract_ts_and_message,
    _filter_tool_events_for_loop,
    _load_round_snapshots,
    _parse_gateway_tool_event,
    _parse_gateway_trace_lines,
    _parse_nav_agent_trace_lines,
    _round_window,
    _scan_token_usage_from_lines,
    _scan_token_usage_from_trace_jsonl,
    _summarize_memory_dir,
)
from habitat_agent.interfaces.tui.helpers import (
    _ANSI_ESCAPE_RE,
    _BRIDGE_TOOLS,
    _LOOP_ID_RE,
    _TOKEN_KV_RE,
    _TOOL_CALL_RE,
    _TOOL_DONE_RE,
    _TOOL_FAIL_RE,
    _all_loops,
    _fmt_duration,
    _fmt_epoch_hms,
    _format_loop_distance,
    _guess_workspace_host_from_nav_status_file,
    _http_get_json,
    _http_post_json,
    _is_bridge_tool,
    _now_iso,
    _parse_iso_to_epoch,
    _poll_bridge,
    _safe_json_load,
    _select_loop,
    _stop_all_nav_loops,
    _stop_single_nav_loop,
    _strip_ansi,
    _tail_lines,
    _truncate,
)
from habitat_agent.runtime.bridge_process import BridgeProcess

__all__ = [
    "ChatAgent",
    "HabitatDashboardApp",
    "BridgeProcess",
    "DashboardState",
    "AgentTraceCollector",
    "build_parser",
    "main",
    "_HAS_TEXTUAL",
]


if __name__ == "__main__":
    sys.exit(main() or 0)
