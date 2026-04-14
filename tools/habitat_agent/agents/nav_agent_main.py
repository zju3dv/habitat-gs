"""CLI entry point for the nav_agent subprocess.

Wraps :class:`habitat_agent.agents.nav_agent.NavAgent` with argument
parsing, environment validation, signal handling, and the
benchmark-stats writer that runs on every termination path.

Phase 1 PR 3 moved this entry point out of ``tools/nav_agent.py`` so
the legacy file can become a thin shim. The shim still supports the
canonical invocation::

    python tools/nav_agent.py <nav_status.json>

by calling ``main()`` from this module.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import traceback

from analytics.session_stats import collect_session_stats
from habitat_agent.runtime.config import load_dotenv_from_project
from habitat_agent.runtime.log import log
from habitat_agent.runtime.terminal_status import mark_terminal_status

from .nav_agent import NavAgent


def main() -> None:
    load_dotenv_from_project()

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <nav_status.json>", file=sys.stderr)
        sys.exit(1)

    status_file = sys.argv[1]
    if not os.path.isfile(status_file):
        print(f"Error: status file not found: {status_file}", file=sys.stderr)
        sys.exit(1)

    # Validate required env vars
    if not os.environ.get("NAV_LLM_API_KEY"):
        print("Error: NAV_LLM_API_KEY env var is required", file=sys.stderr)
        sys.exit(1)

    print("=== nav_agent started ===")
    print(f"  status_file: {status_file}")
    print(f"  events_file: {status_file}.events.jsonl")
    print(f"  bridge:      {os.environ.get('NAV_BRIDGE_HOST', '127.0.0.1')}:{os.environ.get('NAV_BRIDGE_PORT', '18911')}")
    print(f"  model:       {os.environ.get('NAV_LLM_MODEL', 'claude-sonnet-4-20250514')}")
    print(f"  max_iter:    {os.environ.get('NAV_MAX_ITERATIONS', '50')}")
    print(f"  agent_timeout: {os.environ.get('NAV_AGENT_TIMEOUT', '120')}s")
    print()

    agent = NavAgent(status_file)

    # Signal handling
    def _on_sigterm(signum, frame):
        log(f"Received {'SIGTERM' if signum == signal.SIGTERM else 'SIGINT'}; stopping nav_agent")
        if not agent._is_terminal():
            # Try the bridge first; if the signal is from a bridge shutdown
            # the RPC will fail and mark_terminal_status falls back to the
            # locked local write.
            mark_terminal_status(
                status_file,
                "error",
                f"nav_loop interrupted by {'SIGTERM' if signum == signal.SIGTERM else 'SIGINT'}",
                bridge=agent.bridge,
                loop_id=agent.loop_id,
            )
        collect_session_stats(
            status_file, status_file + ".events.jsonl",
            bridge=agent.bridge, loop_id=agent.loop_id,
        )
        sys.exit(143)

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    try:
        agent.run()
    except Exception as exc:
        log(f"Fatal error: {exc}")
        traceback.print_exc()
        # Try the bridge first; the locked fallback handles bridge being
        # the cause of the fatal error.
        mark_terminal_status(
            status_file,
            "error",
            f"nav_agent fatal: {exc}",
            bridge=agent.bridge,
            loop_id=agent.loop_id,
        )
        collect_session_stats(
            status_file, status_file + ".events.jsonl",
            bridge=agent.bridge, loop_id=agent.loop_id,
        )
        sys.exit(1)

    print()
    print("=== nav_agent finished ===")
    try:
        with open(status_file, "r", encoding="utf-8") as f:
            print(json.dumps(json.load(f), indent=2, ensure_ascii=False))
    except Exception:
        pass


__all__ = ["main"]
