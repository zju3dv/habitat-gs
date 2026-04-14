"""TUI entry point — argument parser + engine selection.

The ``main()`` function picks between the Textual app (preferred when
``textual``/``rich`` are installed) and the legacy curses dashboard,
loads the project ``.env`` file, and then hands off. Phase 1 PR 5
moved this verbatim out of ``tools/habitat_agent_tui.py``.
"""

from __future__ import annotations

import argparse
import curses
import os
import sys

from habitat_agent.interfaces.tui.dashboard import _run_dashboard
from habitat_agent.interfaces.tui.textual_app import HabitatDashboardApp, _HAS_TEXTUAL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HabitatAgent — interactive navigation agent terminal.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=18911, type=int)
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to start bridge",
    )
    parser.add_argument(
        "--bridge-script",
        default="habitat_agent_server.py",
        help="Path to bridge server script (absolute, or relative to tools/)",
    )
    parser.add_argument(
        "--bridge-log",
        default=f"/tmp/habitat_agent_bridge_{os.getenv('USER', 'user')}.log",
        help="Bridge stdout/stderr log file path",
    )
    parser.add_argument(
        "--poll-interval",
        default=1.0,
        type=float,
        help="UI polling interval in seconds",
    )
    parser.add_argument("--http-timeout", default=2.5, type=float, help="HTTP timeout seconds")
    parser.add_argument("--log-format", default="text", choices=["text", "json"])
    parser.add_argument("--session-idle-timeout-s", default=900.0, type=float)
    parser.add_argument("--access-log", action="store_true")
    parser.add_argument("--no-start-bridge", action="store_true", help="Do not auto-start bridge when down")
    parser.add_argument("--keep-bridge", action="store_true", help="Do not stop bridge on TUI exit")

    parser.add_argument(
        "--ui-engine",
        default="auto",
        choices=["auto", "textual", "curses"],
        help="UI engine: 'auto' prefers textual if available, 'textual' requires it, 'curses' for legacy.",
    )
    parser.add_argument(
        "--trace-source",
        default="auto",
        choices=["auto", "docker", "file", "none"],
        help="Source for agent tool trace telemetry.",
    )
    parser.add_argument(
        "--gateway-container",
        default="moltbot-openclaw-gateway-1",
        help="Docker container name for gateway logs (trace-source=docker/auto).",
    )
    parser.add_argument(
        "--gateway-log-file",
        default="",
        help="Gateway log file path (trace-source=file or auto fallback).",
    )
    parser.add_argument(
        "--trace-tail-lines",
        default=1200,
        type=int,
        help="How many gateway log lines to scan each polling cycle.",
    )
    parser.add_argument(
        "--workspace-host",
        default="",
        help="Host path of workspace (for memory directory discovery).",
    )
    parser.add_argument(
        "--agent-memory-dir",
        default="",
        help="Optional explicit host path to agent memory directory.",
    )
    return parser


def main() -> int:
    # Auto-load .env from project root if present
    try:
        from dotenv import load_dotenv

        # tools/habitat_agent/interfaces/tui/main.py → ../../../../.env
        # is the project root .env (4 levels up from this file).
        _env_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".env"
        )
        if os.path.isfile(_env_path):
            load_dotenv(_env_path, override=False)
    except ImportError:
        pass

    args = build_parser().parse_args()

    engine = args.ui_engine
    if engine == "auto":
        engine = "textual" if _HAS_TEXTUAL else "curses"

    if engine == "textual":
        if not _HAS_TEXTUAL:
            print(
                "[habitat-agent] textual is not installed; falling back to curses.",
                file=sys.stderr,
            )
            engine = "curses"

    if engine == "textual":
        app = HabitatDashboardApp(args)
        app.run()
        return 0

    try:
        return curses.wrapper(_run_dashboard, args)
    except KeyboardInterrupt:
        return 130


__all__ = ["build_parser", "main"]
