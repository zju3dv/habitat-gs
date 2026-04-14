"""Shared log helper for the habitat_agent subprocess.

This is a minimal stdout logger used by the nav_agent subprocess and
the analytics / terminal_status helpers it calls into. The `[nav_agent]`
prefix identifies the subprocess that produced the line when bridge
logs are aggregated across processes.

Phase 1 moved this out of `nav_agent.py` so that `analytics/` and
`runtime/terminal_status.py` can share it without introducing a
cross-module import cycle with `nav_agent.py`.
"""

from __future__ import annotations

from datetime import datetime, timezone

_LOG_PREFIX = "[nav_agent]"


def log(msg: str) -> None:
    """Print a timestamped log line with the nav_agent prefix.

    Output format matches the pre-Phase-1 behaviour exactly:
        [nav_agent][<ISO-8601 UTC>] <msg>
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{_LOG_PREFIX}[{ts}] {msg}", flush=True)


__all__ = ["log"]
