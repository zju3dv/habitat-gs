"""Shared constants for habitat_agent.

Kept as a minimal module so it can be imported cheaply without pulling in
heavy dependencies (openai, urllib, etc.).
"""

from __future__ import annotations

# Terminal nav_status values — used by nav_agent loop termination checks
# and by collect_session_stats to decide whether to skip bridge refresh.
TERMINAL_STATUSES = frozenset({"reached", "blocked", "error", "timeout"})

__all__ = ["TERMINAL_STATUSES"]
