#!/usr/bin/env python3
"""Legacy entry shim — NavAgent moved to habitat_agent.agents in Phase 1 PR 3.

This file is kept so the canonical invocation
    python tools/nav_agent.py <nav_status.json>
still works without any change. It also re-exports the public symbols
that other code paths may import (NavAgent, PromptBuilder, the analytics
helpers, etc.) so backward-compat holds until **end of Phase 2**, when
all legacy shims are removed.

New code should import directly from:
    from habitat_agent.agents.nav_agent      import NavAgent
    from habitat_agent.agents.nav_agent_main import main
    from habitat_agent.prompts.legacy_builder import PromptBuilder
    from analytics.session_stats             import collect_session_stats
    from analytics.trace_writer              import append_trace, append_round_event
    from habitat_agent.runtime.terminal_status import mark_terminal_status
    from habitat_agent.runtime.log           import log
"""

from __future__ import annotations

import sys

# Re-export the NavAgent class and helpers so any pre-Phase-1 import that
# does `from nav_agent import NavAgent` (or any of the listed symbols)
# continues to resolve.
from analytics.session_stats import collect_session_stats
from analytics.trace_writer import append_round_event, append_trace
from habitat_agent.agents.nav_agent import NavAgent
from habitat_agent.agents.nav_agent_main import main
from habitat_agent.prompts.legacy_builder import PromptBuilder
from habitat_agent.runtime.log import log
from habitat_agent.runtime.terminal_status import (
    _TERMINAL_STATUS_BRIDGE_TIMEOUT_S,
    _write_terminal_status_locally,
    mark_terminal_status,
)

__all__ = [
    "NavAgent",
    "PromptBuilder",
    "main",
    "collect_session_stats",
    "append_trace",
    "append_round_event",
    "mark_terminal_status",
    "_write_terminal_status_locally",
    "_TERMINAL_STATUS_BRIDGE_TIMEOUT_S",
    "log",
]


if __name__ == "__main__":
    sys.exit(main() or 0)
