"""Per-round / per-event trace writers for nav_agent benchmark analytics.

``append_trace`` writes TUI-consumed trace events (one JSON line per
tool call / LLM turn). ``append_round_event`` writes coarser per-round
events that feed into ``collect_session_stats`` for effective-round
counting.

Both functions log on failure but never raise — trace data is
non-critical observability,  losing a line is acceptable but silent
swallow was the bug B3 surfaced in PR #28 review. Logging keeps
operators informed when disk is full or permissions break.

Moved verbatim from ``tools/nav_agent.py`` in Phase 1 PR 2. No
behaviour changes. The old nav_agent.py re-exports these symbols.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

from habitat_agent.runtime.log import log


def append_trace(
    trace_file: str,
    kind: str,
    round_idx: int,
    **kwargs: Any,
) -> None:
    """Append a structured trace entry for TUI consumption.

    Trace data is non-critical (the TUI just won't see this round's
    activity), so we log on failure but do not raise. Silent swallow
    was bug B3 — operators couldn't tell why traces had gaps.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry: Dict[str, Any] = {"ts": ts, "kind": kind, "round": round_idx}
    entry.update(kwargs)
    try:
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        log(f"WARNING: append_trace failed: {exc!r}")


def append_round_event(
    events_file: str,
    phase: str,
    round_idx: int,
    nav_status: Dict[str, Any],
) -> None:
    """Append a round-level event to the events.jsonl trace.

    Same B3 fix as append_trace: log on failure (used to be silent).
    Events are read by collect_session_stats to compute effective_rounds,
    so a write failure here can subtly skew benchmark stats — surfacing
    the failure helps operators notice.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    event = {
        "ts": ts,
        "phase": phase,
        "round": round_idx,
        "status": nav_status.get("status", "unknown"),
        "nav_phase": nav_status.get("nav_phase", "unknown"),
        "total_steps": nav_status.get("total_steps", 0),
        "collisions": nav_status.get("collisions", 0),
        "geodesic_distance": nav_status.get("geodesic_distance"),
    }
    try:
        with open(events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:
        log(f"WARNING: append_round_event failed: {exc!r}")


__all__ = ["append_trace", "append_round_event"]
