"""Session-level benchmark stats collector.

``collect_session_stats`` is called by the nav_agent subprocess at the
end of every nav loop. It reads the current ``nav_status.json`` and the
per-loop ``events.jsonl``, computes derived metrics (SPL, gt_success,
gt_end_geodesic/euclidean distances, tool_usage, effective_rounds),
and appends a single JSON line to the cross-session
``session_stats.jsonl`` at the artifacts root.

Key design decisions carried forward from PR #28:

- **Terminal status short-circuit**: if the file already shows a
  terminal status (reached/blocked/timeout/error), skip the bridge
  refresh to prevent the bridge from overwriting a fallback-written
  terminal state. See the B1 race history in PR #28.

- **No raise on append failure**: disk-full during the final append
  must not crash nav_agent's run loop (it would be caught by main()'s
  fatal handler and silently rewrite a successful loop to
  status="error"). We log loudly instead.

- **Unambiguous end distances**: emit two separate fields
  (``gt_end_geodesic_distance``, ``gt_end_euclidean_distance``) plus
  ``has_navmesh`` so downstream report consumers never merge
  heterogeneous quantities.

Moved verbatim from ``tools/nav_agent.py`` in Phase 1 PR 2. No
behaviour changes. The old nav_agent.py re-exports ``collect_session_stats``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from habitat_agent.runtime.constants import TERMINAL_STATUSES
from habitat_agent.runtime.file_io import append_jsonl_atomic
from habitat_agent.runtime.log import log

if TYPE_CHECKING:
    from habitat_agent.runtime.bridge_client import BridgeClient


def collect_session_stats(
    status_file: str,
    events_file: str,
    bridge: Optional["BridgeClient"] = None,
    loop_id: Optional[str] = None,
) -> None:
    """Write session stats to the artifacts root (not the loop subdir).

    If ``bridge`` and ``loop_id`` are provided AND the file's current
    status is not terminal, refreshes _debug from the live session state
    via get_nav_loop_status before reading nav_status.json. This keeps
    gt_path_length / gt_end_geodesic_distance / gt_end_euclidean_distance
    fresh for live loops.

    Terminal status short-circuit: if the nav_status file already shows
    a terminal status (reached / blocked / timeout / error), the bridge
    refresh is SKIPPED. Rationale: mark_terminal_status writes the
    terminal state directly to the file while the bridge's in-memory
    record.nav_status is still ``in_progress`` (because the patch
    bypassed the bridge). The bridge's get_nav_loop_status path persists
    record.nav_status back to disk, which would silently overwrite the
    terminal status just written — producing wrong ``outcome`` in
    session_stats.jsonl for every timeout / signal / fatal path.
    Preserving the on-file terminal status is more important than having
    slightly-fresher _debug for a loop that already stopped moving.
    """
    # Walk up from loop subdir to find the artifacts root
    # Structure: {artifacts_root}/{session_id}/{loop_id}/nav_status.json
    loop_dir = os.path.dirname(os.path.abspath(status_file))
    candidate_root = os.path.dirname(os.path.dirname(loop_dir))
    # Verify we found a real artifacts root (should contain session subdirs)
    if os.path.isdir(candidate_root) and candidate_root != loop_dir:
        artifacts_root = candidate_root
    else:
        artifacts_root = loop_dir
    stats_file = os.path.join(artifacts_root, "session_stats.jsonl")
    try:
        # Read current file state to decide whether to refresh. If the
        # file already shows a terminal status, skip the bridge refresh
        # (see docstring for the overwrite hazard).
        with open(status_file, "r", encoding="utf-8") as f:
            nav = json.load(f)
        file_status = nav.get("status")
        should_refresh = (
            bridge is not None
            and loop_id
            and file_status not in TERMINAL_STATUSES
        )
        if should_refresh:
            try:
                bridge.call("get_nav_loop_status", {
                    "loop_id": loop_id,
                    "include_nav_status": True,
                })
            except Exception as exc:
                log(f"session_stats: failed to refresh _debug: {exc}")
            # Re-read after refresh to pick up the fresher _debug
            with open(status_file, "r", encoding="utf-8") as f:
                nav = json.load(f)
        events: List[Dict] = []
        if os.path.isfile(events_file):
            with open(events_file, "r", encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
        round_ends = [e for e in events if e.get("phase") == "round_end"]
        effective = 0
        if round_ends and round_ends[0].get("total_steps", 0) > 0:
            effective += 1
        for i in range(1, len(round_ends)):
            if round_ends[i].get("total_steps", 0) > round_ends[i - 1].get("total_steps", 0):
                effective += 1
        history = nav.get("action_history", [])
        tool_counts: Dict[str, int] = {}
        for entry in history:
            action = str(entry.get("action", "") or "").split()[0].lower()
            if action:
                tool_counts[action] = tool_counts.get(action, 0) + 1
        cap_req = nav.get("capability_request")
        cap_requests = [cap_req] if isinstance(cap_req, str) and cap_req else []
        debug = nav.get("_debug", {}) or {}

        # Whether this session has a GT goal coordinate (i.e. metrics computable).
        # Reads eval_goal_position (invisible to agent) which is the
        # evaluation-only GT, independent of agent-facing goal_position.
        eval_goal_position = nav.get("eval_goal_position")
        has_gt_goal = bool(nav.get("has_ground_truth", False)) or (
            isinstance(eval_goal_position, list) and len(eval_goal_position) >= 3
        )

        # Success distance threshold: default 0.5m for all tasks (matches
        # the shipped controller prompts which tell the agent to mark
        # "reached" at 0.5m). Override via hab_nav_loop_start parameter
        # or nav_status.success_distance_threshold for stricter evaluation
        # (e.g., 0.2m for Habitat PointNav benchmark comparison).
        task_type = nav.get("task_type", "pointnav")
        threshold_override = nav.get("success_distance_threshold")
        if isinstance(threshold_override, (int, float)) and threshold_override > 0:
            success_threshold = float(threshold_override)
        else:
            success_threshold = 0.5

        # End-of-loop distances are stored as two SEPARATE fields so that
        # downstream consumers (reports, dashboards, TUIs) can make their
        # own explicit choice. The previous single merged `gt_end_distance`
        # field was ambiguous — readers could not tell whether a number
        # was a navmesh geodesic or an unmeshed euclidean, and reports
        # were averaging the two together as if they were the same quantity.
        gt_end_geodesic: Optional[float] = None
        gt_end_euclidean: Optional[float] = None
        spl = None
        success = None
        if has_gt_goal:
            # For navmesh tasks (has_navmesh=True), require geodesic distance
            # for success judgment: points off-mesh or on disconnected islands
            # may have low euclidean but be unreachable.
            has_navmesh = bool(nav.get("has_navmesh", False))
            gt_end_geodesic = debug.get("gt_geodesic_distance")
            gt_end_euclidean = debug.get("gt_euclidean_distance")

            agent_reached = nav.get("status") == "reached"
            if has_navmesh:
                # Strict: must use geodesic; Euclidean fallback is not valid
                gt_success_judge = gt_end_geodesic
            else:
                # No navmesh available — fall back to euclidean
                gt_success_judge = gt_end_euclidean
            gt_success = (
                gt_success_judge is not None
                and gt_success_judge < success_threshold
            )
            success = bool(agent_reached and gt_success)
            l_opt = debug.get("gt_initial_geodesic_distance")
            l_actual = debug.get("gt_path_length")
            if (
                l_opt is not None
                and l_actual is not None
                and isinstance(l_opt, (int, float))
                and isinstance(l_actual, (int, float))
            ):
                if l_opt == 0:
                    # Trivial success: agent starts at goal
                    spl = 1.0 if success else 0.0
                elif l_opt > 0:
                    spl = round((1.0 if success else 0.0) * (l_opt / max(l_opt, l_actual)), 4)

        stats = {
            "loop_id": nav.get("task_id"),
            "task_type": nav.get("task_type"),
            "nav_mode": nav.get("nav_mode"),
            # has_navmesh is now part of the row so consumers can bucket
            # sessions by scene capability without needing to cross-reference
            # nav_status.json. Reports should split averages by this field.
            "has_navmesh": bool(nav.get("has_navmesh", False)),
            "outcome": nav.get("status"),
            "rounds": len(round_ends),
            "effective_rounds": effective,
            "total_steps": nav.get("total_steps", 0),
            "collisions": nav.get("collisions", 0),
            # GT-based evaluation (only when evaluation GT was provided)
            "has_gt_goal": has_gt_goal,
            "success_distance_threshold": success_threshold if has_gt_goal else None,
            "gt_success": success,
            # Unambiguous end distances — geodesic and euclidean stored
            # independently so reports can split buckets and never
            # average heterogeneous quantities. Replaces the old
            # single merged `gt_end_distance` field.
            "gt_end_geodesic_distance": gt_end_geodesic,
            "gt_end_euclidean_distance": gt_end_euclidean,
            "gt_initial_geodesic_distance": debug.get("gt_initial_geodesic_distance") if has_gt_goal else None,
            "gt_path_length": debug.get("gt_path_length") if has_gt_goal else None,
            "spl": spl,
            "tool_usage": tool_counts,
            "action_history_len": len(history),
            "capability_requests": cap_requests,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except Exception as exc:
        # Read/parse failures are non-fatal: we log and return without
        # writing a row (better than writing a corrupt row).
        log(f"session_stats collection failed during read/build: {exc}")
        return

    # Append phase: log loudly on failure but do NOT raise.
    #
    # History:
    #   - Original code silently swallowed failures (bug B4) → we log now.
    #   - Initial B4 fix re-raised OSError for visibility, but Codex
    #     follow-up review found that a raise here propagates up through
    #     agent.run() and lands in main()'s fatal handler, which then
    #     rewrites a successful loop to status="error". A lost analytics
    #     row is bad, but corrupting the nav outcome is worse.
    #
    # Current contract: the ERROR log line is the operator signal.
    # Analytics I/O must never influence the loop's terminal outcome.
    try:
        append_jsonl_atomic(stats_file, stats)
    except OSError as exc:
        log(
            f"ERROR: failed to append session_stats row "
            f"(outcome {stats.get('outcome')!r} preserved on disk): {exc!r}"
        )


__all__ = ["collect_session_stats"]
