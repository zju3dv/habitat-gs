#!/usr/bin/env python3
"""Generate cross-session analytics report from session_stats.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_stats(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def report_overall(entries: List[Dict[str, Any]]) -> None:
    print_section("Overall Statistics")
    n = len(entries)
    print(f"  Total sessions: {n}")
    if n == 0:
        return

    outcomes = Counter(e.get("outcome", "unknown") for e in entries)
    reached = outcomes.get("reached", 0)
    print(f"  Agent-reported reached: {reached}/{n} ({100 * reached / n:.0f}%)")
    print(f"  Outcomes: {dict(outcomes)}  (note: agent's self-report, not GT-verified)")

    rounds = [e.get("rounds", 0) for e in entries]
    steps = [e.get("total_steps", 0) for e in entries]
    collisions = [e.get("collisions", 0) for e in entries]
    effective = [e.get("effective_rounds", 0) for e in entries]

    print(f"  Avg rounds/session: {sum(rounds) / n:.1f}")
    print(f"  Avg effective rounds: {sum(effective) / n:.1f}")
    print(f"  Avg steps/session: {sum(steps) / n:.1f}")
    print(f"  Avg collisions/session: {sum(collisions) / n:.1f}")

    # GT-based metrics (only for sessions with goal_position set)
    gt_entries = [e for e in entries if e.get("has_gt_goal")]
    non_gt = n - len(gt_entries)
    if non_gt > 0:
        print(f"  Sessions without GT goal (metrics skipped): {non_gt}")

    if gt_entries:
        print_section("GT-evaluated Metrics")
        k = len(gt_entries)
        print(f"  GT-evaluated sessions: {k}")

        # Report the actual thresholds used (from per-session data, not hardcoded)
        thresholds = sorted({
            e.get("success_distance_threshold")
            for e in gt_entries
            if e.get("success_distance_threshold") is not None
        })
        if len(thresholds) == 1:
            thr_text = f"within {thresholds[0]}m"
        elif len(thresholds) > 1:
            thr_text = f"within per-session threshold ({min(thresholds)}–{max(thresholds)}m)"
        else:
            thr_text = "within threshold"

        gt_success = sum(1 for e in gt_entries if e.get("gt_success"))
        print(f"  GT success rate (reached + {thr_text}): {gt_success}/{k} ({100 * gt_success / k:.0f}%)")

        # End distances are reported in two SEPARATE buckets so we never
        # average heterogeneous quantities. Navmesh sessions use the
        # geodesic end distance (authoritative — accounts for walls);
        # non-navmesh sessions use the euclidean end distance (the only
        # thing computable without a pathfinder). The old code merged
        # the two into a single "Avg GT end distance" line, which was
        # mathematically meaningless whenever a report contained a mix.
        nav_entries = [e for e in gt_entries if e.get("has_navmesh")]
        nomesh_entries = [e for e in gt_entries if not e.get("has_navmesh")]

        nav_geo_dists = [
            e["gt_end_geodesic_distance"]
            for e in nav_entries
            if e.get("gt_end_geodesic_distance") is not None
        ]
        if nav_geo_dists:
            avg = sum(nav_geo_dists) / len(nav_geo_dists)
            print(
                f"  Avg end distance (navmesh, geodesic): {avg:.2f}m "
                f"({len(nav_geo_dists)} session{'s' if len(nav_geo_dists) != 1 else ''})"
            )

        nomesh_eu_dists = [
            e["gt_end_euclidean_distance"]
            for e in nomesh_entries
            if e.get("gt_end_euclidean_distance") is not None
        ]
        if nomesh_eu_dists:
            avg = sum(nomesh_eu_dists) / len(nomesh_eu_dists)
            print(
                f"  Avg end distance (no navmesh, euclidean): {avg:.2f}m "
                f"({len(nomesh_eu_dists)} session{'s' if len(nomesh_eu_dists) != 1 else ''})"
            )

        spl_values = [e["spl"] for e in gt_entries if e.get("spl") is not None]
        if spl_values:
            coverage_pct = 100 * len(spl_values) / k
            print(f"  Avg SPL: {sum(spl_values) / len(spl_values):.3f}")
            print(f"  SPL coverage: {len(spl_values)}/{k} ({coverage_pct:.0f}%) of GT sessions")
            if len(spl_values) < k:
                print(f"    (remaining {k - len(spl_values)} GT sessions lacked l_opt/l_actual data)")

        path_lengths = [e["gt_path_length"] for e in gt_entries if e.get("gt_path_length") is not None]
        if path_lengths:
            print(f"  Avg actual path length: {sum(path_lengths) / len(path_lengths):.2f}m")

        init_geos = [e["gt_initial_geodesic_distance"] for e in gt_entries if e.get("gt_initial_geodesic_distance") is not None]
        if init_geos:
            print(f"  Avg initial geodesic distance: {sum(init_geos) / len(init_geos):.2f}m")


def report_by_group(entries: List[Dict[str, Any]]) -> None:
    print_section("By Task Type x Nav Mode")
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        key = f"{e.get('task_type', '?')} x {e.get('nav_mode', '?')}"
        groups[key].append(e)

    for key, group in sorted(groups.items()):
        n = len(group)
        reached = sum(1 for e in group if e.get("outcome") == "reached")
        avg_steps = sum(e.get("total_steps", 0) for e in group) / n
        print(f"\n  [{key}] ({n} sessions)")
        print(f"    Agent-reported reached: {reached}/{n} ({100 * reached / n:.0f}%)")
        print(f"    Avg steps: {avg_steps:.1f}")
        # GT-based success rate (only over GT-evaluated sessions in this group)
        gt_group = [e for e in group if e.get("has_gt_goal")]
        if gt_group:
            k = len(gt_group)
            gt_ok = sum(1 for e in gt_group if e.get("gt_success"))
            print(f"    GT success: {gt_ok}/{k} ({100 * gt_ok / k:.0f}%)")
            spls = [e["spl"] for e in gt_group if e.get("spl") is not None]
            if spls:
                print(f"    Avg SPL: {sum(spls) / len(spls):.3f}")


def report_tool_usage(entries: List[Dict[str, Any]]) -> None:
    print_section("Tool Usage")
    total_usage: Counter = Counter()
    usage_in_success: Counter = Counter()
    usage_in_failure: Counter = Counter()
    n_success = 0
    n_failure = 0

    for e in entries:
        tools = e.get("tool_usage", {})
        # Success is judged by gt_success (the evaluator's verdict) when
        # available, falling back to outcome only for sessions without
        # GT. The previous code used outcome everywhere, which mixed
        # "agent self-reported reached" with "GT-verified success" in
        # the same bucket — a session where the agent gave up claiming
        # reached=True while actually stopping 2m from the goal would
        # be counted as a success.
        if e.get("has_gt_goal"):
            is_success = bool(e.get("gt_success"))
        else:
            is_success = e.get("outcome") == "reached"
        if is_success:
            n_success += 1
        else:
            n_failure += 1
        for tool, count in tools.items():
            total_usage[tool] += count
            if is_success:
                usage_in_success[tool] += count
            else:
                usage_in_failure[tool] += count

    n = len(entries)
    print(f"\n  {'Tool':<25s} {'Total':>6s}  {'Avg/sess':>8s}  {'In success':>10s}  {'In failure':>10s}")
    print(f"  {'-' * 65}")
    for tool, count in total_usage.most_common():
        avg = count / n if n > 0 else 0
        s_count = usage_in_success.get(tool, 0)
        f_count = usage_in_failure.get(tool, 0)
        print(f"  {tool:<25s} {count:>6d}  {avg:>8.1f}  {s_count:>10d}  {f_count:>10d}")

    # Correlation hint
    if n_success > 0 and n_failure > 0:
        print(f"\n  Sessions: {n_success} reached, {n_failure} not reached")
        all_tools = set(total_usage.keys())
        for tool in sorted(all_tools):
            s_avg = usage_in_success[tool] / n_success if n_success else 0
            f_avg = usage_in_failure[tool] / n_failure if n_failure else 0
            if s_avg > f_avg * 1.5 or f_avg > s_avg * 1.5:
                direction = "higher in success" if s_avg > f_avg else "higher in failure"
                print(f"    {tool}: {s_avg:.1f}/success vs {f_avg:.1f}/failure ({direction})")


def report_capability_requests(entries: List[Dict[str, Any]]) -> None:
    requests: List[str] = []
    for e in entries:
        reqs = e.get("capability_requests", [])
        if isinstance(reqs, list):
            requests.extend(r for r in reqs if isinstance(r, str))
    if not requests:
        return

    print_section("Agent Capability Requests")
    counts = Counter(requests)
    for req, count in counts.most_common(20):
        print(f"  [{count}x] {req}")


def report_failure_patterns(entries: List[Dict[str, Any]]) -> None:
    failures = [e for e in entries if e.get("outcome") != "reached"]
    if not failures:
        return

    print_section("Failure Analysis")
    by_outcome = defaultdict(list)
    for e in failures:
        by_outcome[e.get("outcome", "unknown")].append(e)

    for outcome, group in sorted(by_outcome.items()):
        avg_steps = sum(e.get("total_steps", 0) for e in group) / len(group)
        avg_col = sum(e.get("collisions", 0) for e in group) / len(group)
        print(f"\n  {outcome} ({len(group)} sessions)")
        print(f"    Avg steps: {avg_steps:.1f}, avg collisions: {avg_col:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cross-session analytics report")
    parser.add_argument(
        "--stats-file",
        default=None,
        help="Path to session_stats.jsonl (default: auto-detect in artifacts/habitat-gs/)",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of text report")
    args = parser.parse_args()

    if args.stats_file:
        stats_path = Path(args.stats_file)
    else:
        candidates = list(Path(".").rglob("session_stats.jsonl"))
        if not candidates:
            print("No session_stats.jsonl found. Run some nav_loops first.", file=sys.stderr)
            raise SystemExit(1)
        stats_path = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"Using: {stats_path}")

    entries = load_stats(stats_path)
    if not entries:
        print("No entries in stats file.", file=sys.stderr)
        raise SystemExit(1)

    if args.json:
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return

    report_overall(entries)
    report_by_group(entries)
    report_tool_usage(entries)
    report_capability_requests(entries)
    report_failure_patterns(entries)
    print()


if __name__ == "__main__":
    main()
