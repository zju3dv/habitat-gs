"""Score recorded ObjectNav/ordered Multi-ObjectNav trajectories on their navmesh."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def position(value):
    if not isinstance(value, list) or len(value) != 3 or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        for x in value
    ):
        raise ValueError("position must contain three finite numbers")
    return value


def score_episode(episode, submission, nearest, navigable, reference="global_shortest"):
    """Callbacks are evaluator-owned; no GT or target arrival feedback goes to policy."""
    if episode["schema"] != "habitat_gs/objectnav/v1" or episode["evaluation"]["success"] != {
        "metric": "distance_to_approved_viewpoint_set", "threshold_m": 0.1, "comparator": "<"
    }:
        raise ValueError("unsupported episode scoring contract")
    targets = episode["targets"]
    if not targets or (not episode["ordered"] and len(targets) != 1):
        raise ValueError("expected a single goal or ordered goals")
    goals = []
    for target in targets:
        views = target["viewpoints"]
        if not views or any(v["status"] != "approved" for v in views):
            raise ValueError("release contains missing or unapproved viewpoints")
        goals.append([position(v["pose"]["position"]) for v in views])
    reference_length = episode["ground_truth"]["variants"][reference]["total_geodesic_distance"]
    if not math.isfinite(reference_length) or reference_length < 0:
        raise ValueError("invalid reference distance")
    result = {
        "episode_id": episode["episode_id"], "status": "missing_submission",
        "success": False, "spl": 0.0, "visited_targets": 0,
        "target_count": len(targets), "path_length_m": 0.0,
        "reference_variant": reference, "reference_length_m": reference_length,
    }
    if submission is None:
        return result
    events = submission.get("events")
    if not isinstance(events, list) or not events or events[0].get("action") != "start":
        raise ValueError("events must begin with the actual start pose")
    previous = position(events[0]["position"])
    if math.dist(previous, position(episode["start"]["position"])) > 1e-4:
        raise ValueError("trajectory start differs from episode start")
    active, length = 0, 0.0
    result["status"] = "missing_stop"
    for index, event in enumerate(events):
        action = event.get("action")
        if not isinstance(action, str) or not action or (index and action == "start"):
            raise ValueError("each event needs an action; start may occur only once")
        if action in {"runtime_error", "budget_exhausted"}:
            result["status"] = action
            break
        current = position(event["position"])
        if not navigable(current):
            raise ValueError("recorded position is not on the evaluation navmesh")
        delta = math.dist(previous, current)
        if action == "stop" and delta > 1e-4:
            raise ValueError("STOP cannot move the agent")
        length += delta
        previous = current
        if episode["ordered"]:
            # Several successive target regions may overlap at the same observed pose.
            while active < len(goals):
                distance = nearest(current, goals[active])
                if not (math.isfinite(distance) and 0 <= distance < 0.1):
                    break
                active += 1
        if action == "stop":
            if episode["ordered"]:
                success = active == len(goals)
            else:
                distance = nearest(current, goals[0])
                success = math.isfinite(distance) and 0 <= distance < 0.1
                active = int(success)
            result.update(status="stopped", success=success)
            break
    result.update(visited_targets=active, path_length_m=length)
    if result["status"] == "runtime_error":
        result.update(success=None, spl=None)
    elif result["success"]:
        denominator = max(reference_length, length)
        result["spl"] = reference_length / denominator if denominator else 1.0
    return result


def aggregate(results):
    evaluated = [r for r in results if r["status"] != "runtime_error"]
    return {
        "total": len(results), "evaluated": len(evaluated),
        "runtime_errors": len(results) - len(evaluated),
        "successes": sum(bool(r["success"]) for r in evaluated),
        "success_rate": sum(bool(r["success"]) for r in evaluated) / len(evaluated) if evaluated else None,
        "spl": sum(r["spl"] for r in evaluated) / len(evaluated) if evaluated else None,
        "missing_submissions": sum(r["status"] == "missing_submission" for r in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", choices=["global_shortest", "local_shortest"], default="global_shortest")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new file to preserve prior results")
    import habitat_sim

    episodes = json.loads(args.episodes.read_text())["episodes"]
    ids = {e["episode_id"] for e in episodes}
    if len(ids) != len(episodes):
        raise ValueError("duplicate dataset episode IDs")
    submissions = {}
    for row in json.loads(args.submissions.read_text())["episodes"]:
        ident = row["episode_id"]
        if ident not in ids or ident in submissions:
            raise ValueError("unknown or duplicate submission episode ID")
        submissions[ident] = row
    root, current_scene, results = args.scene_root.resolve(), None, []
    pf = habitat_sim.PathFinder()

    def nearest(start, goals):
        query = habitat_sim.MultiGoalShortestPath()
        query.requested_start, query.requested_ends = start, goals
        return float(query.geodesic_distance) if pf.find_path(query) else math.inf

    for episode in sorted(episodes, key=lambda e: e["scene"]["id"]):
        scene = episode["scene"]["asset"]
        if scene != current_scene:
            asset = (root / scene).resolve()
            if not asset.is_relative_to(root) or Path(scene).is_absolute():
                raise ValueError("scene path must stay inside --scene-root")
            navmesh = asset.with_suffix("").with_suffix(".navmesh")
            if not pf.load_nav_mesh(str(navmesh)):
                raise ValueError(f"cannot load navmesh: {navmesh.name}")
            current_scene = scene
        results.append(score_episode(episode, submissions.get(episode["episode_id"]), nearest, pf.is_navigable, args.reference))
    by_id = {row["episode_id"]: row for row in results}
    payload = {"schema": "habitat_gs_navigation_scores/v1", "summary": aggregate(results),
               "episodes": [by_id[e["episode_id"]] for e in episodes]}
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload["summary"]))


if __name__ == "__main__":
    main()
