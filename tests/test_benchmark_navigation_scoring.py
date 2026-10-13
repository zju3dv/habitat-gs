"""Public navigation scoring contract; simulator integration is checked separately."""
import copy
import math

import pytest

from scripts_gs.benchmark.score import aggregate, score_episode


def episode(ordered=False):
    positions = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]] if ordered else [[1.0, 0.0, 0.0]]
    return {
        "episode_id": "test", "schema": "habitat_gs/objectnav/v1", "ordered": ordered,
        "start": {"position": [0.0, 0.0, 0.0]},
        "targets": [{"viewpoints": [{"status": "approved", "pose": {"position": p}}]} for p in positions],
        "evaluation": {"success": {"metric": "distance_to_approved_viewpoint_set", "threshold_m": 0.1, "comparator": "<"}},
        "ground_truth": {"variants": {"global_shortest": {"total_geodesic_distance": float(len(positions))}}},
    }


def events(*pairs):
    return {"events": [{"action": "start", "position": [0.0, 0.0, 0.0]}] + [
        {"action": action, "position": [x, 0.0, 0.0]} for action, x in pairs
    ]}


def score(ep, rows, nearest=None):
    return score_episode(ep, rows, nearest or (lambda a, goals: min(math.dist(a, b) for b in goals)), lambda p: True)


@pytest.mark.parametrize("distance,success", [(0.099999, True), (0.1, False), (0.100001, False), (math.inf, False)])
def test_strict_geodesic_boundary(distance, success):
    result = score(episode(), events(("forward", 1), ("stop", 1)), lambda a, goals: distance)
    assert result["success"] is success


def test_single_requires_stop_at_goal_not_an_earlier_visit():
    assert not score(episode(), events(("forward", 1)))["success"]
    assert not score(episode(), events(("forward", 1), ("backward", 0), ("stop", 0)))["success"]


def test_multi_order_and_first_stop_are_irreversible():
    assert not score(episode(True), events(("forward", 2), ("backward", 1), ("stop", 1)))["success"]
    assert not score(episode(True), events(("forward", 1), ("stop", 1), ("forward", 2), ("stop", 2)))["success"]
    result = score(episode(True), events(("forward", 1), ("forward", 2), ("backward", 1), ("stop", 1)))
    assert result["success"] and result["visited_targets"] == 2
    assert result["path_length_m"] == 3 and result["spl"] == pytest.approx(2 / 3)


def test_same_position_can_visit_overlapping_ordered_regions():
    ep = episode(True)
    ep["targets"][1] = copy.deepcopy(ep["targets"][0])
    assert score(ep, events(("forward", 1), ("stop", 1)))["visited_targets"] == 2


def test_distance_callback_not_bbox_or_reference_endpoint_is_authority():
    ep = episode()
    ep["targets"][0]["bbox"] = {"center": [0, 0, 0], "size": [100, 100, 100]}
    assert not score(ep, events(("forward", 1), ("stop", 1)), lambda *_: math.inf)["success"]


def test_error_missing_and_budget_denominators():
    missing = score(episode(), None)
    error = score(episode(), events(("runtime_error", 0)))
    budget = score(episode(), events(("budget_exhausted", 0), ("stop", 1)))
    assert error["success"] is None and error["spl"] is None
    assert budget["status"] == "budget_exhausted" and not budget["success"]
    summary = aggregate([missing, error, budget])
    assert summary["total"] == 3 and summary["evaluated"] == 2
    assert summary["runtime_errors"] == 1 and summary["success_rate"] == 0


def test_reject_bad_start_nonfinite_pose_and_moving_stop():
    for rows in [events(("stop", 1)), events(("forward", math.nan))]:
        with pytest.raises(ValueError):
            score(episode(), rows)
    rows = events(("stop", 0))
    rows["events"][0]["position"][0] = 1
    with pytest.raises(ValueError, match="start"):
        score(episode(), rows)
