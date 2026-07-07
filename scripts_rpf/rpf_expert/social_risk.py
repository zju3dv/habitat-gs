"""Simple geometric risk checks for the minimal following expert."""

from __future__ import annotations

import numpy as np

from .trajectory_utils import (
    AgentState,
    ExpertConfig,
    distance_point_to_segment,
    forward_from_theta,
    left_from_theta,
    normalize,
    speed,
    state_xy,
)


def detect_same_side_oncoming(
    target_future: list[AgentState],
    pedestrians_future: dict[int, list[AgentState]],
    config: ExpertConfig,
) -> bool:
    """Return True when an oncoming pedestrian threatens the preferred side pocket."""
    side_sign = 1.0 if config.preferred_side >= 0 else -1.0
    for idx, target in enumerate(target_future):
        target_pos = state_xy(target)
        forward = forward_from_theta(target.pose.theta)
        left = left_from_theta(target.pose.theta)
        preferred_ref = (
            target_pos
            - config.default_distance * forward
            + side_sign * config.preferred_lateral_offset * left
        )
        for ped_future in pedestrians_future.values():
            if idx >= len(ped_future):
                continue
            ped = ped_future[idx]
            ped_pos = state_xy(ped)
            side = float(np.dot(ped_pos - target_pos, left))
            closing = float(np.dot(np.asarray(ped.velocity, dtype=np.float32), forward)) < -0.2
            # Keep this threshold fairly tight so the toy expert does not abandon
            # the side-following formation before the pedestrian is actually
            # close to the side-rear pocket.
            near_preferred_pocket = (
                float(np.linalg.norm(ped_pos - preferred_ref)) < config.social_radius
            )
            if side * side_sign > 0.0 and closing and near_preferred_pocket:
                return True
    return False


def detect_visibility_risk(
    robot_state: AgentState,
    target_future: list[AgentState],
    pedestrians_future: dict[int, list[AgentState]],
    corridor_radius: float = 0.5,
) -> tuple[float, int]:
    """Approximate line-of-sight obstruction risk and side of obstruction."""
    robot_pos = state_xy(robot_state)
    max_risk = 0.0
    side_vote = 0
    for idx, target in enumerate(target_future):
        target_pos = state_xy(target)
        target_left = left_from_theta(target.pose.theta)
        for ped_future in pedestrians_future.values():
            if idx >= len(ped_future):
                continue
            ped_pos = state_xy(ped_future[idx])
            dist = distance_point_to_segment(ped_pos, robot_pos, target_pos)
            if dist < corridor_radius:
                risk = 1.0 - dist / corridor_radius
                if risk > max_risk:
                    max_risk = float(risk)
                    side = float(np.dot(ped_pos - target_pos, target_left))
                    side_vote = 1 if side >= 0.0 else -1
    return max_risk, side_vote


def detect_crowd_or_stop(
    target_future: list[AgentState],
    pedestrians_future: dict[int, list[AgentState]],
    config: ExpertConfig,
) -> tuple[bool, float]:
    """Detect stopped target or crowded target neighborhood."""
    if not target_future:
        return False, 0.0
    target_now = target_future[0]
    stopped = speed(target_now) < 0.1
    max_count = 0
    for idx, target in enumerate(target_future):
        target_pos = state_xy(target)
        count = 0
        for ped_future in pedestrians_future.values():
            if idx < len(ped_future):
                if float(np.linalg.norm(state_xy(ped_future[idx]) - target_pos)) < config.social_radius:
                    count += 1
        max_count = max(max_count, count)
    crowded = max_count >= 2
    risk = min(1.0, max_count / 2.0)
    return stopped or crowded, float(risk)


def safety_project(
    points: np.ndarray,
    pedestrians_future: dict[int, list[AgentState]],
    config: ExpertConfig,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Push trajectory points away from pedestrians using a small social buffer."""
    projected = points.copy()
    unsafe = False
    collision = False
    social_risk = 0.0
    for idx in range(projected.shape[0]):
        for ped_future in pedestrians_future.values():
            if idx >= len(ped_future):
                continue
            ped_pos = state_xy(ped_future[idx])
            delta = projected[idx] - ped_pos
            dist = float(np.linalg.norm(delta))
            if dist < config.social_radius:
                social_risk = max(social_risk, 1.0 - dist / config.social_radius)
            if dist < config.personal_radius:
                direction = normalize(delta)
                if float(np.linalg.norm(direction)) < 1.0e-6:
                    direction = np.array([1.0, 0.0], dtype=np.float32)
                projected[idx] = projected[idx] + config.safety_push_beta * direction
                dist_after = float(np.linalg.norm(projected[idx] - ped_pos))
                if dist_after < config.collision_radius:
                    unsafe = True
                    collision = True
                elif dist_after < config.personal_radius:
                    unsafe = True
    return projected.astype(np.float32), {
        "social_risk": round(float(social_risk), 5),
        "unsafe": bool(unsafe),
        "collision": bool(collision),
    }
