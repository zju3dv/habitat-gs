"""Forecast-aware formation expert for minimal robot-person-following data."""

from __future__ import annotations

import numpy as np

from .controller import pure_pursuit_cmd
from .social_risk import (
    detect_crowd_or_stop,
    detect_same_side_oncoming,
    detect_visibility_risk,
    safety_project,
)
from .trajectory_utils import (
    AgentState,
    ExpertConfig,
    as_local_trajectory,
    as_world_trajectory,
    clip,
    forward_from_theta,
    left_from_theta,
    moving_average,
    state_xy,
)


class ForecastAwareFormationExpert:
    """Rule-based expert that returns local trajectory labels and cmd_vel."""

    def __init__(self, config: ExpertConfig):
        self.config = config

    def _formation_parameters(
        self,
        robot_state: AgentState,
        target_future: list[AgentState],
        pedestrians_future: dict[int, list[AgentState]],
    ) -> tuple[float, float, dict[str, float | bool | str]]:
        cfg = self.config
        d = cfg.default_distance
        y = cfg.preferred_side * cfg.preferred_lateral_offset
        scenario_type = "normal_follow"
        speed_scale = 1.0

        same_side_oncoming = detect_same_side_oncoming(target_future, pedestrians_future, cfg)
        visibility_risk, obstruction_side = detect_visibility_risk(
            robot_state, target_future, pedestrians_future
        )
        crowd_or_stop, crowd_risk = detect_crowd_or_stop(target_future, pedestrians_future, cfg)

        if same_side_oncoming:
            d = 2.3
            y = 0.0
            scenario_type = "same_side_oncoming"

        if visibility_risk > 0.0 and obstruction_side != 0:
            # Obstruction from target-left means shift right, and vice versa.
            y = -obstruction_side * cfg.preferred_lateral_offset
            scenario_type = "visibility_avoidance"

        if crowd_or_stop:
            d = min(d + 0.5, cfg.max_distance)
            speed_scale = 0.6
            scenario_type = "crowd_or_stop"

        d = clip(d, cfg.min_distance, cfg.max_distance)
        y = clip(y, -cfg.max_lateral_offset, cfg.max_lateral_offset)
        return d, y, {
            "scenario_type": scenario_type,
            "visibility_risk": round(float(visibility_risk), 5),
            "crowd_risk": round(float(crowd_risk), 5),
            "same_side_oncoming": bool(same_side_oncoming),
            "speed_scale": round(float(speed_scale), 5),
        }

    def _reference_points(
        self,
        target_future: list[AgentState],
        d: float,
        y: float,
    ) -> np.ndarray:
        points = []
        for target in target_future:
            target_pos = state_xy(target)
            forward = forward_from_theta(target.pose.theta)
            left = left_from_theta(target.pose.theta)
            points.append(target_pos - d * forward + y * left)
        return np.asarray(points, dtype=np.float32)

    def step(
        self,
        robot_state: AgentState,
        target_future: list[AgentState],
        pedestrians_future: dict[int, list[AgentState]],
    ) -> dict:
        """Return expert world/local trajectories, cmd_vel, labels, and debug info."""
        if not target_future:
            raise ValueError("target_future must contain at least one AgentState.")

        d, y, rule_info = self._formation_parameters(
            robot_state, target_future, pedestrians_future
        )
        ref_points = self._reference_points(target_future, d=d, y=y)
        projected, safety = safety_project(ref_points, pedestrians_future, self.config)
        smoothed = moving_average(projected, window=3)

        trajectory_world = as_world_trajectory(
            smoothed, fallback_yaw=target_future[0].pose.theta
        )
        trajectory_local = as_local_trajectory(robot_state.pose, smoothed)
        labels = {
            "visibility_risk": rule_info["visibility_risk"],
            "social_risk": safety["social_risk"],
            "unsafe": bool(safety["unsafe"]),
            "collision": bool(safety["collision"]),
            "target_lost": False,
        }
        cmd_vel = pure_pursuit_cmd(
            trajectory_local,
            self.config,
            unsafe=labels["unsafe"],
            speed_scale=float(rule_info["speed_scale"]),
        )
        return {
            "expert_trajectory_world": trajectory_world,
            "expert_trajectory_local": trajectory_local,
            "cmd_vel": cmd_vel,
            "labels": labels,
            "debug": {
                "d": round(float(d), 5),
                "y": round(float(y), 5),
                **rule_info,
            },
        }
