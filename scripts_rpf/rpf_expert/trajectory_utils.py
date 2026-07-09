"""Small geometry and data utilities for robot-person-following experts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass
class AgentState:
    agent_id: int
    pose: Pose2D
    velocity: tuple[float, float]


@dataclass
class ExpertConfig:
    horizon_sec: float = 3.0
    dt: float = 0.2
    default_distance: float = 2.0
    preferred_lateral_offset: float = 0.8
    preferred_side: int = 1
    min_distance: float = 1.5
    max_distance: float = 3.0
    max_lateral_offset: float = 1.2
    collision_radius: float = 0.4
    personal_radius: float = 0.8
    social_radius: float = 1.2
    safety_push_beta: float = 0.3
    v_max: float = 1.0
    omega_max: float = 1.2
    lookahead_index: int = 3
    k_v: float = 0.8
    k_omega: float = 1.5


def clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def pose_xy(pose: Pose2D) -> np.ndarray:
    return np.array([pose.x, pose.y], dtype=np.float32)


def state_xy(state: AgentState) -> np.ndarray:
    return pose_xy(state.pose)


def forward_from_theta(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)


def left_from_theta(theta: float) -> np.ndarray:
    return np.array([-math.sin(theta), math.cos(theta)], dtype=np.float32)


def speed(state: AgentState) -> float:
    return float(math.hypot(state.velocity[0], state.velocity[1]))


def normalize(vec: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def distance_point_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom < 1.0e-8:
        return float(np.linalg.norm(point - start))
    t = clip(float(np.dot(point - start, segment) / denom), 0.0, 1.0)
    closest = start + t * segment
    return float(np.linalg.norm(point - closest))


def world_to_local(robot_pose: Pose2D, point_world: np.ndarray) -> np.ndarray:
    delta = point_world - pose_xy(robot_pose)
    forward = forward_from_theta(robot_pose.theta)
    left = left_from_theta(robot_pose.theta)
    return np.array([float(np.dot(delta, forward)), float(np.dot(delta, left))], dtype=np.float32)


def local_to_world(robot_pose: Pose2D, point_local: np.ndarray) -> np.ndarray:
    forward = forward_from_theta(robot_pose.theta)
    left = left_from_theta(robot_pose.theta)
    return pose_xy(robot_pose) + point_local[0] * forward + point_local[1] * left


def moving_average(points: np.ndarray, window: int = 3) -> np.ndarray:
    if points.shape[0] < 3 or window <= 1:
        return points.copy()
    radius = window // 2
    out = points.copy()
    for idx in range(points.shape[0]):
        lo = max(0, idx - radius)
        hi = min(points.shape[0], idx + radius + 1)
        out[idx] = np.mean(points[lo:hi], axis=0)
    return out.astype(np.float32)


def yaw_from_points(points: np.ndarray, fallback: float = 0.0) -> list[float]:
    yaws = []
    last = fallback
    for idx in range(points.shape[0]):
        if idx < points.shape[0] - 1:
            delta = points[idx + 1] - points[idx]
        elif idx > 0:
            delta = points[idx] - points[idx - 1]
        else:
            delta = np.zeros(2, dtype=np.float32)
        if abs(float(delta[0])) + abs(float(delta[1])) > 1.0e-6:
            last = math.atan2(float(delta[1]), float(delta[0]))
        yaws.append(float(last))
    return yaws


def as_world_trajectory(points: np.ndarray, fallback_yaw: float = 0.0) -> list[list[float]]:
    yaws = yaw_from_points(points, fallback=fallback_yaw)
    return [
        [round(float(point[0]), 5), round(float(point[1]), 5), round(float(theta), 6)]
        for point, theta in zip(points, yaws)
    ]


def as_local_trajectory(robot_pose: Pose2D, points: Iterable[np.ndarray]) -> list[list[float]]:
    return [
        [round(float(local[0]), 5), round(float(local[1]), 5)]
        for local in (world_to_local(robot_pose, np.asarray(point, dtype=np.float32)) for point in points)
    ]
