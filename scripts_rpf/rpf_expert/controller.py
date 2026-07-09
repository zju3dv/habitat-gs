"""Low-level controller for local expert trajectories."""

from __future__ import annotations

import math

from .trajectory_utils import ExpertConfig, clip


def pure_pursuit_cmd(
    expert_trajectory_local: list[list[float]],
    config: ExpertConfig,
    unsafe: bool = False,
    speed_scale: float = 1.0,
) -> dict[str, float]:
    """Convert local waypoints to a differential-drive style cmd_vel."""
    if not expert_trajectory_local:
        return {"v": 0.0, "omega": 0.0}
    idx = min(config.lookahead_index, len(expert_trajectory_local) - 1)
    dx, dy = expert_trajectory_local[idx]
    distance = math.hypot(dx, dy)
    heading_error = math.atan2(dy, dx)
    v = clip(config.k_v * distance, 0.0, config.v_max)
    omega = clip(config.k_omega * heading_error, -config.omega_max, config.omega_max)
    v *= clip(speed_scale, 0.0, 1.0)
    if unsafe:
        v *= 0.5
    return {"v": round(float(v), 5), "omega": round(float(omega), 5)}


def integrate_unicycle(pose, cmd_vel: dict[str, float], dt: float):
    """Forward integrate a 2D unicycle pose."""
    from .trajectory_utils import Pose2D, wrap_angle

    theta = wrap_angle(pose.theta + float(cmd_vel["omega"]) * dt)
    x = pose.x + float(cmd_vel["v"]) * math.cos(theta) * dt
    y = pose.y + float(cmd_vel["v"]) * math.sin(theta) * dt
    return Pose2D(float(x), float(y), float(theta))
