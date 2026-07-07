#!/usr/bin/env python3
"""Run a minimal Forecast-Aware Formation Expert toy episode."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_rpf.rpf_expert import (
    AgentState,
    ExpertConfig,
    ForecastAwareFormationExpert,
    Pose2D,
)
from scripts_rpf.rpf_expert.controller import integrate_unicycle
from scripts_rpf.rpf_expert.logger import JsonlLogger
from scripts_rpf.rpf_expert.validator import validate_record


DEFAULT_OUT = REPO_ROOT / "outputs" / "minimal_expert_episode"


def make_state(agent_id: int, x: float, y: float, theta: float, vx: float, vy: float) -> AgentState:
    return AgentState(agent_id=agent_id, pose=Pose2D(x, y, theta), velocity=(vx, vy))


def future_target(t: float, horizon_steps: int, dt: float) -> list[AgentState]:
    states = []
    for k in range(horizon_steps):
        tau = t + k * dt
        states.append(make_state(1, 0.8 * tau, 0.0, 0.0, 0.8, 0.0))
    return states


def future_oncoming(t: float, horizon_steps: int, dt: float) -> list[AgentState]:
    states = []
    for k in range(horizon_steps):
        tau = t + k * dt
        states.append(make_state(2, 4.0 - 0.7 * tau, 1.0, math.pi, -0.7, 0.0))
    return states


def robot_record(robot: AgentState, last_cmd: dict[str, float]) -> dict:
    return {
        "x": round(float(robot.pose.x), 5),
        "y": round(float(robot.pose.y), 5),
        "theta": round(float(robot.pose.theta), 6),
        "v": round(float(last_cmd["v"]), 5),
        "omega": round(float(last_cmd["omega"]), 5),
    }


def run_episode(args: argparse.Namespace) -> tuple[Path, Path]:
    out_dir = Path(args.out).expanduser().resolve()
    jsonl_path = out_dir / "episode.jsonl"
    plot_path = out_dir / "topdown.png"
    cfg = ExpertConfig(
        preferred_side=args.preferred_side,
        horizon_sec=args.horizon_sec,
        dt=args.dt,
    )
    horizon_steps = max(2, int(round(cfg.horizon_sec / cfg.dt)) + 1)
    expert = ForecastAwareFormationExpert(cfg)

    robot = make_state(0, -2.0, 0.8, 0.0, 0.0, 0.0)
    last_cmd = {"v": 0.0, "omega": 0.0}
    robot_path = []
    target_path = []
    ped_path = []
    records = []

    with JsonlLogger(jsonl_path) as logger:
        for frame in range(args.frames):
            t = frame * cfg.dt
            target_future_states = future_target(t, horizon_steps, cfg.dt)
            ped_future_states = future_oncoming(t, horizon_steps, cfg.dt)
            robot_state = AgentState(
                agent_id=0,
                pose=robot.pose,
                velocity=(
                    last_cmd["v"] * math.cos(robot.pose.theta),
                    last_cmd["v"] * math.sin(robot.pose.theta),
                ),
            )
            result = expert.step(
                robot_state=robot_state,
                target_future=target_future_states,
                pedestrians_future={2: ped_future_states},
            )
            record = {
                "timestamp": round(float(t), 5),
                "episode_id": args.episode_id,
                "scene_id": args.scene_id,
                "scenario_type": result["debug"]["scenario_type"],
                "rgb_path": None,
                "robot_state": robot_record(robot_state, last_cmd),
                "target_id": 1,
                "expert_trajectory_world": result["expert_trajectory_world"],
                "expert_trajectory_local": result["expert_trajectory_local"],
                "expert_action": result["cmd_vel"],
                "labels": result["labels"],
                "debug": result["debug"],
            }
            validate_record(record)
            logger.write(record)
            records.append(record)

            target_now = target_future_states[0]
            ped_now = ped_future_states[0]
            robot_path.append([robot.pose.x, robot.pose.y])
            target_path.append([target_now.pose.x, target_now.pose.y])
            ped_path.append([ped_now.pose.x, ped_now.pose.y])

            print(
                "t={:.1f} robot=({:.2f},{:.2f},{:.2f}) target=({:.2f},{:.2f}) "
                "d={:.2f} y={:.2f} cmd=({:.2f},{:.2f}) social={:.2f} vis={:.2f} unsafe={}".format(
                    t,
                    robot.pose.x,
                    robot.pose.y,
                    robot.pose.theta,
                    target_now.pose.x,
                    target_now.pose.y,
                    result["debug"]["d"],
                    result["debug"]["y"],
                    result["cmd_vel"]["v"],
                    result["cmd_vel"]["omega"],
                    result["labels"]["social_risk"],
                    result["labels"]["visibility_risk"],
                    result["labels"]["unsafe"],
                )
            )

            last_cmd = result["cmd_vel"]
            robot.pose = integrate_unicycle(robot.pose, last_cmd, cfg.dt)

    save_plot(plot_path, np.asarray(robot_path), np.asarray(target_path), np.asarray(ped_path), records)
    return jsonl_path, plot_path


def save_plot(
    path: Path,
    robot_path: np.ndarray,
    target_path: np.ndarray,
    ped_path: np.ndarray,
    records: list[dict],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    times = np.asarray([float(record["timestamp"]) for record in records], dtype=np.float32)
    if times.size == 0:
        times = np.arange(robot_path.shape[0], dtype=np.float32)

    ax.plot(target_path[:, 0], target_path[:, 1], color="#1f77b4", linewidth=1.4, alpha=0.45, label="target path")
    ax.plot(ped_path[:, 0], ped_path[:, 1], color="#d62728", linewidth=1.4, alpha=0.45, label="oncoming path")
    ax.plot(robot_path[:, 0], robot_path[:, 1], color="#111111", linewidth=1.0, alpha=0.35, label="robot path")

    robot_scatter = ax.scatter(
        robot_path[:, 0],
        robot_path[:, 1],
        c=times,
        cmap="viridis",
        s=18,
        label="robot time",
        zorder=4,
    )
    ax.scatter(target_path[:, 0], target_path[:, 1], c=times, cmap="Blues", s=10, alpha=0.65, zorder=3)
    ax.scatter(ped_path[:, 0], ped_path[:, 1], c=times, cmap="Reds", s=10, alpha=0.65, zorder=3)

    ax.scatter(robot_path[0, 0], robot_path[0, 1], color="#111111", marker="o", s=42, zorder=5)
    ax.scatter(robot_path[-1, 0], robot_path[-1, 1], color="#111111", marker="x", s=55, zorder=5)

    snapshot_step = max(1, len(records) // 8)
    for idx in range(0, len(records), snapshot_step):
        traj = np.asarray([[p[0], p[1]] for p in records[idx]["expert_trajectory_world"]])
        ax.plot(traj[:, 0], traj[:, 1], color="#2ca02c", alpha=0.24, linewidth=1.2)
        ax.plot(
            [robot_path[idx, 0], target_path[idx, 0]],
            [robot_path[idx, 1], target_path[idx, 1]],
            color="#888888",
            alpha=0.22,
            linewidth=0.8,
        )
        ax.text(
            robot_path[idx, 0],
            robot_path[idx, 1] + 0.12,
            f"R {records[idx]['timestamp']:.1f}s",
            fontsize=7,
            color="#222222",
            ha="center",
        )
        ax.text(
            target_path[idx, 0],
            target_path[idx, 1] - 0.18,
            f"T {records[idx]['timestamp']:.1f}s",
            fontsize=7,
            color="#1f77b4",
            ha="center",
        )
        ax.text(
            ped_path[idx, 0],
            ped_path[idx, 1] + 0.18,
            f"P {records[idx]['timestamp']:.1f}s",
            fontsize=7,
            color="#d62728",
            ha="center",
        )
        ax.annotate(
            "",
            xy=(robot_path[min(idx + 1, len(robot_path) - 1), 0], robot_path[min(idx + 1, len(robot_path) - 1), 1]),
            xytext=(robot_path[idx, 0], robot_path[idx, 1]),
            arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 0.8, "alpha": 0.7},
        )

    cbar = fig.colorbar(robot_scatter, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("time (s)")
    ax.set_title("Minimal RPF expert episode: time-coded top-down view")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#eeeeee")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory.")
    parser.add_argument("--scene-id", default="lab528", help="Scene id tag for records.")
    parser.add_argument("--episode-id", default="lab528_minimal_rpf_000", help="Episode id.")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--horizon-sec", type=float, default=3.0)
    parser.add_argument("--preferred-side", type=int, default=1, choices=[-1, 1])
    return parser.parse_args()


def main() -> int:
    jsonl_path, plot_path = run_episode(parse_args())
    print(f"[INFO] wrote {jsonl_path}")
    print(f"[INFO] wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
