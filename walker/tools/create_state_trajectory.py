import argparse
from pathlib import Path

import numpy as np


def build_walk_idle_walk(
    fps: int,
    start_x: float,
    start_y: float,
    start_z: float,
    heading: float,
    walk_speed: float,
) -> np.ndarray:
    total_frames = int(8.0 * fps)
    traj = np.zeros((total_frames, 7), dtype=np.float32)

    direction = np.array([np.sin(heading), 0.0, np.cos(heading)], dtype=np.float32)
    position = np.array([start_x, start_y, start_z], dtype=np.float32)

    for i in range(total_frames):
        t = i / fps
        state_id = 0 if 3.0 <= t < 5.0 else 1
        speed = 0.0 if state_id == 0 else walk_speed

        if i > 0 and state_id == 1:
            position = position + direction * (walk_speed / fps)

        traj[i, 0:3] = position
        traj[i, 3] = heading
        traj[i, 4] = speed
        traj[i, 5] = state_id
        traj[i, 6] = t

    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start_x", type=float, default=0.0)
    parser.add_argument("--start_y", type=float, default=0.0)
    parser.add_argument("--start_z", type=float, default=2.0)
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--walk_speed", type=float, default=0.8)
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}.")

    traj = build_walk_idle_walk(
        fps=args.fps,
        start_x=args.start_x,
        start_y=args.start_y,
        start_z=args.start_z,
        heading=args.heading,
        walk_speed=args.walk_speed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, traj)

    print(f"Saved trajectory: {out_path}")
    print(f"shape: {traj.shape}")
    print(f"start: {traj[0]}")
    print(f"at 3.5s: {traj[int(3.5 * args.fps)]}")
    print(f"end: {traj[-1]}")


if __name__ == "__main__":
    main()
