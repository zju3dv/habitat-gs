import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=2.0)
    parser.add_argument("--heading", type=float, default=0.0)
    args = parser.parse_args()

    num_frames = int(args.duration * args.fps)

    # trajectory format:
    # x, y, z, heading, speed, state_id, time
    # state_id: 0 idle, 1 walk
    traj = np.zeros((num_frames, 7), dtype=np.float32)

    direction = np.array(
        [
            np.sin(args.heading),
            0.0,
            np.cos(args.heading),
        ],
        dtype=np.float32,
    )

    start = np.array([args.x, args.y, args.z], dtype=np.float32)

    for i in range(num_frames):
        t = i / args.fps
        pos = start + direction * args.speed * t

        traj[i, 0] = pos[0]
        traj[i, 1] = pos[1]
        traj[i, 2] = pos[2]
        traj[i, 3] = args.heading
        traj[i, 4] = args.speed
        traj[i, 5] = 1  # walk
        traj[i, 6] = t

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, traj)

    print("Saved trajectory:", out_path)
    print("shape:", traj.shape)
    print("start:", traj[0])
    print("end:", traj[-1])


if __name__ == "__main__":
    main()
