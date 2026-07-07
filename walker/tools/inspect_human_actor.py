import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from human_actor import BakedMeshClip, HumanTrajectory, MeshHumanActor


def load_clips(args) -> dict:
    clips = {"walk": BakedMeshClip(args.walk_clip, name="walk")}
    if args.idle_clip is not None:
        idle_path = Path(args.idle_clip)
        if idle_path.exists():
            clips["idle"] = BakedMeshClip(str(idle_path), name="idle")
        else:
            print(f"Idle clip not found, using walk first frame fallback: {idle_path}")
    return clips


def inspect_at(actor: MeshHumanActor, sim_time: float):
    mesh = actor.mesh_at(sim_time)
    capsule = actor.capsule_at(sim_time)
    vertices = mesh["vertices"]
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    root = mesh["root_pose"]

    print(
        f"t={sim_time:.1f} state={mesh['state_name']} "
        f"position=({root[0]:.3f}, {root[1]:.3f}, {root[2]:.3f}) "
        f"heading={root[3]:.3f}"
    )
    print(f"  bbox world min={np.array2string(bbox_min, precision=4)}")
    print(f"  bbox world max={np.array2string(bbox_max, precision=4)}")
    print(
        "  capsule "
        f"center={np.array2string(capsule['center'], precision=4)} "
        f"radius={capsule['radius']:.3f} height={capsule['height']:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk_clip", required=True)
    parser.add_argument("--idle_clip")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    clips = load_clips(args)
    trajectory = HumanTrajectory(args.trajectory, fps=args.fps)
    actor = MeshHumanActor(
        actor_id=1,
        name="human",
        clips=clips,
        trajectory=trajectory,
        fallback_clip="walk",
    )

    print("Loaded MeshHumanActor")
    print(f"  clips: {list(clips.keys())}")
    print(f"  trajectory: {args.trajectory} shape={trajectory.data.shape}")
    print(f"  fallback: walk first frame for idle/stop when idle clip is absent")

    for sim_time in (0.0, 3.5, 6.0):
        inspect_at(actor, sim_time)


if __name__ == "__main__":
    main()
