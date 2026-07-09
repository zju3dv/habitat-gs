import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from human_actor import BakedMeshClip, HumanTrajectory, MeshHumanActor
from human_actor.obj_export import write_obj


def load_clips(args) -> dict:
    clips = {"walk": BakedMeshClip(args.walk_clip, name="walk")}
    if args.idle_clip is not None:
        idle_path = Path(args.idle_clip)
        if idle_path.exists():
            clips["idle"] = BakedMeshClip(str(idle_path), name="idle")
        else:
            print(f"Idle clip not found, using walk first frame fallback: {idle_path}")
    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk_clip", required=True)
    parser.add_argument("--idle_clip")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=240)
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {args.num_frames} OBJ frames to: {out_dir}")
    for i in range(args.num_frames):
        sim_time = i / args.fps
        mesh = actor.mesh_at(sim_time)
        out_path = out_dir / f"human_{i:04d}.obj"
        write_obj(str(out_path), mesh["vertices"], mesh["faces"])

        if i % args.fps == 0 or i == args.num_frames - 1:
            root = mesh["root_pose"]
            print(
                f"frame={i:04d} t={sim_time:.2f} state={mesh['state_name']} "
                f"pos=({root[0]:.3f}, {root[1]:.3f}, {root[2]:.3f}) -> {out_path}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
