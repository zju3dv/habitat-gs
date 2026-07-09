import argparse
from pathlib import Path

from mesh_human_actor import MeshHumanActor


def write_obj(out_path, vertices, faces):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("# moving human obj\n")

        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for face in faces:
            a, b, c = face + 1
            f.write(f"f {a} {b} {c}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--traj", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    actor = MeshHumanActor(
        clip_path=args.clip,
        trajectory_path=args.traj,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_frames):
        sim_time = i / args.fps
        verts_world = actor.world_vertices_at(sim_time)

        out_path = out_dir / f"human_{i:04d}.obj"
        write_obj(out_path, verts_world, actor.faces)

        if i % 10 == 0:
            print("exported:", out_path)

    print("Done.")
    print("Output dir:", out_dir)


if __name__ == "__main__":
    main()
