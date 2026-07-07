import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WALKER_ROOT = REPO_ROOT / "walker"
SRC_PYTHON = REPO_ROOT / "src_python"
for path in (WALKER_ROOT, SRC_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from human_actor import BakedMeshClip, HumanTrajectory, MeshHumanActor
from human_actor.rendering import CameraParams, SimpleMeshRenderer, build_camera_from_habitat_sensor
from human_actor.rendering.depth_compositor import composite_rgbd


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, arr)
    except Exception:
        from PIL import Image

        Image.fromarray(arr).save(path)


def _depth_vis(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not valid.any():
        return out
    dmin = float(depth[valid].min())
    dmax = float(depth[valid].max())
    norm = (depth - dmin) / max(1e-6, dmax - dmin)
    gray = np.clip((1.0 - norm) * 255.0, 0, 255).astype(np.uint8)
    out[valid] = np.stack([gray[valid], gray[valid], gray[valid]], axis=-1)
    return out


def _mask_vis(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int32)
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    out[mask > 0] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def _make_actor(args) -> MeshHumanActor:
    clips = {"walk": BakedMeshClip(args.walk_clip, name="walk")}
    if args.idle_clip:
        idle_path = Path(args.idle_clip)
        if idle_path.exists():
            clips["idle"] = BakedMeshClip(str(idle_path), name="idle")
        else:
            print(f"Idle clip not found, using walk first frame fallback: {idle_path}")
    trajectory = HumanTrajectory(args.trajectory, fps=args.fps)
    return MeshHumanActor(
        actor_id=1,
        name="target",
        clips=clips,
        trajectory=trajectory,
        fallback_clip="walk",
    )


def _synthetic_background(width: int, height: int):
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = (40 + 80 * x).astype(np.uint8)
    rgb[:, :, 1] = (50 + 70 * y).astype(np.uint8)
    rgb[:, :, 2] = 120
    depth = np.full((height, width), 12.0, dtype=np.float32)
    return rgb, depth


def _synthetic_camera(width: int, height: int) -> CameraParams:
    hfov = np.deg2rad(70.0)
    fx = 0.5 * width / np.tan(0.5 * hfov)
    fy = fx
    world_T_camera = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    camera_T_world = np.linalg.inv(world_T_camera).astype(np.float32)
    return CameraParams(
        width=width,
        height=height,
        fx=float(fx),
        fy=float(fy),
        cx=width * 0.5,
        cy=height * 0.5,
        world_T_camera=world_T_camera,
        camera_T_world=camera_T_world,
        near=0.01,
        far=100.0,
    )


def _render_from_habitat(args):
    if not args.scene_config:
        return None
    try:
        import habitat_sim
        from habitat_sim.utils.settings import default_sim_settings, make_cfg
    except Exception as exc:
        print(f"Habitat import failed, using synthetic background: {exc}")
        return None

    try:
        settings = default_sim_settings.copy()
        settings.update(
            {
                "scene": args.scene_config,
                "width": args.width,
                "height": args.height,
                "color_sensor": True,
                "depth_sensor": True,
                "semantic_sensor": False,
                "sensor_height": args.camera_height,
                "hfov": args.hfov,
            }
        )
        if args.scene_dataset_config:
            settings["scene_dataset_config_file"] = args.scene_dataset_config
        sim = habitat_sim.Simulator(make_cfg(settings))
        obs = sim.get_sensor_observations()
        color_sensor = sim.sensors["color_sensor"]
        rgb = np.asarray(obs["color_sensor"])[:, :, :3].copy()
        depth = np.asarray(obs["depth_sensor"], dtype=np.float32).copy()
        camera = build_camera_from_habitat_sensor(color_sensor, debug=True, force_debug=True)
        sim.close()
        return rgb, depth, camera
    except Exception as exc:
        print(f"Habitat render failed, using synthetic background: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_config")
    parser.add_argument("--scene_dataset_config")
    parser.add_argument("--walk_clip", required=True)
    parser.add_argument("--idle_clip")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out_dir", default="outputs/mesh_human_debug")
    parser.add_argument("--frame", type=int, default=60)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera_height", type=float, default=1.0)
    parser.add_argument("--hfov", type=float, default=70.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    habitat_result = _render_from_habitat(args)
    if habitat_result is None:
        rgb_gs, depth_gs = _synthetic_background(args.width, args.height)
        camera = _synthetic_camera(args.width, args.height)
        print(
            "[MeshHumanCamera] "
            f"width={camera.width} height={camera.height} fx={camera.fx:.3f} "
            f"fy={camera.fy:.3f} cx={camera.cx:.3f} cy={camera.cy:.3f}"
        )
        print(
            "[MeshHumanCamera] "
            f"position={np.array2string(camera.world_T_camera[:3, 3], precision=4)} "
            "forward=[0. 0. 1.]"
        )
    else:
        rgb_gs, depth_gs, camera = habitat_result

    actor = _make_actor(args)
    sim_time = args.frame / args.fps
    mesh = actor.mesh_at(sim_time)
    renderer = SimpleMeshRenderer(camera.width, camera.height)
    human = renderer.render([mesh], camera)
    composed = composite_rgbd(
        rgb_gs,
        depth_gs,
        human["rgb"],
        human["depth"],
        human["id_mask"],
        debug=True,
    )

    _save_png(out_dir / "debug_gs_rgb.png", rgb_gs)
    _save_png(out_dir / "debug_human_rgb.png", human["rgb"])
    _save_png(out_dir / "debug_composed_rgb.png", composed["rgb"])
    _save_png(out_dir / "debug_human_id_mask.png", _mask_vis(composed["id_mask"]))
    _save_png(out_dir / "debug_depth_gs.png", _depth_vis(depth_gs))
    _save_png(out_dir / "debug_depth_human.png", _depth_vis(human["depth"]))
    np.save(out_dir / "debug_depth_gs.npy", depth_gs)
    np.save(out_dir / "debug_depth_human.npy", human["depth"])

    print(f"Saved debug outputs to: {out_dir}")
    print(f"human pixels before composite: {int((human['id_mask'] > 0).sum())}")
    print(f"human pixels visible after composite: {int((composed['id_mask'] > 0).sum())}")


if __name__ == "__main__":
    main()
