#!/usr/bin/env python3
"""Capture RGB/depth observations from a camera mounted on the scout robot."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import magnum as mn
import numpy as np

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.utils.settings import default_sim_settings, make_cfg


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    REPO_ROOT
    / "data"
    / "scene_datasets"
    / "Custom_Assets"
    / "lab528"
    / "lab528.scene_dataset_config.json"
)


def _fail(message: str) -> None:
    raise SystemExit(message)


def _parse_xyz(text: str) -> np.ndarray:
    parts = text.split(",")
    if len(parts) != 3:
        _fail(f"Expected x,y,z, got: {text}")
    try:
        return np.asarray([float(v) for v in parts], dtype=np.float32)
    except ValueError:
        _fail(f"Invalid x,y,z value: {text}")


def _parse_route(route_items: list[str] | None) -> list[np.ndarray]:
    if not route_items:
        return [
            np.asarray([-2.0, 0.097, -3.5], dtype=np.float32),
            np.asarray([4.0, 0.097, 4.5], dtype=np.float32),
        ]
    points: list[np.ndarray] = []
    for item in route_items:
        for token in item.split(":"):
            token = token.strip()
            if token:
                points.append(_parse_xyz(token))
    if len(points) < 2:
        _fail("Scout route needs at least two points.")
    return points


def _snap_points(navmesh: Path | None, points: list[np.ndarray]) -> list[np.ndarray]:
    if navmesh is None:
        return points
    try:
        from habitat_sim.nav import PathFinder

        pathfinder = PathFinder()
        if not pathfinder.load_nav_mesh(str(navmesh)):
            return points
        snapped = []
        for point in points:
            candidate = np.asarray(pathfinder.snap_point(point), dtype=np.float32)
            snapped.append(candidate if np.all(np.isfinite(candidate)) else point)
        return snapped
    except Exception:
        return points


def _polyline_samples(
    points: list[np.ndarray],
    *,
    speed: float,
    fps: float,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    segments = []
    total_length = 0.0
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        length = float(np.linalg.norm(delta[[0, 2]]))
        if length <= 1.0e-6:
            continue
        segments.append((start, end, length))
        total_length += length
    if not segments:
        _fail("Scout route has zero horizontal length.")

    duration = total_length / max(speed, 1.0e-6)
    frame_count = max(1, int(math.ceil(duration * fps)))
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)

    samples = []
    for frame_idx in range(frame_count):
        dist = min(frame_idx / fps * speed, total_length)
        remaining = dist
        start, end, length = segments[-1]
        for seg_start, seg_end, seg_length in segments:
            if remaining <= seg_length:
                start, end, length = seg_start, seg_end, seg_length
                break
            remaining -= seg_length
        alpha = float(np.clip(remaining / max(length, 1.0e-6), 0.0, 1.0))
        pos = (1.0 - alpha) * start + alpha * end
        tangent = end - start
        yaw = math.atan2(-float(tangent[0]), -float(tangent[2]))
        samples.append(
            {
                "frame": frame_idx,
                "time": frame_idx / fps,
                "robot_position": pos,
                "robot_yaw": yaw,
            }
        )
    return samples


def _resolve_navmesh(dataset: Path, scene: str, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()
    try:
        with dataset.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        navmeshes = cfg.get("navmesh_instances", {})
        nav_key = f"{scene}_navmesh"
        rel = navmeshes.get(nav_key)
        if rel is None and navmeshes:
            rel = next(iter(navmeshes.values()))
        if rel is not None:
            return (dataset.parent / rel).resolve()
    except Exception:
        return None
    return None


def _find_rigid_object(sim: habitat_sim.Simulator, handle_substring: str):
    try:
        manager = sim.get_rigid_object_manager()
        matches = manager.get_objects_by_handle_substring(handle_substring)
        if matches:
            return next(iter(matches.values()))
    except Exception:
        return None
    return None


def _make_sim(dataset: Path, scene: str, width: int, height: int) -> habitat_sim.Simulator:
    color_spec = habitat_sim.CameraSensorSpec()
    color_spec.uuid = "color_sensor"
    color_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_spec.resolution = [height, width]
    color_spec.position = [0.0, 0.0, 0.0]
    color_spec.hfov = 90.0

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [height, width]
    depth_spec.position = [0.0, 0.0, 0.0]
    depth_spec.hfov = 90.0
    depth_spec.channels = 1

    settings = dict(default_sim_settings)
    settings["scene_dataset_config_file"] = str(dataset)
    settings["scene"] = scene
    settings["enable_physics"] = False
    settings["width"] = width
    settings["height"] = height
    settings["window_width"] = width
    settings["window_height"] = height
    settings["default_agent"] = 0
    settings["default_agent_navmesh"] = False
    settings["enable_hbao"] = False
    settings["gaussian_auto_play"] = False
    settings["gaussian_time"] = 0.0

    cfg = make_cfg(settings)
    cfg.agents[0].sensor_specifications = [color_spec, depth_spec]
    try:
        cfg.sim_cfg.leave_context_with_background_renderer = False
    except Exception:
        pass
    return habitat_sim.Simulator(cfg)


def _rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    return np.ascontiguousarray(rgb.astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture scout front-camera RGB/depth frames from a Habitat-GS scene."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--scene", default="lab528")
    parser.add_argument("--navmesh", default=None)
    parser.add_argument(
        "--route",
        action="append",
        help="Scout route points as x,y,z[:x,y,z...]. Repeat or use ':' to add multiple points.",
    )
    parser.add_argument("--out", default="outputs/scout_capture/lab528")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--speed", type=float, default=0.6)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera-front", type=float, default=0.3)
    parser.add_argument("--camera-height", type=float, default=0.3)
    parser.add_argument("--robot-object", default="scout")
    parser.add_argument("--save-depth", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.exists():
        _fail(f"Dataset config not found: {dataset}")
    out_dir = Path(args.out).expanduser().resolve()
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if args.save_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)

    navmesh = _resolve_navmesh(dataset, args.scene, args.navmesh)
    route = _snap_points(navmesh, _parse_route(args.route))
    samples = _polyline_samples(
        route, speed=float(args.speed), fps=float(args.fps), max_frames=args.max_frames
    )

    print(f"[INFO] dataset={dataset}")
    print(f"[INFO] scene={args.scene}")
    print(f"[INFO] navmesh={navmesh}")
    print(f"[INFO] frames={len(samples)}, out={out_dir}")

    sim = _make_sim(dataset, args.scene, args.width, args.height)
    agent = sim.get_agent(0)
    robot = _find_rigid_object(sim, args.robot_object)
    metadata_path = out_dir / "poses.jsonl"

    with metadata_path.open("w", encoding="utf-8") as meta_f:
        for sample in samples:
            frame = int(sample["frame"])
            t = float(sample["time"])
            robot_pos = np.asarray(sample["robot_position"], dtype=np.float32)
            yaw = float(sample["robot_yaw"])
            forward = np.asarray([-math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float32)
            camera_pos = (
                robot_pos
                + forward * float(args.camera_front)
                + np.asarray([0.0, float(args.camera_height), 0.0], dtype=np.float32)
            )

            if robot is not None:
                robot.translation = mn.Vector3(
                    float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])
                )
                robot.rotation = mn.Quaternion.rotation(
                    mn.Rad(yaw), mn.Vector3.y_axis()
                )

            state = habitat_sim.AgentState()
            state.position = camera_pos
            state.rotation = quat_from_angle_axis(yaw, np.asarray([0.0, 1.0, 0.0]))
            agent.set_state(state)

            sim.gaussian_time = t
            sim.step_world(1.0 / float(args.fps))
            obs = sim.get_sensor_observations()
            rgb = _rgb_to_uint8(obs["color_sensor"])
            imageio.imwrite(rgb_dir / f"{frame:06d}.png", rgb)

            if args.save_depth:
                depth = np.asarray(obs["depth_sensor"], dtype=np.float32)
                np.save(depth_dir / f"{frame:06d}.npy", depth)

            record = {
                "frame": frame,
                "time": t,
                "rgb": f"rgb/{frame:06d}.png",
                "depth": f"depth/{frame:06d}.npy" if args.save_depth else None,
                "robot_position": robot_pos.astype(float).tolist(),
                "robot_yaw": yaw,
                "camera_position": camera_pos.astype(float).tolist(),
                "camera_yaw": yaw,
            }
            meta_f.write(json.dumps(record) + "\n")

            if frame % 20 == 0:
                print(f"[INFO] captured frame {frame}/{len(samples)}")

    sim.close()
    print(f"[INFO] wrote {len(samples)} RGB frames to {rgb_dir}")
    print(f"[INFO] metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
