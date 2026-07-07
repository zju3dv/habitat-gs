from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class CameraParams:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_T_camera: np.ndarray
    camera_T_world: np.ndarray
    near: float = 0.01
    far: float = 100.0


_PRINTED_CAMERA_DEBUG = False


def _matrix_to_numpy(matrix: Any) -> np.ndarray:
    arr = np.array(matrix, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform matrix, got {arr.shape}.")
    return arr


def _sensor_transform(sensor: Any) -> np.ndarray:
    node = getattr(sensor, "node", None)
    if node is None and hasattr(sensor, "sensor_object"):
        node = getattr(sensor.sensor_object, "node", None)
    if node is None:
        raise ValueError("Cannot build camera: sensor has no scene node.")

    fn = getattr(node, "absolute_transformation", None)
    if callable(fn):
        return _matrix_to_numpy(fn())
    transform = getattr(node, "transformation", None)
    if transform is not None:
        return _matrix_to_numpy(transform)
    raise ValueError("Cannot build camera: sensor node has no transform.")


def _resolution(spec: Any) -> tuple[int, int]:
    res = getattr(spec, "resolution", None)
    if res is None:
        raise ValueError("Cannot build camera: sensor spec has no resolution.")
    values = list(res)
    if len(values) != 2:
        raise ValueError(f"Sensor resolution must have 2 values, got {values}.")
    height, width = int(values[0]), int(values[1])
    return width, height


def _hfov_radians(spec: Any) -> float:
    hfov = getattr(spec, "hfov", 90.0)
    try:
        hfov_value = float(hfov)
    except TypeError:
        hfov_value = float(getattr(hfov, "value", 90.0))
    if hfov_value > np.pi * 2.0:
        hfov_value = np.deg2rad(hfov_value)
    return float(hfov_value)


def build_camera_from_habitat_sensor(
    sensor: Any,
    *,
    debug: bool = True,
    force_debug: bool = False,
    world_T_camera: Optional[np.ndarray] = None,
) -> CameraParams:
    """Build pinhole camera params from a Habitat Sensor wrapper or C++ sensor."""

    global _PRINTED_CAMERA_DEBUG

    spec = sensor.spec if hasattr(sensor, "spec") else sensor.specification()
    width, height = _resolution(spec)
    hfov = _hfov_radians(spec)
    fx = 0.5 * width / np.tan(0.5 * hfov)
    fy = fx
    cx = width * 0.5
    cy = height * 0.5

    world_T_camera = (
        np.asarray(world_T_camera, dtype=np.float32)
        if world_T_camera is not None
        else _sensor_transform(sensor)
    )
    if world_T_camera.shape != (4, 4):
        raise ValueError(f"world_T_camera must have shape [4, 4], got {world_T_camera.shape}.")
    camera_T_world = np.linalg.inv(world_T_camera).astype(np.float32)

    near = float(getattr(spec, "near", 0.01))
    far = float(getattr(spec, "far", 100.0))

    camera = CameraParams(
        width=width,
        height=height,
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        world_T_camera=world_T_camera.astype(np.float32),
        camera_T_world=camera_T_world,
        near=near,
        far=far,
    )

    if debug and (force_debug or not _PRINTED_CAMERA_DEBUG):
        position = camera.world_T_camera[:3, 3]
        # Habitat pinhole cameras look along local -Z.
        forward = -(camera.world_T_camera[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float32))
        print(
            "[MeshHumanCamera] "
            f"width={width} height={height} fx={fx:.3f} fy={fy:.3f} "
            f"cx={cx:.3f} cy={cy:.3f}"
        )
        print(
            "[MeshHumanCamera] "
            f"position={np.array2string(position, precision=4)} "
            f"forward={np.array2string(forward, precision=4)}"
        )
        _PRINTED_CAMERA_DEBUG = True

    return camera
