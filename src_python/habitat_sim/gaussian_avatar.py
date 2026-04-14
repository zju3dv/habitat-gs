#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from habitat_sim.logging import logger


def _ensure_inspect_getargspec() -> None:
    import inspect

    if hasattr(inspect, "getargspec"):
        return
    from collections import namedtuple

    ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec  # type: ignore[attr-defined]


def _ensure_legacy_numpy_aliases() -> None:
    # NumPy 2.x removed legacy aliases such as np.float and np.complex.
    # Older SMPL/SMPLX and chumpy codepaths still reference these names.
    legacy_aliases = {
        "bool": getattr(np, "bool_", bool),
        "int": getattr(np, "int_", int),
        "float": getattr(np, "float64", float),
        "complex": getattr(np, "complex128", complex),
        "object": getattr(np, "object_", object),
        "str": getattr(np, "str_", str),
        "long": getattr(np, "int_", int),
        "unicode": getattr(np, "str_", str),
    }
    for name, value in legacy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("GaussianAvatar requires PyTorch to be installed.")
    if not torch.cuda.is_available():
        raise RuntimeError("GaussianAvatar requires CUDA-enabled PyTorch.")


def _resolve_path(base_dir: str, path: Optional[str]) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


DEFAULT_FRAME_TIME = 0.025


def _safe_float(value: Any, default: Optional[float]) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_matrix(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    mat = np.array(value, dtype=np.float32)
    if mat.shape == (4, 4):
        return mat
    if mat.size == 16:
        return mat.reshape(4, 4)
    raise ValueError("coord_transform must be a 4x4 matrix or a flat 16 list.")


def _opencv_to_habitat_transform() -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[1, 1] = -1.0
    mat[2, 2] = -1.0
    return mat


def _coord_transform_from_system(system: str) -> Optional[np.ndarray]:
    if not system:
        return None
    system = system.lower()
    if system in ("opencv", "smpl_opencv", "gaussian_avatar", "opencv_camera"):
        return _opencv_to_habitat_transform()
    if system in ("habitat", "world", "none", "identity"):
        return np.eye(4, dtype=np.float32)
    raise ValueError(f"Unknown coord_system '{system}'.")


def _infer_coord_transform_from_pose(pose_data: torch.Tensor) -> Optional[np.ndarray]:
    if pose_data.numel() < 3:
        return None
    root = pose_data[0, :3].detach().cpu().numpy()
    angle = float(np.linalg.norm(root))
    if angle < 1.0e-4:
        return None
    axis = root / angle
    if abs(abs(axis[0]) - 1.0) < 0.15 and abs(angle - math.pi) < 0.4:
        if abs(axis[1]) < 0.2 and abs(axis[2]) < 0.2:
            return _opencv_to_habitat_transform()
    return None


def _normalize_scene_time(time: float, time_max: float, loop: bool) -> float:
    if time_max > 0.0:
        if loop:
            wrapped = math.fmod(time, time_max)
            if wrapped < 0.0:
                wrapped += time_max
            return wrapped
        return float(np.clip(time, 0.0, time_max))
    return time


def _load_joint_count(path: str) -> int:
    data = np.load(path, mmap_mode="r")
    if "lbs_weights" not in data:
        raise ValueError(f"Missing lbs_weights in {path}.")
    lbs = data["lbs_weights"]
    if lbs.ndim != 2:
        raise ValueError(f"Unexpected lbs_weights shape {lbs.shape} in {path}.")
    return int(lbs.shape[1])


def _load_smpl_params_file(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".npz", ".npy"):
        with np.load(path, allow_pickle=True) as data:
            return {k: np.asarray(v) for k, v in dict(data).items()}
    if ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return data
        return {"data": data}
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        logger.warning("Failed to torch.load %s (%s); falling back to numpy.", path, exc)
        with np.load(path, allow_pickle=True) as data:
            return {k: np.asarray(v) for k, v in dict(data).items()}


def _get_smpl_param(data: Any, *keys: str) -> Any:
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return data[key]
    for key in keys:
        if hasattr(data, key):
            return getattr(data, key)
    return None


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _nlerp_quat(q0: torch.Tensor, q1: torch.Tensor, weight: float) -> torch.Tensor:
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    q = q0 + (q1 - q0) * weight
    return _normalize_quat(q)


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = _normalize_quat(q)
    x, y, z, w = q.unbind(-1)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )


def _matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]

    qw = torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0)) * 0.5
    qx = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)) * 0.5
    qy = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0)) * 0.5
    qz = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0)) * 0.5

    qx = torch.copysign(qx, m21 - m12)
    qy = torch.copysign(qy, m02 - m20)
    qz = torch.copysign(qz, m10 - m01)

    quat = torch.stack([qx, qy, qz, qw], dim=-1)
    return _normalize_quat(quat)


def _build_transform(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    rot = _quat_to_matrix(quat)
    t = torch.eye(4, device=pos.device, dtype=pos.dtype)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


def _compute_frame_position(
    scene_time: float,
    num_frames: int,
    time_begin: float,
    time_end: float,
    frame_time: Optional[float],
) -> float:
    if num_frames <= 1:
        return 0.0
    duration = time_end - time_begin
    if duration <= 0.0:
        return 0.0
    local_time = scene_time - time_begin
    step = frame_time if frame_time is not None and frame_time > 0.0 else None
    if step is None or step <= 0.0:
        step = duration / max(num_frames - 1, 1)
    if step <= 0.0:
        return 0.0
    frame_pos = local_time / step
    return max(0.0, min(frame_pos, num_frames - 1))


def _interp_indices(frame_pos: float, num_frames: int) -> Tuple[int, int, float]:
    if num_frames <= 1:
        return 0, 0, 0.0
    idx0 = int(math.floor(frame_pos))
    idx1 = min(idx0 + 1, num_frames - 1)
    weight = frame_pos - idx0
    return idx0, idx1, weight


def _compute_avatar_world_bounds(
    joint_mats: torch.Tensor, proxy_capsules: np.ndarray, padding: float
) -> Tuple[np.ndarray, np.ndarray]:
    if proxy_capsules.size:
        p0 = proxy_capsules[:, :3]
        p1 = proxy_capsules[:, 3:6]
        radii = proxy_capsules[:, 6:7]
        bounds_min = (np.minimum(p0, p1) - radii).min(axis=0)
        bounds_max = (np.maximum(p0, p1) + radii).max(axis=0)
    else:
        # Fallback when no proxy capsules are available in the driver stream.
        joint_positions = joint_mats[:, :3, 3]
        bounds_min = torch.amin(joint_positions, dim=0).detach().cpu().numpy()
        bounds_max = torch.amax(joint_positions, dim=0).detach().cpu().numpy()

    pad = np.float32(max(0.0, float(padding)))
    bounds_min = np.asarray(bounds_min - pad, dtype=np.float32)
    bounds_max = np.asarray(bounds_max + pad, dtype=np.float32)
    return bounds_min, bounds_max


def _resolve_scene_instance_path(sim) -> str:
    scene_id = sim.config.sim_cfg.scene_id
    if isinstance(scene_id, str) and os.path.exists(scene_id):
        return scene_id

    dataset_cfg = sim.config.sim_cfg.scene_dataset_config_file
    if not dataset_cfg or dataset_cfg == "default":
        return ""
    dataset_cfg = os.path.abspath(dataset_cfg)
    if not os.path.exists(dataset_cfg):
        return ""

    with open(dataset_cfg, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    scene_instances = dataset.get("scene_instances", {})
    paths_cfg = scene_instances.get("paths", {})
    if not isinstance(paths_cfg, dict):
        return ""

    base_dir = os.path.dirname(dataset_cfg)
    search_dirs: List[str] = []
    for _, rel_list in paths_cfg.items():
        if not isinstance(rel_list, list):
            continue
        for rel in rel_list:
            search_dirs.append(os.path.join(base_dir, rel))

    candidates = [
        f"{scene_id}.scene_instance.json",
        f"{scene_id}.json",
    ]
    if isinstance(scene_id, str) and scene_id.endswith(".json"):
        candidates.insert(0, scene_id)

    for search_dir in search_dirs:
        for name in candidates:
            candidate = os.path.join(search_dir, name)
            if os.path.exists(candidate):
                return candidate
    return ""


def _load_avatar_configs(
    scene_instance_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not scene_instance_path or not os.path.exists(scene_instance_path):
        return [], {}
    with open(scene_instance_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    time_cfg: Dict[str, Any] = {}
    if "time_max" in data:
        time_max = _safe_float(data.get("time_max"), None)
        if time_max is None:
            logger.warning("scene time_max in %s is not a number.", scene_instance_path)
        else:
            time_cfg["time_max"] = time_max
    if "time_loop" in data:
        time_cfg["time_loop"] = bool(data.get("time_loop"))

    avatars = data.get("gaussian_avatars")
    tag = "gaussian_avatars"
    if avatars is None:
        avatars = data.get("avatars", [])
        tag = "avatars"
    if not isinstance(avatars, list):
        logger.warning("%s in %s is not a list.", tag, scene_instance_path)
        return [], time_cfg
    return avatars, time_cfg


class GaussianAvatar:
    def __init__(
        self,
        sim,
        avatar_id: int,
        config: Dict[str, Any],
        base_dir: str,
    ) -> None:
        _require_torch()

        self.sim = sim
        self.avatar_id = avatar_id
        self.name = config.get("name", f"gaussian_avatar_{avatar_id}")
        self.device = torch.device("cuda")

        canonical_gaussians = config.get("canonical_gaussians", "")
        driver = config.get("driver", {})
        driver_path = ""
        if isinstance(driver, str):
            driver_path = driver
            driver = {}
        elif driver is None:
            driver = {}
        elif not isinstance(driver, dict):
            logger.warning("GaussianAvatar %s has invalid driver config: %s", self.name, driver)
            driver = {}
        if "proxy_urdf" in config:
            raise ValueError(
                "GaussianAvatar config field `proxy_urdf` is no longer supported. "
                "Remove it from scene_instance.json and provide precomputed "
                "`proxy_capsules` in the driver .pkl."
            )

        self.canonical_gaussians_path = _resolve_path(base_dir, canonical_gaussians)
        self.smpl_parms_path = _resolve_path(
            base_dir, driver.get("smpl_params") or driver.get("smpl_parms") or driver_path
        )
        self.trajectory_path = _resolve_path(base_dir, driver.get("trajectory"))

        self.smpl_model_path = _resolve_path(base_dir, config.get("smpl_model_path"))
        self.smpl_type = config.get("smpl_type", "").lower()
        self.smpl_gender = config.get("smpl_gender", "neutral")
        use_smpl_trans = config.get("use_smpl_trans", None)
        if use_smpl_trans is None:
            has_traj = bool(self.trajectory_path) and os.path.exists(
                self.trajectory_path
            )
            self.use_smpl_trans = not has_traj
        else:
            self.use_smpl_trans = bool(use_smpl_trans)
        self.frame_time = config.get("frame_time", None)
        if self.frame_time is None:
            fps = config.get("fps", None)
            if fps:
                self.frame_time = 1.0 / float(fps)
        self.sync_cuda = bool(config.get("sync_cuda", False))
        self.trajectory_in_smpl = bool(config.get("trajectory_in_smpl", False))
        scale_val = config.get("scale", 1.0)
        try:
            self.scale = float(scale_val)
        except (TypeError, ValueError):
            logger.warning("Invalid scale for %s: %s; using 1.0", self.name, scale_val)
            self.scale = 1.0
        if self.scale <= 0.0:
            logger.warning("Non-positive scale for %s: %s; using 1.0", self.name, self.scale)
            self.scale = 1.0
        offset_val = config.get("offset_y", 0.0)
        self.offset_y = 0.0
        try:
            self.offset_y = float(offset_val)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid offset_y for %s: %s; using 0.0", self.name, offset_val
            )
        time_begin = _safe_float(config.get("time_begin"), 0.0)
        self.time_begin = 0.0 if time_begin is None else time_begin
        self.time_end = _safe_float(config.get("time_end"), None)
        self.scene_time_max = 0.0
        self.scene_time_loop = True
        self._active = True
        self._visible = True
        self.inv_bind_root = None
        self.expression_data = None
        self.coord_system = str(config.get("coord_system", "")).lower()
        coord_transform = _parse_matrix(config.get("coord_transform"))
        if coord_transform is None:
            coord_transform = _coord_transform_from_system(self.coord_system)
        if coord_transform is not None:
            self.coord_transform = torch.from_numpy(coord_transform).to(self.device)
        else:
            self.coord_transform = None

        self.joint_count = None
        self._joint_count_warned = False
        self._bounds_update_failed = False
        self._empty_proxy_capsules = np.empty((0, 7), dtype=np.float32)
        self._proxy_capsules = self._empty_proxy_capsules
        self.precomputed_proxy_capsules: Optional[np.ndarray] = None
        self.precomputed_proxy_capsule_count: int = 0
        self.precomputed_proxy_capsules_version: int = 0
        self.precomputed_proxy_capsules_fps: float = 0.0
        self.precomputed_joint_mats: Optional[torch.Tensor] = None
        self.precomputed_joint_count: int = 0
        self.precomputed_joint_fps: float = 0.0
        self.precomputed_joint_space: str = ""
        self.precomputed_joint_version: int = 0

        if not self.smpl_type:
            self.joint_count = _load_joint_count(self.canonical_gaussians_path)
            self.smpl_type = "smplx" if self.joint_count == 55 else "smpl"

        required_paths = {
            "canonical_gaussians": self.canonical_gaussians_path,
            "smpl_params": self.smpl_parms_path,
            "smpl_model_path": self.smpl_model_path,
        }
        missing = [name for name, path in required_paths.items() if not path]
        if missing:
            raise ValueError(f"Missing GaussianAvatar paths: {', '.join(missing)}")

        self._load_canonical_assets()
        self._load_driver()
        self._finalize_time_settings()

    def _load_canonical_assets(self) -> None:
        if not self.canonical_gaussians_path or not os.path.exists(
            self.canonical_gaussians_path
        ):
            raise ValueError(
                f"Canonical gaussians not found: {self.canonical_gaussians_path}"
            )
        if self.joint_count is None:
            self.joint_count = _load_joint_count(self.canonical_gaussians_path)

        try:
            _ensure_inspect_getargspec()
            _ensure_legacy_numpy_aliases()
            import smplx as smplx_module
        except Exception as exc:
            raise RuntimeError(f"Failed to import SMPL/SMPLX modules: {exc}") from exc
        self.smplx_module = smplx_module
        self.inv_bind_root = None
        self.inv_bind_root_is_identity = False
        try:
            with np.load(self.canonical_gaussians_path) as data:
                inv_bind = data.get("joints_inv_bind_matrix")
                if inv_bind is not None and inv_bind.ndim == 3 and inv_bind.shape[1:] == (4, 4):
                    self.inv_bind_root = torch.from_numpy(inv_bind[0]).to(self.device)
                    eye = torch.eye(
                        4,
                        device=self.inv_bind_root.device,
                        dtype=self.inv_bind_root.dtype,
                    )
                    self.inv_bind_root_is_identity = torch.allclose(
                        self.inv_bind_root, eye, atol=1.0e-5
                    )
        except Exception as exc:
            logger.warning(
                "GaussianAvatar %s failed to read joints_inv_bind_matrix from %s: %s",
                self.name,
                self.canonical_gaussians_path,
                exc,
            )

    def _load_driver(self) -> None:
        smpl_data = _load_smpl_params_file(self.smpl_parms_path)
        if self.frame_time is None:
            fps = _get_smpl_param(smpl_data, "fps")
            if fps:
                self.frame_time = 1.0 / float(fps)
            else:
                frame_time = _get_smpl_param(smpl_data, "frame_time")
                if frame_time:
                    self.frame_time = float(frame_time)
        body_pose = _get_smpl_param(smpl_data, "body_pose", "pose", "poses", "full_pose")
        global_orient = _get_smpl_param(smpl_data, "global_orient", "root_orient")
        trans = _get_smpl_param(smpl_data, "trans", "transl")
        betas = _get_smpl_param(smpl_data, "beta", "betas")
        jaw_pose = _get_smpl_param(smpl_data, "jaw_pose")
        leye_pose = _get_smpl_param(smpl_data, "leye_pose")
        reye_pose = _get_smpl_param(smpl_data, "reye_pose")
        left_hand_pose = _get_smpl_param(smpl_data, "left_hand_pose")
        right_hand_pose = _get_smpl_param(smpl_data, "right_hand_pose")
        expression = _get_smpl_param(smpl_data, "expression")

        if body_pose is None or trans is None or betas is None:
            raise ValueError(f"smpl_params missing required keys in {self.smpl_parms_path}.")

        def _as_frame_tensor(value: Any, name: str, expected_dim: Optional[int] = None) -> torch.Tensor:
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 2:
                raise ValueError(f"{name} should be 1D or 2D, got {tensor.shape}.")
            if expected_dim is not None and tensor.shape[1] != expected_dim:
                raise ValueError(
                    f"{name} should have last dim {expected_dim}, got {tensor.shape}."
                )
            return tensor

        def _broadcast_frames(tensor: torch.Tensor, frames: int, name: str) -> torch.Tensor:
            if tensor.shape[0] == frames:
                return tensor
            if tensor.shape[0] == 1:
                return tensor.repeat(frames, 1)
            raise ValueError(
                f"{name} has {tensor.shape[0]} frames, expected {frames}."
            )

        body_pose = _as_frame_tensor(body_pose, "body_pose")
        pose_len = body_pose.shape[0]
        if global_orient is not None:
            global_orient = _as_frame_tensor(global_orient, "global_orient", 3)
            global_orient = _broadcast_frames(global_orient, pose_len, "global_orient")
            pose_data = torch.cat([global_orient, body_pose], dim=-1)
        else:
            if body_pose.shape[1] in (63, 69):
                zeros = torch.zeros((pose_len, 3), dtype=body_pose.dtype)
                pose_data = torch.cat([zeros, body_pose], dim=-1)
            else:
                pose_data = body_pose

        pose_dim = pose_data.shape[-1]
        has_smplx_keys = any(
            value is not None
            for value in (
                jaw_pose,
                leye_pose,
                reye_pose,
                left_hand_pose,
                right_hand_pose,
                expression,
            )
        )
        data_suggests_smplx = has_smplx_keys or pose_dim > 72

        if data_suggests_smplx and self.smpl_type != "smplx":
            logger.warning(
                "SMPL params suggest SMPLX, overriding smpl_type=%s.",
                self.smpl_type,
            )
            self.smpl_type = "smplx"
        elif not data_suggests_smplx and self.smpl_type == "smplx":
            logger.warning(
                "SMPL params suggest SMPL, overriding smpl_type=smplx.",
            )
            self.smpl_type = "smpl"
        elif not self.smpl_type:
            self.smpl_type = "smplx" if data_suggests_smplx else "smpl"

        rest_pose = None
        expression_data = None
        if self.smpl_type == "smplx":
            if has_smplx_keys:
                jaw_pose = _as_frame_tensor(
                    jaw_pose if jaw_pose is not None else np.zeros((1, 3), np.float32),
                    "jaw_pose",
                    3,
                )
                jaw_pose = _broadcast_frames(jaw_pose, pose_len, "jaw_pose")
                leye_pose = _as_frame_tensor(
                    leye_pose if leye_pose is not None else np.zeros((1, 3), np.float32),
                    "leye_pose",
                    3,
                )
                leye_pose = _broadcast_frames(leye_pose, pose_len, "leye_pose")
                reye_pose = _as_frame_tensor(
                    reye_pose if reye_pose is not None else np.zeros((1, 3), np.float32),
                    "reye_pose",
                    3,
                )
                reye_pose = _broadcast_frames(reye_pose, pose_len, "reye_pose")
                left_hand_pose = _as_frame_tensor(
                    left_hand_pose
                    if left_hand_pose is not None
                    else np.zeros((1, 45), np.float32),
                    "left_hand_pose",
                    45,
                )
                left_hand_pose = _broadcast_frames(
                    left_hand_pose, pose_len, "left_hand_pose"
                )
                right_hand_pose = _as_frame_tensor(
                    right_hand_pose
                    if right_hand_pose is not None
                    else np.zeros((1, 45), np.float32),
                    "right_hand_pose",
                    45,
                )
                right_hand_pose = _broadcast_frames(
                    right_hand_pose, pose_len, "right_hand_pose"
                )
                rest_pose = torch.cat(
                    [
                        jaw_pose,
                        leye_pose,
                        reye_pose,
                        left_hand_pose,
                        right_hand_pose,
                    ],
                    dim=-1,
                )
                if expression is not None:
                    expression_data = _as_frame_tensor(expression, "expression", 10)
                    expression_data = _broadcast_frames(
                        expression_data, pose_len, "expression"
                    )
            elif pose_dim > 66:
                rest_pose = pose_data[:, 66:]
                pose_data = pose_data[:, :66]
            else:
                rest_pose = torch.zeros((pose_len, 99), dtype=pose_data.dtype)
        else:
            rest_pose = None

        trans = _as_frame_tensor(trans, "trans", 3)
        trans = _broadcast_frames(trans, pose_len, "trans")
        betas = torch.as_tensor(betas, dtype=torch.float32)
        if betas.ndim == 1:
            betas = betas.unsqueeze(0)

        joint_mats = _get_smpl_param(smpl_data, "joint_mats")
        if joint_mats is None:
            raise ValueError(
                f"smpl_params missing required key 'joint_mats' in {self.smpl_parms_path}."
            )
        joint_mats_space = _get_smpl_param(smpl_data, "joint_mats_space")
        if joint_mats_space != "smpl_with_trans":
            raise ValueError(
                f"joint_mats_space must be 'smpl_with_trans', got {joint_mats_space}."
            )
        joint_mats_version = _get_smpl_param(smpl_data, "joint_mats_version")
        try:
            joint_mats_version = int(joint_mats_version)
        except (TypeError, ValueError):
            raise ValueError(
                f"joint_mats_version must be an integer, got {joint_mats_version}."
            )
        if joint_mats_version != 1:
            raise ValueError(
                f"Unsupported joint_mats_version={joint_mats_version}, expected 1."
            )

        joint_mats_fps = _get_smpl_param(smpl_data, "joint_mats_fps")
        try:
            joint_mats_fps = float(joint_mats_fps)
        except (TypeError, ValueError):
            raise ValueError(f"joint_mats_fps must be a float, got {joint_mats_fps}.")
        if joint_mats_fps <= 0.0:
            raise ValueError(f"joint_mats_fps must be > 0, got {joint_mats_fps}.")
        fps_in_driver = _get_smpl_param(smpl_data, "fps")
        if fps_in_driver is not None:
            try:
                fps_in_driver = float(fps_in_driver)
            except (TypeError, ValueError):
                raise ValueError(f"fps must be a float, got {fps_in_driver}.")
            if abs(fps_in_driver - joint_mats_fps) > 1.0e-4:
                raise ValueError(
                    f"joint_mats_fps ({joint_mats_fps}) does not match fps ({fps_in_driver})."
                )

        try:
            joint_mats_joint_count = int(
                _get_smpl_param(smpl_data, "joint_mats_joint_count")
            )
        except (TypeError, ValueError):
            raise ValueError("joint_mats_joint_count must be an integer.")

        if isinstance(joint_mats, torch.Tensor):
            joint_mats_np = joint_mats.detach().cpu().numpy()
        else:
            try:
                joint_mats_np = np.asarray(joint_mats, dtype=np.float32)
            except Exception as exc:
                raise ValueError(
                    f"joint_mats cannot be converted to float32 array: {exc}"
                ) from exc
        if joint_mats_np.ndim != 4 or joint_mats_np.shape[2:] != (4, 4):
            raise ValueError(
                f"joint_mats must be [T, J, 4, 4], got {joint_mats_np.shape}."
            )
        if joint_mats_np.shape[0] != pose_len:
            raise ValueError(
                "joint_mats frame count mismatch: "
                f"{joint_mats_np.shape[0]} vs pose frames {pose_len}."
            )
        if joint_mats_np.shape[1] != joint_mats_joint_count:
            raise ValueError(
                "joint_mats_joint_count mismatch: "
                f"metadata={joint_mats_joint_count} vs tensor={joint_mats_np.shape[1]}."
            )
        if self.joint_count is not None and joint_mats_np.shape[1] != self.joint_count:
            raise ValueError(
                "joint_mats joint count mismatch with canonical gaussians: "
                f"{joint_mats_np.shape[1]} vs {self.joint_count}."
            )

        self.pose_data = pose_data
        self.rest_pose = rest_pose
        self.expression_data = expression_data
        self.trans_data = trans
        self.betas = betas[0:1]

        self.pose_data = self.pose_data.to(self.device)
        self.trans_data = self.trans_data.to(self.device)
        self.betas = self.betas.to(self.device)
        if self.rest_pose is not None:
            self.rest_pose = self.rest_pose.to(self.device)
        if self.expression_data is not None:
            self.expression_data = self.expression_data.to(self.device)
        self.precomputed_joint_mats = (
            torch.from_numpy(joint_mats_np.astype(np.float32))
            .to(self.device)
            .contiguous()
        )
        self.precomputed_joint_count = int(joint_mats_np.shape[1])
        self.precomputed_joint_fps = float(joint_mats_fps)
        self.precomputed_joint_space = str(joint_mats_space)
        self.precomputed_joint_version = int(joint_mats_version)
        proxy_capsules = _get_smpl_param(smpl_data, "proxy_capsules")
        if proxy_capsules is None:
            self.precomputed_proxy_capsules = np.empty((pose_len, 0, 7), dtype=np.float32)
            self.precomputed_proxy_capsule_count = 0
            self.precomputed_proxy_capsules_version = 0
            self.precomputed_proxy_capsules_fps = 0.0
        else:
            if isinstance(proxy_capsules, torch.Tensor):
                proxy_capsules_np = proxy_capsules.detach().cpu().numpy()
            else:
                try:
                    proxy_capsules_np = np.asarray(proxy_capsules, dtype=np.float32)
                except Exception as exc:
                    raise ValueError(
                        f"proxy_capsules cannot be converted to float32 array: {exc}"
                    ) from exc
            if proxy_capsules_np.ndim != 3 or proxy_capsules_np.shape[2] != 7:
                raise ValueError(
                    f"proxy_capsules must be [T, C, 7], got {proxy_capsules_np.shape}."
                )
            if proxy_capsules_np.shape[0] != pose_len:
                raise ValueError(
                    "proxy_capsules frame count mismatch: "
                    f"{proxy_capsules_np.shape[0]} vs pose frames {pose_len}."
                )

            proxy_capsules_version = _get_smpl_param(
                smpl_data, "proxy_capsules_version"
            )
            if proxy_capsules_version is None:
                proxy_capsules_version = 1
            try:
                proxy_capsules_version = int(proxy_capsules_version)
            except (TypeError, ValueError):
                raise ValueError(
                    "proxy_capsules_version must be an integer."
                )
            if proxy_capsules_version != 1:
                raise ValueError(
                    f"Unsupported proxy_capsules_version={proxy_capsules_version}, expected 1."
                )

            proxy_capsules_fps = _get_smpl_param(smpl_data, "proxy_capsules_fps")
            if proxy_capsules_fps is None:
                proxy_capsules_fps = joint_mats_fps
            try:
                proxy_capsules_fps = float(proxy_capsules_fps)
            except (TypeError, ValueError):
                raise ValueError(
                    f"proxy_capsules_fps must be a float, got {proxy_capsules_fps}."
                )
            if proxy_capsules_fps <= 0.0:
                raise ValueError(
                    f"proxy_capsules_fps must be > 0, got {proxy_capsules_fps}."
                )
            if abs(proxy_capsules_fps - joint_mats_fps) > 1.0e-4:
                raise ValueError(
                    "proxy_capsules_fps does not match joint_mats_fps: "
                    f"{proxy_capsules_fps} vs {joint_mats_fps}."
                )

            self.precomputed_proxy_capsules = proxy_capsules_np.astype(
                np.float32, copy=False
            )
            self.precomputed_proxy_capsule_count = int(proxy_capsules_np.shape[1])
            self.precomputed_proxy_capsules_version = int(proxy_capsules_version)
            self.precomputed_proxy_capsules_fps = float(proxy_capsules_fps)

        traj = None
        if self.trajectory_path and os.path.exists(self.trajectory_path):
            traj_npz = np.load(self.trajectory_path)
            if "trajectory" in traj_npz:
                traj = traj_npz["trajectory"]
            elif traj_npz.files:
                traj = traj_npz[traj_npz.files[0]]

        if traj is not None:
            if traj.ndim != 2 or traj.shape[1] != 7:
                raise ValueError(
                    f"trajectory should be Nx7 (pos + xyzw quat), got {traj.shape}."
                )
            traj = torch.from_numpy(traj.astype(np.float32)).to(self.device)
            self.traj_pos = traj[:, :3]
            self.traj_quat = _normalize_quat(traj[:, 3:])
        else:
            self.traj_pos = None
            self.traj_quat = None

        if self.smpl_type == "smplx":
            self.smpl_model = self.smplx_module.SMPLX(
                model_path=self.smpl_model_path,
                gender=self.smpl_gender,
                use_pca=False,
                num_pca_comps=45,
                flat_hand_mean=True,
                batch_size=1,
            ).to(self.device)
        else:
            self.smpl_model = self.smplx_module.SMPL(
                model_path=self.smpl_model_path,
                gender=self.smpl_gender,
                batch_size=1,
            ).to(self.device)

        self.smpl_model.eval()

        if self.coord_transform is None and not self.coord_system:
            inferred = _infer_coord_transform_from_pose(self.pose_data)
            if inferred is not None:
                self.coord_transform = torch.from_numpy(inferred).to(self.device)
                logger.info(
                    "GaussianAvatar: inferred OpenCV->Habitat coord transform from pose data."
                )

    def _sample_precomputed_proxy_capsules(self, frame_pos: float) -> np.ndarray:
        if self.precomputed_proxy_capsules is None:
            return self._empty_proxy_capsules
        if self.precomputed_proxy_capsule_count <= 0:
            return self._empty_proxy_capsules

        idx0, idx1, weight = _interp_indices(
            frame_pos, self.precomputed_proxy_capsules.shape[0]
        )
        caps0 = self.precomputed_proxy_capsules[idx0]
        if idx1 == idx0 or weight <= 0.0:
            return caps0

        caps1 = self.precomputed_proxy_capsules[idx1]
        if weight >= 1.0:
            return caps1
        return caps0 + (caps1 - caps0) * np.float32(weight)

    def _sample_precomputed_joint_mats(self, frame_pos: float) -> torch.Tensor:
        if self.precomputed_joint_mats is None:
            raise RuntimeError("Precomputed joint_mats are not initialized.")

        idx0, idx1, weight = _interp_indices(
            frame_pos, self.precomputed_joint_mats.shape[0]
        )
        mats0 = self.precomputed_joint_mats[idx0]
        if idx1 == idx0:
            # Return an owned tensor: downstream path applies in-place scale/offset.
            return mats0.clone()

        mats1 = self.precomputed_joint_mats[idx1]
        if weight <= 0.0:
            return mats0.clone()
        if weight >= 1.0:
            return mats1.clone()

        trans = mats0[:, :3, 3] + (mats1[:, :3, 3] - mats0[:, :3, 3]) * weight
        quat0 = _matrix_to_quat(mats0[:, :3, :3])
        quat1 = _matrix_to_quat(mats1[:, :3, :3])
        quat = _nlerp_quat(quat0, quat1, float(weight))
        rot = _quat_to_matrix(quat)

        out = torch.zeros_like(mats0)
        out[:, :3, :3] = rot
        out[:, :3, 3] = trans
        out[:, 3, 3] = 1.0
        return out.contiguous()

    def _sample_trajectory(self, frame_pos: float) -> Tuple[torch.Tensor, torch.Tensor]:
        idx0, idx1, weight = _interp_indices(frame_pos, self.traj_pos.shape[0])
        pos = self.traj_pos[idx0]
        if idx1 != idx0:
            pos = pos + (self.traj_pos[idx1] - pos) * weight

        quat = self.traj_quat[idx0]
        if idx1 != idx0:
            quat = _nlerp_quat(quat, self.traj_quat[idx1], weight)
        return pos, quat

    def _apply_world_transforms(
        self, joint_mats: torch.Tensor, traj_transform: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if traj_transform is not None:
            if self.coord_transform is not None:
                if self.trajectory_in_smpl:
                    joint_mats = torch.matmul(
                        self.coord_transform, torch.matmul(traj_transform, joint_mats)
                    )
                else:
                    joint_mats = torch.matmul(
                        traj_transform, torch.matmul(self.coord_transform, joint_mats)
                    )
            else:
                joint_mats = torch.matmul(traj_transform, joint_mats)
        elif self.coord_transform is not None:
            joint_mats = torch.matmul(self.coord_transform, joint_mats)

        if self.offset_y != 0.0:
            joint_mats[:, 1, 3] += self.offset_y
        return joint_mats

    def _finalize_time_settings(self) -> None:
        frame_count = int(self.pose_data.shape[0]) if self.pose_data is not None else 0
        if self.frame_time is not None and abs(self.frame_time - DEFAULT_FRAME_TIME) > 1.0e-6:
            logger.warning(
                "GaussianAvatar %s overriding frame_time %.4f to %.4f per README.",
                self.name,
                float(self.frame_time),
                DEFAULT_FRAME_TIME,
            )
        self.frame_time = DEFAULT_FRAME_TIME

        if self.time_end is None or self.time_end <= 0.0:
            duration = self.frame_time * frame_count if frame_count > 0 else 0.0
            self.time_end = self.time_begin + duration
        if self.time_end < self.time_begin:
            logger.warning(
                "GaussianAvatar %s time_end < time_begin (%.3f < %.3f); clamping.",
                self.name,
                self.time_end,
                self.time_begin,
            )
            self.time_end = self.time_begin

    def set_scene_time_config(self, time_max: float, loop: bool) -> None:
        self.scene_time_max = float(time_max)
        self.scene_time_loop = bool(loop)

    def _set_avatar_visible(self, sim, visible: bool) -> None:
        if self._visible == visible:
            return
        try:
            sim.set_gaussian_avatar_visible(self.avatar_id, bool(visible))
            self._visible = visible
        except Exception:
            self._visible = visible

    def _remove_physics_proxy(self, sim) -> None:
        self._proxy_capsules = self._empty_proxy_capsules

    def get_navmesh_capsules(self) -> np.ndarray:
        return self._proxy_capsules

    def _update_render_bounds(
        self, sim, joint_mats: torch.Tensor, proxy_capsules: np.ndarray
    ) -> None:
        if self._bounds_update_failed:
            return
        if not hasattr(sim, "update_gaussian_avatar_bounds"):
            self._bounds_update_failed = True
            return

        try:
            padding = max(0.02, 0.03 * float(self.scale))
            bounds_min, bounds_max = _compute_avatar_world_bounds(
                joint_mats, proxy_capsules, padding
            )
            if not (
                np.all(np.isfinite(bounds_min)) and np.all(np.isfinite(bounds_max))
            ):
                raise ValueError("non-finite avatar bounds")
            updated = bool(
                sim.update_gaussian_avatar_bounds(
                    self.avatar_id,
                    np.ascontiguousarray(bounds_min, dtype=np.float32),
                    np.ascontiguousarray(bounds_max, dtype=np.float32),
                )
            )
            if not updated:
                raise RuntimeError("backend rejected avatar bounds update")
        except Exception as exc:
            logger.warning(
                "GaussianAvatar %s render-bounds update failed: %s",
                self.name,
                exc,
            )
            self._bounds_update_failed = True

    def _set_active(self, sim, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        self._set_avatar_visible(sim, active)
        if not active:
            self._remove_physics_proxy(sim)

    def update(self, sim=None) -> bool:
        if sim is None:
            sim = self.sim

        with torch.no_grad():
            scene_time = _normalize_scene_time(
                float(sim.gaussian_time), self.scene_time_max, self.scene_time_loop
            )
            active = self.time_begin <= scene_time <= self.time_end
            if not active:
                self._set_active(sim, False)
                return False
            self._set_active(sim, True)

            frame_pos = _compute_frame_position(
                scene_time,
                self.pose_data.shape[0],
                self.time_begin,
                self.time_end,
                self.frame_time,
            )
            if self.precomputed_joint_mats is None:
                raise RuntimeError(
                    "GaussianAvatar strict mode requires precomputed joint_mats."
                )
            joint_mats = self._sample_precomputed_joint_mats(frame_pos)
            proxy_capsules = self._sample_precomputed_proxy_capsules(frame_pos)
            if (
                self.precomputed_joint_count > 0
                and joint_mats.shape[0] != self.precomputed_joint_count
            ):
                raise RuntimeError(
                    "Precomputed joint_mats joint count mismatch: "
                    f"{joint_mats.shape[0]} vs {self.precomputed_joint_count}."
                )
            if self.joint_count is not None and joint_mats.shape[0] != self.joint_count:
                if not self._joint_count_warned:
                    logger.warning(
                        "GaussianAvatar %s joint count mismatch: expected %d, got %d.",
                        self.name,
                        self.joint_count,
                        joint_mats.shape[0],
                    )
                    self._joint_count_warned = True

            if self.scale != 1.0:
                root_t = joint_mats[0, :3, 3].clone()
                joint_mats[:, :3, :3] *= self.scale
                joint_mats[:, :3, 3] = (
                    joint_mats[:, :3, 3] * self.scale + (1.0 - self.scale) * root_t
                )
                if proxy_capsules.size:
                    root_np = root_t.detach().cpu().numpy()
                    proxy_capsules = proxy_capsules.copy()
                    proxy_capsules[:, :3] = (
                        root_np + (proxy_capsules[:, :3] - root_np) * self.scale
                    )
                    proxy_capsules[:, 3:6] = (
                        root_np + (proxy_capsules[:, 3:6] - root_np) * self.scale
                    )
                    proxy_capsules[:, 6] *= self.scale

            traj_transform = None
            if self.traj_pos is not None:
                traj_frame = _compute_frame_position(
                    scene_time,
                    self.traj_pos.shape[0],
                    self.time_begin,
                    self.time_end,
                    self.frame_time,
                )
                traj_pos, traj_quat = self._sample_trajectory(traj_frame)
                traj_transform = _build_transform(traj_pos, traj_quat)

            joint_mats = self._apply_world_transforms(joint_mats, traj_transform)
            if proxy_capsules.size:
                capsule_world_transform = None
                if traj_transform is not None:
                    if self.coord_transform is not None:
                        if self.trajectory_in_smpl:
                            capsule_world_transform = torch.matmul(
                                self.coord_transform, traj_transform
                            )
                        else:
                            capsule_world_transform = torch.matmul(
                                traj_transform, self.coord_transform
                            )
                    else:
                        capsule_world_transform = traj_transform
                elif self.coord_transform is not None:
                    capsule_world_transform = self.coord_transform

                if capsule_world_transform is not None:
                    mat = capsule_world_transform.detach().cpu().numpy()
                    proxy_capsules = proxy_capsules.copy()
                    p0 = proxy_capsules[:, :3]
                    p1 = proxy_capsules[:, 3:6]
                    proxy_capsules[:, :3] = (mat[:3, :3] @ p0.T).T + mat[:3, 3]
                    proxy_capsules[:, 3:6] = (mat[:3, :3] @ p1.T).T + mat[:3, 3]
                if self.offset_y != 0.0:
                    proxy_capsules = proxy_capsules.copy()
                    proxy_capsules[:, 1] += self.offset_y
                    proxy_capsules[:, 4] += self.offset_y
            self._proxy_capsules = proxy_capsules
            self._update_render_bounds(sim, joint_mats, proxy_capsules)

        if self.sync_cuda:
            torch.cuda.current_stream().synchronize()

        return bool(
            sim.update_gaussian_avatar_pose(
                self.avatar_id, int(joint_mats.contiguous().data_ptr())
            )
        )


def load_gaussian_avatars(
    sim,
    scene_instance_path: Optional[str] = None,
    configs: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[GaussianAvatar]:
    avatar_ids = list(sim.get_gaussian_avatar_ids())
    if not avatar_ids:
        return []

    time_cfg: Dict[str, Any] = {}
    if configs is None:
        if scene_instance_path is None:
            scene_instance_path = _resolve_scene_instance_path(sim)
        configs, time_cfg = _load_avatar_configs(scene_instance_path)
        if not configs:
            logger.warning("No gaussian_avatars or avatars found in scene instance config.")
            return []

    base_dir = os.path.dirname(scene_instance_path) if scene_instance_path else os.getcwd()
    if len(configs) != len(avatar_ids):
        logger.warning(
            "GaussianAvatar config count (%d) does not match instance count (%d).",
            len(configs),
            len(avatar_ids),
        )

    avatars: List[GaussianAvatar] = []
    for idx, avatar_id in enumerate(avatar_ids):
        if idx >= len(configs):
            break
        avatars.append(GaussianAvatar(sim, avatar_id, configs[idx], base_dir))
    if not avatars:
        return []

    if "time_max" in time_cfg:
        scene_time_max = float(time_cfg["time_max"])
    else:
        scene_time_max = max((avatar.time_end for avatar in avatars), default=0.0)
    if scene_time_max < 0.0:
        scene_time_max = 0.0
    scene_time_loop = bool(time_cfg.get("time_loop", True))

    for avatar in avatars:
        avatar.set_scene_time_config(scene_time_max, scene_time_loop)

    try:
        sim.gaussian_time_min = 0.0
        sim.gaussian_time_max = float(scene_time_max)
        sim.gaussian_time_loop = bool(scene_time_loop)
    except Exception:
        pass

    return avatars


class GaussianAvatarManager:
    def __init__(self, avatars: Sequence[GaussianAvatar]):
        self.avatars = list(avatars)
        self._navmesh_capsule_update_failed = False

    def update(self, sim=None) -> None:
        if sim is None and self.avatars:
            sim = self.avatars[0].sim

        capsule_blocks: List[np.ndarray] = []
        total_capsules = 0
        for avatar in self.avatars:
            avatar.update(sim)
            capsules = avatar.get_navmesh_capsules()
            if capsules.size == 0:
                continue
            capsule_blocks.append(capsules)
            total_capsules += int(capsules.shape[0])

        if sim is None or self._navmesh_capsule_update_failed:
            return
        if not hasattr(sim, "set_navmesh_dynamic_capsules"):
            return

        try:
            if total_capsules == 0:
                if hasattr(sim, "clear_navmesh_dynamic_capsules"):
                    sim.clear_navmesh_dynamic_capsules()
                return

            merged = np.empty((total_capsules, 7), dtype=np.float32)
            cursor = 0
            for block in capsule_blocks:
                count = int(block.shape[0])
                merged[cursor : cursor + count, :] = block
                cursor += count
            sim.set_navmesh_dynamic_capsules(merged)
        except Exception as exc:
            logger.warning("GaussianAvatar navmesh capsule update failed: %s", exc)
            self._navmesh_capsule_update_failed = True
