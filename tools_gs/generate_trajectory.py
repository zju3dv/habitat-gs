#!/usr/bin/env python3

"""
Generate coupled SMPL pose + trajectory using GAMMA and a Habitat navmesh.

Output format:
  A .pkl containing dense per-frame SMPL params:
    - transl: [T, 3]
    - global_orient: [T, 3]
    - body_pose: [T, 63] for SMPL-X or padded to [T, 69] for SMPL trajectories
    - left_hand_pose: [T, 45] (zeros by default for SMPL-X trajectories)
    - right_hand_pose: [T, 45] (zeros by default for SMPL-X trajectories)
    - betas: [10]
    - gender: str
    - smpl_type: "smpl" or "smplx"
    - fps: float
    - joint_mats: [T, J, 4, 4] (float32)
    - joint_mats_space: "smpl_with_trans"
    - joint_mats_version: 1
    - joint_mats_fps: float
    - joint_mats_joint_count: int
    - proxy_capsules: [T, C, 7] (float32, optional)
    - proxy_capsules_version: 1 (when proxy_capsules is present)
    - proxy_capsules_fps: float (when proxy_capsules is present)
    - proxy_capsules_count: int (when proxy_capsules is present)
"""

import argparse
import glob
import math
import os
import pickle
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


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


try:
    import torch
except ImportError as exc:
    raise SystemExit("generate_trajectory.py requires PyTorch.") from exc

try:
    from scipy.spatial.transform import Rotation as SciRotation
except Exception as exc:
    raise SystemExit("generate_trajectory.py requires scipy.") from exc

from habitat_sim.nav import PathFinder, ShortestPath


def _parse_vec3(value: Sequence[float]) -> np.ndarray:
    return np.array(value, dtype=np.float32)


def _expand_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    if x.shape[0] == batch_size:
        return x
    if x.shape[0] != 1:
        raise ValueError(f"Expected batch size 1 or {batch_size}, got {x.shape[0]}")
    repeat = [batch_size] + [1] * (x.dim() - 1)
    return x.repeat(*repeat)


def _resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in ("1", "true", "t", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _infer_smpl_model_type(model_path: str, preferred: str = "auto") -> str:
    preferred = str(preferred or "auto").strip().lower()
    if preferred in {"smpl", "smplx"}:
        return preferred
    if preferred != "auto":
        raise ValueError(
            f"Unsupported smpl_type={preferred!r}. Expected one of: auto, smpl, smplx."
        )

    basename = os.path.basename(model_path).lower()
    if "smplx" in basename:
        return "smplx"
    if basename.startswith("smpl") or "smpl_" in basename:
        return "smpl"

    if os.path.isdir(model_path):
        entries = [name.lower() for name in os.listdir(model_path)]
        has_smplx = any(name.startswith("smplx_") for name in entries)
        has_smpl = any(
            name.startswith("smpl_") and not name.startswith("smplx_")
            for name in entries
        )
        if has_smplx and has_smpl:
            raise ValueError(
                "SMPL model path contains both SMPL and SMPL-X assets; "
                "pass --smpl-type smpl or --smpl-type smplx explicitly."
            )
        if has_smplx:
            return "smplx"
        if has_smpl:
            return "smpl"

    return "smpl"


def _build_proxy_model(model_path: str, model_type: str, gender: str):
    _ensure_inspect_getargspec()
    _ensure_legacy_numpy_aliases()
    try:
        import smplx
    except Exception as exc:
        raise RuntimeError("Failed to import smplx for proxy generation.") from exc

    if model_type == "smplx":
        return smplx.SMPLX(
            model_path=model_path,
            gender=gender,
            use_pca=False,
            num_pca_comps=45,
            flat_hand_mean=True,
            batch_size=1,
        )
    return smplx.SMPL(
        model_path=model_path,
        gender=gender,
        batch_size=1,
    )


def _load_proxy_rest_joints(model, model_type: str, betas: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    betas_t = torch.as_tensor(betas, dtype=torch.float32, device=device).reshape(1, -1)
    num_betas = int(getattr(model, "num_betas", betas_t.shape[1]))
    if betas_t.shape[1] < num_betas:
        pad = torch.zeros((1, num_betas - betas_t.shape[1]), dtype=betas_t.dtype, device=device)
        betas_t = torch.cat([betas_t, pad], dim=1)
    elif betas_t.shape[1] > num_betas:
        betas_t = betas_t[:, :num_betas]

    zeros = torch.zeros((1, 3), device=device)
    with torch.no_grad():
        if model_type == "smplx":
            hand_dims = model.NUM_HAND_JOINTS * 3
            output = model.forward(
                betas=betas_t,
                global_orient=zeros,
                transl=zeros,
                body_pose=torch.zeros((1, model.NUM_BODY_JOINTS * 3), device=device),
                jaw_pose=zeros,
                leye_pose=zeros,
                reye_pose=zeros,
                left_hand_pose=torch.zeros((1, hand_dims), device=device),
                right_hand_pose=torch.zeros((1, hand_dims), device=device),
            )
        else:
            output = model.forward(
                betas=betas_t,
                global_orient=zeros,
                transl=zeros,
                body_pose=torch.zeros((1, model.NUM_JOINTS * 3), device=device),
            )

    joints = output.joints[0].detach().cpu().numpy()
    num_joints = int(model.parents.shape[0])
    return joints[:num_joints].astype(np.float32)


def _resolve_proxy_joint_names(model_type: str, num_joints: int) -> List[str]:
    try:
        from smplx import joint_names as smplx_joint_names
    except Exception:
        smplx_joint_names = None

    if model_type == "smpl":
        names = (
            list(getattr(smplx_joint_names, "SMPL_JOINT_NAMES", []))
            if smplx_joint_names is not None
            else []
        )
        if not names:
            names = [
                "pelvis",
                "left_hip",
                "right_hip",
                "spine1",
                "left_knee",
                "right_knee",
                "spine2",
                "left_ankle",
                "right_ankle",
                "spine3",
                "left_foot",
                "right_foot",
                "neck",
                "left_collar",
                "right_collar",
                "head",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hand",
                "right_hand",
            ]
    else:
        names = (
            list(getattr(smplx_joint_names, "JOINT_NAMES", []))
            if smplx_joint_names is not None
            else []
        )
        if not names and smplx_joint_names is not None:
            names = list(getattr(smplx_joint_names, "SMPLH_JOINT_NAMES", []))

    if len(names) < num_joints:
        names.extend([f"joint_{i}" for i in range(len(names), num_joints)])
    return names[:num_joints]


def _rotation_from_z(vec: np.ndarray) -> np.ndarray:
    v = vec / np.linalg.norm(vec)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.dot(z, v))
    if dot > 0.999999:
        return np.eye(3, dtype=np.float64)
    if dot < -0.999999:
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )

    axis = np.cross(z, v)
    axis = axis / np.linalg.norm(axis)
    angle = math.acos(dot)
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z_axis = axis
    k = np.array(
        [
            [0.0, -z_axis, y],
            [z_axis, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + s * k + (1.0 - c) * (k @ k)


def _matrix_to_rpy(matrix: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
    singular = sy < 1.0e-6
    if not singular:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        pitch = math.atan2(-matrix[2, 0], sy)
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        pitch = math.atan2(-matrix[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def _format_vec(vec: np.ndarray) -> str:
    return "{:.6f} {:.6f} {:.6f}".format(float(vec[0]), float(vec[1]), float(vec[2]))


def _add_proxy_inertial(link: ET.Element, mass: float) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="{:.6f}".format(float(mass)))
    inertia_val = "{:.6f}".format(float(mass) * 1.0e-4)
    ET.SubElement(
        inertial,
        "inertia",
        ixx=inertia_val,
        ixy="0",
        ixz="0",
        iyy=inertia_val,
        iyz="0",
        izz=inertia_val,
    )


def _add_proxy_capsule_collision(
    link: ET.Element,
    center: np.ndarray,
    rpy: Tuple[float, float, float],
    radius: float,
    length: float,
) -> None:
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", xyz=_format_vec(center), rpy=_format_vec(np.array(rpy)))
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(
        geometry,
        "capsule",
        radius="{:.6f}".format(float(radius)),
        length="{:.6f}".format(float(length)),
    )


def _add_proxy_revolute_joint(
    robot: ET.Element,
    name: str,
    parent: str,
    child: str,
    offset: np.ndarray,
    axis: Tuple[float, float, float],
) -> None:
    joint = ET.SubElement(robot, "joint", name=name, type="revolute")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", xyz=_format_vec(offset), rpy="0 0 0")
    ET.SubElement(joint, "axis", xyz=_format_vec(np.array(axis, dtype=np.float32)))
    ET.SubElement(
        joint,
        "limit",
        lower="-3.14159",
        upper="3.14159",
        effort="1.0",
        velocity="1.0",
    )


def _build_proxy_urdf(
    joint_positions: np.ndarray,
    parents: np.ndarray,
    joint_names: List[str],
) -> ET.Element:
    robot = ET.Element("robot", name="smpl_proxy")
    collisions_by_link: Dict[str, List[Tuple[np.ndarray, Tuple[float, float, float], float, float]]] = {}
    min_radius = 0.01
    min_length = 0.001

    for idx in range(1, int(joint_positions.shape[0])):
        parent = int(parents[idx])
        if parent < 0:
            continue
        parent_name = joint_names[parent]
        vec = joint_positions[idx] - joint_positions[parent]
        length = float(np.linalg.norm(vec))
        if length < 1.0e-6:
            continue
        radius = max(length * 0.1, min_radius)
        radius = min(radius, length * 0.5)
        capsule_length = max(length - 2.0 * radius, min_length)
        rot = _rotation_from_z(vec)
        rpy = _matrix_to_rpy(rot)
        center = vec * 0.5
        collisions_by_link.setdefault(parent_name, []).append((center, rpy, radius, capsule_length))

    root_link = ET.SubElement(robot, "link", name=joint_names[0])
    _add_proxy_inertial(root_link, 1.0)
    for center, rpy, radius, length in collisions_by_link.get(joint_names[0], []):
        _add_proxy_capsule_collision(root_link, center, rpy, radius, length)

    for idx in range(1, int(joint_positions.shape[0])):
        parent = int(parents[idx])
        if parent < 0:
            continue
        parent_name = joint_names[parent]
        child_name = joint_names[idx]
        offset = joint_positions[idx] - joint_positions[parent]

        dummy_x_name = f"{child_name}_dummy_x"
        dummy_y_name = f"{child_name}_dummy_y"

        dummy_x = ET.SubElement(robot, "link", name=dummy_x_name)
        _add_proxy_inertial(dummy_x, 0.1)
        dummy_y = ET.SubElement(robot, "link", name=dummy_y_name)
        _add_proxy_inertial(dummy_y, 0.1)

        child_link = ET.SubElement(robot, "link", name=child_name)
        _add_proxy_inertial(child_link, 1.0)
        for center, rpy, radius, length in collisions_by_link.get(child_name, []):
            _add_proxy_capsule_collision(child_link, center, rpy, radius, length)

        _add_proxy_revolute_joint(
            robot,
            f"{child_name}_joint_x",
            parent_name,
            dummy_x_name,
            offset,
            (1.0, 0.0, 0.0),
        )
        _add_proxy_revolute_joint(
            robot,
            f"{child_name}_joint_y",
            dummy_x_name,
            dummy_y_name,
            np.zeros(3, dtype=np.float32),
            (0.0, 1.0, 0.0),
        )
        _add_proxy_revolute_joint(
            robot,
            f"{child_name}_joint_z",
            dummy_y_name,
            child_name,
            np.zeros(3, dtype=np.float32),
            (0.0, 0.0, 1.0),
        )

    return robot


def _parse_vec3_attr(raw: Optional[str]) -> np.ndarray:
    out = np.zeros(3, dtype=np.float32)
    if not raw:
        return out
    values = raw.replace(",", " ").split()
    for idx, value in enumerate(values[:3]):
        try:
            out[idx] = float(value)
        except (TypeError, ValueError):
            out[idx] = 0.0
    return out


def _rpy_to_matrix_np(rpy: np.ndarray) -> np.ndarray:
    roll = float(rpy[0])
    pitch = float(rpy[1])
    yaw = float(rpy[2])
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float32,
    )


def _axis_angle_matrix_np(axis: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1.0e-8:
        return np.eye(3, dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis / norm
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float32,
    )


def _axis_angle_to_euler_xyz(axis_angles: np.ndarray) -> np.ndarray:
    if axis_angles.size == 0:
        return axis_angles.reshape(0, 3).astype(np.float32)
    return SciRotation.from_rotvec(axis_angles).as_euler("XYZ", degrees=False).astype(np.float32)


@dataclass
class _ProxyJointSpec:
    parent_link: int
    child_link: int
    origin_xyz: np.ndarray
    origin_rot: np.ndarray
    axis: np.ndarray
    dof_index: int = -1


@dataclass
class _ProxyCollisionSpec:
    link_index: int
    local_p0: np.ndarray
    local_p1: np.ndarray
    radius: float


class _ProxyKinematicDriver:
    def __init__(self, urdf_path: str) -> None:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        link_elems = list(root.findall("link"))
        if not link_elems:
            raise ValueError(f"No links found in proxy URDF: {urdf_path}")

        self._link_names: List[str] = []
        self._link_index: Dict[str, int] = {}
        for link_elem in link_elems:
            name = link_elem.attrib.get("name", f"link_{len(self._link_names)}")
            if name in self._link_index:
                continue
            self._link_index[name] = len(self._link_names)
            self._link_names.append(name)

        self._collisions: List[_ProxyCollisionSpec] = []
        for link_elem in link_elems:
            name = link_elem.attrib.get("name")
            if not name or name not in self._link_index:
                continue
            self._collisions.extend(self._parse_link_collisions(link_elem, self._link_index[name]))

        self._joints: List[_ProxyJointSpec] = []
        child_link_indices = set()
        dof_index = 0
        for joint_elem in root.findall("joint"):
            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")
            if parent_elem is None or child_elem is None:
                continue
            parent_name = parent_elem.attrib.get("link")
            child_name = child_elem.attrib.get("link")
            if (
                not parent_name
                or not child_name
                or parent_name not in self._link_index
                or child_name not in self._link_index
            ):
                continue

            origin_elem = joint_elem.find("origin")
            origin_xyz = _parse_vec3_attr(origin_elem.attrib.get("xyz") if origin_elem is not None else None)
            origin_rpy = _parse_vec3_attr(origin_elem.attrib.get("rpy") if origin_elem is not None else None)

            axis_elem = joint_elem.find("axis")
            axis = _parse_vec3_attr(axis_elem.attrib.get("xyz") if axis_elem is not None else None)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1.0e-8:
                axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                axis = axis / axis_norm

            joint_type = str(joint_elem.attrib.get("type", "fixed")).lower()
            has_dof = joint_type in ("revolute", "continuous")
            this_dof = dof_index if has_dof else -1
            if has_dof:
                dof_index += 1

            parent_idx = self._link_index[parent_name]
            child_idx = self._link_index[child_name]
            self._joints.append(
                _ProxyJointSpec(
                    parent_link=parent_idx,
                    child_link=child_idx,
                    origin_xyz=origin_xyz.astype(np.float32),
                    origin_rot=_rpy_to_matrix_np(origin_rpy),
                    axis=axis.astype(np.float32),
                    dof_index=this_dof,
                )
            )
            child_link_indices.add(child_idx)

        self.dof_count = dof_index
        roots = [idx for idx in range(len(self._link_names)) if idx not in child_link_indices]
        self._root_link = roots[0] if roots else 0

        self._children_by_link: List[List[int]] = [[] for _ in self._link_names]
        for joint_idx, joint in enumerate(self._joints):
            self._children_by_link[joint.parent_link].append(joint_idx)

        self._joint_eval_order = self._build_joint_eval_order()
        self._identity = np.eye(3, dtype=np.float32)
        self._link_rot = np.tile(self._identity[None, :, :], (len(self._link_names), 1, 1))
        self._link_trans = np.zeros((len(self._link_names), 3), dtype=np.float32)
        self._empty_capsules = np.empty((0, 7), dtype=np.float32)

    def _build_joint_eval_order(self) -> List[int]:
        order: List[int] = []
        queue: List[int] = [self._root_link]
        q_index = 0
        while q_index < len(queue):
            link_idx = queue[q_index]
            q_index += 1
            for joint_idx in self._children_by_link[link_idx]:
                order.append(joint_idx)
                queue.append(self._joints[joint_idx].child_link)
        if len(order) < len(self._joints):
            seen = set(order)
            for joint_idx in range(len(self._joints)):
                if joint_idx not in seen:
                    order.append(joint_idx)
        return order

    def _parse_link_collisions(
        self,
        link_elem: ET.Element,
        link_idx: int,
    ) -> List[_ProxyCollisionSpec]:
        collisions: List[_ProxyCollisionSpec] = []
        for collision_elem in link_elem.findall("collision"):
            geometry_elem = collision_elem.find("geometry")
            if geometry_elem is None:
                continue
            capsule_elem = geometry_elem.find("capsule")
            if capsule_elem is None:
                continue

            try:
                radius = float(capsule_elem.attrib.get("radius", "0"))
                length = float(capsule_elem.attrib.get("length", "0"))
            except (TypeError, ValueError):
                continue
            if radius <= 0.0:
                continue

            origin_elem = collision_elem.find("origin")
            origin_xyz = _parse_vec3_attr(origin_elem.attrib.get("xyz") if origin_elem is not None else None)
            origin_rpy = _parse_vec3_attr(origin_elem.attrib.get("rpy") if origin_elem is not None else None)
            origin_rot = _rpy_to_matrix_np(origin_rpy)
            half_axis = origin_rot @ np.array([0.0, 0.0, 0.5 * length], dtype=np.float32)
            center = origin_xyz.astype(np.float32)
            collisions.append(
                _ProxyCollisionSpec(
                    link_index=link_idx,
                    local_p0=center - half_axis,
                    local_p1=center + half_axis,
                    radius=float(radius),
                )
            )
        return collisions

    def compute_capsules(self, root_transform: np.ndarray, joint_positions: np.ndarray) -> np.ndarray:
        if not self._collisions:
            return self._empty_capsules

        joint_values = np.asarray(joint_positions, dtype=np.float32).reshape(-1)
        if self.dof_count and joint_values.shape[0] < self.dof_count:
            raise ValueError(
                f"proxy DOF mismatch: expected {self.dof_count}, got {joint_values.shape[0]}"
            )
        if joint_values.shape[0] > self.dof_count:
            joint_values = joint_values[: self.dof_count]

        root_rot = np.asarray(root_transform[:3, :3], dtype=np.float32)
        root_trans = np.asarray(root_transform[:3, 3], dtype=np.float32)
        self._link_rot[:] = self._identity
        self._link_trans[:] = root_trans
        self._link_rot[self._root_link] = root_rot
        self._link_trans[self._root_link] = root_trans

        for joint_idx in self._joint_eval_order:
            joint = self._joints[joint_idx]
            parent_rot = self._link_rot[joint.parent_link]
            parent_trans = self._link_trans[joint.parent_link]
            child_rot = parent_rot @ joint.origin_rot
            child_trans = parent_trans + parent_rot @ joint.origin_xyz
            if joint.dof_index >= 0:
                angle = float(joint_values[joint.dof_index])
                if abs(angle) > 1.0e-8:
                    child_rot = child_rot @ _axis_angle_matrix_np(joint.axis, angle)
            self._link_rot[joint.child_link] = child_rot
            self._link_trans[joint.child_link] = child_trans

        capsules = np.empty((len(self._collisions), 7), dtype=np.float32)
        for idx, collision in enumerate(self._collisions):
            link_rot = self._link_rot[collision.link_index]
            link_trans = self._link_trans[collision.link_index]
            p0 = link_trans + link_rot @ collision.local_p0
            p1 = link_trans + link_rot @ collision.local_p1
            capsules[idx, :3] = p0
            capsules[idx, 3:6] = p1
            capsules[idx, 6] = collision.radius
        return capsules


def _precompute_proxy_capsules(
    urdf_path: str,
    model_type: str,
    root_rest_joint: np.ndarray,
    transl: np.ndarray,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    left_hand_pose: np.ndarray,
    right_hand_pose: np.ndarray,
) -> np.ndarray:
    frame_count = int(transl.shape[0])
    if global_orient.shape[0] != frame_count or body_pose.shape[0] != frame_count:
        raise ValueError("Proxy precompute expects transl/global_orient/body_pose with same frame count.")
    if left_hand_pose.shape[0] != frame_count or right_hand_pose.shape[0] != frame_count:
        raise ValueError("Proxy precompute expects hand pose arrays with same frame count.")

    body_pose = _adapt_body_pose_for_model_type(body_pose, model_type)
    if model_type == "smplx":
        zeros = np.zeros((frame_count, 3), dtype=np.float32)
        full_pose = np.concatenate(
            [
                body_pose.astype(np.float32),
                zeros,
                zeros,
                zeros,
                left_hand_pose.astype(np.float32),
                right_hand_pose.astype(np.float32),
            ],
            axis=1,
        )
    else:
        full_pose = body_pose.astype(np.float32)

    if full_pose.shape[1] % 3 != 0:
        raise ValueError(f"Unexpected pose dimension for proxy precompute: {full_pose.shape}.")

    axis_angles = full_pose.reshape(frame_count, -1, 3).astype(np.float32)
    eulers = _axis_angle_to_euler_xyz(axis_angles.reshape(-1, 3)).reshape(frame_count, -1)
    root_rest_joint = np.asarray(root_rest_joint, dtype=np.float32).reshape(3)

    driver = _ProxyKinematicDriver(urdf_path)
    if driver.dof_count > eulers.shape[1]:
        raise ValueError(
            f"Proxy DOF mismatch: URDF expects {driver.dof_count}, but pose provides {eulers.shape[1]}."
        )

    out = np.empty((frame_count, len(driver._collisions), 7), dtype=np.float32)
    for frame_idx in range(frame_count):
        root_transform = np.eye(4, dtype=np.float32)
        root_rot = _rotation_from_rotvec(global_orient[frame_idx])
        root_transform[:3, :3] = root_rot
        # SMPL transl is the model-space translation, not the pelvis world position.
        # The proxy FK root must match the actual root joint used by avatar skinning.
        root_transform[:3, 3] = transl[frame_idx] + root_rot @ root_rest_joint
        out[frame_idx] = driver.compute_capsules(root_transform, eulers[frame_idx])
    return out


@contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _dedupe_points(points: np.ndarray, eps: float = 1.0e-4) -> np.ndarray:
    if points.shape[0] <= 1:
        return points
    kept = [points[0]]
    for pt in points[1:]:
        if np.linalg.norm(pt - kept[-1]) > eps:
            kept.append(pt)
    return np.array(kept, dtype=np.float32)


def _find_path_segment(
    pathfinder: PathFinder, start: np.ndarray, end: np.ndarray
) -> Optional[np.ndarray]:
    path = ShortestPath()
    path.requested_start = start
    path.requested_end = end
    if not pathfinder.find_path(path) or not path.points:
        return None
    pts = np.array([[p[0], p[1], p[2]] for p in path.points], dtype=np.float32)
    return pts


def _build_navmesh_path(
    pathfinder: PathFinder,
    start: np.ndarray,
    via: Sequence[np.ndarray],
    end: np.ndarray,
) -> np.ndarray:
    waypoints = [start] + list(via) + [end]
    snapped = [np.array(pathfinder.snap_point(p), dtype=np.float32) for p in waypoints]
    segments: List[np.ndarray] = []
    for idx in range(len(snapped) - 1):
        seg = _find_path_segment(pathfinder, snapped[idx], snapped[idx + 1])
        if seg is None:
            raise RuntimeError(
                f"Failed to find path segment {idx} from {snapped[idx]} to {snapped[idx + 1]}"
            )
        if segments:
            seg = seg[1:]
        segments.append(seg)
    points = np.concatenate(segments, axis=0)
    points = _dedupe_points(points)
    if points.shape[0] < 2:
        raise RuntimeError("Navmesh path must contain at least 2 points.")
    return points


def _path_length(points: np.ndarray, up_axis: Optional[int] = None) -> float:
    cumulative = _cumulative_path_lengths(points, up_axis=up_axis)
    return float(cumulative[-1]) if cumulative.size else 0.0


def _planar_points(points: np.ndarray, up_axis: int) -> np.ndarray:
    axes = [axis for axis in range(3) if axis != up_axis]
    return np.asarray(points, dtype=np.float32)[..., axes]


def _distance_to_navmesh_edge(
    pathfinder: PathFinder,
    point: np.ndarray,
    max_search_radius: float = 2.0,
) -> float:
    try:
        distance = float(pathfinder.distance_to_closest_obstacle(point, max_search_radius))
    except TypeError:
        try:
            distance = float(pathfinder.distance_to_closest_obstacle(point))
        except Exception:
            return 0.0
    except Exception:
        return 0.0
    if not np.isfinite(distance):
        return float(max_search_radius)
    return float(max(0.0, min(distance, max_search_radius)))


def _sample_path_clearances(
    pathfinder: PathFinder,
    points: np.ndarray,
    sample_step: float = 0.6,
    max_search_radius: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    dense = _subdivide_path(points, max_segment_length=sample_step)
    clearances = np.array(
        [
            _distance_to_navmesh_edge(
                pathfinder, np.asarray(point, dtype=np.float32), max_search_radius=max_search_radius
            )
            for point in dense
        ],
        dtype=np.float32,
    )
    return dense, clearances


def _sample_repeat_penalty(
    existing_points: np.ndarray,
    candidate_points: np.ndarray,
    up_axis: int = 1,
    near_threshold: float = 0.75,
) -> float:
    existing_points = np.asarray(existing_points, dtype=np.float32)
    candidate_points = np.asarray(candidate_points, dtype=np.float32)
    if existing_points.size == 0 or candidate_points.shape[0] <= 1:
        return 0.0
    cand_planar = _planar_points(candidate_points[1:], up_axis=up_axis)
    if cand_planar.size == 0:
        return 0.0
    exist_planar = _planar_points(existing_points, up_axis=up_axis)
    deltas = cand_planar[:, None, :] - exist_planar[None, :, :]
    min_dists = np.linalg.norm(deltas, axis=-1).min(axis=1)
    hard_overlap = np.mean(min_dists < near_threshold)
    soft_threshold = near_threshold * 2.0
    soft_overlap = np.clip((soft_threshold - min_dists) / max(soft_threshold, 1.0e-6), 0.0, 1.0)
    return float(hard_overlap + 0.35 * np.mean(soft_overlap))


def _segment_direction(points: np.ndarray, up_axis: int = 1) -> Optional[np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 2:
        return None
    direction = pts[-1] - pts[0]
    direction[up_axis] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-6:
        return None
    return (direction / norm).astype(np.float32)


def _select_sampling_island(
    pathfinder: PathFinder,
    reference_point: Optional[np.ndarray],
) -> Tuple[int, Optional[np.ndarray]]:
    if reference_point is not None:
        snapped = np.array(pathfinder.snap_point(reference_point), dtype=np.float32)
        if not _is_navigable_point(pathfinder, snapped):
            raise RuntimeError(f"Reference point is not on the navmesh: {reference_point}")
        island_index = int(pathfinder.get_island(snapped))
        return island_index, snapped

    best_island = -1
    best_area = -1.0
    num_islands = int(getattr(pathfinder, "num_islands", 0))
    for island_index in range(num_islands):
        try:
            area = float(pathfinder.island_area(island_index))
        except Exception:
            area = -1.0
        if area > best_area:
            best_area = area
            best_island = island_index
    return best_island, None


def _sample_candidate_point(
    pathfinder: PathFinder,
    island_index: int,
    reference_point: Optional[np.ndarray],
    reference_radius: Optional[float],
    min_clearance: float,
    sample_max_tries: int = 24,
    edge_search_radius: float = 2.0,
) -> Tuple[np.ndarray, float]:
    best_point = None
    best_clearance = -1.0
    for _ in range(sample_max_tries):
        try:
            if reference_point is not None and reference_radius is not None:
                point = np.array(
                    pathfinder.get_random_navigable_point_near(
                        reference_point,
                        float(reference_radius),
                        100,
                        island_index,
                    ),
                    dtype=np.float32,
                )
            else:
                point = np.array(
                    pathfinder.get_random_navigable_point(100, island_index),
                    dtype=np.float32,
                )
        except Exception:
            continue
        if not _is_navigable_point(pathfinder, point):
            continue
        clearance = _distance_to_navmesh_edge(
            pathfinder,
            point,
            max_search_radius=edge_search_radius,
        )
        if clearance > best_clearance:
            best_point = point
            best_clearance = clearance
        if clearance >= min_clearance:
            return point.astype(np.float32), float(clearance)
    if best_point is None:
        raise RuntimeError("Failed to sample a navigable point on the navmesh.")
    return best_point.astype(np.float32), float(best_clearance)


def _build_auto_navmesh_path(
    pathfinder: PathFinder,
    target_length: float,
    reference_point: Optional[np.ndarray] = None,
    up_axis: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if target_length <= 0.0:
        raise ValueError(f"path_length must be > 0, got {target_length}.")

    island_index, snapped_ref = _select_sampling_island(pathfinder, reference_point)
    if island_index < 0:
        raise RuntimeError("Failed to determine a valid navmesh island for auto path sampling.")

    try:
        island_radius = float(pathfinder.island_radius(island_index))
    except Exception:
        island_radius = float(target_length)
    if not np.isfinite(island_radius) or island_radius <= 0.0:
        island_radius = float(target_length)

    if snapped_ref is not None:
        reference_radius = min(
            max(target_length * 0.85, 3.0),
            max(island_radius * 0.95, 3.0),
        )
    else:
        reference_radius = None

    min_required_length = float(target_length) + 1.0e-3
    min_segment_length = max(1.2, min(4.0, 0.18 * target_length))
    max_segment_length = min(max(4.0, 0.55 * target_length), 12.0)
    if reference_radius is not None:
        max_segment_length = min(max_segment_length, max(3.0, reference_radius * 1.1))

    edge_search_radius = 2.0
    start_clearance = 0.55
    segment_clearance = 0.25
    candidate_count = 56
    max_segments = max(4, int(math.ceil(target_length / max(min_segment_length, 1.0))) + 4)
    max_restarts = 12

    best_path: Optional[np.ndarray] = None
    best_length = -1.0
    best_meta: Dict[str, Any] = {}

    for _ in range(max_restarts):
        start_point, start_point_clearance = _sample_candidate_point(
            pathfinder=pathfinder,
            island_index=island_index,
            reference_point=snapped_ref,
            reference_radius=reference_radius,
            min_clearance=start_clearance,
            sample_max_tries=48,
            edge_search_radius=edge_search_radius,
        )

        path_points = start_point.reshape(1, 3).astype(np.float32)
        visited_points = path_points.copy()
        current = start_point.astype(np.float32)
        prev_direction: Optional[np.ndarray] = None
        total_length = 0.0
        segment_count = 0

        while segment_count < max_segments and total_length < min_required_length:
            remaining = max(min_required_length - total_length, 0.0)
            best_candidate = None

            for _ in range(candidate_count):
                try:
                    candidate_point, endpoint_clearance = _sample_candidate_point(
                        pathfinder=pathfinder,
                        island_index=island_index,
                        reference_point=snapped_ref,
                        reference_radius=reference_radius,
                        min_clearance=segment_clearance,
                        sample_max_tries=8,
                        edge_search_radius=edge_search_radius,
                    )
                except RuntimeError:
                    continue

                planar_delta = candidate_point - current
                planar_delta[up_axis] = 0.0
                if float(np.linalg.norm(planar_delta)) < 0.75:
                    continue

                segment = _find_path_segment(pathfinder, current, candidate_point)
                if segment is None or segment.shape[0] < 2:
                    continue
                segment = _dedupe_points(segment)
                segment_length = _path_length(segment, up_axis=up_axis)
                if segment_length < min_segment_length:
                    continue
                if segment_length > max_segment_length and remaining > max_segment_length * 0.65:
                    continue

                dense_segment, clearances = _sample_path_clearances(
                    pathfinder,
                    segment,
                    sample_step=0.6,
                    max_search_radius=edge_search_radius,
                )
                min_clear = float(np.min(clearances)) if clearances.size else 0.0
                mean_clear = float(np.mean(clearances)) if clearances.size else 0.0
                if min_clear < segment_clearance:
                    continue

                repeat_penalty = _sample_repeat_penalty(
                    visited_points,
                    dense_segment,
                    up_axis=up_axis,
                    near_threshold=0.8,
                )
                segment_direction = _segment_direction(dense_segment, up_axis=up_axis)
                turn_alignment = 0.0
                if prev_direction is not None and segment_direction is not None:
                    turn_alignment = float(np.clip(np.dot(prev_direction, segment_direction), -1.0, 1.0))

                ref_penalty = 0.0
                if snapped_ref is not None and reference_radius is not None:
                    ref_dists = np.linalg.norm(
                        _planar_points(dense_segment, up_axis=up_axis)
                        - _planar_points(snapped_ref.reshape(1, 3), up_axis=up_axis)[0][None, :],
                        axis=1,
                    )
                    ref_penalty = float(np.mean(ref_dists) / max(reference_radius, 1.0e-6))

                overshoot_penalty = 0.0
                if remaining > 0.0 and segment_length > remaining * 1.5:
                    overshoot_penalty = (segment_length - remaining * 1.5) / max(remaining, 1.0)

                score = (
                    2.4 * min(mean_clear, edge_search_radius)
                    + 2.0 * min(min_clear, edge_search_radius)
                    + 0.25 * min(segment_length, max_segment_length)
                    + 0.5 * turn_alignment
                    - 5.0 * repeat_penalty
                    - 0.8 * ref_penalty
                    - 0.5 * overshoot_penalty
                    + 0.3 * min(endpoint_clearance, edge_search_radius)
                )

                if best_candidate is None or score > best_candidate["score"]:
                    best_candidate = {
                        "segment": segment,
                        "dense_segment": dense_segment,
                        "segment_length": float(segment_length),
                        "segment_direction": segment_direction,
                        "score": float(score),
                    }

            if best_candidate is None:
                break

            segment = best_candidate["segment"]
            dense_segment = best_candidate["dense_segment"]
            path_points = np.concatenate([path_points, segment[1:]], axis=0)
            visited_points = np.concatenate([visited_points, dense_segment[1:]], axis=0)
            total_length += float(best_candidate["segment_length"])
            current = path_points[-1].astype(np.float32)
            prev_direction = best_candidate["segment_direction"]
            segment_count += 1

        total_length = _path_length(path_points, up_axis=up_axis)
        if total_length > best_length:
            best_length = total_length
            best_path = _dedupe_points(path_points.astype(np.float32))
            best_meta = {
                "island_index": island_index,
                "reference_point": snapped_ref.astype(np.float32) if snapped_ref is not None else None,
                "reference_radius": float(reference_radius) if reference_radius is not None else None,
                "start_point_clearance": float(start_point_clearance),
                "path_length": float(total_length),
                "segment_count": int(segment_count),
            }

        if total_length >= min_required_length and best_path is not None:
            return best_path, best_meta

    if best_path is None or best_path.shape[0] < 2:
        raise RuntimeError(
            f"Failed to auto-sample a valid navmesh path for target length {target_length:.3f}m."
        )
    raise RuntimeError(
        "Failed to auto-sample a navmesh path longer than "
        f"{target_length:.3f}m. Best path length was {best_length:.3f}m."
    )


def _subdivide_path(points: np.ndarray, max_segment_length: float) -> np.ndarray:
    if max_segment_length <= 0.0:
        raise ValueError(
            f"max_segment_length must be > 0, got {max_segment_length}."
        )
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points should be [N, 3], got {points.shape}.")
    if points.shape[0] <= 1:
        return points.copy()

    dense: List[np.ndarray] = [points[0]]
    for idx in range(points.shape[0] - 1):
        start = points[idx]
        end = points[idx + 1]
        segment = end - start
        seg_len = float(np.linalg.norm(segment))
        n_steps = max(1, int(math.ceil(seg_len / max_segment_length)))
        for step in range(1, n_steps + 1):
            alpha = float(step) / float(n_steps)
            point = (1.0 - alpha) * start + alpha * end
            if float(np.linalg.norm(point - dense[-1])) > 1.0e-4:
                dense.append(point.astype(np.float32))

    dense_points = np.array(dense, dtype=np.float32)
    return _dedupe_points(dense_points)


def _cumulative_path_lengths(points: np.ndarray, up_axis: Optional[int] = None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points should be [N, 3], got {points.shape}.")
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    coords = points
    if up_axis is not None:
        axes = [axis for axis in range(points.shape[1]) if axis != up_axis]
        coords = points[:, axes]
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(seg_lengths, dtype=np.float32)],
        axis=0,
    )


def _sample_path_by_length(
    points: np.ndarray,
    sample_lengths: np.ndarray,
    up_axis: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    sample_lengths = np.asarray(sample_lengths, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points should be [N, 3], got {points.shape}.")
    if points.shape[0] == 0:
        raise ValueError("points must not be empty.")

    cumulative = _cumulative_path_lengths(points, up_axis=up_axis)
    total_length = float(cumulative[-1]) if cumulative.size else 0.0
    if points.shape[0] == 1 or total_length <= 1.0e-6:
        fallback = np.repeat(points[:1], sample_lengths.shape[0], axis=0)
        tangents = np.repeat(
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32), sample_lengths.shape[0], axis=0
        )
        return fallback, tangents

    coords = points
    if up_axis is not None:
        axes = [axis for axis in range(points.shape[1]) if axis != up_axis]
        coords = points[:, axes]
    seg_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)

    clipped = np.clip(sample_lengths, 0.0, total_length)
    seg_idx = np.searchsorted(cumulative, clipped, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, points.shape[0] - 2)

    out_points = np.zeros((clipped.shape[0], 3), dtype=np.float32)
    out_tangents = np.zeros((clipped.shape[0], 3), dtype=np.float32)

    for row, idx in enumerate(seg_idx.tolist()):
        while idx < seg_lengths.shape[0] - 1 and seg_lengths[idx] <= 1.0e-6:
            idx += 1
        while idx > 0 and seg_lengths[idx] <= 1.0e-6:
            idx -= 1

        length0 = float(cumulative[idx])
        seg_len = max(float(seg_lengths[idx]), 1.0e-6)
        alpha = (float(clipped[row]) - length0) / seg_len
        alpha = min(max(alpha, 0.0), 1.0)

        start = points[idx]
        end = points[idx + 1]
        out_points[row] = (1.0 - alpha) * start + alpha * end
        out_tangents[row] = end - start

    return out_points, out_tangents


def _fit_translations_to_path(
    transl: np.ndarray,
    path_points: np.ndarray,
    up_axis: int,
    heading_lookahead: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    transl = np.asarray(transl, dtype=np.float32)
    path_points = np.asarray(path_points, dtype=np.float32)
    if transl.ndim != 2 or transl.shape[1] != 3:
        raise ValueError(f"transl should be [T, 3], got {transl.shape}.")
    if transl.shape[0] == 0:
        return transl.copy(), np.zeros((0, 3), dtype=np.float32), 0.0, 0.0, 1.0

    raw_cumulative = _cumulative_path_lengths(transl, up_axis=up_axis)
    path_cumulative = _cumulative_path_lengths(path_points, up_axis=up_axis)
    raw_total = float(raw_cumulative[-1]) if raw_cumulative.size else 0.0
    path_total = float(path_cumulative[-1]) if path_cumulative.size else 0.0

    if path_points.shape[0] == 0 or path_total <= 1.0e-6:
        return (
            np.repeat(path_points[:1], transl.shape[0], axis=0),
            np.repeat(
                np.array([[0.0, 0.0, 1.0]], dtype=np.float32), transl.shape[0], axis=0
            ),
            raw_total,
            path_total,
            1.0,
        )

    if raw_total <= 1.0e-6:
        progress = np.linspace(0.0, path_total, transl.shape[0], dtype=np.float32)
        scale = 0.0
    else:
        scale = path_total / raw_total
        progress = raw_cumulative * scale
        progress[-1] = path_total

    fitted_transl, _ = _sample_path_by_length(path_points, progress, up_axis=up_axis)

    lookahead = max(float(heading_lookahead), 1.0e-3)
    lookbehind_progress = np.clip(progress - 0.25 * lookahead, 0.0, path_total)
    lookahead_progress = np.clip(progress + lookahead, 0.0, path_total)
    behind_pts, _ = _sample_path_by_length(path_points, lookbehind_progress, up_axis=up_axis)
    ahead_pts, _ = _sample_path_by_length(path_points, lookahead_progress, up_axis=up_axis)
    tangents = ahead_pts - behind_pts

    fitted_transl[0] = path_points[0]
    fitted_transl[-1] = path_points[-1]
    if tangents.shape[0] >= 2 and float(np.linalg.norm(tangents[-1])) <= 1.0e-6:
        tangents[-1] = tangents[-2]
    return fitted_transl, tangents.astype(np.float32), raw_total, path_total, float(scale)


def _rotation_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    return SciRotation.from_rotvec(rotvec).as_matrix()


def _rotvec_from_rotation(rotmat: np.ndarray) -> np.ndarray:
    return SciRotation.from_matrix(rotmat).as_rotvec()


def _apply_rotation_to_rotvecs(rot: np.ndarray, rotvecs: np.ndarray) -> np.ndarray:
    mats = SciRotation.from_rotvec(rotvecs).as_matrix()
    mats = np.einsum("ij,tjk->tik", rot, mats)
    return SciRotation.from_matrix(mats).as_rotvec()


def _max_rotvec_angle(a: np.ndarray, b: np.ndarray) -> float:
    rel = SciRotation.from_rotvec(a.reshape(-1, 3)).inv() * SciRotation.from_rotvec(
        b.reshape(-1, 3)
    )
    angles = np.linalg.norm(rel.as_rotvec(), axis=1)
    return float(np.max(angles)) if angles.size else 0.0


def _smooth_pose_spikes(
    transl: np.ndarray,
    pose_rotvecs: np.ndarray,
    angle_thresh: float = 1.0,
    trans_thresh: float = 0.2,
    span_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if pose_rotvecs.shape[0] < 3:
        return transl, pose_rotvecs, 0

    pose_in = pose_rotvecs
    pose_out = pose_rotvecs.copy()
    transl_in = transl
    transl_out = transl.copy()
    spikes = 0

    for i in range(1, pose_in.shape[0] - 1):
        prev_pose = pose_in[i - 1]
        curr_pose = pose_in[i]
        next_pose = pose_in[i + 1]

        ang_prev = _max_rotvec_angle(prev_pose, curr_pose)
        ang_next = _max_rotvec_angle(curr_pose, next_pose)
        ang_span = _max_rotvec_angle(prev_pose, next_pose)

        trans_prev = float(np.linalg.norm(transl_in[i] - transl_in[i - 1]))
        trans_next = float(np.linalg.norm(transl_in[i + 1] - transl_in[i]))
        trans_span = float(np.linalg.norm(transl_in[i + 1] - transl_in[i - 1]))

        rot_spike = (
            ang_prev > angle_thresh
            and ang_next > angle_thresh
            and ang_span < angle_thresh * span_ratio
        )
        trans_spike = (
            trans_prev > trans_thresh
            and trans_next > trans_thresh
            and trans_span < trans_thresh * span_ratio
        )
        if not (rot_spike or trans_spike):
            continue

        rot_prev = SciRotation.from_rotvec(prev_pose.reshape(-1, 3))
        rot_next = SciRotation.from_rotvec(next_pose.reshape(-1, 3))
        rel = rot_prev.inv() * rot_next
        rot_mid = rot_prev * SciRotation.from_rotvec(0.5 * rel.as_rotvec())
        pose_out[i] = rot_mid.as_rotvec().reshape(prev_pose.shape)
        transl_out[i] = 0.5 * (transl_in[i - 1] + transl_in[i + 1])
        spikes += 1

    return transl_out, pose_out, spikes


def _is_navigable_point(pathfinder: PathFinder, point: np.ndarray) -> bool:
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return False
    if not hasattr(pathfinder, "is_navigable"):
        return True
    try:
        return bool(pathfinder.is_navigable(point))
    except Exception:
        return False


def _snap_to_navmesh_or_fallback(
    pathfinder: PathFinder, point: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    try:
        snapped = np.array(pathfinder.snap_point(point), dtype=np.float32)
    except Exception:
        snapped = fallback
    if not _is_navigable_point(pathfinder, snapped):
        return fallback.astype(np.float32, copy=True)
    return snapped.astype(np.float32, copy=True)


def _project_transl_to_navmesh(
    pathfinder: PathFinder, transl: np.ndarray
) -> Tuple[np.ndarray, int, int]:
    """Project each translation frame onto navmesh.

    Returns:
        projected_transl, num_snapped_frames, num_connectivity_fallback_frames
    """
    transl_np = np.asarray(transl, dtype=np.float32)
    if transl_np.ndim != 2 or transl_np.shape[1] != 3:
        raise ValueError(f"transl should be [T, 3], got {transl_np.shape}.")
    if transl_np.shape[0] == 0:
        return transl_np.copy(), 0, 0

    projected = transl_np.copy()
    snapped_count = 0
    connectivity_fallback_count = 0

    first = _snap_to_navmesh_or_fallback(pathfinder, projected[0], projected[0])
    if float(np.linalg.norm(first - projected[0])) > 1.0e-5:
        snapped_count += 1
    projected[0] = first
    prev = first

    for idx in range(1, projected.shape[0]):
        raw = projected[idx]
        snapped = _snap_to_navmesh_or_fallback(pathfinder, raw, prev)
        if float(np.linalg.norm(snapped - raw)) > 1.0e-5:
            snapped_count += 1

        # Keep temporal continuity on the same navigable component.
        if _find_path_segment(pathfinder, prev, snapped) is None:
            snapped = prev.copy()
            connectivity_fallback_count += 1

        projected[idx] = snapped
        prev = snapped

    return projected, snapped_count, connectivity_fallback_count


def _transform_points(rot: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (rot @ points.T).T


def _rotation_matrix_from_forward(forward: np.ndarray, up_axis: int) -> np.ndarray:
    forward = np.asarray(forward, dtype=np.float64).reshape(3)
    up = np.zeros((3,), dtype=np.float64)
    up[up_axis] = 1.0

    planar = forward.copy()
    planar[up_axis] = 0.0
    norm = float(np.linalg.norm(planar))
    if norm <= 1.0e-6:
        fallback_axis = 2 if up_axis != 2 else 1
        planar[fallback_axis] = 1.0
        norm = float(np.linalg.norm(planar))
    forward_n = planar / norm

    right = np.cross(up, forward_n)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1.0e-6:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if up_axis == 0:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = fallback
        right_norm = float(np.linalg.norm(right))
    right = right / right_norm
    up_ortho = np.cross(forward_n, right)
    up_ortho = up_ortho / max(float(np.linalg.norm(up_ortho)), 1.0e-6)
    return np.stack([right, up_ortho, forward_n], axis=1)


def _rotvecs_from_path_tangents(tangents: np.ndarray, up_axis: int) -> np.ndarray:
    tangents = np.asarray(tangents, dtype=np.float32)
    if tangents.ndim != 2 or tangents.shape[1] != 3:
        raise ValueError(f"tangents should be [T, 3], got {tangents.shape}.")
    if tangents.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    rotmats: List[np.ndarray] = []
    prev_rotmat: Optional[np.ndarray] = None
    for tangent in tangents:
        planar = tangent.copy()
        planar[up_axis] = 0.0
        if float(np.linalg.norm(planar)) <= 1.0e-6 and prev_rotmat is not None:
            rotmats.append(prev_rotmat)
            continue
        rotmat = _rotation_matrix_from_forward(planar, up_axis=up_axis)
        prev_rotmat = rotmat
        rotmats.append(rotmat)

    if prev_rotmat is None:
        prev_rotmat = _rotation_matrix_from_forward(np.array([0.0, 0.0, 1.0]), up_axis=up_axis)
        rotmats = [prev_rotmat for _ in range(tangents.shape[0])]

    return SciRotation.from_matrix(np.stack(rotmats, axis=0)).as_rotvec().astype(np.float32)


def _gamma_to_hab_rotation() -> np.ndarray:
    # GAMMA: x-right, y-forward, z-up. Habitat: x-right, y-up, z-forward(-Z).
    return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32)


def _adapt_body_pose_for_model_type(body_pose: np.ndarray, model_type: str) -> np.ndarray:
    body_pose = np.asarray(body_pose, dtype=np.float32)
    if body_pose.ndim != 2 or body_pose.shape[1] % 3 != 0:
        raise ValueError(f"Unexpected body_pose shape {body_pose.shape}.")

    joint_count = body_pose.shape[1] // 3
    target_joint_count = 21 if model_type == "smplx" else 23
    if joint_count == target_joint_count:
        return body_pose

    if model_type == "smpl" and joint_count == 21:
        # GAMMA uses a SMPL-X-style 21-joint body pose. SMPL expects two extra
        # wrist/hand joints in body_pose, so pad them with zeros.
        pad = np.zeros((body_pose.shape[0], 6), dtype=body_pose.dtype)
        return np.concatenate([body_pose, pad], axis=1)

    if model_type == "smplx" and joint_count == 23:
        return body_pose[:, : 21 * 3]

    raise ValueError(
        f"body_pose joint count {joint_count} is incompatible with {model_type.upper()}."
    )


def _compute_joint_mats_from_smpl_output(
    smpl_model,
    smplx_lbs,
    smpl_output,
    betas: torch.Tensor,
    model_type: str,
    expression: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    joint_mats = getattr(smpl_output, "A", None)
    if joint_mats is not None:
        return joint_mats

    full_pose = getattr(smpl_output, "full_pose", None)
    if full_pose is None:
        raise RuntimeError(f"{model_type.upper()} output missing both A and full_pose.")

    shape_components = betas
    shapedirs = smpl_model.shapedirs
    if model_type == "smplx":
        expr_dirs = getattr(smpl_model, "expr_dirs", None)
        if expression is not None and expr_dirs is not None and expression.shape[1] > 0:
            shape_components = torch.cat([betas, expression], dim=-1)
            shapedirs = torch.cat([smpl_model.shapedirs, expr_dirs], dim=-1)

    v_shaped = smpl_model.v_template + smplx_lbs.blend_shapes(shape_components, shapedirs)
    joints = smplx_lbs.vertices2joints(smpl_model.J_regressor, v_shaped)
    rot_mats = smplx_lbs.batch_rodrigues(full_pose.reshape(-1, 3)).view(
        full_pose.shape[0], -1, 3, 3
    )
    _, joint_mats = smplx_lbs.batch_rigid_transform(
        rot_mats, joints, smpl_model.parents, dtype=full_pose.dtype
    )
    return joint_mats


def _precompute_joint_mats(
    transl: np.ndarray,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    left_hand_pose: np.ndarray,
    right_hand_pose: np.ndarray,
    betas: np.ndarray,
    gender: str,
    smpl_model_path: str,
    smpl_type: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not os.path.exists(smpl_model_path):
        raise FileNotFoundError(f"SMPL model path not found: {smpl_model_path}")
    if batch_size <= 0:
        raise ValueError(f"joint_mats batch_size must be > 0, got {batch_size}")

    total_frames = int(transl.shape[0])
    if (
        global_orient.shape[0] != total_frames
        or body_pose.shape[0] != total_frames
        or left_hand_pose.shape[0] != total_frames
        or right_hand_pose.shape[0] != total_frames
    ):
        raise ValueError("SMPL driver arrays must share the same frame count.")

    body_pose = _adapt_body_pose_for_model_type(body_pose, smpl_type)

    _ensure_inspect_getargspec()
    _ensure_legacy_numpy_aliases()
    try:
        import smplx as smplx_module
        import smplx.lbs as smplx_lbs
    except Exception as exc:
        raise RuntimeError(
            "Failed to import smplx for joint_mats precompute. Install smplx."
        ) from exc

    smpl_type = _infer_smpl_model_type(smpl_model_path, smpl_type)
    model_batch_size = min(max(int(batch_size), 1), max(total_frames, 1))
    if smpl_type == "smplx":
        smpl_model = smplx_module.SMPLX(
            model_path=smpl_model_path,
            gender=gender,
            use_pca=False,
            num_pca_comps=45,
            flat_hand_mean=True,
            batch_size=model_batch_size,
        ).to(device)
    else:
        smpl_model = smplx_module.SMPL(
            model_path=smpl_model_path,
            gender=gender,
            batch_size=model_batch_size,
        ).to(device)
    smpl_model.eval()

    betas = np.asarray(betas, dtype=np.float32).reshape(1, -1)
    betas_base = torch.from_numpy(betas).to(device)
    num_betas = int(getattr(smpl_model, "num_betas", betas_base.shape[1]))
    if betas_base.shape[1] < num_betas:
        betas_base = torch.cat(
            [
                betas_base,
                torch.zeros((1, num_betas - betas_base.shape[1]), dtype=betas_base.dtype, device=device),
            ],
            dim=1,
        )
    elif betas_base.shape[1] > num_betas:
        betas_base = betas_base[:, :num_betas]
    expr_dim = 0
    expression_template = None
    if smpl_type == "smplx":
        expr_dim = int(
            getattr(smpl_model, "num_expression_coeffs", smpl_model.expression.shape[1])
        )
        expression_template = torch.zeros((1, expr_dim), dtype=torch.float32, device=device)
    output_chunks: List[np.ndarray] = []

    for start in range(0, total_frames, batch_size):
        end = min(start + batch_size, total_frames)
        bs = end - start
        transl_batch = torch.from_numpy(transl[start:end].astype(np.float32)).to(device)
        global_batch = torch.from_numpy(global_orient[start:end].astype(np.float32)).to(device)
        body_batch = torch.from_numpy(body_pose[start:end].astype(np.float32)).to(device)
        lhand_batch = torch.from_numpy(left_hand_pose[start:end].astype(np.float32)).to(device)
        rhand_batch = torch.from_numpy(right_hand_pose[start:end].astype(np.float32)).to(device)
        betas_batch = betas_base.repeat(bs, 1)
        expression_batch = None
        zeros_3 = None
        if smpl_type == "smplx":
            expression_batch = expression_template.repeat(bs, 1)
            zeros_3 = torch.zeros((bs, 3), dtype=torch.float32, device=device)

        prev_bs = getattr(smpl_model, "batch_size", None)
        if prev_bs is not None and prev_bs != bs:
            smpl_model.batch_size = bs

        with torch.no_grad():
            if smpl_type == "smplx":
                smpl_output = smpl_model.forward(
                    betas=betas_batch,
                    global_orient=global_batch,
                    transl=torch.zeros_like(transl_batch),
                    body_pose=body_batch,
                    left_hand_pose=lhand_batch,
                    right_hand_pose=rhand_batch,
                    expression=expression_batch,
                    jaw_pose=zeros_3,
                    leye_pose=zeros_3,
                    reye_pose=zeros_3,
                    return_full_pose=True,
                )
            else:
                smpl_output = smpl_model.forward(
                    betas=betas_batch,
                    global_orient=global_batch,
                    transl=torch.zeros_like(transl_batch),
                    body_pose=body_batch,
                    return_full_pose=True,
                )
            joint_mats = _compute_joint_mats_from_smpl_output(
                smpl_model,
                smplx_lbs,
                smpl_output,
                betas_batch,
                smpl_type,
                expression_batch,
            ).clone()
            joint_mats[:, :, :3, 3] += transl_batch.unsqueeze(1)

        if prev_bs is not None and prev_bs != bs:
            smpl_model.batch_size = prev_bs

        output_chunks.append(joint_mats.detach().cpu().numpy().astype(np.float32))

    if not output_chunks:
        raise RuntimeError("No joint mats were generated.")
    return np.concatenate(output_chunks, axis=0)


class GammaRunner:
    def __init__(
        self,
        gamma_root: str,
        cfg_policy: str,
        body_model_path: str,
        marker_path: str,
        device: torch.device,
        gpu_index: int,
        random_seed: int,
        use_policy: bool,
        use_policy_mean: bool,
        n_gens_root: int,
        n_gens_leaf: int,
        goal_thresh: float,
        reproj_factor: float,
    ) -> None:
        self.device = device
        self.gpu_index = gpu_index
        self.random_seed = random_seed
        self.use_policy = use_policy
        self.use_policy_mean = use_policy_mean
        self.n_gens_root = n_gens_root
        self.n_gens_leaf = n_gens_leaf
        self.goal_thresh = goal_thresh
        self.reproj_factor = reproj_factor

        if gamma_root not in sys.path:
            sys.path.append(gamma_root)

        from exp_GAMMAPrimitive.utils import config_env as gamma_env
        from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
        from human_body_prior.tools.model_loader import load_vposer
        from models.baseops import SMPLXParser
        from models.models_GAMMA_primitive import GAMMAPrimitiveComboGenOP
        from models.models_policy import GAMMAPolicy
        import models.baseops as baseops

        gamma_env.get_body_model_path = lambda: body_model_path
        gamma_env.get_body_marker_path = lambda: marker_path
        baseops.get_body_model_path = lambda: body_model_path
        baseops.get_body_marker_path = lambda: marker_path

        self.ConfigCreator = ConfigCreator
        self.GAMMAPrimitiveComboGenOP = GAMMAPrimitiveComboGenOP
        self.GAMMAPolicy = GAMMAPolicy
        self.SMPLXParser = SMPLXParser

        self.vposer, _ = load_vposer(os.path.join(body_model_path, "vposer_v1_0"), vp_model="snapshot")
        self.vposer.eval()
        self.vposer = self.vposer.to(self.device)

        self.cfg_policy = ConfigCreator(cfg_policy)
        self.cfg_1frame_male = self.cfg_policy.trainconfig["cfg_1frame_male"]
        self.cfg_2frame_male = self.cfg_policy.trainconfig["cfg_2frame_male"]
        self.cfg_1frame_female = self.cfg_policy.trainconfig["cfg_1frame_female"]
        self.cfg_2frame_female = self.cfg_policy.trainconfig["cfg_2frame_female"]
        self.body_repr = self.cfg_policy.modelconfig["body_repr"]

        self.genop_1frame_male = self._configure_model(self.cfg_1frame_male)
        self.genop_1frame_female = self._configure_model(self.cfg_1frame_female)
        self.genop_2frame_male = self._configure_model(self.cfg_2frame_male)
        self.genop_2frame_female = self._configure_model(self.cfg_2frame_female)

        self.policy_model = GAMMAPolicy(self.cfg_policy.modelconfig).to(self.device)
        self.policy_model.eval()
        self._load_policy_checkpoint(self.cfg_policy.trainconfig["save_dir"])

        pconfig_mp = {
            "n_batch": 10 * self.n_gens_root * self.n_gens_leaf,
            "device": self.device,
            "marker_placement": "ssm2_67",
        }
        self.smplxparser_mp = SMPLXParser(pconfig_mp)

        pconfig_1f_root = {
            "n_batch": self.n_gens_root,
            "device": self.device,
            "marker_placement": "ssm2_67",
        }
        self.smplxparser_1f_root = SMPLXParser(pconfig_1f_root)

        pconfig_mp_root = {
            "n_batch": 10 * self.n_gens_root,
            "device": self.device,
            "marker_placement": "ssm2_67",
        }
        self.smplxparser_mp_root = SMPLXParser(pconfig_mp_root)

    def _configure_model(self, cfg_name: str):
        cfgall = self.ConfigCreator(cfg_name)
        modelcfg = cfgall.modelconfig
        traincfg = cfgall.trainconfig
        predictorcfg = self.ConfigCreator(modelcfg["predictor_config"])
        regressorcfg = self.ConfigCreator(modelcfg["regressor_config"])

        testcfg = {
            "gpu_index": self.gpu_index,
            "ckpt_dir": traincfg["save_dir"],
            "result_dir": cfgall.cfg_result_dir,
            "seed": self.random_seed,
            "log_dir": cfgall.cfg_log_dir,
        }
        testop = self.GAMMAPrimitiveComboGenOP(predictorcfg, regressorcfg, testcfg)
        testop.build_model(load_pretrained_model=True)
        return testop

    def _load_policy_checkpoint(self, ckpt_dir: str) -> None:
        ckpt_list = sorted(glob.glob(os.path.join(ckpt_dir, "epoch-*.ckp")), key=os.path.getmtime)
        if not ckpt_list:
            print("[INFO] GAMMA policy checkpoint not found; using initial weights.")
            return
        ckpt_path = ckpt_list[-1]
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.policy_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[INFO] Loaded GAMMA policy checkpoint: {ckpt_path}")

    def _gen_motion_one_step(self, motion_model, states, bparam_seed, prev_betas, t_his: int):
        n_gens = self.n_gens_root if t_his == 1 else self.n_gens_leaf
        pred_markers, pred_params, act, _, _ = motion_model.generate_ppo(
            self.policy_model,
            states.permute([1, 0, 2]),
            bparam_seed.permute([1, 0, 2]),
            prev_betas,
            n_gens=n_gens,
            to_numpy=False,
            param_blending=True,
            use_policy_mean=self.use_policy_mean,
            use_policy=self.use_policy,
        )
        pred_markers = pred_markers.reshape(pred_markers.shape[0], pred_markers.shape[1], -1, 3)
        return pred_markers.permute([1, 0, 2, 3]), pred_params.permute([1, 0, 2]), act

    def _canonicalize_static_pose(self, data):
        smplx_transl = data["transl"]
        smplx_glorot = data["global_orient"]
        smplx_poses = data["body_pose"]
        gender = data["gender"]
        smplx_handposes = torch.zeros((smplx_transl.shape[0], 24), device=self.device)
        prev_params = torch.cat([smplx_transl, smplx_glorot, smplx_poses, smplx_handposes], dim=-1)
        if self.n_gens_root != 1:
            raise RuntimeError("GAMMA root expansion expects n_gens_root == 1.")
        prev_params = prev_params.repeat(self.n_gens_root, 1, 1)
        prev_betas = data["betas"]
        nb, nt = prev_params.shape[:2]
        body_param_seed = prev_params.reshape(nt * nb, -1)

        R0, T0 = self.smplxparser_1f_root.get_new_coordinate(
            betas=prev_betas, gender=gender, xb=prev_params[:, 0], to_numpy=False
        )
        body_param_seed = self.smplxparser_1f_root.update_transl_glorot(
            R0, T0, betas=prev_betas, gender=gender, xb=body_param_seed, to_numpy=False
        ).reshape(nb, nt, -1)

        return body_param_seed, prev_betas, gender, R0, T0

    def _get_wpath_feature(self, Y_l, pel, R0, T0, pt_wpath):
        nb, nt = pel.shape[:2]
        Y_l = Y_l.reshape(nb, nt, -1, 3)
        pt_wpath_l_3d = torch.einsum("bij,btj->bti", R0.permute(0, 2, 1), pt_wpath[None, ...] - T0)

        fea_wpathxy = pt_wpath_l_3d[:, :, :2] - pel[:, :, :2]
        dist_xy = torch.norm(fea_wpathxy, dim=-1, keepdim=True)
        fea_wpathxy = fea_wpathxy / dist_xy
        fea_wpathz = pt_wpath_l_3d[:, :, -1:] - pel[:, :, -1:]
        fea_wpath = torch.cat([fea_wpathxy, fea_wpathz], dim=-1)

        fea_marker = pt_wpath_l_3d[:, :, None, :] - Y_l
        dist_m_3d = torch.norm(fea_marker, dim=-1, keepdim=True)
        fea_marker_3d_n = (fea_marker / dist_m_3d).reshape(nb, nt, -1)

        dist_m_2d = torch.norm(fea_marker[:, :, :, :2], dim=-1, keepdim=True)
        fea_marker_xyn = fea_marker[:, :, :, :2] / dist_m_2d
        fea_marker_h = torch.cat([fea_marker_xyn, fea_marker[:, :, :, -1:]], dim=-1).reshape(
            nb, nt, -1
        )

        return dist_xy, fea_wpath, fea_marker_3d_n, fea_marker_h

    def _update_local_target(self, marker, pelvis_loc, idx_target_curr, R0, T0, wpath):
        dist, _, fea_marker, _ = self._get_wpath_feature(
            marker, pelvis_loc, R0, T0, wpath[idx_target_curr][None, ...]
        )
        while torch.any(dist < self.goal_thresh) and idx_target_curr < wpath.shape[0] - 1:
            idx_target_curr += 1
            dist, _, fea_marker, _ = self._get_wpath_feature(
                marker, pelvis_loc, R0, T0, wpath[idx_target_curr][None, ...]
            )
        if torch.any(dist < self.goal_thresh) and idx_target_curr == wpath.shape[0] - 1:
            return None
        return idx_target_curr, fea_marker

    def _get_cost_search(self, bparams, joints, Y_l, R0, T0, pt_wpath):
        R0 = R0.repeat(self.n_gens_leaf, 1, 1)
        T0 = T0.repeat(self.n_gens_leaf, 1, 1)
        pel_loc = joints[:, :, 0]
        nb, nt = Y_l.shape[:2]
        Y_w = torch.einsum("bij,btpj->btpi", R0, Y_l) + T0[:, None, ...]
        pel_w = torch.einsum("bij,btj->bti", R0, pel_loc) + T0
        Y_wz = Y_w[:, :, :, -1].reshape(nb, -1)
        h = 1 / 40
        Y_w_speed = torch.norm(Y_w[:, 2:] - Y_w[:, :-2], dim=-1) / (2 * h)
        Y_w_speed = Y_w_speed.reshape(nb, -1)
        dist2gp = (torch.abs(Y_wz.min(dim=-1)[0]) - 0.05).clamp(min=0)
        dist2skat = (torch.abs(Y_w_speed.min(dim=-1)[0]) - 0.075).clamp(min=0)

        pel_w_speed = torch.norm(pel_w[:, 2:, :2] - pel_w[:, :-2, :2], dim=-1) / (2 * h)
        dist2run = (pel_w_speed.mean(dim=-1) - 2.25).clamp(min=0)

        target_wpath_l = torch.einsum("bij,btj->bti", R0.permute(0, 2, 1), pt_wpath[None, ...] - T0)[:, :, :2]
        dist2target = torch.norm(target_wpath_l[:, 0] - pel_loc[:, -1, :2], dim=-1)

        joints_end = joints[:, -1]
        x_axis = joints_end[:, 2, :] - joints_end[:, 1, :]
        x_axis[:, -1] = 0
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)
        z_axis = torch.zeros_like(x_axis)
        z_axis[:, -1] = 1.0
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        b_ori = y_axis[:, :2]
        t_ori = target_wpath_l[:, 0] - pel_loc[:, -1, :2]
        t_ori = t_ori / torch.norm(t_ori, dim=-1, keepdim=True)
        dist2ori = -torch.einsum("bi,bi->b", t_ori, b_ori)

        body_pose = bparams[:, :, 6:69].reshape(nb, nt, -1, 3)
        pose_angle = torch.norm(body_pose, dim=-1)
        dist2pose = (pose_angle.mean(dim=(-1, -2)) - 0.65).clamp(min=0)

        torso_joint_idx = min(12, joints_end.shape[1] - 1)
        torso_vec = joints_end[:, torso_joint_idx, :] - joints_end[:, 0, :]
        torso_vec_w = torch.einsum("bij,bj->bi", R0, torso_vec)
        torso_upright = torso_vec_w[:, -1] / torch.norm(torso_vec_w, dim=-1).clamp(min=1.0e-6)
        dist2tilt = (0.75 - torso_upright).clamp(min=0)

        cost = (
            dist2gp
            + dist2skat
            + 0.05 * dist2target
            + 0.1 * dist2ori
            + 0.5 * dist2run
            + 0.35 * dist2pose
            + 2.0 * dist2tilt
        )
        return cost

    def _gen_tree_roots(self, data_mp, wpath, idx_target_curr):
        body_param_seed, prev_betas, gender, R0, T0 = data_mp
        nb, nt = body_param_seed.shape[:2]
        t_his = 1

        motion_model = self.genop_1frame_male if gender == "male" else self.genop_1frame_female

        marker_seed = self.smplxparser_1f_root.get_markers(
            betas=prev_betas, gender=gender, xb=body_param_seed.reshape(nb * nt, -1), to_numpy=False
        ).reshape(nb, nt, -1)

        pelvis_loc = self.smplxparser_1f_root.get_jts(
            betas=prev_betas, gender=gender, xb=body_param_seed.reshape(nb * nt, -1), to_numpy=False
        )[:, 0].reshape(nb, nt, -1)

        wpath_feature = self._update_local_target(marker_seed, pelvis_loc[:, :t_his], idx_target_curr, R0, T0, wpath)
        if wpath_feature is None:
            return None
        idx_target_curr, fea_marker = wpath_feature

        if self.body_repr != "ssm2_67_condi_marker":
            raise RuntimeError(f"Unsupported body_repr: {self.body_repr}")

        states = torch.cat([marker_seed, fea_marker], dim=-1)
        pred_markers, pred_params, _ = self._gen_motion_one_step(motion_model, states, body_param_seed, prev_betas, t_his)

        nb, nt = pred_params.shape[:2]
        pred_joints = self.smplxparser_mp_root.get_jts(
            betas=prev_betas, gender=gender, xb=pred_params.reshape(nb * nt, -1), to_numpy=False
        ).reshape(nb, nt, -1, 3)
        pred_pelvis_loc = pred_joints[:, :, 0]
        pred_markers_proj = self.smplxparser_mp_root.get_markers(
            betas=prev_betas, gender=gender, xb=pred_params.reshape(nb * nt, -1), to_numpy=False
        ).reshape(nb, nt, -1, 3)

        pred_marker_b = self.reproj_factor * pred_markers_proj + (1 - self.reproj_factor) * pred_markers
        return [pred_marker_b, pred_params, prev_betas, gender, R0, T0, pred_pelvis_loc, "start-frame"], idx_target_curr

    def _expand_tree(self, data_mp, wpath, idx_target_curr):
        prev_markers_b, prev_params, prev_betas, prev_gender, prev_rotmat, prev_transl, prev_pelvis_loc, _ = data_mp
        t_his = 2
        body_param_seed = prev_params[:, -t_his:]
        nb, nt = body_param_seed.shape[:2]

        R_, T_ = self.smplxparser_1f_root.get_new_coordinate(
            betas=prev_betas, gender=prev_gender, xb=body_param_seed[:, 0], to_numpy=False
        )
        T0 = torch.einsum("bij,btj->bti", prev_rotmat, T_) + prev_transl
        R0 = torch.einsum("bij,bjk->bik", prev_rotmat, R_)

        marker_seed = torch.einsum(
            "bij,btpj->btpi", R_.permute(0, 2, 1), prev_markers_b[:, -t_his:] - T_[..., None, :]
        )
        pel_loc_seed = torch.einsum(
            "bij,btj->bti", R_.permute(0, 2, 1), prev_pelvis_loc[:, -t_his:] - T_
        )
        wpath_feature = self._update_local_target(
            marker_seed, pel_loc_seed, idx_target_curr, R0, T0, wpath
        )
        if wpath_feature is None:
            return None
        idx_target_curr, fea_marker = wpath_feature

        if self.body_repr != "ssm2_67_condi_marker":
            raise RuntimeError(f"Unsupported body_repr: {self.body_repr}")

        marker_seed = marker_seed.reshape(nb, t_his, -1)
        states = torch.cat([marker_seed, fea_marker], dim=-1)
        motion_model = self.genop_2frame_male if prev_gender == "male" else self.genop_2frame_female
        pred_markers, pred_params, _ = self._gen_motion_one_step(
            motion_model, states, body_param_seed, prev_betas, t_his
        )
        nb, nt = pred_params.shape[:2]
        pred_joints = self.smplxparser_mp.get_jts(
            betas=prev_betas, gender=prev_gender, xb=pred_params.reshape(nb * nt, -1), to_numpy=False
        ).reshape(nb, nt, -1, 3)
        pred_pelvis_loc = pred_joints[:, :, 0]
        pred_markers_proj = self.smplxparser_mp.get_markers(
            betas=prev_betas, gender=prev_gender, xb=pred_params.reshape(nb * nt, -1), to_numpy=False
        ).reshape(nb, nt, -1, 3)

        pred_marker_b = self.reproj_factor * pred_markers_proj + (1 - self.reproj_factor) * pred_markers
        traj_cost = self._get_cost_search(pred_params, pred_joints, pred_marker_b, R0, T0, wpath[idx_target_curr])
        rank_idx = torch.topk(traj_cost, k=self.n_gens_root, dim=0, largest=False, sorted=True)[1]
        pred_marker_b = pred_marker_b[rank_idx]
        pred_params = pred_params[rank_idx]
        pred_pelvis_loc = pred_pelvis_loc[rank_idx]

        return [pred_marker_b, pred_params, prev_betas, prev_gender, R0, T0, pred_pelvis_loc, "2-frame"], idx_target_curr

    def rollout(self, body_s, max_depth: int):
        outmps = []
        data_mp0 = self._canonicalize_static_pose(body_s)
        wpath = body_s["wpath"]
        idx_target_curr = 1

        rootdata = self._gen_tree_roots(data_mp0, wpath, idx_target_curr)
        if rootdata is None:
            return []
        data_mp, idx_target_curr = rootdata
        outmps.append(data_mp)

        for _ in range(1, max_depth):
            leafdata = self._expand_tree(data_mp, wpath, idx_target_curr)
            if leafdata is None:
                break
            data_mp, idx_target_curr = leafdata
            outmps.append(data_mp)

        return outmps

    def _calc_delta_t(self, betas: np.ndarray, body_pose: np.ndarray, gender: str) -> np.ndarray:
        bm = self.smplxparser_mp.bm_male if gender == "male" else self.smplxparser_mp.bm_female

        body_pose_t = torch.as_tensor(body_pose, dtype=torch.float32)
        if body_pose_t.ndim == 1:
            body_pose_t = body_pose_t.view(1, -1)
        if body_pose_t.device != self.device:
            body_pose_t = body_pose_t.to(self.device)
        n_batches = body_pose_t.shape[0]

        betas_t = torch.as_tensor(betas, dtype=torch.float32)
        if betas_t.ndim == 1:
            betas_t = betas_t.view(1, -1)
        if betas_t.device != self.device:
            betas_t = betas_t.to(self.device)
        betas_t = betas_t.repeat(n_batches, 1)

        left_hand_dim = bm.left_hand_pose.shape[1]
        right_hand_dim = bm.right_hand_pose.shape[1]
        expr_dim = getattr(bm, "num_expression_coeffs", bm.expression.shape[1])

        bodyconfig = {
            "body_pose": body_pose_t,
            "betas": betas_t,
            "transl": torch.zeros((n_batches, 3), dtype=torch.float32, device=self.device),
            "global_orient": torch.zeros((n_batches, 3), dtype=torch.float32, device=self.device),
            "left_hand_pose": torch.zeros((n_batches, left_hand_dim), device=self.device),
            "right_hand_pose": torch.zeros((n_batches, right_hand_dim), device=self.device),
            "expression": torch.zeros((n_batches, expr_dim), device=self.device),
            "jaw_pose": torch.zeros((n_batches, 3), device=self.device),
            "leye_pose": torch.zeros((n_batches, 3), device=self.device),
            "reye_pose": torch.zeros((n_batches, 3), device=self.device),
        }

        prev_bs = getattr(bm, "batch_size", None)
        if prev_bs is not None and prev_bs != n_batches:
            bm.batch_size = n_batches
        with torch.no_grad():
            smplx_out = bm(return_verts=True, **bodyconfig)
        if prev_bs is not None and prev_bs != n_batches:
            bm.batch_size = prev_bs

        delta_t = smplx_out.joints[:, 0, :]
        return delta_t.detach().cpu().numpy()

    def _to_world_params(
        self,
        params: np.ndarray,
        transf_rotmat: np.ndarray,
        transf_transl: np.ndarray,
        betas: np.ndarray,
        gender: str,
    ) -> np.ndarray:
        transl = params[:, :3]
        rotvec = params[:, 3:6]
        body_pose = params[:, 6:69]
        delta_t = self._calc_delta_t(betas, body_pose, gender)

        transl_world = (transf_rotmat @ (transl + delta_t).T).T + transf_transl - delta_t
        rot_world = _apply_rotation_to_rotvecs(transf_rotmat, rotvec)

        out = params.copy()
        out[:, :3] = transl_world
        out[:, 3:6] = rot_world
        return out

    def primitives_to_dense_params(self, primitives) -> Tuple[np.ndarray, str, np.ndarray]:
        frames: List[np.ndarray] = []
        gender = primitives[0][3]
        betas = primitives[0][2]
        if isinstance(betas, torch.Tensor):
            betas = betas.detach().cpu().numpy()
        betas = np.array(betas).reshape(-1)

        for idx, mp in enumerate(primitives):
            params = mp[1]
            if isinstance(params, torch.Tensor):
                params = params.detach().cpu().numpy()
            if params.ndim == 3:
                params = params[0]

            R0 = mp[4]
            T0 = mp[5]
            if isinstance(R0, torch.Tensor):
                R0 = R0.detach().cpu().numpy()
            if isinstance(T0, torch.Tensor):
                T0 = T0.detach().cpu().numpy()
            if R0.ndim == 3:
                R0 = R0[0]
            if T0.ndim == 3:
                T0 = T0[0, 0]

            mp_type = mp[7]
            seed_len = 1 if mp_type in ("start-frame", "1-frame") else 2
            if idx > 0:
                params = params[seed_len:]

            params = self._to_world_params(params, R0, T0, betas, gender)
            frames.append(params)

        dense = np.concatenate(frames, axis=0)
        return dense, gender, betas


def _make_body_seed(
    gamma: GammaRunner,
    wpath: np.ndarray,
    gender: str,
    betas: Optional[np.ndarray],
    snap_to_ground: bool,
    randomize_start_pose: bool,
    start_pose_lookahead: float,
) -> dict:
    if betas is None:
        betas = np.random.randn(1, 10).astype(np.float32)
    else:
        betas = np.array(betas, dtype=np.float32).reshape(1, 10)

    transl = wpath[:1].astype(np.float32)
    wpath_cumulative = _cumulative_path_lengths(wpath, up_axis=2)
    lookahead_idx = 1
    min_lookahead = max(float(start_pose_lookahead), 1.0e-3)
    for idx in range(1, wpath.shape[0]):
        if float(wpath_cumulative[idx]) >= min_lookahead:
            lookahead_idx = idx
            break
    forward = wpath[lookahead_idx] - wpath[0]
    rotmat = _rotation_matrix_from_forward(forward, up_axis=2).astype(np.float32)
    global_orient = SciRotation.from_matrix(rotmat).as_rotvec().astype(np.float32)

    try:
        vposer_device = next(gamma.vposer.parameters()).device
    except StopIteration:
        vposer_device = gamma.device

    pose_latent = torch.zeros((1, 32), device=vposer_device)
    if randomize_start_pose:
        pose_latent = torch.randn((1, 32), device=vposer_device)
    with torch.no_grad():
        body_pose = gamma.vposer.decode(pose_latent, output_type="aa").view(1, -1)
    body_pose = body_pose.detach().cpu().numpy().astype(np.float32)

    if snap_to_ground:
        bm = gamma.smplxparser_1f_root.bm_male if gender == "male" else gamma.smplxparser_1f_root.bm_female
        expr_dim = getattr(bm, "num_expression_coeffs", bm.expression.shape[1])
        bparam = {
            "transl": torch.from_numpy(transl).to(gamma.device),
            "global_orient": torch.from_numpy(global_orient[None, ...]).to(gamma.device),
            "betas": torch.from_numpy(betas).to(gamma.device),
            "body_pose": torch.from_numpy(body_pose).to(gamma.device),
            "left_hand_pose": torch.zeros((1, 12), device=gamma.device),
            "right_hand_pose": torch.zeros((1, 12), device=gamma.device),
            "expression": torch.zeros((1, expr_dim), device=gamma.device),
            "jaw_pose": torch.zeros((1, 3), device=gamma.device),
            "leye_pose": torch.zeros((1, 3), device=gamma.device),
            "reye_pose": torch.zeros((1, 3), device=gamma.device),
        }
        batch_size = int(getattr(bm, "batch_size", bparam["transl"].shape[0]))
        if batch_size != bparam["transl"].shape[0]:
            # Match SMPLX internal batch size to avoid landmark concatenation errors.
            bparam = {key: _expand_batch(val, batch_size) for key, val in bparam.items()}
        with torch.no_grad():
            verts = bm(return_verts=True, **bparam).vertices[0].detach().cpu().numpy()
        delta_z = float(np.min(verts[:, -1]) - transl[0, -1])
        transl[0, -1] -= delta_z

    out = {
        "transl": torch.from_numpy(transl).to(gamma.device),
        "global_orient": torch.from_numpy(global_orient[None, ...]).to(gamma.device),
        "body_pose": torch.from_numpy(body_pose).to(gamma.device),
        "betas": torch.from_numpy(betas).to(gamma.device),
        "gender": gender,
        "wpath": torch.from_numpy(wpath).to(gamma.device),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate coupled SMPL driver (.pkl) using GAMMA and a navmesh path."
    )
    parser.add_argument("--navmesh", required=True, help="Path to .navmesh file")
    parser.add_argument("--output", required=True, help="Output .pkl path")
    parser.add_argument(
        "--start",
        type=float,
        nargs=3,
        default=None,
        help="Start XYZ in Habitat coords.",
    )
    parser.add_argument(
        "--end",
        type=float,
        nargs=3,
        default=None,
        help="End XYZ in Habitat coords.",
    )
    parser.add_argument(
        "--via",
        type=float,
        nargs=3,
        action="append",
        default=[],
        help="Optional via point XYZ; repeat flag for multiple points.",
    )
    parser.add_argument(
        "--path-length",
        type=float,
        default=None,
        help=(
            "Auto-sample a navmesh path whose total length exceeds this value. "
            "When set, do not pass --start/--end/--via."
        ),
    )
    parser.add_argument(
        "--path-ref",
        type=float,
        nargs=3,
        default=None,
        help=(
            "Optional navmesh reference point for --path-length mode. "
            "Sampling will prefer trajectories around this point."
        ),
    )
    parser.add_argument(
        "--length",
        type=int,
        default=100,
        help="Max GAMMA depth; each primitive is 0.25s (default: 100).",
    )
    parser.add_argument("--gamma-root", default="", help="Path to GAMMA-release root")
    parser.add_argument("--gamma-cfg-policy", default="MPVAEPolicy_v0", help="GAMMA policy config name")
    parser.add_argument("--body-model-path", default="", help="Path to GAMMA body model (VPoser)")
    parser.add_argument("--marker-path", default="", help="Path to GAMMA marker set")
    parser.add_argument(
        "--smpl-model-path",
        default="",
        help="Path to SMPL or SMPL-X model directory used for joint_mats precompute.",
    )
    parser.add_argument(
        "--smpl-type",
        choices=["auto", "smpl", "smplx"],
        default="auto",
        help="Type of assets under --smpl-model-path (default: auto-detect).",
    )
    parser.add_argument(
        "--joint-mats-batch-size",
        type=int,
        default=256,
        help="Batch size for offline joint_mats precompute (default: 256).",
    )
    parser.add_argument(
        "--include-proxy",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to generate proxy URDF and precompute proxy capsules (default: true).",
    )
    parser.add_argument(
        "--proxy-urdf-output",
        action="store_true",
        default=False,
        help="If set, write generated proxy URDF to <output>_proxy.urdf.",
    )
    parser.add_argument("--gender", choices=["male", "female"], default="male")
    parser.add_argument("--betas", type=float, nargs=10, default=None, help="Optional 10 betas")
    parser.add_argument("--fps", type=float, default=40.0, help="Output FPS (default: 40)")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index, set <0 for CPU")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--no-use-policy", action="store_true", default=False)
    parser.add_argument("--use-policy-mean", action="store_true", default=False)
    parser.add_argument("--n-gens-root", type=int, default=1)
    parser.add_argument("--n-gens-leaf", type=int, default=32)
    parser.add_argument("--goal-thresh", type=float, default=0.4)
    parser.add_argument("--reproj-factor", type=float, default=0.5)
    parser.add_argument("--no-snap-ground", action="store_true", default=False)
    parser.add_argument(
        "--path-segment-length",
        type=float,
        default=0.9,
        help="Max segment length of the shortest-path polyline fed to GAMMA (default: 0.9m).",
    )
    parser.add_argument(
        "--heading-lookahead",
        type=float,
        default=0.6,
        help="Lookahead distance along the navmesh path used for root heading alignment (default: 0.6m).",
    )
    parser.add_argument(
        "--start-pose-lookahead",
        type=float,
        default=1.0,
        help="Lookahead distance used to initialize the avatar facing direction (default: 1.0m).",
    )
    parser.add_argument(
        "--randomize-start-pose",
        action="store_true",
        default=False,
        help="Use a random VPoser latent for the start pose instead of a neutral pose.",
    )
    parser.add_argument(
        "--smooth-spikes",
        action="store_true",
        default=False,
        help="Enable spike smoothing for single-frame pose outliers.",
    )
    parser.add_argument(
        "--no-coord-convert",
        action="store_true",
        default=False,
        help="Skip GAMMA(Z-up Y-forward) -> Habitat(Y-up -Z-forward) rotation.",
    )

    args = parser.parse_args()

    orig_cwd = os.getcwd()
    navmesh_path = _resolve_path(args.navmesh, orig_cwd)
    output_path = _resolve_path(args.output, orig_cwd)

    if not os.path.exists(navmesh_path):
        print(f"Error: navmesh not found: {navmesh_path}")
        return 1

    gamma_root = args.gamma_root
    if not gamma_root:
        gamma_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "GAMMA-release"))
    gamma_root = _resolve_path(gamma_root, orig_cwd)
    if not os.path.exists(gamma_root):
        print(f"Error: GAMMA root not found: {gamma_root}")
        return 1

    body_model_path = args.body_model_path or os.path.join(
        gamma_root, "exp_GAMMAPrimitive", "data", "VPoser"
    )
    body_model_path = _resolve_path(body_model_path, orig_cwd)
    marker_path = args.marker_path or os.path.join(
        gamma_root, "exp_GAMMAPrimitive", "data", "Mosh"
    )
    marker_path = _resolve_path(marker_path, orig_cwd)
    if not os.path.exists(body_model_path):
        print(f"Error: body model path not found: {body_model_path}")
        return 1
    if not os.path.exists(marker_path):
        print(f"Error: marker path not found: {marker_path}")
        return 1
    if not args.smpl_model_path:
        print("Error: --smpl-model-path is required for joint_mats precompute.")
        return 1
    smpl_model_path = _resolve_path(args.smpl_model_path, orig_cwd)
    if not os.path.exists(smpl_model_path):
        print(f"Error: smpl model path not found: {smpl_model_path}")
        return 1
    try:
        smpl_type = _infer_smpl_model_type(smpl_model_path, args.smpl_type)
    except Exception as exc:
        print(f"Error: failed to resolve SMPL model type: {exc}")
        return 1
    print(f"[INFO] Using {smpl_type.upper()} assets for joint_mats/proxy precompute.")

    explicit_path_mode = args.path_length is None
    if explicit_path_mode:
        if args.start is None or args.end is None:
            print("Error: explicit path mode requires both --start and --end.")
            return 1
        if args.path_ref is not None:
            print("Error: --path-ref is only supported together with --path-length.")
            return 1
        start = _parse_vec3(args.start)
        end = _parse_vec3(args.end)
        via = [_parse_vec3(v) for v in args.via]
    else:
        if args.path_length <= 0.0:
            print(f"Error: --path-length must be > 0, got {args.path_length}.")
            return 1
        if args.start is not None or args.end is not None or args.via:
            print("Error: --path-length mode does not accept --start/--end/--via.")
            return 1
        start = None
        end = None
        via = []
        path_ref = _parse_vec3(args.path_ref) if args.path_ref is not None else None

    pathfinder = PathFinder()
    if not pathfinder.load_nav_mesh(navmesh_path):
        print(f"Error: failed to load navmesh {navmesh_path}")
        return 1
    try:
        pathfinder.seed(int(args.random_seed))
    except Exception:
        pass

    if explicit_path_mode:
        path_points = _build_navmesh_path(pathfinder, start, via, end)
        print(
            "[INFO] Planned navmesh path from explicit waypoints: "
            f"polyline_points={path_points.shape[0]}"
        )
    else:
        try:
            path_points, auto_meta = _build_auto_navmesh_path(
                pathfinder,
                target_length=float(args.path_length),
                reference_point=path_ref,
                up_axis=1,
            )
        except Exception as exc:
            print(f"Error: failed to auto-sample navmesh path: {exc}")
            return 1
        ref_msg = ""
        if auto_meta.get("reference_point") is not None:
            ref_pt = np.asarray(auto_meta["reference_point"], dtype=np.float32)
            ref_msg = (
                " "
                f"reference_point=[{ref_pt[0]:.3f}, {ref_pt[1]:.3f}, {ref_pt[2]:.3f}],"
                f" reference_radius={float(auto_meta['reference_radius']):.3f}m,"
            )
        print(
            "[INFO] Auto-sampled navmesh path: "
            f"island={int(auto_meta['island_index'])},"
            f"{ref_msg}"
            f" start_clearance={float(auto_meta['start_point_clearance']):.3f}m,"
            f" segments={int(auto_meta['segment_count'])},"
            f" length={float(auto_meta['path_length']):.3f}m"
        )
    motion_path_points = _subdivide_path(path_points, max_segment_length=float(args.path_segment_length))
    path_total_length = float(_cumulative_path_lengths(motion_path_points, up_axis=1)[-1])
    print(
        "[INFO] Planned navmesh path: "
        f"polyline_points={path_points.shape[0]}, "
        f"motion_points={motion_path_points.shape[0]}, "
        f"length={path_total_length:.3f}m"
    )

    rot_gamma_to_hab = _gamma_to_hab_rotation()
    rot_hab_to_gamma = rot_gamma_to_hab.T
    if args.no_coord_convert:
        rot_gamma_to_hab = np.eye(3, dtype=np.float32)
        rot_hab_to_gamma = np.eye(3, dtype=np.float32)

    wpath_gamma = _transform_points(rot_hab_to_gamma, motion_path_points)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.gpu_index >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
        device = torch.device("cuda", index=args.gpu_index)
    else:
        device = torch.device("cpu")
    torch.set_grad_enabled(False)

    with _pushd(gamma_root):
        gamma = GammaRunner(
            gamma_root=gamma_root,
            cfg_policy=args.gamma_cfg_policy,
            body_model_path=body_model_path,
            marker_path=marker_path,
            device=device,
            gpu_index=args.gpu_index,
            random_seed=args.random_seed,
            use_policy=not args.no_use_policy,
            use_policy_mean=args.use_policy_mean,
            n_gens_root=args.n_gens_root,
            n_gens_leaf=args.n_gens_leaf,
            goal_thresh=args.goal_thresh,
            reproj_factor=args.reproj_factor,
        )

        body_seed = _make_body_seed(
            gamma=gamma,
            wpath=wpath_gamma,
            gender=args.gender,
            betas=args.betas,
            snap_to_ground=not args.no_snap_ground,
            randomize_start_pose=bool(args.randomize_start_pose),
            start_pose_lookahead=float(args.start_pose_lookahead),
        )

        primitives = gamma.rollout(body_seed, max_depth=args.length)
        if not primitives:
            print("Error: GAMMA failed to generate any motion primitives.")
            return 1

        dense_params, gender, betas = gamma.primitives_to_dense_params(primitives)
    global_orient_hab = _apply_rotation_to_rotvecs(rot_gamma_to_hab, dense_params[:, 3:6])
    transl_hab = _transform_points(rot_gamma_to_hab, dense_params[:, :3])
    body_pose = dense_params[:, 6:69]
    hand_zeros = np.zeros((dense_params.shape[0], 45), dtype=np.float32)

    if args.smooth_spikes:
        pose_rotvecs = np.concatenate(
            [global_orient_hab[:, None, :], body_pose.reshape(body_pose.shape[0], -1, 3)],
            axis=1,
        )
        transl_hab, pose_rotvecs, spikes = _smooth_pose_spikes(transl_hab, pose_rotvecs)
        if spikes:
            print(f"[INFO] Smoothed {spikes} pose spike frame(s).")
        else:
            print("[INFO] No pose spikes detected.")
        body_joints = body_pose.shape[1] // 3
        global_orient_hab = pose_rotvecs[:, 0]
        body_pose = pose_rotvecs[:, 1 : 1 + body_joints].reshape(body_pose.shape)

    transl_hab, path_tangents_hab, generated_distance, planned_distance, distance_scale = _fit_translations_to_path(
        transl_hab,
        motion_path_points,
        up_axis=1,
        heading_lookahead=float(args.heading_lookahead),
    )
    global_orient_hab = _rotvecs_from_path_tangents(path_tangents_hab, up_axis=1)
    body_pose_model = _adapt_body_pose_for_model_type(body_pose, smpl_type)
    print(
        "[INFO] Path-constrained root fit applied: "
        f"generated_distance={generated_distance:.3f}m, "
        f"planned_distance={planned_distance:.3f}m, "
        f"distance_scale={distance_scale:.3f}, "
        f"total_frames={transl_hab.shape[0]}"
    )

    try:
        joint_mats = _precompute_joint_mats(
            transl=transl_hab.astype(np.float32),
            global_orient=global_orient_hab.astype(np.float32),
            body_pose=body_pose_model.astype(np.float32),
            left_hand_pose=hand_zeros,
            right_hand_pose=hand_zeros,
            betas=betas.astype(np.float32),
            gender=gender,
            smpl_model_path=smpl_model_path,
            smpl_type=smpl_type,
            device=device,
            batch_size=int(args.joint_mats_batch_size),
        )
    except Exception as exc:
        print(f"Error: failed to precompute joint_mats: {exc}")
        return 1

    proxy_capsules = None
    proxy_urdf_path = ""
    if args.include_proxy:
        base, _ = os.path.splitext(output_path)
        requested_proxy_urdf_path = f"{base}_proxy.urdf"
        persist_proxy_urdf = bool(args.proxy_urdf_output)
        proxy_urdf_compute_path = ""
        remove_temp_proxy_urdf = False
        if persist_proxy_urdf:
            proxy_urdf_path = requested_proxy_urdf_path
            proxy_dir = os.path.dirname(proxy_urdf_path)
            if proxy_dir:
                os.makedirs(proxy_dir, exist_ok=True)
            proxy_urdf_compute_path = proxy_urdf_path
        else:
            temp_dir = os.path.dirname(output_path) or orig_cwd
            os.makedirs(temp_dir, exist_ok=True)
            fd, proxy_urdf_compute_path = tempfile.mkstemp(
                prefix="proxy_capsules_",
                suffix=".urdf",
                dir=temp_dir,
            )
            os.close(fd)
            remove_temp_proxy_urdf = True

        try:
            proxy_model_type = smpl_type
            proxy_model = _build_proxy_model(smpl_model_path, proxy_model_type, gender)
            proxy_joints = _load_proxy_rest_joints(proxy_model, proxy_model_type, betas.astype(np.float32))
            proxy_parents = proxy_model.parents.detach().cpu().numpy()
            proxy_joint_names = _resolve_proxy_joint_names(
                proxy_model_type, int(proxy_joints.shape[0])
            )
            proxy_robot = _build_proxy_urdf(proxy_joints, proxy_parents, proxy_joint_names)
            proxy_tree = ET.ElementTree(proxy_robot)
            ET.indent(proxy_tree, space="  ", level=0)
            proxy_tree.write(
                proxy_urdf_compute_path, encoding="utf-8", xml_declaration=True
            )

            proxy_capsules = _precompute_proxy_capsules(
                urdf_path=proxy_urdf_compute_path,
                model_type=proxy_model_type,
                root_rest_joint=proxy_joints[0],
                transl=transl_hab.astype(np.float32),
                global_orient=global_orient_hab.astype(np.float32),
                body_pose=body_pose_model.astype(np.float32),
                left_hand_pose=hand_zeros.astype(np.float32),
                right_hand_pose=hand_zeros.astype(np.float32),
            )
            if persist_proxy_urdf:
                print(
                    f"[INFO] Generated proxy URDF: {proxy_urdf_path} "
                    f"(capsules={proxy_capsules.shape[1]}, frames={proxy_capsules.shape[0]})"
                )
        except Exception as exc:
            print(f"Error: failed to generate/precompute proxy capsules: {exc}")
            return 1
        finally:
            if remove_temp_proxy_urdf and proxy_urdf_compute_path:
                try:
                    os.remove(proxy_urdf_compute_path)
                except OSError:
                    pass

    output = {
        "transl": transl_hab.astype(np.float32),
        "global_orient": global_orient_hab.astype(np.float32),
        "body_pose": body_pose_model.astype(np.float32),
        "betas": betas.astype(np.float32),
        "gender": gender,
        "smpl_type": smpl_type,
        "fps": float(args.fps),
        "navmesh_path": motion_path_points.astype(np.float32),
        "navmesh_path_space": "habitat",
        "joint_mats": joint_mats.astype(np.float32),
        "joint_mats_space": "smpl_with_trans",
        "joint_mats_version": 1,
        "joint_mats_fps": float(args.fps),
        "joint_mats_joint_count": int(joint_mats.shape[1]),
    }
    if smpl_type == "smplx":
        output["left_hand_pose"] = hand_zeros
        output["right_hand_pose"] = hand_zeros
    if proxy_capsules is not None:
        output["proxy_capsules"] = proxy_capsules.astype(np.float32)
        output["proxy_capsules_version"] = 1
        output["proxy_capsules_fps"] = float(args.fps)
        output["proxy_capsules_count"] = int(proxy_capsules.shape[1])
        if proxy_urdf_path:
            output["proxy_urdf_path"] = proxy_urdf_path

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"Saved coupled driver with {dense_params.shape[0]} frames to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
