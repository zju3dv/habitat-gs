#!/usr/bin/env python3
"""
Create a minimal driver.pkl for testing avatar rendering.
This generates a procedural SMPL-X walking motion without requiring GAMMA.
For full motion trajectories, use tools_gs/generate_trajectory.py with GAMMA.
"""

from pathlib import Path
import pickle

import numpy as np
import torch


def _ensure_legacy_numpy_aliases() -> None:
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


def set_joint(body_pose: np.ndarray, joint_id: int, rotvec: np.ndarray) -> None:
    """Set SMPL-X body joint rotvec. joint_id is the full skeleton id, 1..21."""
    start = (joint_id - 1) * 3
    body_pose[:, start : start + 3] = rotvec


def make_walk_pose(num_frames: int, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(num_frames, dtype=np.float32) / np.float32(fps)
    phase = 2.0 * np.pi * 1.55 * t
    s = np.sin(phase)
    c = np.cos(phase)

    transl = np.zeros((num_frames, 3), dtype=np.float32)
    transl[:, 0] = np.linspace(0.0, 3.2, num_frames, dtype=np.float32)
    transl[:, 1] = 0.035 * (1.0 - np.cos(phase * 2.0)).astype(np.float32)

    global_orient = np.zeros((num_frames, 3), dtype=np.float32)
    global_orient[:, 1] = -0.5 * np.pi  # Face the +X walking direction.
    global_orient[:, 2] = 0.035 * s  # Subtle pelvis sway.

    body_pose = np.zeros((num_frames, 63), dtype=np.float32)

    left_hip = np.stack([0.42 * s, 0.04 * c, 0.05 * s], axis=1).astype(np.float32)
    right_hip = np.stack([-0.42 * s, -0.04 * c, -0.05 * s], axis=1).astype(np.float32)
    left_knee = np.stack([0.18 + 0.62 * np.maximum(0.0, -s), np.zeros_like(s), np.zeros_like(s)], axis=1).astype(np.float32)
    right_knee = np.stack([0.18 + 0.62 * np.maximum(0.0, s), np.zeros_like(s), np.zeros_like(s)], axis=1).astype(np.float32)
    left_ankle = np.stack([-0.20 * np.maximum(0.0, -s) + 0.08 * c, np.zeros_like(s), np.zeros_like(s)], axis=1).astype(np.float32)
    right_ankle = np.stack([-0.20 * np.maximum(0.0, s) - 0.08 * c, np.zeros_like(s), np.zeros_like(s)], axis=1).astype(np.float32)
    left_foot = np.stack([0.10 * c, np.zeros_like(s), np.zeros_like(s)], axis=1).astype(np.float32)
    right_foot = np.stack([-0.10 * c, np.zeros_like(s), np.zeros_like(s)], axis=1).astype(np.float32)

    spine1 = np.stack([0.04 * s, np.zeros_like(s), -0.05 * s], axis=1).astype(np.float32)
    spine2 = np.stack([-0.03 * s, np.zeros_like(s), -0.04 * s], axis=1).astype(np.float32)
    spine3 = np.stack([-0.02 * s, np.zeros_like(s), -0.03 * s], axis=1).astype(np.float32)
    neck = np.stack([np.zeros_like(s), np.zeros_like(s), 0.02 * s], axis=1).astype(np.float32)

    left_shoulder = np.stack(
        [-0.30 * s - 0.05, np.zeros_like(s), -1.10 + 0.08 * c], axis=1
    ).astype(np.float32)
    right_shoulder = np.stack(
        [0.30 * s - 0.05, np.zeros_like(s), 1.10 - 0.08 * c], axis=1
    ).astype(np.float32)
    left_elbow = np.stack(
        [-0.28 - 0.14 * np.maximum(0.0, s), np.zeros_like(s), np.zeros_like(s)],
        axis=1,
    ).astype(np.float32)
    right_elbow = np.stack(
        [-0.28 - 0.14 * np.maximum(0.0, -s), np.zeros_like(s), np.zeros_like(s)],
        axis=1,
    ).astype(np.float32)

    # SMPL-X body joints: 1 L hip, 2 R hip, 3 spine1, 4 L knee, ...
    set_joint(body_pose, 1, left_hip)
    set_joint(body_pose, 2, right_hip)
    set_joint(body_pose, 3, spine1)
    set_joint(body_pose, 4, left_knee)
    set_joint(body_pose, 5, right_knee)
    set_joint(body_pose, 6, spine2)
    set_joint(body_pose, 7, left_ankle)
    set_joint(body_pose, 8, right_ankle)
    set_joint(body_pose, 9, spine3)
    set_joint(body_pose, 10, left_foot)
    set_joint(body_pose, 11, right_foot)
    set_joint(body_pose, 12, neck)
    set_joint(body_pose, 16, left_shoulder)
    set_joint(body_pose, 17, right_shoulder)
    set_joint(body_pose, 18, left_elbow)
    set_joint(body_pose, 19, right_elbow)

    return transl, global_orient, body_pose


def estimate_avatar_min_y(
    canonical_path: Path,
    joint_mats: np.ndarray,
    sample_count: int = 12,
) -> float:
    with np.load(canonical_path) as canonical:
        means = canonical["means"].astype(np.float32)
        lbs = canonical["lbs_weights"].astype(np.float32)
        inv_bind = canonical["joints_inv_bind_matrix"].astype(np.float32)

    point_h = np.concatenate(
        [means, np.ones((means.shape[0], 1), dtype=np.float32)], axis=1
    )
    frame_ids = np.linspace(
        0, joint_mats.shape[0] - 1, min(sample_count, joint_mats.shape[0]), dtype=int
    )
    min_y = np.inf
    for frame_id in frame_ids:
        mats = np.matmul(joint_mats[frame_id], inv_bind)
        transformed = np.einsum("jmn,pn->pjm", mats, point_h)[..., :3]
        points = (transformed * lbs[:, :, None]).sum(axis=1)
        min_y = min(min_y, float(points[:, 1].min()))
    return min_y


def precompute_joint_mats(
    transl: np.ndarray,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    betas: np.ndarray,
    smpl_model_path: Path,
    batch_size: int = 128,
) -> np.ndarray:
    _ensure_legacy_numpy_aliases()
    import smplx
    import smplx.lbs as smplx_lbs

    device = torch.device("cpu")
    model = smplx.SMPLX(
        model_path=str(smpl_model_path),
        gender="neutral",
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=True,
        batch_size=batch_size,
    ).to(device)
    model.eval()

    output_chunks = []
    total_frames = int(transl.shape[0])
    betas_base = torch.as_tensor(betas.reshape(1, -1), dtype=torch.float32, device=device)

    for start in range(0, total_frames, batch_size):
        end = min(start + batch_size, total_frames)
        bs = end - start
        if getattr(model, "batch_size", None) != bs:
            model.batch_size = bs

        transl_batch = torch.from_numpy(transl[start:end]).to(device)
        betas_batch = betas_base.repeat(bs, 1)
        global_batch = torch.from_numpy(global_orient[start:end]).to(device)
        body_batch = torch.from_numpy(body_pose[start:end]).to(device)
        zeros_3 = torch.zeros((bs, 3), dtype=torch.float32, device=device)
        zeros_hand = torch.zeros((bs, 45), dtype=torch.float32, device=device)

        with torch.no_grad():
            smpl_output = model.forward(
                betas=betas_batch,
                global_orient=global_batch,
                transl=torch.zeros_like(transl_batch),
                body_pose=body_batch,
                left_hand_pose=zeros_hand,
                right_hand_pose=zeros_hand,
                expression=torch.zeros((bs, 10), dtype=torch.float32, device=device),
                jaw_pose=zeros_3,
                leye_pose=zeros_3,
                reye_pose=zeros_3,
                return_full_pose=True,
            )
            v_shaped = model.v_template + smplx_lbs.blend_shapes(
                betas_batch, model.shapedirs
            )
            joints = smplx_lbs.vertices2joints(model.J_regressor, v_shaped)
            rot_mats = smplx_lbs.batch_rodrigues(
                smpl_output.full_pose.reshape(-1, 3)
            ).view(bs, -1, 3, 3)
            _, joint_mats = smplx_lbs.batch_rigid_transform(
                rot_mats, joints, model.parents, dtype=torch.float32
            )
            joint_mats[:, :, :3, 3] += transl_batch.unsqueeze(1)
        output_chunks.append(joint_mats.cpu().numpy().astype(np.float32))

    return np.concatenate(output_chunks, axis=0)

# Avatar parameters
fps = 40.0  # Habitat-GS GaussianAvatar runtime uses 0.025s/frame.
num_frames = 800  # 20 seconds at 40 fps (matches time_max: 20.0)
repo_root = Path(__file__).resolve().parent
avatar_dir = repo_root / "data/scene_datasets/gs_scenes/avatars/avatar1"
canonical_path = avatar_dir / "canonical_gs.npz"
smpl_model_path = repo_root / "data/scene_datasets/gs_scenes/avatars/smplx"
output_path = avatar_dir / "driver.pkl"

with np.load(canonical_path) as canonical:
    joint_count = int(canonical["lbs_weights"].shape[1])

if joint_count != 55:
    raise ValueError(
        f"avatar1 canonical_gs.npz has {joint_count} joints; expected 55 for SMPL-X."
    )

transl, global_orient, body_pose = make_walk_pose(num_frames, fps)
betas = np.zeros(10, dtype=np.float32)
joint_mats = precompute_joint_mats(
    transl=transl,
    global_orient=global_orient,
    body_pose=body_pose,
    betas=betas,
    smpl_model_path=smpl_model_path,
)
ground_clearance = 0.02
min_y = estimate_avatar_min_y(canonical_path, joint_mats)
ground_delta = ground_clearance - min_y
transl[:, 1] += np.float32(ground_delta)
joint_mats[:, :, 1, 3] += np.float32(ground_delta)

# Create minimal SMPL-X pose parameters
driver_data = {
    # Translation trajectory: simple linear motion
    "transl": transl,

    # Global orientation (rotation around Y axis)
    "global_orient": global_orient,

    # Body pose (63 dims for SMPL-X)
    "body_pose": body_pose,

    # Hand poses (45 dims each, zeros by default)
    "left_hand_pose": np.zeros((num_frames, 45), dtype=np.float32),
    "right_hand_pose": np.zeros((num_frames, 45), dtype=np.float32),

    # Shape parameters
    "betas": betas,

    # Metadata
    "gender": "neutral",
    "smpl_type": "smplx",
    "fps": fps,
    "num_frames": num_frames,
    "joint_mats": joint_mats,
    "joint_mats_space": "smpl_with_trans",
    "joint_mats_version": 1,
    "joint_mats_fps": fps,
    "joint_mats_joint_count": joint_count,
}

# Save
with open(output_path, "wb") as f:
    pickle.dump(driver_data, f)

print(f"✓ Created minimal driver.pkl at: {output_path}")
print(f"  - {num_frames} frames at {fps} fps ({num_frames/fps:.1f} seconds)")
print(f"  - {joint_count} SMPL-X joint matrices")
print("  - Procedural walk cycle with leg, arm, torso, and root motion")
print(f"  - Ground calibration: min_y={min_y:.3f}, delta_y={ground_delta:.3f}")
print("")
print("Note: This is a procedural walking driver for testing.")
print("For realistic motions, use: python tools_gs/generate_trajectory.py")
