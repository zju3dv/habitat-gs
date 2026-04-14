#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch


# SH constant must match renderer (src/esp/gfx/gaussian_cuda_rasterizer/auxiliary.h).
SH_C0 = 0.28209479177387814


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export AnimatableGaussians canonical Gaussians to unified NPZ."
    )
    parser.add_argument("--config", required=True, help="AnimatableGaussians config YAML.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint dir or net.pt path.")
    parser.add_argument("--out", required=True, help="Output canonical NPZ path.")
    parser.add_argument("--anim-root", default="", help="AnimatableGaussians repo root.")
    parser.add_argument("--data-dir", default="", help="Override data_dir.")
    parser.add_argument("--smpl-params", default="", help="Override smpl_params.npz path.")
    parser.add_argument("--smpl-pos-map", default="", help="Override cano_smpl_pos_map.exr path.")
    parser.add_argument("--smpl-model-path", default="", help="Override SMPLX model path.")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda or cpu).")
    return parser.parse_args()


def _resolve_anim_root(config_path: str, anim_root: str) -> str:
    if anim_root:
        return anim_root
    config_dir = os.path.abspath(os.path.dirname(config_path))
    return os.path.abspath(os.path.join(config_dir, os.pardir))


def _resolve_data_dir(opt: Dict[str, Any], override: str) -> str:
    if override:
        return override
    for key in ("train", "test"):
        data = opt.get(key, {}).get("data", {})
        data_dir = data.get("data_dir")
        if data_dir:
            return data_dir
    return ""


def _patch_data_dir(opt: Dict[str, Any], data_dir: str) -> None:
    for split in ("train", "test"):
        opt.setdefault(split, {}).setdefault("data", {})["data_dir"] = data_dir


def _infer_data_dir_from_smpl_pos_map(path: str) -> str:
    if not path:
        return ""
    return os.path.abspath(os.path.join(os.path.dirname(path), os.pardir))


def _resolve_pose_index(opt: Dict[str, Any]) -> Optional[int]:
    test_opt = opt.get("test", {})
    if isinstance(test_opt, dict):
        fix_hand_id = test_opt.get("fix_hand_id")
        if fix_hand_id is not None:
            try:
                return int(fix_hand_id)
            except (TypeError, ValueError):
                pass

    train_opt = opt.get("train", {})
    for key in ("eval_training_ids", "eval_testing_ids"):
        ids = None
        if isinstance(train_opt, dict):
            ids = train_opt.get(key)
        if ids is None and isinstance(test_opt, dict):
            ids = test_opt.get(key)
        if isinstance(ids, (list, tuple)) and ids:
            try:
                return int(ids[0])
            except (TypeError, ValueError):
                pass
    return None


def _load_pose_map(cano_path: str, target_size: Optional[int] = None) -> np.ndarray:
    import cv2 as cv

    pose_map = cv.imread(cano_path, cv.IMREAD_UNCHANGED)
    if pose_map is None:
        raise ValueError(f"Failed to read pose map: {cano_path}")
    if pose_map.shape[1] % 2 != 0:
        raise ValueError(f"Unexpected pose map width: {pose_map.shape}")
    pos_map_size = pose_map.shape[1] // 2
    front = pose_map[:, :pos_map_size]
    if target_size and front.shape[0] != target_size:
        front = cv.resize(front, (target_size, target_size), interpolation=cv.INTER_NEAREST)
    pose_map = front.transpose((2, 0, 1))
    return pose_map


def _load_smpl_params(path: str) -> Dict[str, np.ndarray]:
    smpl_data = np.load(path, allow_pickle=True)
    return {k: np.asarray(v) for k, v in dict(smpl_data).items()}


def _load_checkpoint(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "net.pt")
    return path


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device=device, dtype=torch.float32)


def _colors_to_sh0(colors: np.ndarray) -> np.ndarray:
    """Convert precomputed RGB to SH0 coefficients."""
    return (colors - 0.5) / SH_C0


def main() -> None:
    args = _parse_args()

    anim_root = _resolve_anim_root(args.config, args.anim_root)
    if not os.path.isdir(anim_root):
        raise ValueError(f"AnimatableGaussians root not found: {anim_root}")

    sys.path.insert(0, anim_root)

    import config as ag_config

    ag_config.load_global_opt(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    ag_config.device = device

    data_dir = _resolve_data_dir(ag_config.opt, args.data_dir)
    if not data_dir:
        raise ValueError("Failed to resolve data_dir from config; pass --data-dir.")

    ckpt_path = _load_checkpoint(args.ckpt)
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint not found: {ckpt_path}")

    if args.smpl_pos_map:
        inferred_data_dir = _infer_data_dir_from_smpl_pos_map(args.smpl_pos_map)
        if inferred_data_dir and os.path.isdir(inferred_data_dir):
            if args.data_dir and os.path.abspath(data_dir) != inferred_data_dir:
                raise ValueError(
                    f"--data-dir ({data_dir}) does not match --smpl-pos-map base ({inferred_data_dir})."
                )
            data_dir = inferred_data_dir

    if not os.path.isdir(data_dir):
        raise ValueError(f"data_dir not found: {data_dir}")

    _patch_data_dir(ag_config.opt, data_dir)

    smpl_params_path = args.smpl_params or os.path.join(data_dir, "smpl_params.npz")
    if not os.path.exists(smpl_params_path):
        raise ValueError(f"smpl_params.npz not found: {smpl_params_path}")

    smpl_pos_map_path = args.smpl_pos_map or os.path.join(data_dir, "smpl_pos_map", "cano_smpl_pos_map.exr")
    if not os.path.exists(smpl_pos_map_path):
        raise ValueError(f"cano_smpl_pos_map.exr not found: {smpl_pos_map_path}")

    pose_map_path = smpl_pos_map_path
    pose_idx = _resolve_pose_index(ag_config.opt)
    if pose_idx is not None:
        candidate = os.path.join(data_dir, "smpl_pos_map", f"{pose_idx:08d}.exr")
        if os.path.exists(candidate):
            pose_map_path = candidate
        else:
            print(
                f"Warning: pose map for index {pose_idx} not found at {candidate}. "
                "Falling back to canonical pose map."
            )

    smpl_model_path = args.smpl_model_path or os.path.join(
        anim_root, "smpl_files", "smplx"
    )

    from network.avatar import AvatarNet

    avatar_net = AvatarNet(ag_config.opt.get("model", {})).to(device).eval()
    net_dict = torch.load(ckpt_path, map_location="cpu")
    avatar_net.load_state_dict(net_dict.get("avatar_net", net_dict))

    pose_map = _load_pose_map(
        pose_map_path, getattr(avatar_net.position_net, "inp_size", None)
    )
    pose_map = torch.from_numpy(pose_map).to(torch.float32).to(device)
    pose_map = pose_map[:3]

    with torch.no_grad():
        positions = avatar_net.get_positions(pose_map)
        opacity, scales, rotations = avatar_net.get_others(pose_map)
        colors, _ = avatar_net.get_colors(pose_map, None, None)

    means = positions.detach().cpu().numpy().astype(np.float32)
    colors = colors.detach().cpu().numpy().astype(np.float32)
    shs = _colors_to_sh0(colors).astype(np.float32)
    shs = shs[:, [2, 1, 0]]
    scales = scales.detach().cpu().numpy().astype(np.float32)
    quats = rotations.detach().cpu().numpy().astype(np.float32)
    opacities = opacity.detach().cpu().numpy().astype(np.float32)
    lbs_weights = avatar_net.lbs.detach().cpu().numpy().astype(np.float32)

    smpl_data = _load_smpl_params(smpl_params_path)
    betas = smpl_data.get("betas")
    if betas is None:
        raise ValueError("smpl_params missing betas.")
    betas = _to_device(torch.from_numpy(betas[:1]), device)

    jaw_pose = smpl_data.get("jaw_pose")
    expression = smpl_data.get("expression")
    if jaw_pose is None:
        jaw_pose = np.zeros((1, 3), dtype=np.float32)
    if expression is None:
        expression = np.zeros((1, 10), dtype=np.float32)

    try:
        import smplx
    except Exception as exc:
        raise RuntimeError("Failed to import smplx.") from exc

    smpl_model = smplx.SMPLX(
        model_path=smpl_model_path,
        gender="neutral",
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=True,
        batch_size=1,
    ).to(device)

    with torch.no_grad():
        cano_out = smpl_model.forward(
            betas=betas,
            global_orient=_to_device(ag_config.cano_smpl_global_orient[None], device),
            transl=_to_device(ag_config.cano_smpl_transl[None], device),
            body_pose=_to_device(ag_config.cano_smpl_body_pose[None], device),
            jaw_pose=_to_device(torch.from_numpy(jaw_pose[:1]), device),
            expression=_to_device(torch.from_numpy(expression[:1]), device),
        )
        inv_bind = torch.linalg.inv(cano_out.A[0]).cpu().numpy().astype(np.float32)

    np.savez(
        args.out,
        means=means,
        colors=colors,
        shs=shs,
        scales=scales,
        quats=quats,
        opacities=opacities,
        lbs_weights=lbs_weights,
        joints_inv_bind_matrix=inv_bind,
    )

    print(f"Saved canonical gaussians to {args.out}")


if __name__ == "__main__":
    main()
