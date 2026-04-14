#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch


def _ensure_legacy_numpy_aliases() -> None:
    # NumPy 2.x removed legacy aliases such as np.float and np.complex.
    # Older GaussianAvatar dependencies still import these names.
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


def _is_gaussian_avatar_root(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "model", "network.py")) and os.path.isfile(
        os.path.join(path, "model", "modules.py")
    )


def _find_gaussian_avatar_root(candidates) -> str:
    env_root = os.getenv("GAUSSIAN_AVATAR_ROOT")
    if env_root and _is_gaussian_avatar_root(env_root):
        return env_root

    for candidate in candidates:
        if not candidate:
            continue
        cur_dir = os.path.dirname(os.path.abspath(candidate))
        for _ in range(6):
            if _is_gaussian_avatar_root(cur_dir):
                return cur_dir
            parent = os.path.dirname(cur_dir)
            if parent == cur_dir:
                break
            cur_dir = parent
    return ""


def _import_gaussian_avatar_net(root: str):
    if root and root not in sys.path:
        sys.path.insert(0, root)

    _ensure_inspect_getargspec()
    _ensure_legacy_numpy_aliases()

    try:
        from model.network import POP_no_unet
    except Exception as exc:
        raise RuntimeError(
            "Failed to import GaussianAvatar network. Set GAUSSIAN_AVATAR_ROOT "
            "or pass --ga-root."
        ) from exc

    return POP_no_unet


def _load_posmap(path: str) -> np.ndarray:
    data = np.load(path)
    key = ""
    for candidate in data.files:
        if candidate.startswith("posmap"):
            key = candidate
            break
    if not key:
        raise ValueError(f"No posmap entry found in {path}.")
    posmap = data[key]
    if posmap.ndim != 3 or posmap.shape[2] != 3:
        raise ValueError(f"Unexpected posmap shape {posmap.shape} in {path}.")
    return posmap.astype(np.float32)


def _load_lbs_map(path: str) -> np.ndarray:
    lbs = np.load(path)
    if lbs.ndim not in (2, 3):
        raise ValueError(f"Unexpected LBS map shape {lbs.shape} in {path}.")
    return lbs.astype(np.float32)


def _resolve_uv_mask_path(
    posmap_path: str, lbs_path: str, posmap_size: int, smpl_type: str, override: str
) -> str:
    if override and os.path.exists(override):
        return override

    filename = f"uv_mask{posmap_size}_with_faceid_{smpl_type}.npy"

    def _search(base: str) -> str:
        cur_dir = os.path.dirname(os.path.abspath(base))
        for _ in range(6):
            candidate = os.path.join(cur_dir, "assets", "uv_masks", filename)
            if os.path.exists(candidate):
                return candidate
            candidate = os.path.join(cur_dir, filename)
            if os.path.exists(candidate):
                return candidate
            parent = os.path.dirname(cur_dir)
            if parent == cur_dir:
                break
            cur_dir = parent
        return ""

    found = _search(posmap_path)
    if found:
        return found
    return _search(lbs_path)


def _load_uv_mask(path: str, posmap_size: int) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    mask = np.load(path)
    return mask.reshape(posmap_size, posmap_size)


def _build_uv_coord_map(size: int, device: torch.device) -> torch.Tensor:
    try:
        ys, xs = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
            indexing="ij",
        )
    except TypeError:
        ys, xs = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
        )
    coords = torch.stack([ys, xs], dim=-1).view(-1, 2).float()
    coords = coords / float(size - 1)
    return coords


def _infer_net_params(
    state_dict: Dict[str, torch.Tensor], geo_feature: torch.Tensor
) -> Dict[str, Any]:
    geom_type = "conv"
    for key in state_dict.keys():
        if key.startswith("geom_proc_layers.up"):
            geom_type = "bottleneck"
            break
        if key.startswith("geom_proc_layers.conv5") or key.startswith(
            "geom_proc_layers.up4"
        ):
            geom_type = "unet"
            break

    hsize = 256
    if "decoder.conv1.weight" in state_dict:
        hsize = int(state_dict["decoder.conv1.weight"].shape[0])

    nf = 64
    if geom_type == "unet" and "geom_proc_layers.conv1.weight" in state_dict:
        nf = int(state_dict["geom_proc_layers.conv1.weight"].shape[0])

    return {
        "c_geom": int(geo_feature.shape[1]),
        "geom_layer_type": geom_type,
        "nf": nf,
        "hsize": hsize,
        "up_mode": "upconv",
        "use_dropout": False,
        "uv_feat_dim": 2,
    }


def _flatten_lbs(lbs: np.ndarray, flat_count: int, mask_flat: np.ndarray) -> np.ndarray:
    if lbs.ndim == 3:
        h, w, j = lbs.shape
        flat = lbs.reshape(h * w, j)
    else:
        flat = lbs
    if flat.shape[0] != flat_count:
        raise ValueError(f"LBS map size mismatch: expected {flat_count}, got {flat.shape[0]}.")
    return flat[mask_flat]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GaussianAvatar canonical Gaussians to unified NPZ."
    )
    parser.add_argument("--posmap", required=True, help="Path to query_posemap_.npz.")
    parser.add_argument("--lbs-map", required=True, help="Path to lbs_map_.npy.")
    parser.add_argument("--joint-mat", required=True, help="Path to smpl_cano_joint_mat.pth.")
    parser.add_argument("--net-ckpt", required=True, help="Path to net.pth.")
    parser.add_argument("--out", required=True, help="Output canonical NPZ path.")
    parser.add_argument("--uv-mask", default="", help="Optional UV mask path.")
    parser.add_argument("--ga-root", default="", help="GaussianAvatar repo root.")
    parser.add_argument("--device", default="cuda", help="Torch device (cuda or cpu).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    posmap = _load_posmap(args.posmap)
    posmap_size = posmap.shape[0]
    lbs = _load_lbs_map(args.lbs_map)
    joint_count = lbs.shape[2] if lbs.ndim == 3 else lbs.shape[1]
    smpl_type = "smplx" if joint_count == 55 else "smpl"

    uv_mask_path = _resolve_uv_mask_path(
        args.posmap, args.lbs_map, posmap_size, smpl_type, args.uv_mask
    )
    uv_mask = _load_uv_mask(uv_mask_path, posmap_size)
    if uv_mask is None:
        mask_flat = np.ones((posmap_size * posmap_size,), dtype=bool)
    else:
        mask_flat = uv_mask.reshape(-1) != -1

    flat_pos = posmap.reshape(-1, 3)
    positions = flat_pos[mask_flat]
    lbs_weights = _flatten_lbs(lbs, flat_pos.shape[0], mask_flat)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    net_root = args.ga_root or _find_gaussian_avatar_root(
        [args.net_ckpt, args.posmap, args.lbs_map]
    )
    if not net_root:
        raise RuntimeError("Failed to locate GaussianAvatar repo root.")

    POP_no_unet = _import_gaussian_avatar_net(net_root)

    ckpt = torch.load(args.net_ckpt, map_location="cpu")
    net_state = ckpt.get("net", {})
    geo_feature = ckpt.get("geo_feature", None)
    if geo_feature is None:
        raise ValueError(f"geo_feature missing in {args.net_ckpt}.")

    geo_feature = geo_feature.detach().float().to(device)
    net_params = _infer_net_params(net_state, geo_feature)
    net = POP_no_unet(**net_params).to(device).eval()
    net.load_state_dict(net_state, strict=False)

    mask_tensor = torch.from_numpy(mask_flat).to(device)
    uv_coords = _build_uv_coord_map(posmap_size, device)
    uv_coords = uv_coords[None, ...].contiguous()

    with torch.no_grad():
        geom_featmap = geo_feature.expand(1, -1, -1, -1).contiguous()
        pred_res, pred_scales, pred_shs = net.forward(
            pose_featmap=None, geom_featmap=geom_featmap, uv_loc=uv_coords
        )

        pred_res = pred_res.permute(0, 2, 1) * 0.02
        pred_scales = pred_scales.permute(0, 2, 1)
        pred_shs = pred_shs.permute(0, 2, 1)

        pred_res = pred_res[:, mask_tensor, :].contiguous()
        pred_scales = pred_scales[:, mask_tensor, :].contiguous()
        pred_scales = pred_scales.repeat(1, 1, 3)
        pred_shs = pred_shs[:, mask_tensor, :].contiguous()

    offsets = pred_res[0].detach().cpu().numpy()
    shs = pred_shs[0].detach().cpu().numpy()
    scales = pred_scales[0].detach().cpu().numpy()

    means = positions + offsets

    quats = np.zeros((means.shape[0], 4), dtype=np.float32)
    quats[:, 0] = 1.0
    opacities = np.ones((means.shape[0], 1), dtype=np.float32)

    joint_mats = torch.load(args.joint_mat, map_location="cpu")
    if isinstance(joint_mats, dict):
        if len(joint_mats) == 1:
            joint_mats = next(iter(joint_mats.values()))
        else:
            raise ValueError(f"Unexpected joint_mat dict keys: {list(joint_mats.keys())}")
    if not isinstance(joint_mats, torch.Tensor):
        raise ValueError(f"Unexpected joint_mat type: {type(joint_mats)}")
    if joint_mats.ndim == 4 and joint_mats.shape[0] == 1:
        joint_mats = joint_mats[0]
    elif joint_mats.ndim == 2 and joint_mats.shape[1] == 16:
        joint_mats = joint_mats.view(-1, 4, 4)
    if joint_mats.ndim != 3 or joint_mats.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected joint_mat shape: {tuple(joint_mats.shape)}")
    inv_bind = torch.linalg.inv(joint_mats).cpu().numpy().astype(np.float32)

    np.savez(
        args.out,
        means=means.astype(np.float32),
        shs=shs.astype(np.float32),
        scales=scales.astype(np.float32),
        quats=quats,
        opacities=opacities,
        lbs_weights=lbs_weights.astype(np.float32),
        joints_inv_bind_matrix=inv_bind,
    )

    print(f"Saved canonical gaussians to {args.out}")


if __name__ == "__main__":
    main()
