import numpy as np


def _rgb3(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"RGB image must have shape [H, W, 3/4], got {arr.shape}.")
    return arr[:, :, :3]


def composite_rgbd(
    rgb_gs: np.ndarray,
    depth_gs: np.ndarray,
    rgb_human: np.ndarray,
    depth_human: np.ndarray,
    id_mask: np.ndarray,
    *,
    debug: bool = False,
) -> dict:
    rgb_gs_arr = np.asarray(rgb_gs)
    rgb_gs_3 = _rgb3(rgb_gs_arr)
    rgb_human_3 = _rgb3(rgb_human)
    depth_gs_arr = np.asarray(depth_gs, dtype=np.float32)
    depth_human_arr = np.asarray(depth_human, dtype=np.float32)
    id_mask_arr = np.asarray(id_mask, dtype=np.int32)

    if depth_gs_arr.shape != depth_human_arr.shape or depth_gs_arr.shape != id_mask_arr.shape:
        raise ValueError(
            "Depth/id shapes must match: "
            f"depth_gs={depth_gs_arr.shape} depth_human={depth_human_arr.shape} "
            f"id_mask={id_mask_arr.shape}."
        )
    if rgb_gs_3.shape[:2] != depth_gs_arr.shape or rgb_human_3.shape[:2] != depth_gs_arr.shape:
        raise ValueError(
            "RGB/depth shapes must match: "
            f"rgb_gs={rgb_gs_arr.shape} rgb_human={rgb_human_3.shape} "
            f"depth={depth_gs_arr.shape}."
        )

    valid_human = id_mask_arr > 0
    gs_valid = np.isfinite(depth_gs_arr) & (depth_gs_arr > 0.0)
    human_valid = np.isfinite(depth_human_arr) & (depth_human_arr > 0.0)
    human_front = human_valid & (~gs_valid | (depth_human_arr < depth_gs_arr))
    visible = valid_human & human_front

    rgb_final_3 = rgb_gs_3.copy()
    rgb_final_3[visible] = rgb_human_3[visible].astype(rgb_final_3.dtype, copy=False)

    if rgb_gs_arr.shape[2] == 4:
        rgb_final = rgb_gs_arr.copy()
        rgb_final[:, :, :3] = rgb_final_3
    else:
        rgb_final = rgb_final_3

    depth_final = depth_gs_arr.copy()
    depth_final[visible] = depth_human_arr[visible]

    id_mask_final = np.zeros_like(id_mask_arr, dtype=np.int32)
    id_mask_final[visible] = id_mask_arr[visible]

    if debug:
        print(
            "[MeshHumanComposite] "
            f"rgb_gs={rgb_gs_arr.shape}/{rgb_gs_arr.dtype} "
            f"depth_gs_minmax=({np.nanmin(depth_gs_arr):.4f}, {np.nanmax(depth_gs_arr):.4f}) "
            f"depth_human_minmax=({np.nanmin(depth_human_arr):.4f}, {np.nanmax(depth_human_arr):.4f}) "
            f"visible_pixels={int(visible.sum())}"
        )

    return {
        "rgb": rgb_final,
        "depth": depth_final.astype(np.float32),
        "id_mask": id_mask_final,
    }
