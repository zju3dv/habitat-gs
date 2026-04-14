"""Shared helpers for the Tool subclasses.

Holds the two utility functions that multiple Tool categories share —
`visual_payload` (the `output_dir` block every visual bridge call needs)
and `collect_images` (extracts `visuals` / `images` entries from a bridge
response and updates `ctx.round_state`).

These replace the legacy `ToolExecutor._visual_payload` /
`_collect_images` instance methods. Keeping them as module-level pure
functions lets the Tool subclasses stay stateless; all the state lives
on `ctx.round_state`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import ToolContext


def visual_payload(ctx: ToolContext) -> Dict[str, Any]:
    """Build the output_dir block for visual bridge calls."""
    return {
        "output_dir": ctx.output_dir,
    }


def collect_images(result: Dict[str, Any], ctx: ToolContext) -> None:
    """Extract captured image paths from a bridge response into RoundState.

    The bridge returns captured visuals in two possible shapes:
      - `result["visuals"]`: dict keyed by sensor name, each entry with
        `path`. Used by `step_and_capture`, `get_visuals`.
      - `result["images"]`: list of dicts with `path`
        and optional `direction`. Used by `get_panorama`.

    Appends every extracted path to `ctx.round_state.captured_images`
    and updates `ctx.round_state.last_visual_path` using the priority
    color_sensor > front panorama > any image. The last_visual_path
    falls back to its previous value if nothing new is collected so
    the mapless auto-injection in `update_nav_status` stays stable
    across turns where no image was taken.
    """
    color_path: Optional[str] = None
    any_path: Optional[str] = None
    front_pano_path: Optional[str] = None

    visuals = result.get("visuals", {})
    if isinstance(visuals, dict):
        for sensor_name, sensor_data in visuals.items():
            if isinstance(sensor_data, dict):
                mp = sensor_data.get("path")
                if mp and isinstance(mp, str):
                    ctx.round_state.captured_images.append(mp)
                    any_path = mp
                    if "color" in sensor_name:
                        color_path = mp

    images = result.get("images", [])
    if isinstance(images, list):
        for img in images:
            if isinstance(img, dict):
                mp = img.get("path")
                if mp and isinstance(mp, str):
                    ctx.round_state.captured_images.append(mp)
                    any_path = mp
                    if img.get("direction", "") == "front":
                        front_pano_path = mp
                    if color_path is None:
                        color_path = mp  # panorama images are always color

    # Priority: color_sensor > front panorama > any captured image.
    # Fall back to previous value so mapless auto-injection stays stable.
    ctx.round_state.last_visual_path = (
        color_path or front_pano_path or any_path or ctx.round_state.last_visual_path
    )


_FALSE_STRINGS = frozenset({"false", "0", "no", "n", "off", ""})
_TRUE_STRINGS = frozenset({"true", "1", "yes", "y", "on"})


def parse_bool_flag(value: Any, default: bool = False) -> bool:
    """Tolerantly parse a boolean flag from heterogeneous caller input.

    Python's built-in `bool()` has a well-known pitfall: `bool("false")`
    returns True because any non-empty string is truthy. Some LLMs emit
    JSON string booleans ("false" / "true") even when the schema
    declares a boolean parameter, so a naive `bool(args.get(...))` on
    LLM-supplied tool arguments would misinterpret their intent.

    Parsing rules (case-insensitive for strings):
      - Native `bool` → passed through
      - `int`/`float` → `bool(value)` (0 → False, non-zero → True)
      - String `"true"`/`"yes"`/`"y"`/`"on"`/`"1"` → True
      - String `"false"`/`"no"`/`"n"`/`"off"`/`"0"`/`""` → False
      - `None` or any other type → `default`

    Used by `NavLoopStartTool` to parse `has_ground_truth` from LLM
    tool-call args without the string-truthiness bug. Any future Tool
    that takes a boolean from untrusted args should prefer this over
    `bool()`.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE_STRINGS:
            return True
        if v in _FALSE_STRINGS:
            return False
        return default
    return default


__all__ = ["visual_payload", "collect_images", "parse_bool_flag"]
