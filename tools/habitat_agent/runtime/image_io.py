"""Image encoding helpers for LLM function-calling image inputs.

Kept in `runtime/` because agents and tools across the whole package need
them, and they have no dependencies beyond `base64` / `os`.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional


def read_image_as_data_url(path: str) -> Optional[str]:
    """Read image file and return as base64 data URL.

    Returns None on any failure (missing file, permission error, unknown
    extension). Callers should treat the return as "best effort".
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
            ext, "image/png"
        )
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def read_image_b64(path: str) -> Optional[str]:
    """Read image file and return raw base64 string (no data URL prefix).

    Returns None on any failure.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


def build_image_content_parts(
    image_paths: List[str],
    max_images: int = 4,
) -> List[Dict[str, Any]]:
    """Build OpenAI-compatible image content parts from file paths.

    Caps at `max_images` to keep LLM input bounded. Paths that fail to
    load are silently skipped.

    DEPRECATED: prefer `build_image_user_message`, which composes a full
    user message with prompt text + panorama direction labels and does
    NOT cap the image count. The cap here is preserved only because
    this function is re-exported via the legacy `habitat_agent_core`
    shim and may have external users.
    """
    parts: List[Dict[str, Any]] = []
    for path in image_paths[:max_images]:
        data_url = read_image_as_data_url(path)
        if data_url:
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
    return parts


_PANO_DIRECTION_RE = None  # lazy-compiled to avoid module-import-time regex cost


def build_image_user_message(
    prompt_text: str,
    image_paths: List[str],
) -> Optional[Dict[str, Any]]:
    """Build a complete OpenAI user message containing the LLM prompt
    text plus all relevant images from `image_paths`.

    The output is the ephemeral image message both `nav_agent` and
    `chat_agent` inject into the LLM call (it lives in `messages` for
    one API call only, never in `self.conversation`, so base64
    payloads are not persisted across rounds).

    Selection rules:
      - Depth-sensor images are filtered out (paths containing "depth").
        If filtering would leave nothing, fall back to the raw list so
        a depth-only response still produces a message.
      - Panorama images get a "[FRONT view]" / "[RIGHT view]" /
        "[BACK view]" / "[LEFT view]" label inserted before each
        image so the LLM can spatially orient itself.
      - Returns `None` if no images load successfully (all paths
        invalid or all reads failed).

    NO IMAGE COUNT CAP. Phase 2 PR review round 6 confirmed that the
    previous `[:4]` truncation was actively harmful: agents that
    chained `panorama` (4 images) + `look` (1 image) in a single turn
    saw their newest observation silently dropped, and the LLM then
    reasoned over stale visuals. Token cost is not a concern here
    because the message is ephemeral — it lives in one API call and
    is never written into the persisted conversation history.
    """
    if not image_paths:
        return None

    color_paths = [p for p in image_paths if "depth" not in p]
    if not color_paths:
        color_paths = list(image_paths)

    parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]

    global _PANO_DIRECTION_RE
    if _PANO_DIRECTION_RE is None:
        import re
        _PANO_DIRECTION_RE = re.compile(r"_pano_(front|right|back|left)_")

    loaded = 0
    for path in color_paths:
        match = _PANO_DIRECTION_RE.search(path)
        if match:
            parts.append({"type": "text", "text": f"[{match.group(1).upper()} view]"})
        data_url = read_image_as_data_url(path)
        if data_url:
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
            loaded += 1

    if loaded == 0:
        return None
    return {"role": "user", "content": parts}


__all__ = [
    "read_image_as_data_url",
    "read_image_b64",
    "build_image_content_parts",
    "build_image_user_message",
]
