"""Phase 2 PR review round 6 — regression locks for the image helper.

Covers `build_image_user_message`, the helper extracted from the
inline image-message-construction logic that previously lived in 3
separate sites in `nav_agent.py` / `chat_agent.py`. The most
load-bearing invariant: there is NO image count cap, so chained
visual tool calls (panorama + look) deliver every captured frame to
the LLM.
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from habitat_agent.runtime.image_io import build_image_user_message  # noqa: E402


@pytest.fixture
def fake_pngs(tmp_path, monkeypatch):
    """Create N small valid PNGs and return their paths.

    The helper actually opens each file and base64-encodes it, so
    real bytes on disk are needed (not just paths). One-pixel PNG
    bytes are enough — we're testing message structure, not pixels.
    """
    # 1x1 PNG signature + minimum IHDR/IDAT/IEND
    png_bytes = bytes.fromhex(
        "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4"
        "890000000D49444154789C6300010000000500010D0A2DB40000000049454E44"
        "AE426082"
    )
    paths = []
    for i in range(8):
        p = tmp_path / f"frame_{i}.png"
        p.write_bytes(png_bytes)
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------------------
# Round 6 P0 regression lock — no image count cap
# ---------------------------------------------------------------------------


def test_build_image_user_message_sends_all_5_images_no_cap(fake_pngs):
    """The Codex round-6 scenario: agent chained `panorama` (4 images)
    + `look` (1 image) in one turn. The helper MUST surface ALL 5
    images to the LLM. The previous `[:4]` cap silently dropped the
    newest frame and made the next LLM round reason over stale
    visuals."""
    paths = fake_pngs[:5]  # 4 panorama + 1 look = 5 frames
    msg = build_image_user_message("Here are 5 frames:", paths)
    assert msg is not None
    parts = msg["content"]
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 5, (
        f"expected all 5 images delivered to LLM, got {len(image_parts)}. "
        f"This is the round-6 regression: a [:4] cap or similar truncation "
        f"silently dropped the most recent frame."
    )


def test_build_image_user_message_no_cap_at_8_images(fake_pngs):
    """Sanity check: bigger fan-out (8 images) also works.
    There is genuinely no cap."""
    msg = build_image_user_message("Here are 8 frames:", fake_pngs)
    assert msg is not None
    image_parts = [p for p in msg["content"] if p.get("type") == "image_url"]
    assert len(image_parts) == 8


# ---------------------------------------------------------------------------
# Other invariants the previous inline code preserved
# ---------------------------------------------------------------------------


def test_build_image_user_message_filters_depth_paths(fake_pngs, tmp_path):
    """Depth-sensor images are filtered out by file name. (Color
    images contain "color" in the path; depth images contain "depth".
    The helper drops anything with "depth" in the path so the LLM
    doesn't get one-channel depth maps mixed in with RGB.)"""
    # Rename one path to look like a depth image
    depth_path = str(tmp_path / "frame_depth_0.png")
    os.rename(fake_pngs[0], depth_path)
    paths = [depth_path] + fake_pngs[1:4]  # 1 depth + 3 color
    msg = build_image_user_message("Here are 4 frames:", paths)
    assert msg is not None
    image_parts = [p for p in msg["content"] if p.get("type") == "image_url"]
    assert len(image_parts) == 3  # depth was filtered out


def test_build_image_user_message_falls_back_when_only_depth_present(
    fake_pngs, tmp_path
):
    """If the depth filter would leave nothing (rare — all sensors
    were depth-only), fall back to the raw list so the LLM still
    gets *something*. Mirrors the legacy fallback in nav_agent."""
    depth_path = str(tmp_path / "frame_depth_0.png")
    os.rename(fake_pngs[0], depth_path)
    msg = build_image_user_message("depth only", [depth_path])
    assert msg is not None
    image_parts = [p for p in msg["content"] if p.get("type") == "image_url"]
    assert len(image_parts) == 1


def test_build_image_user_message_inserts_panorama_view_labels(
    fake_pngs, tmp_path
):
    """Panorama images get a directional text label inserted before
    each `image_url` part so the LLM can reason about orientation.
    Looks for `_pano_(front|right|back|left)_` in the path name."""
    pano_paths = []
    for direction in ("front", "right", "back", "left"):
        p = tmp_path / f"capture_pano_{direction}_0001.png"
        os.rename(fake_pngs[len(pano_paths)], str(p))
        pano_paths.append(str(p))
    msg = build_image_user_message("Look around:", pano_paths)
    assert msg is not None
    parts = msg["content"]
    text_parts = [p["text"] for p in parts if p.get("type") == "text"]
    assert "[FRONT view]" in text_parts
    assert "[RIGHT view]" in text_parts
    assert "[BACK view]" in text_parts
    assert "[LEFT view]" in text_parts


def test_build_image_user_message_returns_none_when_no_paths():
    assert build_image_user_message("nothing", []) is None


def test_build_image_user_message_returns_none_when_all_paths_invalid(tmp_path):
    """If every path fails to load, the helper returns None instead
    of an empty user message — callers can then skip the inject
    entirely (matches the legacy `if len(parts) > 1` guard)."""
    bogus = [str(tmp_path / "does_not_exist.png")]
    assert build_image_user_message("none of these exist", bogus) is None


def test_build_image_user_message_includes_prompt_text(fake_pngs):
    """The first content part is always the prompt text the caller
    supplied (so the LLM has a clear instruction header)."""
    msg = build_image_user_message("Describe these:", fake_pngs[:2])
    assert msg is not None
    assert msg["content"][0] == {"type": "text", "text": "Describe these:"}
