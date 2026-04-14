"""Perception tools — `look`, `panorama`, `depth_analyze`.

Phase 2 PR 2 migrates these 3 tools out of
`ToolExecutor._tool_look` / `_tool_panorama` / `_tool_depth_analyze`.
All three are read-only (they capture sensor data without moving the
agent). `look` and `panorama` populate `ctx.round_state.captured_images`
and `last_visual_path` via the shared `collect_images` helper;
`depth_analyze` is pure bridge pass-through with no image capture.
"""

from __future__ import annotations

from typing import Any, Dict

from ._common import collect_images, visual_payload
from .base import (
    PermissionLevel,
    Tool,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)


# ---------------------------------------------------------------------------
# LookTool
# ---------------------------------------------------------------------------


class LookTool:
    """Capture the current camera view (RGB + depth)."""

    metadata = ToolMetadata(
        name="look",
        category=ToolCategory.PERCEPTION,
        description=(
            "Capture current camera view (RGB image + depth). Returns "
            "metrics and saves image to disk."
        ),
        parameters_schema={"type": "object", "properties": {}},
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        payload: Dict[str, Any] = {"include_metrics": True}
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("get_visuals", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        collect_images(result, ctx)
        ctx.round_state.round_actions.append("look")
        return ToolResult(
            ok=True,
            body=result,
            captured_images=list(ctx.round_state.captured_images),
        )


# ---------------------------------------------------------------------------
# PanoramaTool
# ---------------------------------------------------------------------------


class PanoramaTool:
    """Capture a 4-direction panorama (front/right/back/left)."""

    metadata = ToolMetadata(
        name="panorama",
        category=ToolCategory.PERCEPTION,
        description=(
            "Capture 4-direction panorama (front/right/back/left). Agent "
            "heading is restored after scan. Returns 4 images + depth "
            "analysis."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "include_depth": {
                    "type": "boolean",
                    "description": "Include depth analysis per direction",
                    "default": True,
                },
            },
        },
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        payload: Dict[str, Any] = {
            "include_depth_analysis": args.get("include_depth", True),
            "include_metrics": True,
        }
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("get_panorama", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        collect_images(result, ctx)
        ctx.round_state.round_actions.append("panorama")
        return ToolResult(
            ok=True,
            body=result,
            captured_images=list(ctx.round_state.captured_images),
        )


# ---------------------------------------------------------------------------
# DepthAnalyzeTool
# ---------------------------------------------------------------------------


class DepthAnalyzeTool:
    """Analyze depth sensor data in 3 front-facing regions."""

    metadata = ToolMetadata(
        name="depth_analyze",
        category=ToolCategory.PERCEPTION,
        description=(
            "Analyze depth sensor data in 3 regions (front_left, "
            "front_center, front_right). Returns min/mean distance per "
            "region to detect obstacles."
        ),
        parameters_schema={"type": "object", "properties": {}},
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = ctx.bridge.call("analyze_depth", {})
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body=result)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

ToolRegistry.register(LookTool())
ToolRegistry.register(PanoramaTool())
ToolRegistry.register(DepthAnalyzeTool())


__all__ = ["LookTool", "PanoramaTool", "DepthAnalyzeTool"]
