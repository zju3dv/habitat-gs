"""Mapping tools — `topdown`.

Currently contains `TopdownTool`, the overhead 2D map view.
Navmesh-only: a top-down map requires the scene's navmesh to project
the agent's world coordinates correctly.
"""

from __future__ import annotations

from typing import Any, Dict

from ._common import visual_payload
from .base import (
    PermissionLevel,
    Tool,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)


class TopdownTool:
    """Overhead 2D map view; optionally overlay goal and planned path."""

    metadata = ToolMetadata(
        name="topdown",
        category=ToolCategory.MAPPING,
        description=(
            "Get overhead 2D map view. Optionally show goal and planned path."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "goal_x": {"type": "number"},
                "goal_y": {"type": "number"},
                "goal_z": {"type": "number"},
                "show_path": {"type": "boolean", "default": False},
            },
        },
        allowed_nav_modes={"navmesh"},
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        payload: Dict[str, Any] = {}
        if "goal_x" in args:
            payload["goal"] = [
                args["goal_x"],
                args.get("goal_y", 0),
                args.get("goal_z", 0),
            ]
        if args.get("show_path"):
            payload["show_path"] = True
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("get_topdown_map", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body=result)


ToolRegistry.register(TopdownTool())


__all__ = ["TopdownTool"]
