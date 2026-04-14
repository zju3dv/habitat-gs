"""Navigation tools — `forward`, `turn`, `navigate`, `find_path`, `sample_point`.

Phase 2 PR 2 migrates these 5 tools out of the legacy
`ToolExecutor._tool_*` methods into proper `Tool` subclasses. Behaviour
is preserved exactly — same bridge action names, same payloads, same
state mutations on `ctx.round_state`. The only structural change is
that state now lives on the context rather than on a long-lived
`ToolExecutor` instance.

The 4 navmesh-only tools (`navigate`, `find_path`, `sample_point`,
`topdown`) are gated via `metadata.allowed_nav_modes={"navmesh"}` so
`ToolRegistry.available_for("mapless", ...)` filters them out
automatically — replacing the legacy hand-written
`if nav_mode != "mapless":` branch in `build_tool_schemas`.
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
# ForwardTool
# ---------------------------------------------------------------------------


class ForwardTool:
    """Move the agent forward by distance_m metres.

    The bridge auto-decomposes `distance_m` into 0.25m atomic steps
    (the underlying unit of motion) and stops early on collision."""

    metadata = ToolMetadata(
        name="forward",
        category=ToolCategory.NAVIGATION,
        description=(
            "Move agent forward by `distance_m` metres. The bridge "
            "auto-decomposes the request into 0.25m atomic steps and "
            "stops early on collision. Returns metrics with position, "
            "heading, collision status, and euclidean_distance_to_goal."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "distance_m": {
                    "type": "number",
                    "description": (
                        "Distance in metres. Must be a positive multiple "
                        "of the 0.25m atomic step. Stops early on collision."
                    ),
                    "default": 0.5,
                },
            },
        },
        permission=PermissionLevel.MUTATING,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        dist = args.get("distance_m", 0.5)
        payload: Dict[str, Any] = {
            "action": "move_forward",
            "distance": dist,
            "include_metrics": True,
        }
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("step_and_capture", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))

        collect_images(result, ctx)
        ctx.round_state.last_collided = bool(result.get("collided", False))
        ctx.round_state.last_movement_action = "move_forward"
        col_tag = "!" if ctx.round_state.last_collided else ""
        ctx.round_state.round_actions.append(f"forward({dist}m){col_tag}")
        return ToolResult(
            ok=True,
            body=result,
            captured_images=list(ctx.round_state.captured_images),
        )


# ---------------------------------------------------------------------------
# TurnTool
# ---------------------------------------------------------------------------


class TurnTool:
    """Rotate the agent left or right by `degrees` degrees.

    The bridge auto-decomposes `degrees` into 10° atomic steps
    (the underlying unit of rotation)."""

    metadata = ToolMetadata(
        name="turn",
        category=ToolCategory.NAVIGATION,
        description=(
            "Rotate agent left or right by `degrees` degrees. The "
            "bridge auto-decomposes the request into 10° atomic steps. "
            "Returns metrics with updated heading and "
            "euclidean_distance_to_goal."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": "Turn direction",
                },
                "degrees": {
                    "type": "number",
                    "description": (
                        "Degrees to turn. Must be a positive multiple "
                        "of the 10° atomic step."
                    ),
                    "default": 10,
                },
            },
            "required": ["direction"],
        },
        permission=PermissionLevel.MUTATING,
    )

    # Defensive enum check — the JSON schema declares enum=["left",
    # "right"] but neither OpenAI function calling nor FastMCP enforce
    # that constraint at the protocol layer, so the Tool itself has to
    # validate. Without this, an LLM hallucinated direction (e.g.
    # "up", "around") would form `action="turn_up"` and the bridge
    # would reject it with an opaque "Unknown action" error.
    _ALLOWED_DIRECTIONS = frozenset(("left", "right"))

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        direction = args.get("direction", "left")
        if direction not in self._ALLOWED_DIRECTIONS:
            return ToolResult(
                ok=False,
                body={},
                error=(
                    f"turn direction must be 'left' or 'right', "
                    f"got {direction!r}"
                ),
            )
        degrees = args.get("degrees", 10)
        action = f"turn_{direction}"
        payload: Dict[str, Any] = {
            "action": action,
            "degrees": degrees,
            "include_metrics": True,
        }
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("step_and_capture", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))

        collect_images(result, ctx)
        ctx.round_state.last_collided = bool(result.get("collided", False))
        ctx.round_state.last_movement_action = action
        ctx.round_state.round_actions.append(f"turn_{direction}({degrees}°)")
        return ToolResult(
            ok=True,
            body=result,
            captured_images=list(ctx.round_state.captured_images),
        )


# ---------------------------------------------------------------------------
# NavigateTool — navmesh only
# ---------------------------------------------------------------------------


class NavigateTool:
    """Navmesh-based multi-step navigation to absolute coordinates."""

    metadata = ToolMetadata(
        name="navigate",
        category=ToolCategory.NAVIGATION,
        description=(
            "Navigate toward absolute coordinates using navmesh greedy "
            "follower. Executes multiple steps."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Target X coordinate"},
                "y": {"type": "number", "description": "Target Y coordinate"},
                "z": {"type": "number", "description": "Target Z coordinate"},
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum steps (default 10)",
                    "default": 10,
                },
            },
            "required": ["x", "y", "z"],
        },
        allowed_nav_modes={"navmesh"},
        permission=PermissionLevel.MUTATING,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            goal = [args["x"], args["y"], args["z"]]
        except KeyError as exc:
            return ToolResult(
                ok=False, body={}, error=f"navigate requires x, y, z (missing {exc})"
            )
        payload: Dict[str, Any] = {
            "goal": goal,
            "max_steps": args.get("max_steps", 10),
            "include_metrics": True,
        }
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("navigate_step", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        collect_images(result, ctx)
        return ToolResult(
            ok=True,
            body=result,
            captured_images=list(ctx.round_state.captured_images),
        )


# ---------------------------------------------------------------------------
# FindPathTool — navmesh only
# ---------------------------------------------------------------------------


class FindPathTool:
    """Plan the shortest navmesh path to absolute coordinates."""

    metadata = ToolMetadata(
        name="find_path",
        category=ToolCategory.NAVIGATION,
        description=(
            "Plan shortest path to coordinates. Returns waypoints and "
            "geodesic distance."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
            "required": ["x", "y", "z"],
        },
        allowed_nav_modes={"navmesh"},
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            end = [args["x"], args["y"], args["z"]]
        except KeyError as exc:
            return ToolResult(
                ok=False, body={}, error=f"find_path requires x, y, z (missing {exc})"
            )
        try:
            result = ctx.bridge.call("find_shortest_path", {"end": end})
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body=result)


# ---------------------------------------------------------------------------
# SamplePointTool — navmesh only
# ---------------------------------------------------------------------------


class SamplePointTool:
    """Sample a random navigable point for exploration."""

    metadata = ToolMetadata(
        name="sample_point",
        category=ToolCategory.NAVIGATION,
        description="Sample a random navigable point for exploration.",
        parameters_schema={"type": "object", "properties": {}},
        allowed_nav_modes={"navmesh"},
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = ctx.bridge.call("sample_navigable_point", {})
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body=result)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

ToolRegistry.register(ForwardTool())
ToolRegistry.register(TurnTool())
ToolRegistry.register(NavigateTool())
ToolRegistry.register(FindPathTool())
ToolRegistry.register(SamplePointTool())


__all__ = [
    "ForwardTool",
    "TurnTool",
    "NavigateTool",
    "FindPathTool",
    "SamplePointTool",
]
