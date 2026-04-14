"""Session tools — `init_scene`, `close_session`, `nav_loop_*`.

Phase 2 PR 3 migrates the 5 chat-only tools out of
`ChatAgent._tool_*` methods (and also the richer `mcp_server.hab_*`
variants) into a single set of `Tool` subclasses. All 5 have
`metadata.allowed_task_types = {"chat"}` so `ToolRegistry.available_for`
hides them from the nav_agent loop (which sees `task_type="pointnav"`
etc.).

`NavLoopStartTool` takes the **richer** mcp_server version of the tool
as the single source of truth — it supports `task_type`, `goal_type`,
`goal_x/y/z`, `has_ground_truth`, `success_distance_threshold`,
`max_iterations`, `reference_image`, `nav_mode` — and preserves the
codex P1/P2 fix verbatim:

  - `is_pointnav_task = task_type == "pointnav"` (not `goal_type`)
  - PointNav always requires explicit coordinates, even without
    `has_ground_truth=True`, and even if `goal_type="instruction"`.
    This closes the hole where a caller could spin up a PointNav loop
    with no target at all.
  - Non-PointNav tasks never receive `goal_position` in the agent
    prompt, regardless of `goal_type`. Coordinates (when provided)
    only go into `eval_goal_position` for GT scoring. This closes
    the hole where `task_type="objectnav", goal_type="position"`
    would leak coordinates to the coordinate-blind objectnav prompt.

The chat-mode session persistence (writing chat history JSONL files)
stays in `ChatAgent` — it is chat-UX business, not a tool behaviour.
`InitSceneTool` only handles the bridge `init_scene` call and the
resulting context mutations (`session_id`, `is_gaussian`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from ._common import parse_bool_flag
from .base import (
    PermissionLevel,
    Tool,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)

_SESSION_TASK_TYPES = {"chat"}


# ---------------------------------------------------------------------------
# InitSceneTool
# ---------------------------------------------------------------------------


def _default_scene_dataset_config() -> str:
    """Fallback scene dataset config path: NAV_DEFAULT_DATASET_CONFIG env
    → the bundled gs-test GS train dataset if present → empty string.

    The path resolution walks up 3 directories from this file
    (tools/habitat_agent/tools/session.py → tools/habitat_agent →
    tools → project root) and then into data/gs-test/gs_train/.
    Mirrors the legacy ChatAgent fallback that was hard-coded in
    `_tool_init_scene`.
    """
    env = os.environ.get("HAB_DEFAULT_DATASET_CONFIG", "")
    if env:
        return env
    candidate = str(
        Path(__file__).resolve().parents[3]
        / "data"
        / "gs-test"
        / "gs_train"
        / "train.scene_dataset_config.json"
    )
    return candidate if os.path.isfile(candidate) else ""


_DEFAULT_SCENE_FALLBACK = "interior_0405_840145"


class InitSceneTool:
    """Initialize a fresh simulator session. Mutates ctx.session_id and
    ctx.is_gaussian from the bridge's response."""

    metadata = ToolMetadata(
        name="init_scene",
        category=ToolCategory.SESSION,
        description=(
            "Initialize a simulation session for a given scene. Must be "
            "called first, before any movement or observation tools."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "scene": {
                    "type": "string",
                    "description": "Scene ID (optional; uses bridge default when omitted)",
                },
                "scene_dataset_config_file": {
                    "type": "string",
                    "description": "Path to scene_dataset_config.json (optional)",
                },
                "depth": {
                    "type": "boolean",
                    "description": "Enable depth sensor (default true)",
                    "default": True,
                },
                "semantic": {
                    "type": "boolean",
                    "description": "Enable semantic sensor (default false)",
                    "default": False,
                },
                "third_person": {
                    "type": "boolean",
                    "description": (
                        "Enable third-person over-the-shoulder RGB camera "
                        "(default false). When true, the bridge injects an "
                        "extra CameraSensorSpec at ~1.5m behind and 1.2m "
                        "above the agent and exposes it as `third_rgb_sensor` "
                        "in visuals/observations."
                    ),
                    "default": False,
                },
            },
        },
        allowed_task_types=_SESSION_TASK_TYPES,
        permission=PermissionLevel.MUTATING,
        requires_session=False,  # init_scene creates the session
        # MCP backward-compat: legacy clients hardcoded `hab_init`,
        # which maps to the InitSceneTool via the `init` alias.
        legacy_names={"init"},
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        payload: Dict[str, Any] = {}

        # Scene fallback chain (mirrors legacy mcp_server.hab_init):
        #   1. explicit caller arg
        #   2. HAB_DEFAULT_SCENE env var
        #   3. hardcoded "interior_0405_840145"
        # The bridge requires a non-empty `payload.scene`, so we MUST
        # always populate it — a no-arg init_scene() call cannot be
        # allowed to reach the bridge with the key missing.
        scene = (
            args.get("scene")
            or os.environ.get("HAB_DEFAULT_SCENE")
            or _DEFAULT_SCENE_FALLBACK
        )
        payload["scene"] = scene

        ds_config = args.get("scene_dataset_config_file") or _default_scene_dataset_config()
        if ds_config:
            payload["scene_dataset_config_file"] = ds_config

        # Sensor config. depth/semantic flags come from caller args
        # via parse_bool_flag (LLM string-boolean tolerance), with
        # defaults that match legacy mcp_server.hab_init exactly:
        # depth=True, semantic=False. Width/height/hfov stay
        # hardcoded to the values both legacy paths (ChatAgent +
        # mcp_server) used.
        depth_enabled = parse_bool_flag(args.get("depth", True), default=True)
        semantic_enabled = parse_bool_flag(args.get("semantic", False), default=False)
        third_person_enabled = parse_bool_flag(
            args.get("third_person", False), default=False
        )
        payload["sensor"] = {
            "width": 512,
            "height": 512,
            "sensor_height": 1.5,
            "hfov": 90,
            "color_sensor": True,
            "depth_sensor": depth_enabled,
            "semantic_sensor": semantic_enabled,
            "third_person_color_sensor": third_person_enabled,
        }

        # Clear any prior session_id so the bridge knows this is a new
        # session, not a re-init of the current one. Clear BOTH sides
        # (ctx.bridge.session_id AND ctx.session_id) so the failure
        # path leaves a coherent "no active session" state — a previous
        # version only cleared the bridge side, which left ctx.session_id
        # stale on a failed re-init.
        ctx.bridge.session_id = None
        ctx.session_id = ""

        try:
            result = ctx.bridge.call("init_scene", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))

        # Propagate bridge's new session_id and scene flavour into ctx
        # so subsequent tools (which key off ctx.session_id) see it.
        new_session_id = result.get("session_id") if isinstance(result, dict) else None
        if new_session_id:
            ctx.bridge.session_id = new_session_id
            ctx.session_id = new_session_id
        ctx.is_gaussian = bool(result.get("is_gaussian", False)) if isinstance(result, dict) else False

        return ToolResult(ok=True, body=result if isinstance(result, dict) else {})


# ---------------------------------------------------------------------------
# CloseSessionTool
# ---------------------------------------------------------------------------


class CloseSessionTool:
    """Close the active simulator session and free its resources."""

    metadata = ToolMetadata(
        name="close_session",
        category=ToolCategory.SESSION,
        description="Close the current simulation session and free resources.",
        parameters_schema={"type": "object", "properties": {}},
        allowed_task_types=_SESSION_TASK_TYPES,
        permission=PermissionLevel.DESTRUCTIVE,
        # MCP backward-compat: legacy clients hardcoded `hab_close`.
        legacy_names={"close"},
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = ctx.bridge.call("close_session", {})
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        ctx.bridge.session_id = None
        ctx.session_id = ""
        return ToolResult(ok=True, body=result if isinstance(result, dict) else {})


# ---------------------------------------------------------------------------
# NavLoopStartTool — hosts the codex P1/P2 fix
# ---------------------------------------------------------------------------


def _artifacts_dir() -> str:
    return os.environ.get(
        "NAV_ARTIFACTS_DIR",
        str(Path(__file__).resolve().parents[3] / "data" / "nav_artifacts"),
    )


class NavLoopStartTool:
    """Launch an autonomous nav_loop sub-agent.

    This is the single canonical implementation of the nav-loop
    startup logic. Both the chat-mode `ChatAgent` and the MCP
    `hab_nav_loop_start` will funnel through this class in PR 4. The
    codex P1/P2 fix lives in `_validate_goal_args` so it cannot be
    bypassed by either caller path.
    """

    metadata = ToolMetadata(
        name="nav_loop_start",
        category=ToolCategory.SESSION,
        description=(
            "Launch an autonomous navigation sub-agent. The nav agent "
            "runs independently and updates nav_status. For imagenav "
            "you MUST provide reference_image."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": [
                        "pointnav",
                        "objectnav",
                        "imagenav",
                        "instruction_following",
                        "eqa",
                    ],
                    "default": "pointnav",
                },
                "goal_description": {"type": "string"},
                "goal_type": {
                    "type": "string",
                    "enum": ["", "position", "instruction"],
                    "description": (
                        "Override auto-derivation. Normally leave empty: "
                        "pointnav → position, others → instruction."
                    ),
                },
                "nav_mode": {
                    "type": "string",
                    "enum": ["", "navmesh", "mapless"],
                    "description": "Override bridge auto-detection.",
                },
                "goal_x": {"type": "number"},
                "goal_y": {"type": "number"},
                "goal_z": {"type": "number"},
                "has_ground_truth": {"type": "boolean", "default": False},
                "success_distance_threshold": {
                    "type": "number",
                    "description": "Override success threshold in meters (0 = default)",
                },
                "max_iterations": {"type": "integer", "default": 50},
                "reference_image": {
                    "type": "string",
                    "description": "Required for imagenav",
                },
            },
        },
        allowed_task_types=_SESSION_TASK_TYPES,
        permission=PermissionLevel.MUTATING,
    )

    # ── Public entry point ───────────────────────────────────────

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        # 1. Session precondition
        if not ctx.bridge.session_id:
            return ToolResult(
                ok=False,
                body={},
                error="No active session. Call init_scene first.",
            )

        task_type = args.get("task_type", "pointnav")

        # 2. Task-type-specific preconditions
        if task_type == "imagenav" and not args.get("reference_image"):
            return ToolResult(
                ok=False,
                body={},
                error=(
                    "imagenav requires reference_image. Use look first to "
                    "capture a view, then pass its file path here."
                ),
            )

        # 3. Codex P1/P2 validation + coordinate completeness.
        # The single gate for both "pointnav must have coords" and
        # "non-pointnav must not leak coords to the agent prompt".
        goal_x = args.get("goal_x")
        goal_y = args.get("goal_y")
        goal_z = args.get("goal_z")
        # parse_bool_flag handles LLM-emitted string booleans ("false",
        # "true") correctly — a naive `bool(...)` here would treat
        # "false" as truthy and incorrectly trigger the strict GT
        # coordinate check for callers that said "no ground truth".
        err = self._validate_goal_args(
            task_type=task_type,
            goal_x=goal_x,
            goal_y=goal_y,
            goal_z=goal_z,
            has_ground_truth=parse_bool_flag(args.get("has_ground_truth", False)),
        )
        if err is not None:
            return ToolResult(ok=False, body={}, error=err)

        # 4. Build the bridge payload
        payload = self._build_payload(args, ctx, task_type, goal_x, goal_y, goal_z)

        # 5. Call the bridge
        try:
            result = ctx.bridge.call("start_nav_loop", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))

        return ToolResult(ok=True, body=result if isinstance(result, dict) else {})

    # ── Validation (codex P1/P2 fix lives here) ──────────────────

    @staticmethod
    def _validate_goal_args(
        *,
        task_type: str,
        goal_x: Any,
        goal_y: Any,
        goal_z: Any,
        has_ground_truth: bool,
    ) -> "str | None":
        """Apply the P1/P2 rules to goal coordinates.

        Returns None on success, or an error string. MUST key every
        check off `task_type`, NOT off any LLM-supplied `goal_type`.

        Rules:
          - Coords must be all-or-none: 2/3 partial is an error.
          - `task_type == "pointnav"` ALWAYS requires all three coords,
            regardless of `goal_type` (P2 regression lock).
          - `has_ground_truth=True` requires all three coords, regardless
            of task type (so callers never accidentally mark the origin
            as GT).
        """
        coords_fully_provided = (
            goal_x is not None and goal_y is not None and goal_z is not None
        )
        any_provided = (
            goal_x is not None or goal_y is not None or goal_z is not None
        )
        if any_provided and not coords_fully_provided:
            return (
                "goal_x, goal_y, and goal_z must be passed together. "
                "Either provide all three or omit all three."
            )

        if task_type == "pointnav" and not coords_fully_provided:
            return (
                "pointnav requires target coordinates. "
                "Pass goal_x, goal_y, and goal_z explicitly."
            )

        if has_ground_truth and not coords_fully_provided:
            return (
                "has_ground_truth=True requires explicit goal_x, goal_y, "
                "and goal_z. If the GT really is the origin, pass "
                "goal_x=0, goal_y=0, goal_z=0 explicitly."
            )

        return None

    # ── Payload assembly (P1 leak gate lives here) ───────────────

    @staticmethod
    def _build_payload(
        args: Dict[str, Any],
        ctx: ToolContext,
        task_type: str,
        goal_x: Any,
        goal_y: Any,
        goal_z: Any,
    ) -> Dict[str, Any]:
        # Artifact dir — writable scratch space the nav_agent will use.
        # If we cannot create it (bridge running in a container where
        # the path is read-only, permission denied, etc.), omit
        # `output_dir` from the payload entirely so the bridge picks
        # its own default. Previously this block caught the OSError
        # but still wrote the unwritable path into the payload,
        # defeating the fallback.
        artifacts_dir = ctx.output_dir or _artifacts_dir()
        artifacts_dir_writable = True
        try:
            os.makedirs(artifacts_dir, exist_ok=True)
        except OSError:
            artifacts_dir_writable = False

        goal_desc = args.get("goal_description") or f"Navigate ({task_type})"
        payload: Dict[str, Any] = {
            "session_id": ctx.bridge.session_id,
            "task_type": task_type,
            "goal_description": goal_desc,
            "max_iterations": args.get("max_iterations", 50),
        }
        if artifacts_dir_writable:
            payload["output_dir"] = artifacts_dir

        # Optional nav_mode override
        nav_mode = args.get("nav_mode") or ""
        if nav_mode:
            payload["nav_mode"] = nav_mode

        # goal_type: explicit override wins; otherwise derive from
        # task_type. Mirrors legacy mcp_server.hab_nav_loop_start.
        goal_type = args.get("goal_type", "")
        if goal_type in ("position", "instruction"):
            payload["goal_type"] = goal_type
        elif task_type == "pointnav":
            payload["goal_type"] = "position"
        else:
            payload["goal_type"] = "instruction"

        coords_fully_provided = (
            goal_x is not None and goal_y is not None and goal_z is not None
        )
        if coords_fully_provided:
            coords = [goal_x, goal_y, goal_z]
            # P1 leak gate: agent-facing goal_position ONLY for pointnav.
            # Keys off task_type, NOT goal_type.
            if task_type == "pointnav":
                payload["goal_position"] = coords
            # Evaluation-only GT: ANY task type that opts in via coords
            payload["eval_goal_position"] = coords
            payload["has_ground_truth"] = True

        # Optional success distance threshold override
        threshold = args.get("success_distance_threshold", 0) or 0
        if threshold and threshold > 0:
            payload["success_distance_threshold"] = threshold

        # Pass-through reference_image for imagenav (validated earlier)
        ref = args.get("reference_image", "")
        if ref:
            payload["reference_image"] = ref

        return payload


# ---------------------------------------------------------------------------
# NavLoopStatusTool / NavLoopStopTool
# ---------------------------------------------------------------------------


def _resolve_loop_id(args: Dict[str, Any], ctx: ToolContext) -> str:
    """Explicit `loop_id` arg wins; otherwise query the bridge for the
    most-recent active loop and fall back to the most-recent closed
    one. Mirrors the legacy `ChatAgent._resolve_loop_id` and
    `mcp_server._resolve_mcp_loop_id` verbatim."""
    lid = args.get("loop_id", "")
    if lid:
        return str(lid)
    try:
        runtime = ctx.bridge.call("get_runtime_status", {"include_nav_status": False})
    except Exception:
        return ""
    if not isinstance(runtime, dict):
        return ""
    for key in ("nav_loops", "recently_closed_nav_loops"):
        loops = runtime.get(key, [])
        if isinstance(loops, list) and loops:
            last = loops[-1]
            if isinstance(last, dict):
                loop_id_val = last.get("loop_id", "")
                if loop_id_val:
                    return str(loop_id_val)
    return ""


class NavLoopStatusTool:
    metadata = ToolMetadata(
        name="nav_loop_status",
        category=ToolCategory.SESSION,
        description=(
            "Check the status of an autonomous nav_loop sub-agent. "
            "Omit loop_id to fetch the most recent one."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "loop_id": {
                    "type": "string",
                    "description": "Loop ID (optional; empty = most recent)",
                },
            },
        },
        allowed_task_types=_SESSION_TASK_TYPES,
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        loop_id = _resolve_loop_id(args, ctx)
        if not loop_id:
            return ToolResult(
                ok=False,
                body={},
                error="No active nav loop found. Start one with nav_loop_start first.",
            )
        try:
            result = ctx.bridge.call(
                "get_nav_loop_status",
                {"loop_id": loop_id, "include_nav_status": True},
            )
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body=result if isinstance(result, dict) else {})


class NavLoopStopTool:
    metadata = ToolMetadata(
        name="nav_loop_stop",
        category=ToolCategory.SESSION,
        description=(
            "Stop an autonomous nav_loop sub-agent. Omit loop_id to "
            "stop the most recent one."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "loop_id": {
                    "type": "string",
                    "description": "Loop ID (optional; empty = most recent)",
                },
            },
        },
        allowed_task_types=_SESSION_TASK_TYPES,
        permission=PermissionLevel.DESTRUCTIVE,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        loop_id = _resolve_loop_id(args, ctx)
        if not loop_id:
            return ToolResult(
                ok=False, body={}, error="No active nav loop found."
            )
        try:
            result = ctx.bridge.call("stop_nav_loop", {"loop_id": loop_id})
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body=result if isinstance(result, dict) else {})


class SceneGraphQueryTool:
    """Query the precomputed scene graph for rooms or objects by type/label.

    The scene graph (room_object_scene_graph.json) is generated offline using
    ``tools/scene_graph/generate_room_object_scene_graph.py`` and auto-loaded when a
    scene session is initialized. Each node includes a position and bounding box,
    allowing the agent to navigate to specific rooms or object instances.
    """

    metadata = ToolMetadata(
        name="scene_graph",
        category=ToolCategory.SESSION,
        description=(
            "Query the precomputed scene graph to get 3D positions and bounding boxes "
            "of objects in the scene. Use this before navigating to a target. "
            "IMPORTANT: Room nodes have NO semantic labels (only numeric IDs). "
            "To locate a functional area such as a kitchen or bedroom, query objects "
            "typically found there — e.g. query_type='object', object_label='sink' to "
            "find the kitchen area, then check the room_id field to identify the room. "
            "The room_type parameter does NOT filter rooms by name and should be omitted."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["all", "room", "object"],
                    "description": (
                        "'object' — find objects by label (RECOMMENDED for navigation). "
                        "'room' — list all rooms with centroids and areas (no label filtering). "
                        "'all' — objects and rooms mixed; apply object_label to filter objects."
                    ),
                },
                "room_type": {
                    "type": "string",
                    "description": (
                        "NOT SUPPORTED — room nodes carry no semantic labels in this scene graph. "
                        "Omit this parameter. To find a kitchen, use query_type='object' with "
                        "object_label='sink' or 'refrigerator' and follow the room_id field."
                    ),
                },
                "object_label": {
                    "type": "string",
                    "description": (
                        "Substring match on object label (case-insensitive). "
                        "Examples: 'chair', 'sofa', 'sink', 'table', 'door'. "
                        "Used with query_type='object' or query_type='all'."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of nodes to return (default 10).",
                    "default": 10,
                },
            },
            "required": ["query_type"],
        },
        permission=PermissionLevel.READ_ONLY,
        mcp_visible=True,
        allowed_nav_modes={"navmesh"},  # SG gives absolute 3D positions — equivalent to a map
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = ctx.bridge.call("get_scene_graph", args)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        if not isinstance(result, dict):
            return ToolResult(ok=False, body={}, error="Unexpected bridge response")
        return ToolResult(ok=True, body=result)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

ToolRegistry.register(InitSceneTool())
ToolRegistry.register(CloseSessionTool())
ToolRegistry.register(NavLoopStartTool())
ToolRegistry.register(NavLoopStatusTool())
ToolRegistry.register(NavLoopStopTool())
ToolRegistry.register(SceneGraphQueryTool())


__all__ = [
    "InitSceneTool",
    "CloseSessionTool",
    "NavLoopStartTool",
    "NavLoopStatusTool",
    "NavLoopStopTool",
]
