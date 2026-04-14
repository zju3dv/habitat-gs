"""Status tools — `update_nav_status`, `export_video`.

Phase 2 PR 2. These are the tools the nav_agent uses to report
progress / terminate a loop / produce the final video trace.

`UpdateNavStatusTool` is the most complex migration in the entire
Phase 2 refactor because it has three layers of legacy behaviour that
MUST be preserved exactly:

  1. **Flat-args to nested patch coercion.** The LLM often sends patch
     fields flat at the top level instead of under a `patch` key. The
     legacy executor normalises by taking every arg except "patch"
     and rebuilding a `patch` dict. We do the same.

  2. **Auto-injection of round context.** Every action_history entry
     inherits `collided` from the most recent movement and an
     `action` chain built from `ctx.round_state.round_actions`
     (e.g. "forward(0.5m) → turn_left(45°) → look"). Mapless mode
     additionally requires `last_visual` on motion patches, so we
     auto-fill it from `ctx.round_state.last_visual_path` if the LLM
     didn't provide one.

  3. **Terminal-status synthesis.** When the LLM sets a terminal status
     (reached / blocked / error / timeout) without supplying any
     action_history_append, the bridge's mapless validation would
     reject the patch. The legacy executor synthesises a minimal
     one-entry action_history_append from the `finding` field so the
     terminal transition still goes through. We preserve this
     synthesis behaviour verbatim.

After the bridge call succeeds, the response's `state_version` is
written back to `ctx.state_version_ref[0]` so the next round's
`expected_version` matches what the bridge now holds. The list-of-one
indirection mirrors the legacy pattern exactly and keeps the nav_agent
read site unchanged (it still dereferences `ref[0]`).

Finally, `round_actions` is cleared so the next cycle starts fresh.
This clearing is the reason `update_nav_status` must not be a purely
read-only tool from the RoundState perspective — it is the boundary
where per-round state is committed and reset.
"""

from __future__ import annotations

from typing import Any, Dict, List

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


_TERMINAL_STATUSES = ("reached", "blocked", "error", "timeout")


# ---------------------------------------------------------------------------
# UpdateNavStatusTool
# ---------------------------------------------------------------------------


class UpdateNavStatusTool:
    """Commit a batch of nav-status updates, advance state_version,
    and close out the current round by clearing `round_actions`."""

    metadata = ToolMetadata(
        name="update_nav_status",
        category=ToolCategory.STATUS,
        description=(
            "Update navigation status. Call after movements to record "
            "progress, action history, and spatial memory."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["in_progress", "reached", "blocked", "error", "timeout"],
                    "description": "Navigation status",
                },
                "nav_phase": {
                    "type": "string",
                    "description": "Current phase: decomposing, navigating, exploring, verifying",
                },
                "total_steps": {
                    "type": "integer",
                    "description": "Total movement steps taken so far",
                },
                "collisions": {
                    "type": "integer",
                    "description": "Total collision count",
                },
                "last_action": {
                    "type": "string",
                    "description": "Last action performed (e.g. move_forward, turn_left, panorama)",
                },
                "action_history_append": {
                    "type": "array",
                    "description": (
                        "Entries to append. Each must have: perception "
                        "(what you see), analysis (interpretation), "
                        "decision (what you chose and why)"
                    ),
                    "items": {"type": "object"},
                },
                "spatial_memory_append": {
                    "type": "array",
                    "description": (
                        "Spatial memory entries: {heading_deg, "
                        "scene_description, room_label, objects_detected}"
                    ),
                    "items": {"type": "object"},
                },
                "substeps": {
                    "type": "array",
                    "description": "Task decomposition substeps",
                    "items": {"type": "object"},
                },
                "current_substep_index": {
                    "type": "integer",
                    "description": "Index of current substep being executed",
                },
                "finding": {
                    "type": "string",
                    "description": "Final finding/answer when task is complete",
                },
                "error": {
                    "type": "string",
                    "description": "Error description if status=error",
                },
                "rooms_discovered": {
                    "type": "array",
                    "description": "List of discovered room names/labels",
                    "items": {"type": "string"},
                },
                "last_visual": {
                    "type": "object",
                    "description": (
                        "Last captured visual: {path: '/path/to/image.png'}. "
                        "Required for mapless mode."
                    ),
                },
                "geodesic_distance": {
                    "type": "number",
                    "description": "Geodesic distance to goal (navmesh mode only)",
                },
            },
        },
        permission=PermissionLevel.MUTATING,
        # nav_loop-internal: requires ctx.loop_id and ctx.state_version_ref
        # which only NavAgent subprocesses can provide. Excluded from
        # the dynamic MCP registration so the MCP tool inventory does
        # not advertise a tool that would always fail with a cryptic
        # 'loop_id must be non-empty' bridge error.
        mcp_visible=False,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        patch = self._normalize_patch(args)
        self._inject_last_visual(patch, ctx)
        action_chain = self._build_action_chain(ctx)
        self._inject_action_chain_into_history(patch, action_chain, ctx)
        self._synthesize_terminal_history(patch, action_chain, ctx)

        # Clear per-round state BEFORE the bridge call: if the bridge
        # raises, the round is still considered "closed" from the
        # agent's perspective because the patch was assembled. This
        # matches legacy semantics.
        ctx.round_state.round_actions = []

        try:
            result = ctx.bridge.call(
                "update_nav_loop_status",
                {
                    "loop_id": ctx.loop_id,
                    "patch": patch,
                    "expected_version": ctx.state_version_ref[0],
                },
            )
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))

        nav_status = result.get("nav_status", {}) if isinstance(result, dict) else {}
        new_ver = nav_status.get("state_version")
        if isinstance(new_ver, int):
            ctx.state_version_ref[0] = new_ver

        return ToolResult(
            ok=True,
            body={"ok": True, "state_version": ctx.state_version_ref[0]},
        )

    # ── Helpers (private) ────────────────────────────────────────

    @staticmethod
    def _normalize_patch(args: Dict[str, Any]) -> Dict[str, Any]:
        """LLMs often send patch fields flat instead of nested under
        `patch`. Accept both shapes; the nested form wins if present."""
        patch = args.get("patch")
        if isinstance(patch, dict):
            return dict(patch)  # defensive copy
        return {k: v for k, v in args.items() if k != "patch"}

    @staticmethod
    def _inject_last_visual(patch: Dict[str, Any], ctx: ToolContext) -> None:
        """Mapless mode requires last_visual on motion patches. Inject
        from RoundState if the LLM didn't provide one."""
        if "last_visual" not in patch and ctx.round_state.last_visual_path:
            patch["last_visual"] = {"path": ctx.round_state.last_visual_path}

    @staticmethod
    def _build_action_chain(ctx: ToolContext) -> str:
        """Compose the human-readable action chain for this round."""
        if not ctx.round_state.round_actions:
            return ""
        # round_actions holds strings in the legacy model; keep the
        # same " → " separator.
        return " → ".join(str(a) for a in ctx.round_state.round_actions)

    @staticmethod
    def _inject_action_chain_into_history(
        patch: Dict[str, Any], action_chain: str, ctx: ToolContext
    ) -> None:
        """Fill in `collided` / `action` defaults on each
        action_history_append entry so the LLM doesn't have to repeat
        information the executor already has."""
        ah = patch.get("action_history_append")
        if not isinstance(ah, list):
            return
        for entry in ah:
            if not isinstance(entry, dict):
                continue
            if "collided" not in entry:
                entry["collided"] = ctx.round_state.last_collided
            if "action" not in entry and action_chain:
                entry["action"] = action_chain

    @staticmethod
    def _synthesize_terminal_history(
        patch: Dict[str, Any], action_chain: str, ctx: ToolContext
    ) -> None:
        """If the LLM set a terminal status without supplying
        action_history_append, synthesize a minimal one so the bridge's
        mapless validation accepts the patch."""
        status_val = patch.get("status", "")
        if status_val not in _TERMINAL_STATUSES:
            return
        existing = patch.get("action_history_append")
        if isinstance(existing, list) and existing:
            return  # LLM already provided entries — don't clobber
        finding = patch.get("finding") or status_val
        patch["action_history_append"] = [
            {
                "perception": finding,
                "analysis": f"Terminal status: {status_val}",
                "decision": f"Set status={status_val}",
                "action": action_chain or "terminal",
                "collided": ctx.round_state.last_collided,
            }
        ]


# ---------------------------------------------------------------------------
# ExportVideoTool
# ---------------------------------------------------------------------------


class ExportVideoTool:
    """Export the accumulated frame trace as an mp4 video file."""

    metadata = ToolMetadata(
        name="export_video",
        category=ToolCategory.STATUS,
        description="Export video trace of the navigation session.",
        parameters_schema={"type": "object", "properties": {}},
        permission=PermissionLevel.READ_ONLY,
    )

    def execute(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        payload: Dict[str, Any] = {}
        payload.update(visual_payload(ctx))
        try:
            result = ctx.bridge.call("export_video_trace", payload)
        except Exception as exc:
            return ToolResult(ok=False, body={}, error=str(exc))
        return ToolResult(ok=True, body=result)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

ToolRegistry.register(UpdateNavStatusTool())
ToolRegistry.register(ExportVideoTool())


__all__ = ["UpdateNavStatusTool", "ExportVideoTool"]
