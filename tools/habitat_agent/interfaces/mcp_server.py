#!/usr/bin/env python3
"""MCP server for habitat-gs — exposes bridge actions as MCP tools.

Runs as an independent process alongside the bridge HTTP server.
Any MCP-compatible client (Claude Code, Codex, OpenClaw, Cursor, etc.) can connect.

Usage:
    python3 tools/mcp_server.py                          # stdio transport (default)
    python3 tools/mcp_server.py --transport sse           # SSE transport
    python3 tools/mcp_server.py --transport streamable-http  # Streamable HTTP

Env vars:
    NAV_BRIDGE_HOST  — Bridge host (default: 127.0.0.1)
    NAV_BRIDGE_PORT  — Bridge port (default: 18911)
    NAV_ARTIFACTS_DIR — Artifacts directory for resource serving

Phase 2 PR 4: all 16 `hab_*` tools are now **dynamically registered**
from `ToolRegistry` at startup. There are no hand-written
`@mcp.tool()` functions any more — adding a new tool only requires
writing a `Tool` subclass under `habitat_agent.tools.*`, and it
automatically appears here as a `hab_<name>` MCP tool.

The dynamic wrapper factory (`_make_typed_wrapper`) uses the
`inspect.Signature` + `__annotations__` recipe validated by the
Phase 2 PR 1 FastMCP spike. Every wrapper:

  1. builds a fresh `ToolContext` from the global `_bridge`
  2. dispatches to `ToolRegistry.dispatch(tool_name, kwargs, ctx)`
  3. serialises the `ToolResult` back to the JSON string shape MCP
     callers expect (success → `result.body`, failure → `{"error": ...}`)

Wrappers are also stashed as module attributes (`mcp_server.hab_forward`,
etc.) so existing test fixtures that poke the functions directly
keep working.
"""

from __future__ import annotations

import base64
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from habitat_agent.runtime.bridge_client import BridgeClient
from habitat_agent.runtime.config import load_dotenv_from_project
from habitat_agent.runtime.image_io import read_image_b64

# Importing habitat_agent.tools triggers ToolRegistry.register for
# every tool as a side effect, so the registry is populated before
# we try to enumerate it below.
import habitat_agent.tools  # noqa: F401
from habitat_agent.tools.base import (
    RoundState,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    sys.exit("ERROR: mcp SDK not installed. Run: pip install mcp>=1.27.0")


# ---------------------------------------------------------------------------
# Bridge client + MCP state
# ---------------------------------------------------------------------------
# NOTE: Single-client only. All tools share one global BridgeClient.
# For multi-client transports (sse/streamable-http), per-connection
# isolation requires FastMCP to expose connection-scoped context,
# which is not yet available. Using concurrent MCP clients against
# the same server will cause session conflicts.
_bridge = BridgeClient()
_mcp_is_gaussian: bool = False  # mirrored from InitSceneTool after hab_init


def _artifacts_dir() -> str:
    return os.environ.get(
        "NAV_ARTIFACTS_DIR",
        str(Path(__file__).resolve().parents[3] / "data" / "nav_artifacts"),
    )


# ---------------------------------------------------------------------------
# MCP response shapers — restore the legacy hand-written tool shapes
# ---------------------------------------------------------------------------
#
# Phase 2 PR 4 dynamic registration originally returned `result.body`
# verbatim, which silently broke external MCP clients (and the
# `skills/habitat-gs/` reference docs) that read fields like
# `result["images"]` or `result["scene_info"]` produced by the legacy
# hand-written `@mcp.tool()` functions. The shapers below restore
# those exact shapes so any pre-Phase-2 MCP integration keeps working.
#
# Each shaper takes the raw `ToolResult.body` (which is the bridge
# response dict) and returns a dict ready to be JSON-serialised into
# the MCP tool reply. Tools without an entry in _MCP_RESPONSE_SHAPERS
# fall through to `body` unchanged (most tools don't need shaping —
# the shaping is only for the visual / session-init tools where
# legacy clients depend on a specific top-level layout).


def _shape_init_scene_response(body: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy hab_init wrapped the full bridge response under
    `scene_info` and surfaced `session_id` + `is_gaussian` at the
    top level."""
    return {
        "session_id": body.get("session_id"),
        "is_gaussian": body.get("is_gaussian", False),
        "scene_info": body,
    }


def _extract_visual_paths(body: Dict[str, Any]) -> List[str]:
    """Pull image file paths out of `visuals` (dict-of-sensor) and/or
    `images` (list-of-dicts) entries on a bridge response. Returns a
    flat ordered list of paths."""
    paths: List[str] = []
    visuals = body.get("visuals")
    if isinstance(visuals, dict):
        for sensor_data in visuals.values():
            if isinstance(sensor_data, dict):
                mp = sensor_data.get("path")
                if mp and isinstance(mp, str):
                    paths.append(mp)
    images = body.get("images")
    if isinstance(images, list):
        for img in images:
            if isinstance(img, dict):
                mp = img.get("path")
                if mp and isinstance(mp, str):
                    paths.append(mp)
    return paths


def _shape_visuals_to_image_paths(body: Dict[str, Any]) -> Dict[str, Any]:
    """For look / turn: strip the `visuals` dict and add a top-level
    `images` list of file paths.

    Legacy hab_look:
        text_result = {k: v for k, v in result.items() if k != "visuals"}
        text_result["images"] = image_paths
    """
    paths = _extract_visual_paths(body)
    out = {k: v for k, v in body.items() if k != "visuals"}
    out["images"] = paths
    return out


def _shape_panorama_response(body: Dict[str, Any]) -> Dict[str, Any]:
    """For panorama: same idea as look but the bridge returns
    images as a `list-of-dicts` under `images`, not `visuals`."""
    paths = _extract_visual_paths(body)
    out = {k: v for k, v in body.items() if k not in ("visuals", "images")}
    out["images"] = paths
    return out


def _shape_strip_visuals(body: Dict[str, Any]) -> Dict[str, Any]:
    """For forward / navigate: legacy stripped `visuals` from the top
    level (the LLM context didn't need the per-sensor dict — it could
    fetch images via the sidecar artifact resource if needed)."""
    return {k: v for k, v in body.items() if k != "visuals"}


def _shape_topdown_response(body: Dict[str, Any]) -> Dict[str, Any]:
    """For topdown: legacy collected paths from BOTH the standard
    visual sources AND the response's `topdown_map` field, then
    stripped all three keys from the top level."""
    paths = _extract_visual_paths(body)
    td = body.get("topdown_map")
    if isinstance(td, dict):
        mp = td.get("path")
        if mp and isinstance(mp, str):
            paths.append(mp)
    out = {
        k: v for k, v in body.items()
        if k not in ("visuals", "images", "topdown_map")
    }
    out["images"] = paths
    return out


_MCP_RESPONSE_SHAPERS = {
    "init_scene": _shape_init_scene_response,
    "look": _shape_visuals_to_image_paths,
    "turn": _shape_visuals_to_image_paths,
    "panorama": _shape_panorama_response,
    "forward": _shape_strip_visuals,
    "navigate": _shape_strip_visuals,
    "topdown": _shape_topdown_response,
}


def _build_mcp_context() -> ToolContext:
    """Build a fresh ToolContext for one MCP tool call.

    MCP callers are higher-level agents / humans, not nav_loop
    subprocesses, so we use `task_type="chat"` to make the 5 session
    tools (init_scene, close_session, nav_loop_*) visible. `nav_mode`
    defaults to "navmesh" because the 4 navmesh-only tools (navigate,
    find_path, sample_point, topdown) should also be exposed via MCP
    — a mapless caller can still request them, and the bridge will
    return a real error if the scene has no navmesh.

    Each call gets its own fresh `RoundState` because MCP calls are
    one-shot (no cross-call accumulation, unlike nav_agent's turn loop).
    """
    return ToolContext(
        bridge=_bridge,
        session_id=_bridge.session_id or "",
        loop_id="",
        output_dir=_artifacts_dir(),
        nav_mode="navmesh",
        task_type="chat",
        is_gaussian=_mcp_is_gaussian,
        round_state=RoundState(),
    )


# ---------------------------------------------------------------------------
# Dynamic wrapper factory (PR 1 spike recipe)
# ---------------------------------------------------------------------------

# JSON Schema types → Python types. FastMCP uses __annotations__ to
# build the MCP protocol schema, so the annotations must resolve to
# real Python types the caller can pass by kwarg.
_JSON_TYPE_TO_PYTHON = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _extract_params(tool: Tool) -> List[inspect.Parameter]:
    """Walk a Tool's parameters_schema and produce an ordered list of
    `inspect.Parameter` objects suitable for setting on a wrapper's
    `__signature__`.

    The ordering is: required params first, then optional params.
    This matters because FastMCP surfaces `required: [...]` in the MCP
    schema based on whether each parameter has a default.
    """
    schema = tool.metadata.parameters_schema or {}
    properties: Dict[str, Dict[str, Any]] = schema.get("properties", {}) or {}
    required: set = set(schema.get("required", []) or [])

    required_params: List[inspect.Parameter] = []
    optional_params: List[inspect.Parameter] = []

    for pname, pdef in properties.items():
        ptype_name = pdef.get("type", "string") if isinstance(pdef, dict) else "string"
        ptype = _JSON_TYPE_TO_PYTHON.get(ptype_name, str)

        if pname in required:
            required_params.append(
                inspect.Parameter(
                    pname,
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=ptype,
                )
            )
        else:
            default = pdef.get("default", None) if isinstance(pdef, dict) else None
            optional_params.append(
                inspect.Parameter(
                    pname,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=ptype,
                )
            )

    return required_params + optional_params


def _make_typed_wrapper(tool: Tool, mcp_name: str):
    """Build an MCP-ready wrapper function for `tool`.

    FastMCP uses `inspect.signature(fn)` + `fn.__annotations__` to
    derive the MCP protocol schema. A bare `**kwargs` wrapper
    collapses the schema to a single string param (verified by the
    PR 1 spike), so we set BOTH `__signature__` and `__annotations__`
    to the extracted parameter list. The resulting schema is
    indistinguishable from a hand-written `def hab_forward(...)`.
    """
    params = _extract_params(tool)
    tool_name = tool.metadata.name

    def _wrapper(**kwargs) -> str:
        # Drop keyword args whose value is None (optional, not
        # provided). Legacy MCP functions had concrete defaults like
        # `goal_x: float = 0`; Phase 2 Tool parameters_schema uses
        # "not in dict" to mean "omitted", so stripping None-valued
        # entries preserves that semantics through the wrapper layer.
        cleaned_args = {k: v for k, v in kwargs.items() if v is not None}
        ctx = _build_mcp_context()
        result = ToolRegistry.dispatch(tool_name, cleaned_args, ctx)
        if not result.ok:
            return json.dumps({"error": result.error or "unknown error"})

        # InitSceneTool mutates ctx.is_gaussian; mirror to the
        # module-level state so subsequent MCP calls see the flag.
        if tool_name == "init_scene":
            global _mcp_is_gaussian
            _mcp_is_gaussian = bool(ctx.is_gaussian)

        # Apply per-tool MCP response shaping. The shapers exist to
        # preserve the exact response shape that legacy hand-written
        # `@mcp.tool()` functions produced — see _MCP_RESPONSE_SHAPERS
        # docstring. Tools without an entry pass through unchanged.
        body = result.body
        shaper = _MCP_RESPONSE_SHAPERS.get(tool_name)
        if shaper is not None:
            body = shaper(body)
        return json.dumps(body, default=str)

    # Critical: both __signature__ and __annotations__ must be set
    # so FastMCP generates a real schema instead of collapsing to
    # `**kwargs`.
    _wrapper.__signature__ = inspect.Signature(
        parameters=params, return_annotation=str
    )
    _wrapper.__annotations__ = {p.name: p.annotation for p in params}
    _wrapper.__annotations__["return"] = str
    _wrapper.__name__ = mcp_name
    _wrapper.__doc__ = tool.metadata.description
    return _wrapper


# ---------------------------------------------------------------------------
# MCP Server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "habitat-gs",
    instructions=(
        "Habitat-GS navigation simulator bridge. Use these tools to "
        "control a robot navigating indoor 3D scenes."
    ),
    stateless_http=True,
)


def _register_tools_from_registry() -> int:
    """Iterate `ToolRegistry` and register every tool as an MCP tool.

    For each tool, registers:
      1. the canonical `hab_<tool.metadata.name>` entry point
      2. one `hab_<legacy_name>` alias for every entry in
         `tool.metadata.legacy_names` — this preserves backward
         compatibility for external MCP clients that hardcoded the
         pre-Phase-2 tool names (e.g. `hab_init` → now
         `hab_init_scene`).

    Returns the count of MCP-visible names (canonical + aliases) so
    callers can log / verify. Also stashes each generated wrapper on
    this module's globals as `hab_<name>` so test fixtures can poke
    them directly and so legacy imports like
    `from mcp_server import hab_init` keep working.

    Idempotent: safe to call twice (re-registration overwrites).
    """
    count = 0
    for tool in ToolRegistry.list_all():
        # Skip tools that opt out of MCP exposure (e.g.
        # UpdateNavStatusTool, which needs nav_loop-internal context
        # like loop_id and state_version_ref that no top-level MCP
        # caller can provide).
        if not tool.metadata.mcp_visible:
            continue
        canonical_name = f"hab_{tool.metadata.name}"
        names_to_register = [canonical_name] + [
            f"hab_{legacy}" for legacy in sorted(tool.metadata.legacy_names)
        ]
        for mcp_name in names_to_register:
            wrapper = _make_typed_wrapper(tool, mcp_name)
            mcp.add_tool(
                fn=wrapper,
                name=mcp_name,
                description=tool.metadata.description,
            )
            # Stash on module globals for direct-call test access.
            globals()[mcp_name] = wrapper
            count += 1
    return count


# Register tools eagerly at module import so `from mcp_server import
# hab_forward` works without having to call `main()` first.
_registered_count = _register_tools_from_registry()


# ---------------------------------------------------------------------------
# MCP Resources (stay manual — not tools)
# ---------------------------------------------------------------------------


@mcp.resource("habitat://artifacts/list")
def list_artifacts() -> str:
    """List all artifact files in the artifacts directory."""
    d = _artifacts_dir()
    if not os.path.isdir(d):
        return json.dumps({"files": [], "dir": d})
    files = sorted(os.listdir(d))
    return json.dumps({"dir": d, "count": len(files), "files": files})


@mcp.resource("habitat://artifacts/{filename}")
def read_artifact(filename: str) -> str:
    """Read an artifact file. Images return base64, JSON/text return content."""
    d = _artifacts_dir()
    path = os.path.join(d, os.path.basename(filename))  # prevent traversal
    if not os.path.isfile(path):
        return json.dumps({"error": f"File not found: {filename}"})
    ext = os.path.splitext(path)[1].lower()
    _image_mime_by_ext = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    if ext in _image_mime_by_ext:
        b64 = read_image_b64(path)
        return json.dumps(
            {
                "type": "image",
                "filename": filename,
                "data": b64,
                "mimeType": _image_mime_by_ext[ext],
            }
        )
    elif ext == ".mp4":
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return json.dumps(
            {"type": "video", "filename": filename, "data": b64, "mimeType": "video/mp4"}
        )
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    load_dotenv_from_project()

    # Re-configure bridge after dotenv load
    host = os.environ.get("NAV_BRIDGE_HOST", "127.0.0.1")
    port = int(os.environ.get("NAV_BRIDGE_PORT", "18911"))
    _bridge.base_url = f"http://{host}:{port}"

    parser = argparse.ArgumentParser(description="habitat-gs MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=18912,
        help="HTTP port for sse/streamable-http (default: 18912)",
    )
    args = parser.parse_args()

    print(
        f"habitat-gs MCP server starting (transport={args.transport}, "
        f"bridge={_bridge.base_url}, tools={_registered_count})",
        file=sys.stderr,
    )

    if args.transport in ("sse", "streamable-http"):
        # FastMCP reads host/port from self.settings, set them directly
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = args.port

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
