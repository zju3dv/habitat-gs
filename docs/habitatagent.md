---
title: HabitatAgent
parent: Habitat-GS
---

# HabitatAgent

*Part of [Habitat-GS](../README.md)*

habitat-gs includes an AI agent system (HabitatAgent) for natural-language-driven scene exploration, goal-driven navigation, and visual grounding inside 3DGS environments.

### Architecture

```
HabitatAgent TUI (chat-first, like Claude Code)
  → ChatAgent (streaming LLM + 16 bridge tools)
  → habitat_agent_server.py (HTTP bridge, port 18911)
  → nav_agent.py (autonomous navigation subagent, isolated LLM per loop)

MCP Server (mcp_server.py, 16 tools, stdio/SSE/streamable-http)
  → habitat_agent_server.py (HTTP bridge)

Agent client (Claude Code, Codex, OpenClaw, Cursor, etc.)
  → mcp_server.py (stdio/SSE/HTTP transport)
  → habitat_agent_server.py (HTTP bridge)
```

> **Note on file layout**: `tools/habitat_agent_server.py` / `tools/nav_agent.py` / `tools/mcp_server.py` are thin shims at the canonical paths. The real implementation lives under `tools/habitat_agent/` (package: `agents/`, `tools/`, `memory/`, `prompts/`, `runtime/`, `interfaces/`). You can invoke the shims or `python -m habitat_agent ...` directly; both work.

### Quick Start

```bash
# One command: TUI + bridge
python tools/habitat_agent.py

# With MCP server (for MCP client integration)
python tools/habitat_agent.py --mcp

# All services: TUI + bridge + MCP + web dashboard
python tools/habitat_agent.py --all

# Headless (bridge + MCP, no TUI — for CI/deployment)
python tools/habitat_agent.py --headless

# Bridge only
python tools/habitat_agent.py --bridge-only
```

### MCP Integration

habitat-gs exposes 16 MCP tools (scene loading, movement, observation, navigation, scene-graph query, video export). Any MCP-compatible client can connect.

**Available tools**: `hab_init_scene` (legacy alias `hab_init`), `hab_close_session` (legacy alias `hab_close`), `hab_look`, `hab_forward`, `hab_turn`, `hab_panorama`, `hab_depth_analyze`, `hab_navigate`, `hab_find_path`, `hab_sample_point`, `hab_topdown`, `hab_scene_graph`, `hab_nav_loop_start`, `hab_nav_loop_status`, `hab_nav_loop_stop`, `hab_export_video`

> `hab_scene_graph` returns room / object metadata (labels, centroids, 3D bounding boxes) for a loaded scene. It is **only registered when `nav_mode="navmesh"`** — under `nav_mode="mapless"` the tool is intentionally hidden because the scene graph provides absolute coordinates, which would defeat the point of a mapless benchmark.

#### stdio transport (recommended for local clients)

All stdio clients use the same config pattern:

```json
{
  "habitat-gs": {
    "command": "python3",
    "args": ["/path/to/habitat-gs/tools/mcp_server.py", "--transport", "stdio"]
  }
}
```

> **Prerequisite**: The bridge must be running. Start it with `python tools/habitat_agent.py --bridge-only` or let the TUI auto-start it.

| Client | Config file | Config key |
|--------|-------------|------------|
| **Claude Code** | `.claude/settings.json` | `mcpServers` |
| **Codex** | `.codex/config.json` | `mcpServers` |
| **OpenCode** | `opencode.json` | `mcpServers` |
| **Cursor** | `.cursor/mcp.json` | `mcpServers` |

#### HTTP transport (for remote/multi-client access)

```bash
# Start MCP server with HTTP transport
python tools/mcp_server.py --transport streamable-http --port 18912
# Or via unified launcher:
python tools/habitat_agent.py --mcp
```

Connect at `http://<host>:18912/mcp` with streamable-http transport.

#### OpenClaw

Register as an MCP skill (requires OpenClaw gateway ≥2026.3.31):

```bash
openclaw mcp set habitat-gs '{"command":"python3","args":["/path/to/habitat-gs/tools/mcp_server.py","--transport","stdio"]}'
```

Then install the skill for agent behavior instructions, pointing at the OpenClaw workspace directory:

```bash
bash tools/manage_habitat_gs_skill.sh install --workspace /path/to/openclaw/workspace
```

> **Docker note**: If OpenClaw runs in Docker, mount the habitat-gs artifacts directory into the container so the agent can read exported images and videos directly (no path translation needed):
> ```yaml
> volumes:
>   - /path/to/habitat-gs/data/nav_artifacts:/nav_artifacts:ro
> ```

### TUI Chat Interface

The TUI provides a chat-first terminal (like Claude Code) with streaming LLM responses:

```bash
python tools/habitat_agent_tui.py
```

- Chat view (default): type messages to interact with the agent
- Monitor view (`Ctrl+M`): Timeline / Trace / Actions / Memory tabs
- Slash commands: `/help`, `/status`, `/copy`, `/sessions`, `/resume`, `/clear`

### Live Visualization

`tools/vis/rerun_nav_viewer.py` is a [rerun](https://rerun.io/)-based viewer that attaches to a running bridge and streams multiple panels side-by-side:

- **Scene Graph 3D** — rooms and object bounding boxes (requires `nav_mode="navmesh"` and a pre-generated scene graph file)
- **RGB / Depth / Third Person** — first-person sensors plus the optional third-person camera (see [Third-Person Camera & Visual Robot](#third-person-camera--visual-robot))
- **BEV** — top-down map with the agent's trajectory and planned waypoints
- **Text panels** — recent tool calls and `nav_status` updates

Start it in a separate terminal after the bridge is up:

```bash
# Default: attach to local bridge on 18911, open rerun viewer on 9091
python tools/vis/rerun_nav_viewer.py

# Or specify bridge / port / scene explicitly
python tools/vis/rerun_nav_viewer.py \
    --bridge-url http://127.0.0.1:18911 \
    --rrd-port 9091
```

Open [http://localhost:9091](http://localhost:9091) in any rerun client. The viewer is read-only — it cannot inject actions back into the simulator; use the TUI or an MCP client to drive the agent. Configuration (entity filters, panel layout) lives in `tools/vis/config.yaml`.

Requires `rerun-sdk` from `requirements-agent.txt`.

### Third-Person Camera & Visual Robot

By default, `init_scene` loads only first-person sensors (`color`, `depth`, `semantic` as requested). If you want a third-person view (useful for recording demos or for the rerun Third Person panel), set `third_person=True` at scene init:

```python
hab_init_scene(
    scene_path="/path/to/scene.glb",
    third_person=True,            # inject a CameraSensorSpec behind-and-above the agent
)
```

The sensor is positioned 1.5 m behind and 1.2 m above the agent, with a −20° pitch down so the agent sits in the lower-center of the frame.

**Robot mesh** — for the third-person view to actually show a robot body, the bridge needs to load a visual robot model. This is off by default (so benchmarks don't pay the loading cost) and is controlled entirely by environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `HAB_VISUAL_ROBOT_ENABLE` | `0` (off) | Set `1` to attempt to attach a visual robot mesh to the agent node |
| `HAB_VISUAL_ROBOT_ROOT` | unset | Optional root directory for robot assets, joined with `URDF` / `MESH` paths if they are relative |
| `HAB_VISUAL_ROBOT_URDF` | unset | Path to a URDF file; preferred over raw mesh when both are set |
| `HAB_VISUAL_ROBOT_MESH` | unset | Path to a GLB / PLY / OBJ mesh file (fallback if no URDF is given) |
| `HAB_VISUAL_ROBOT_SCALE` | `1.0` | Uniform scale applied to the loaded asset |
| `HAB_VISUAL_ROBOT_OFFSET_Y` | `0.0` | Vertical offset applied after ground-align |
| `HAB_VISUAL_ROBOT_YAW_OFFSET_DEG` | `0.0` | Yaw correction if the asset's forward axis does not match agent forward |
| `HAB_VISUAL_ROBOT_AUTO_GROUND_ALIGN` | `1` (on) | Auto-adjust vertical position so the robot sits on the navmesh floor |
| `HAB_VISUAL_ROBOT_HIDE_FROM_FIRST_PERSON` | `1` (on) | Hide the robot mesh from the first-person sensors so it doesn't occlude them |

Example (demo recording):

```bash
HAB_VISUAL_ROBOT_ENABLE=1 \
HAB_VISUAL_ROBOT_URDF=/path/to/robot.urdf \
python tools/habitat_agent.py --all
```

Without `HAB_VISUAL_ROBOT_ENABLE=1`, or without a valid `HAB_VISUAL_ROBOT_URDF` / `HAB_VISUAL_ROBOT_MESH`, the third-person camera still renders the scene correctly but the agent itself is invisible.

### Additional dependencies for HabitatAgent

The TUI / MCP server / LLM agent / rerun visualization need a few extra packages that are NOT in the base `requirements.txt` (because core habitat-gs can be built without them). They're all listed in a dedicated file:

```bash
pip install -r requirements-agent.txt
```

Contents (as of this release):

| Package | Purpose |
|---------|---------|
| `textual` ≥1.0 | Terminal UI (TUI chat + monitor dashboard) |
| `rich` ≥13.0 | Pretty-printing used by TUI and CLI tools |
| `pyyaml` ≥6.0 | Controller prompt YAML + config parsing |
| `mcp` ≥1.0 | MCP server for LLM client integration |
| `openai` ≥1.0 | LLM API calls (OpenAI-compatible, works with Anthropic / OpenAI / self-hosted) |
| `python-dotenv` ≥1.0 | Auto-load `.env` in every entry point |
| `rerun-sdk` | Optional: needed only if you want `tools/vis/rerun_nav_viewer.py` (see [Live Visualization](#live-visualization)) |

### `.env` Configuration

All entry points (bridge, TUI, nav_agent.py, MCP server) auto-load `.env` from project root via `python-dotenv`. No manual `source .env` needed.

```bash
cp .env.example .env
# Edit .env with your LLM API key and other settings
```

Key variables of the **project-root** `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `NAV_LLM_API_KEY` | Yes | LLM API key (any OpenAI-compatible provider) |
| `NAV_LLM_BASE_URL` | No | API base URL (default: `https://api.anthropic.com/v1/`) |
| `NAV_LLM_MODEL` | No | Model name (default: `claude-sonnet-4-20250514`) |
| `NAV_ARTIFACTS_DIR` | No | Artifacts output directory (default: `data/nav_artifacts` under project root) |

There is a **second, separate** `.env` for the MCP skill installed into an agent workspace. It's created during skill install — see [Agent Skill](#agent-skill) below for the one-liner.

### Nav Loop

The `start_nav_loop` action (via `hab_nav_loop_start` MCP tool) launches the NavAgent (`tools/nav_agent.py` shim → `tools/habitat_agent/agents/nav_agent_main.py`) as a background process. Each nav loop gets its own isolated LLM conversation.

Artifacts are organized per session → per loop:

```
data/nav_artifacts/
  <session_id>/
    navloop-<loop_id>/
      nav_status.json                 # bridge-managed persisted snapshot
      nav_status.json.last-good.json  # last-known-good snapshot
      nav_status.json.loop.log        # human-readable loop log
      nav_status.json.events.jsonl    # machine-readable per-round events
      nav_status.json.trace.jsonl     # tool-call / tool-result trace
      spatial_memory.json             # SpatialMemory dump
```

```bash
# Follow the loop log of the most recent loop in real time
tail -f data/nav_artifacts/*/navloop-*/nav_status.json.loop.log

# Follow machine-readable round events
tail -f data/nav_artifacts/*/navloop-*/nav_status.json.events.jsonl

# View nav_status.json (bridge-managed persisted snapshot)
cat data/nav_artifacts/*/navloop-*/nav_status.json | python3 -m json.tool
```

> If you know the session / loop you care about, replace the glob with the exact `<session_id>/navloop-<loop_id>/` directory.

### Ground-Truth Evaluation (SPL & success rate)

For quantitative evaluation, pass the target coordinate as evaluation ground truth when starting a nav loop. GT metrics are computed automatically only for sessions where GT was provided.

**How to provide GT**:

```python
hab_nav_loop_start(
    task_type="objectnav",
    goal_description="find the door",
    # All three coordinates MUST be passed together explicitly. The
    # defaults are None (not 0), and the MCP entry point rejects
    # partial coords or a bare has_ground_truth=True without coords
    # so a forgotten parameter cannot silently target world origin.
    goal_x=-4.0, goal_y=0.2, goal_z=-1.0,
    has_ground_truth=True,                    # required when the GT is the
                                              # origin (0,0,0); otherwise
                                              # optional — GT is auto-detected
                                              # from the provided coords
    success_distance_threshold=0.2,           # optional override (default 0.5m)
)
```

**Upfront reachability check**: if the scene has a navmesh, `hab_nav_loop_start` pre-validates that the provided GT is reachable from the agent's current position. An unreachable goal raises a clear error *before* spawning the nav loop, so you never waste benchmark budget on sessions whose SPL/success can never be computed. Scenes without a navmesh (no pathfinder loaded) skip this check.

**Agent visibility** (strict isolation for benchmark validity):

| Task type | What agent sees | What agent does NOT see |
|-----------|-----------------|-------------------------|
| **pointnav** | Absolute coords (navmesh mode) OR polar `euclidean_distance_to_goal` + `goal_direction_deg` (mapless mode) | — |
| **objectnav / imagenav / instruction_following / eqa** | Only `goal_description` / `reference_image` | Coordinates, polar distance, direction — NONE |

The evaluator (bridge) always has the full GT; the *agent prompt* is what's filtered. This is enforced by keeping `session.last_goal` (agent-facing) strictly separate from `session.eval_goal` (evaluation-only) in the adapter, and by the `_debug` snapshot being invisible to the agent's tool schema.

**`nav_mode` is agent policy, not bridge capability**: `nav_mode="mapless"` only affects which tools the agent can see (no `find_path` / `navigate` / `topdown` / `sample_point`) and whether absolute coordinates appear in `state_summary`. The bridge still uses the navmesh for evaluation if one is loaded, so SPL and all GT distance metrics remain computable for `mapless` runs on navmesh-backed scenes — this is actually the standard benchmark configuration for testing "agent reaches goal without map tools".

**Metrics computed when GT is provided** (written to `session_stats.jsonl`):
- `gt_initial_geodesic_distance` (l_opt): start-of-episode walkable distance to the GT
- `gt_path_length` (l_actual): cumulative distance the agent actually walked
- `gt_end_geodesic_distance`: terminal geodesic distance to GT — *authoritative on navmesh scenes*; `null` when no pathfinder was loaded
- `gt_end_euclidean_distance`: terminal straight-line distance to GT — always available when GT is set; the coarse fallback for no-navmesh scenes
- `gt_success`: `True` iff the agent declared `status="reached"` **and** the authoritative end distance (`gt_end_geodesic_distance` on navmesh, `gt_end_euclidean_distance` otherwise) is less than `success_distance_threshold`
- `spl`: `S × (l_opt / max(l_opt, l_actual))` — standard Habitat SPL
- `has_navmesh`: tells consumers which end distance is authoritative for this session

> **Note** — the old single `gt_end_distance` field (which merged geodesic and euclidean into one number) was removed. Averaging it across a mixed benchmark run would compare heterogeneous quantities. Reports now bucket explicitly by `has_navmesh`.

**Success threshold**:
- **Default**: 0.5m for all task types (matches the shipped prompt instruction "Mark reached when distance < 0.5m").
- **Override**: pass `success_distance_threshold` to `hab_nav_loop_start` for stricter evaluation. Habitat PointNav benchmark standard is 0.2m; use `success_distance_threshold=0.2` for direct comparison.
- **Per-session**: the threshold used is recorded in `session_stats.jsonl` under `success_distance_threshold`.

**Reports** (`tools/analytics/generate_report.py`):
- Split sessions by `has_gt_goal` — only GT-evaluated sessions contribute to `gt_success` rate and SPL.
- Further split GT sessions by `has_navmesh` for end-distance averages: navmesh sessions report average **geodesic** end distance, non-navmesh sessions report average **euclidean** — the two are never merged into a single number.
- Tool-usage analysis classifies success by `gt_success` (the evaluator's verdict) for GT runs, falling back to `outcome` only for non-GT runs.

**TUI distance column** (`tools/habitat_agent_tui.py`) labels every number with its source so benchmark operators can never mistake one quantity for another:
- `5.170 g` — GT geodesic (navmesh + pathfinder, authoritative)
- `4.230 e` — GT euclidean (no navmesh, coarse)
- `3.140 wp` — the agent's most recent `find_path` waypoint distance; **not a GT distance** — shown only for non-GT live monitoring
- `-` — no distance info of any kind

### Cross-Session Analytics

Nav loops automatically collect session statistics to `session_stats.jsonl` on exit (all termination paths including SIGTERM).

```bash
# Generate a cross-session report (success rate, tool usage, failure analysis)
python3 tools/analytics/generate_report.py \
  --stats-file /path/to/workspace/artifacts/habitat-gs/session_stats.jsonl

# Or auto-detect the stats file
python3 tools/analytics/generate_report.py

# Output raw JSON instead of text report
python3 tools/analytics/generate_report.py --json
```

The report includes:
- Overall success rate and outcome distribution
- Per task_type × nav_mode breakdown
- Tool usage frequency with success/failure correlation
- Agent capability requests (aggregated from `capability_request` patch field)
- Failure pattern analysis

### Endpoints

The bridge exposes two endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/healthz` | GET | Liveness check, active session count |
| `/v1/request` | POST | All simulator operations |

### Request Format

```json
{
  "request_id": "<unique string>",
  "action": "<action name>",
  "session_id": "<from init_scene response>",
  "payload": {}
}
```

### Available Actions

| Category | Actions |
|---|---|
| Session | `init_scene`, `close_session`, `describe_api` |
| Perception | `get_scene_info`, `get_observation`, `get_visuals`, `get_panorama`, `analyze_depth`, `query_depth`, `get_metrics`, `get_runtime_status` |
| Scene Graph | `get_scene_graph` (navmesh-only) |
| Navigation | `sample_navigable_point`, `find_shortest_path`, `get_topdown_map`, `navigate_step`, `set_agent_state` |
| Execution | `step_action`, `step_and_capture` (both support compound `degrees`/`distance` params) |
| Nav Loop | `start_nav_loop`, `get_nav_loop_status`, `update_nav_loop_status`, `stop_nav_loop` |
| Export | `export_video_trace` |

The full authoritative schema is what `describe_api` returns at runtime — use that if you need to wire a new client.

### Agent Skill

The habitat-gs skill lives in [`skills/habitat-gs`](skills/habitat-gs) and provides behavior instructions for MCP-compatible agent clients (OpenClaw, Claude Code, Codex, etc.).

Use the skill manager to install/uninstall:

```bash
# Default: copy the skill files into the workspace (safe, no cross-host issues)
tools/manage_habitat_gs_skill.sh install --workspace /path/to/agent/workspace

# Alternative: symlink the skill files (only safe if the symlink target is
# visible to the agent process, e.g. not when the agent runs in Docker)
tools/manage_habitat_gs_skill.sh install --workspace /path/to/agent/workspace --mode symlink

tools/manage_habitat_gs_skill.sh status    --workspace /path/to/agent/workspace
tools/manage_habitat_gs_skill.sh uninstall --workspace /path/to/agent/workspace
```

`install` only copies or symlinks the skill files; it does **not** auto-create the per-workspace `.env`. Do that once after install:

```bash
cp /path/to/agent/workspace/skills/habitat-gs/.env.example \
   /path/to/agent/workspace/skills/habitat-gs/.env
# Then edit the copy to point at your bridge host / API key
```

### Developer Test Suite (Prompt/Asset Pipeline)

The test-suite tooling is developer-facing and lives under `tools/test_suite/` (not in the skill payload):

- scripts: `tools/test_suite/`
- assets/schema/prompts: `tools/test_suite/v1/`
- full guide: `tools/test_suite/v1/README.md`

Quick start:

```bash
# 1) Rebuild inventory and audit (skip live bridge probe)
./tools/test_suite/hab-suite prepare --skip-api --json

# 2) Regenerate prompts from inventory (default: pointnav=2, imagenav=1, eqa=5, if=9)
./tools/test_suite/hab-suite generate-prompts

# 3) Audit prompt/inventory consistency
./tools/test_suite/hab-suite audit --skip-api --json
```

### Coordinate System and Turn Direction

Habitat-sim uses a **right-handed Y-up** coordinate system (OpenGL convention), configured via `train.scene_dataset_config.json`:

```json
{
  "up": [0, 1, 0],
  "front": [0, 0, -1],
  "origin": [0, 0, 0],
  "units_to_meters": 1.0
}
```

- **+X** = right (in world space)
- **+Y** = up
- **+Z** = toward the viewer (backward in camera view)
- **Agent forward** = local **-Z** axis (matching `"front": [0, 0, -1]`)

**Heading convention** (as reported by `state_summary`):

- `heading_deg = atan2(forward_z, forward_x)` computed in the XZ plane
- Initial heading (agent facing `-Z`) is therefore **-90°** — this is not a magic value, it is just `atan2(-1, 0)` reported in degrees
- Calling `turn_right` makes the heading **increase** (e.g. -90° → 0° means the agent rotated from facing -Z to facing +X, which is a visual right turn)

### Known Limitations

- **Docker artifact access**: the bridge writes PNG frames and videos to the host filesystem. If an MCP client runs in Docker, mount the artifacts directory into the container (see the Docker note under [MCP Integration § OpenClaw](#openclaw)) so the agent can read the paths returned by `hab_export_video` and `hab_look` directly.
- **Rerun viewer is view-only**: `rerun_nav_viewer.py` streams events out of the bridge; it cannot inject actions back. Use the TUI or an MCP client to drive the agent.
- **`hab_scene_graph` requires a pre-generated SG file**: run `tools/scene_graph/generate_room_object_scene_graph.py` once per scene dataset before starting a navmesh session, otherwise `init_scene` succeeds but `hab_scene_graph` returns empty.
