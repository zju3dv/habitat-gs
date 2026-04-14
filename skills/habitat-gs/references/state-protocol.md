# State Protocol

nav_status.json patch protocol and field definitions for nav-loop orchestration.

## Patch submission flow

The nav agent (`nav_agent.py`) handles state updates automatically via the `update_nav_status` tool call. The bridge validates the patch, increments `state_version`, and atomically writes the updated `nav_status.json`.

For external monitoring:
- TUI: `Ctrl+M` → monitor panel shows live nav_status
- MCP: `hab_nav_loop_status(loop_id)`

## Mutable fields

| Field | Type | Description |
|---|---|---|
| `substeps` | array | Plan decomposition steps |
| `current_substep_index` | int | Which substep is active |
| `status` | string | `in_progress` / `reached` / `blocked` / `error` / `timeout` |
| `nav_phase` | string | `decomposing` / `navigating` / `approaching` / `verifying` / ... |
| `total_steps` | int | Cumulative simulator steps |
| `collisions` | int | Cumulative collision count |
| `geodesic_distance` | float | Distance to goal (navmesh mode) |
| `rooms_discovered` | array | Room labels visited |
| `last_visual` | object | `{path, summary}` — most recent visual output |
| `last_action` | object | `{action, collided, position, saw}` — most recent action result |
| `action_history_append` | array | Append-only action log entries |
| `spatial_memory_append` | array | Environment snapshot entries (written to spatial memory file) |
| `spatial_memory_file` | string | Path to spatial memory JSON |
| `finding` | any | Final task result/report |
| `error` | string | Error message |
| `capability_request` | string | Agent's request for a missing capability |

## Immutable fields (do not include in patch)

`task_id`, `task_type`, `nav_mode`, `has_navmesh`, `goal_type`, `goal_description`, `goal_position`, `target_object`, `reference_image`, `session_id`, `state_version`, `updated_at`

## action_history_append entry format

```json
{
  "step": 12,
  "action": "move_forward",
  "collided": false,
  "perception": "Open corridor ahead with doorway on the left. Depth shows 3.2m clearance.",
  "analysis": "The doorway likely leads to a hallway. Worth exploring for the target.",
  "decision": "Turn left 45° to face the doorway, then move forward."
}
```

- `perception` (or `saw`) is required for mapless motion patches (visual-first validation)
- `step`, `action`, `collided`, `pos` are auto-populated by bridge normalizer if omitted

## spatial_memory_append entry format

```json
{
  "heading_deg": 90.0,
  "scene_description": "Kitchen area with dining table",
  "room_label": "kitchen",
  "objects_detected": ["table", "chair", "microwave"]
}
```

Bridge writes entries atomically to `spatial_memory_{loop_id}.json` and auto-maintains `rooms{}` and `object_sightings{}` aggregation.

## Mapless motion validation

For mapless mode patches where `total_steps` increased or a movement action was appended, bridge requires:
1. Non-empty `last_visual.path` (auto-injected by ToolExecutor from recent look/panorama)
2. At least one `action_history_append` entry with non-empty `perception` or `saw`
3. At least one `spatial_memory_append` entry

## Optimistic concurrency

`expected_version` must match current `state_version`. If another update happened between read and write, the patch is rejected with a version conflict error. Re-read state and retry.

## capability_request

If the agent encounters a problem it cannot solve with available tools, it can submit a natural language request:

```json
{"capability_request": "I need door detection to find room exits"}
```

These requests are aggregated across sessions for developer analysis.
