# Navigation Modes

habitat-gs supports two navigation modes that define what tools the agent can use.

## navmesh mode

Full map-planning tools available. Use when `nav_mode=navmesh`.

### Available tools
- `navigate X Y Z --steps N`: greedy path follower
- `find-path X Y Z`: shortest path query
- `topdown`: 2D overhead map
- `sample-point`: random navigable point
- All perception tools (look, depth-analyze, panorama)
- All movement tools (forward, turn)
- `state_summary` includes absolute `position` and `goal` coordinates

### Strategy
1. Use `navigate` for efficient batch stepping toward goal
2. Use `topdown --show-path` for route overview
3. Use `look` + VLM for arrival verification
4. Read `euclidean_distance_to_goal` from `state_summary` to track progress

## mapless mode

Map-planning tools forbidden. Use when `nav_mode=mapless`.

### Forbidden tools
- `navigate` (greedy path follower requires navmesh)
- `find-path` (shortest path query)
- `topdown` (overhead map)
- `sample-point` (random navigable point)

### Available tools
- All perception tools (look, depth-analyze, panorama, depth-query)
- All movement tools (forward, turn)
- `state_summary` includes `goal_direction_deg` and `euclidean_distance_to_goal` but NOT absolute coordinates

### Key constraint: no GT coordinates

In mapless mode, the agent has **zero access to absolute ground-truth coordinates**:

| Channel | Filtered? |
|---|---|
| `state_summary.position` | Hidden |
| `state_summary.goal` | Hidden |
| `metrics.agent_state.position` | Hidden |
| `metrics.current_goal` | Hidden |
| nav_loop prompt goal line | Polar only |
| nav_loop action history | No `pos` field |

The only navigation signals are:
- `goal_direction_deg`: relative bearing to goal [-180, 180]. 0=ahead, +right, -left
- `euclidean_distance_to_goal`: straight-line distance in meters
- Visual perception (VLM scene understanding, depth sensing)
- Collision feedback (`COLLIDED: true/false`)

### Strategy
1. Read `goal_direction_deg` → turn to face goal
2. Use `depth-analyze` to check for obstacles
3. If clear: move forward. If blocked: adjust direction
4. Read `euclidean_distance_to_goal` after movement — should decrease
5. If distance increases for 3+ rounds: try a different direction
6. Use `panorama --include-depth` when entering new areas or stuck

## Mode selection

- `hab nav-loop start --mode navmesh`: force navmesh mode
- `hab nav-loop start --mode mapless`: force mapless mode
- Omit `--mode`: auto-detect based on scene navmesh availability

Even if a scene has a navmesh, `--mode mapless` forbids the agent from using map tools. This is the standard setting for embodied navigation research.
