# Tool Reference

MCP tools for habitat-gs. Call these directly — do NOT use `exec` or CLI scripts.

## Session

| Tool | Purpose |
|---|---|
| `hab_init(scene?, depth?, semantic?)` | Load scene, create session |
| `hab_close()` | End session, free resources |

## Observation

| Tool | Purpose |
|---|---|
| `hab_look()` | Current camera view (RGB image + metrics) |
| `hab_panorama(include_depth?)` | 4-direction scan (front/left/back/right) with optional depth |
| `hab_depth_analyze()` | Obstacle distances: front_left / front_center / front_right |

## Movement

| Tool | Purpose |
|---|---|
| `hab_forward(distance_m?)` | Move forward (default 0.5m) |
| `hab_turn(direction, degrees?)` | Turn left or right (default 10°) |

## Navigation (navmesh mode only)

| Tool | Purpose |
|---|---|
| `hab_navigate(x, y, z, max_steps?)` | Auto-walk toward coordinates |
| `hab_find_path(x, y, z)` | Plan shortest path, return waypoints |
| `hab_sample_point()` | Random navigable point for exploration |
| `hab_topdown(goal_x?, goal_y?, goal_z?, show_path?)` | Overhead 2D map |

## Autonomous Navigation

| Tool | Purpose |
|---|---|
| `hab_nav_loop_start(task_type, goal_description, ...)` | Launch autonomous sub-agent |
| `hab_nav_loop_status(loop_id?)` | Check progress (only when asked) |
| `hab_nav_loop_stop(loop_id?)` | Stop a running loop |

## Export

| Tool | Purpose |
|---|---|
| `hab_export_video()` | Encode accumulated frames into mp4 |
