---
name: habitat-gs
description: Control a robot in 3D Gaussian Splatting indoor environments. Load scenes, navigate, observe with RGB/depth, and export video.
---

# habitat-gs

Control a robot navigating through photo-realistic 3D indoor environments. You have MCP tools for scene loading, movement, observation, and autonomous navigation.

**IMPORTANT**: Always use MCP tools (`hab_init`, `hab_look`, `hab_forward`, etc.) directly. Do NOT use `exec` or shell commands — the MCP tools are the only supported interface.

## What you can do

- **Observe**: `hab_look` (current view), `hab_panorama` (4-direction scan with depth), `hab_depth_analyze` (obstacle distances)
- **Move**: `hab_forward` (walk forward), `hab_turn` (rotate left/right)
- **Navigate**: `hab_navigate` (auto-walk to coordinates), `hab_find_path` (plan route), `hab_topdown` (overhead map)
- **Long tasks**: `hab_nav_loop_start` (launch autonomous navigation sub-agent), `hab_nav_loop_status` (check progress), `hab_nav_loop_stop`
- **Session**: `hab_init` (load scene), `hab_close` (end session), `hab_export_video` (replay video)

## How to use

1. **Always init first**: call `hab_init` to load a scene before anything else.
2. **Look before you move**: use `hab_look` or `hab_panorama` to see the environment, then decide where to go.
3. **For simple tasks** (move forward, look around): use tools directly.
4. **For complex navigation** (find an object, reach a goal, follow instructions): use `hab_nav_loop_start` with a goal description. It launches an autonomous sub-agent that navigates independently. Check progress with `hab_nav_loop_status` only when asked.
5. **Export video** when the task is done: `hab_export_video` encodes all captured frames into an mp4.

## Navigation modes

The scene may run in two modes (auto-detected):

| | **navmesh** | **mapless** |
|---|---|---|
| Map tools (`hab_navigate`, `hab_find_path`, `hab_topdown`) | available | not available |
| Coordinates visible | yes | no — only `euclidean_distance_to_goal` and `goal_direction_deg` |
| How to navigate | use `hab_navigate` for efficient multi-step movement | use `hab_forward`/`hab_turn` with depth checks |

## Task types for nav_loop

| Type | What it does |
|---|---|
| `pointnav` | Navigate to a target coordinate |
| `objectnav` | Find a specific object in the scene |
| `imagenav` | Find the viewpoint matching a reference image |
| `eqa` | Explore and answer a question about the environment |
| `instruction_following` | Follow step-by-step natural language instructions |

Examples:
- PointNav: `hab_nav_loop_start(task_type="pointnav", goal_x=1.0, goal_y=0.2, goal_z=-3.0)`
- ObjectNav: `hab_nav_loop_start(task_type="objectnav", goal_description="find the kitchen door")`
- EQA: `hab_nav_loop_start(task_type="eqa", goal_description="Is there a microwave in this scene?")`
- ImageNav: **requires reference_image** — see below

### ImageNav workflow

ImageNav needs a reference image to match. You must capture it first:

1. Call `hab_look` — note the image **path** from the result (e.g. `"/path/to/session_step000001_color_sensor.png"`)
2. Move the robot away: `hab_forward`, `hab_turn`
3. Start imagenav with the saved path: `hab_nav_loop_start(task_type="imagenav", goal_description="match the reference view", reference_image="/path/to/session_step000001_color_sensor.png")`

If you skip `reference_image`, the sub-agent will not know what view to look for.

## Sending results to users

Use the `media` parameter in the `message` tool to send images or videos:

```
message(action="send", message="Here is the navigation replay", media="/path/to/video.mp4", filename="replay.mp4")
```

Do NOT paste file paths as plain text — always use the `media` parameter.

## Troubleshooting

| Problem | Solution |
|---|---|
| `Connection refused` | Bridge is not running — ask the user to start it |
| `Unknown session_id` | Session expired — call `hab_init` again |
| `ESP_CHECK failed` | Scene failed to load — try a different scene |
