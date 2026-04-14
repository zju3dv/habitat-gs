# PointNav Task

Navigate to a target position specified by coordinates or natural language description.

## Input

- `--goal-type position --goal-pos X Y Z`: target coordinates
- `--goal-type instruction --goal "go to the kitchen"`: natural language (agent infers direction)

## Output

- `status: reached` when `euclidean_distance_to_goal < 0.5m`
- `status: blocked` when unable to make progress

## Starting

```bash
# Navmesh mode (uses path planner)
hab nav-loop start --task-type pointnav --mode navmesh --goal-type position --goal-pos 2.0 0.2 1.5 --goal "reach target"

# Mapless mode (polar coordinates only, no map tools)
hab nav-loop start --task-type pointnav --mode mapless --goal-type position --goal-pos 0.0 0.2 1.5 --goal "reach target" --depth
```

## Strategy

### Navmesh mode
- Use `hab navigate X Y Z --steps N` for efficient batch stepping
- Use `hab topdown --show-path` for route overview
- Use `hab look` + VLM for arrival verification
- Track `state_summary.euclidean_distance_to_goal` from responses

### Mapless mode
- Navigate using ONLY polar signals: `goal_direction_deg` (turn direction) + `euclidean_distance_to_goal` (progress, straight-line)
- No absolute coordinates available
- Use `hab depth-analyze` before moving to check obstacles
- Use `hab panorama --include-depth` when stuck or entering new areas
- Turn by ~`goal_direction_deg` to face goal, then move forward

## Common issues

- Distance increases for 3+ rounds → wrong direction, try turning more
- Repeated collisions → use depth-analyze, try different angle
- Stuck in narrow passage → panorama to find alternative route

## Capability requests

If you encounter problems beyond available tools, submit via `capability_request` in your patch:
- "I need a local obstacle map to plan around tight corridors"
- "I need a way to detect openings/doorways in the depth image"
