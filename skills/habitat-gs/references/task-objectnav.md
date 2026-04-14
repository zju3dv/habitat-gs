# ObjectNav Task

Find a target object in the environment by exploring and using visual recognition.

## Input

- `--goal "find a chair"`: natural language description of the target object

## Output

- `status: reached` when the target object is clearly visible and within reach
- `status: blocked` when the environment has been thoroughly explored without finding the target
- `finding`: description of what was found and where

## Starting

```bash
# Navmesh mode (can use map tools for efficient exploration)
hab nav-loop start --task-type objectnav --mode navmesh --goal "find a chair"

# Mapless mode (visual exploration only)
hab nav-loop start --task-type objectnav --mode mapless --goal "find the door" --depth
```

## Strategy

### General approach
1. **Scan current area**: use `hab panorama` to see all directions
2. **Detect target**: use VLM on each image — "Is there a {target} visible? If yes, where in the frame?"
3. **If found**: navigate toward the target, verify when close
4. **If not found**: explore systematically — prioritize doorways, unexplored rooms
5. **Track progress**: use spatial_memory to remember visited areas and avoid revisiting

### Navmesh mode extras
- Use `hab topdown` for exploration overview
- Use `hab sample-point` to find frontier exploration points
- Use `hab navigate X Y Z` for efficient movement to exploration targets

### Mapless mode
- Use `hab depth-analyze` before moving
- Use `hab panorama --include-depth` to find passable directions
- Prioritize doorways and openings visible in depth/VLM analysis

## VLM prompts

Use these with the `image` tool:
- Detection: `"Is there a {target} visible? Answer YES or NO. If YES, where in the frame (left/center/right)?"`
- Room identification: `"What room is this? What objects do you see?"`
- Direction guidance: `"I'm looking for {target}. Which direction seems most promising?"`

## Common issues

- Target not in current room → explore other rooms via doorways
- Similar objects confused (e.g., cabinet vs door) → get closer, look from multiple angles
- Revisiting same areas → check spatial_memory summary, try unexplored directions

## Capability requests

- "I need more reliable object detection for the target category"
- "I need a way to systematically track which areas I have already explored"
- "I need object category labels from the scene semantic data"
