# ImageNav Task

Navigate to the viewpoint that matches a reference image.

## Input

- `--goal "find the matching viewpoint"`: description of what to match
- `--reference-image /path/to/ref.png`: reference image to compare against

### Getting the reference image

Two sources (agent decides which):
1. **User sends image via Feishu**: the image is downloaded to a local path available as `MediaPath` in the agent envelope. Use this path with `--reference-image`.
2. **User provides a file path**: pass directly via `--reference-image`.

If the user sends an image attachment, prefer that over any text-described path.

## Output

- `status: reached` when VLM rates current view similarity ≥ 8/10 vs reference
- `status: blocked` when explored thoroughly without finding a match
- `finding`: similarity score and description of the matching viewpoint

## Starting

```bash
# With reference image from Feishu
hab nav-loop start --task-type imagenav --mode navmesh --goal "find this viewpoint" --reference-image /path/from/MediaPath.png

# With reference image from file path
hab nav-loop start --task-type imagenav --mode navmesh --goal "match this kitchen view" --reference-image /path/to/reference.png
```

## Strategy

### Core loop
1. **Look and compare**: capture current view with `hab look`, then use VLM to compare with reference image
2. **Score similarity**: ask VLM "Compare this view with the reference. Similarity 1-10?"
3. **If similarity ≥ 7**: make fine adjustments (small turns, slight forward/backward)
4. **If similarity < 7**: explore — move to a different area and try again
5. **If similarity ≥ 8**: declare reached

### VLM comparison prompt

```
image(image="/path/to/reference.png", prompt="This is a reference image. Now look at my current view and rate similarity 1-10. What key elements match or differ? Which direction might lead closer to the matching viewpoint?")
```

### Exploration hints
- Use `hab panorama` to scan all directions and compare each with reference
- Use spatial_memory to avoid revisiting areas already checked
- Look for distinctive landmarks in the reference image (furniture, windows, doors) and navigate toward similar features

## Common issues

- Low similarity everywhere → the target viewpoint may be in a different room, explore further
- High similarity but not exact → fine-tune position with small movements and turns
- Reference image is ambiguous → focus on the most distinctive elements

## Capability requests

- "I need a way to compare the current view with the reference image more precisely"
- "I need a way to extract distinctive features from the reference image"
