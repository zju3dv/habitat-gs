# Depth Sensing

Detailed guide for depth perception capabilities.

## Sensor Specification

| Parameter | Value |
|---|---|
| Format | float32, meters |
| Resolution | 640×480 (configurable) |
| Horizontal FOV | 90° (configurable) |
| Camera intrinsics | fx=fy=320, cx=320, cy=240 |
| Pixel coordinate system | Top-left origin, u=column, v=row |
| Activation | `hab init --depth` (or enabled by default in nav-loop sessions) |

## analyze_depth

Splits the camera's forward FOV into three vertical regions:
- `front_left` ≈ forward 15°–45° to the left
- `front_center` ≈ forward ±15° (straight ahead)
- `front_right` ≈ forward 15°–45° to the right

**All three regions are within the forward-facing view.** For full 360° awareness, use `hab panorama --include-depth`.

### Response

```json
{
  "front_left":   {"min_dist": 2.3, "mean_dist": 3.1, "clear": true},
  "front_center": {"min_dist": 1.5, "mean_dist": 2.0, "clear": true},
  "front_right":  {"min_dist": 0.4, "mean_dist": 0.8, "clear": false},
  "closest_obstacle_dist": 0.4,
  "closest_obstacle_direction": "front_right",
  "recommended_action": "turn_left",
  "depth_unit": "meters",
  "clearance_threshold": 0.5
}
```

- `clear`: true if `min_dist >= clearance_threshold`
- `recommended_action`: `forward` / `turn_left` / `turn_right` / `turn_around`

### When to use

- **Before moving** in mapless mode to avoid collisions
- After a collision to find a clear direction
- When VLM scene analysis is ambiguous about obstacle distance

## query_depth

Query precise depth at specific pixel locations or regions.

### Point query

```bash
hab depth-query --points 320 240 100 200
```

Returns `depths: [2.35, 4.12]` (meters). Out-of-bounds coordinates return `null`.

### Bounding box query

```bash
hab depth-query --bbox 100 150 300 350
```

Returns `min_depth`, `max_depth`, `mean_depth`, `pixel_count`, `valid_pixel_count`.

### When to use

- Measure distance to a specific object seen in the camera image

## panorama with depth

```bash
hab panorama --include-depth
```

Returns 4 RGB images (front/right/back/left) plus depth analysis per direction. Agent rotation is automatically restored after capture.

Use this for comprehensive environment scanning when entering a new area or when stuck.
