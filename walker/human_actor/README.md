# Mesh Human Actor

This package is an independent debug module for baked mesh humans. It does not
modify the Habitat-GS renderer.

From `walker/`, run:

```bash
python3 tools/create_state_trajectory.py \
  --out assets/humans/target_walk_idle_walk_traj.npy \
  --fps 30 \
  --start_x 0 \
  --start_y 0 \
  --start_z 2 \
  --heading 0 \
  --walk_speed 0.8

python3 tools/inspect_human_actor.py \
  --walk_clip assets/humans/walk_meters_inplace.npz \
  --trajectory assets/humans/target_walk_idle_walk_traj.npy

python3 tools/export_actor_obj_sequence.py \
  --walk_clip assets/humans/walk_meters_inplace.npz \
  --trajectory assets/humans/target_walk_idle_walk_traj.npy \
  --out_dir assets/debug_objs/target_walk_idle_walk \
  --num_frames 240 \
  --fps 30
```

Open OBJ files such as `human_0000.obj`, `human_0030.obj`, `human_0105.obj`,
and `human_0180.obj` in Blender to verify the walk-idle-walk trajectory.

## Habitat-GS RGBD Overlay

Task2 adds a minimal mesh-human RGBD overlay:

- `human_actor/rendering/camera.py`
- `human_actor/rendering/simple_mesh_renderer.py`
- `human_actor/rendering/depth_compositor.py`
- `src_python/habitat_sim/mesh_human_overlay.py`
- `src_python/habitat_sim/simulator.py`
- `tools/debug_render_mesh_human_in_habitat.py`

Integration point:

```text
Integrated mesh human overlay after GS RGB/depth sensor rendering in
Simulator.get_sensor_observations(), before returning the observation dict.
```

Scene instance config example:

```json
{
  "mesh_humans": {
    "enabled": true,
    "debug": true,
    "fps": 30,
    "actors": [
      {
        "actor_id": 1,
        "name": "target",
        "clips": {
          "walk": "assets/humans/walk_meters_inplace.npz"
        },
        "trajectory": "assets/humans/target_walk_idle_walk_traj.npy",
        "fallback_clip": "walk",
        "capsule_radius": 0.35,
        "capsule_height": 1.70
      }
    ]
  }
}
```

Run the debug renderer from `walker/`:

```bash
python3 tools/debug_render_mesh_human_in_habitat.py \
  --walk_clip assets/humans/walk_meters_inplace.npz \
  --trajectory assets/humans/target_walk_idle_walk_traj.npy \
  --out_dir outputs/mesh_human_debug
```

With a Habitat-GS scene:

```bash
python3 tools/debug_render_mesh_human_in_habitat.py \
  --scene_config /path/to/scene_instance.json \
  --walk_clip assets/humans/walk_meters_inplace.npz \
  --trajectory assets/humans/target_walk_idle_walk_traj.npy \
  --out_dir outputs/mesh_human_debug
```

Debug outputs:

- `debug_gs_rgb.png`
- `debug_human_rgb.png`
- `debug_composed_rgb.png`
- `debug_human_id_mask.png`
- `debug_depth_gs.npy`
- `debug_depth_human.npy`

If the human is not visible, check the printed projected bbox, camera position,
camera forward direction, and human/depth min-max logs.
