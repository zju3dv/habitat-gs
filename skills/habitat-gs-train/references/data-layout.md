# Data layout

Everything hangs off a single shared root: **`data/scene_datasets/gs_scenes/`**. Habitat-Lab's conventional `data/datasets/` directory is **not** used by this project.

```
data/scene_datasets/gs_scenes/
├── train.scene_dataset_config.json   # registers train .gs.ply stages + per-scene navmesh
├── val.scene_dataset_config.json     # same for val
├── train/<scene>/<scene>.gs.ply (+ .navmesh, optional .mesh.ply)
├── val/<scene>/...
├── configs/                          # Hydra task configs (ddppo_*_gs_{train,eval}.yaml, vln_*.yaml)
├── episodes/                         # nav episode datasets (.json.gz)  ← generators write here
│   ├── pointnav/{train,val}/{<split>.json.gz, content/<scene>.json.gz}
│   ├── imagenav/{train,val}/...
│   ├── objectnav/{train,val}/...     (+ objectnav_interiorgs_gt/ — GT-object variant)
│   └── vln/{train,val}/<split>.json.gz   # VLN is one monolithic file, no content/ shards
└── trajectory_data/                  # VLN supervised demonstrations
    ├── vln/{annotations.json, images/...}          # StreamVLN
    └── uninavid/{nav_gs_*.json, nav_videos/*.mp4}  # Uni-NaVid
```

`scenes_dir` in every config is `data/scene_datasets`; episode `scene_id` is relative to it, e.g. `gs_scenes/train/scene04/scene04.gs.ply`.

## Per-task episode location (train/eval read these)

- PointNav: `episodes/pointnav/{train,val}/{split}.json.gz`
- ImageNav: `episodes/imagenav/{train,val}/{split}.json.gz`
- ObjectNav: `episodes/objectnav/{train,val}/{split}.json.gz`
- VLN (shared by StreamVLN & Uni-NaVid): `episodes/vln/{train,val}/{split}.json.gz`

For the RL tasks, the top-level `{split}.json.gz` is an **empty index** (`{"episodes": []}`); the real episodes live in `content/<scene>.json.gz` per scene (Habitat's standard `content/` convention — the loader auto-discovers them). VLN is the exception: one big file.

## Episode JSON schema

**PointNav / ImageNav** (identical):
```json
{"episode_id": "0",
 "scene_id": "gs_scenes/train/scene04/scene04.gs.ply",
 "start_position": [12.02, 0.22, 2.24],
 "start_rotation": [x, y, z, w],
 "goals": [{"position": [20.24, 0.70, -7.38], "radius": 0.2}],
 "info": {"geodesic_distance": 18.94}}
```
ImageNav stores no goal image — it is rendered on the fly at train time from the goal pose.

**ObjectNav** — file has extra top-level keys `category_to_task_category_id`, `category_to_scene_annotation_category_id`, `goals_by_category`. Each episode carries `"object_category"` but an **empty `"goals": []`** — the real goals (with `view_points`) are in `goals_by_category["<scene>.gs.ply_<category>"]`.

**VLN** (R2RVLN-v1) — top level `instruction_vocab` + `episodes`; each episode adds `reference_path` (waypoint list), `instruction.{instruction_text, instruction_tokens}`, `trajectory_id`.

## VLN trajectory data (supervised demos)

- StreamVLN `trajectory_data/vln/annotations.json`: list of `{"id", "video":"images/<scene>_gs_<id>", "instructions":[...], "actions":[-1,1,2,...]}` with RGB frames under `images/<...>/rgb/NNN.jpg` (640×480). Action codes: `-1` init, `0` stop, `1` forward, `2` turn-left, `3` turn-right.
- Uni-NaVid `trajectory_data/uninavid/nav_gs_<split>.json`: list of LLaVA-conversation records `{"id":"NAV_ID_gs_<scene>_<id>", "video":"nav_videos/<scene>_gs_<id>.mp4", "conversations":[{human, gpt}]}`; the `gpt` value is a space-joined action-word sequence ending in `stop`. Videos 640×480 @ 10 fps.
