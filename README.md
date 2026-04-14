<div align="center">

<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/gs_assets/logo_black.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/gs_assets/logo_white.png">
    <img alt="Habitat-GS" src="docs/gs_assets/logo_black.png" width="50%">
  </picture><br>
  A High-Fidelity Navigation Simulator with Dynamic Gaussian Splatting
</h1>

<div align="center">
  <a href=""><img src='https://img.shields.io/badge/arXiv-Habitat--GS-red' alt='Paper PDF'></a>
  <a href='https://zju3dv.github.io/habitat-gs/'><img src='https://img.shields.io/badge/Project_Page-Habitat--GS-green' alt='Project Page'></a>
  <a href="https://huggingface.co/datasets/RukawaY/gs_scenes"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-GS_Dataset-blue' alt='Hugging Face'></a>
</div>
  
<p align="center">
  <a href="https://ziyuan-xia.com">Ziyuan Xia</a> •
  <a href="https://github.com/echo636">Jingyi Xu</a> •
  <a href="https://github.com/Kinchite17">Chong Cui</a> •
  <a href="https://yuanhongyu.xyz">Yuanhong Yu</a> •
  <a href="https://jzhzhang.github.io">Jiazhao Zhang</a> •
  <a href="https://yanqswhu.top">Qingsong Yan</a> •
  <a href="https://orcid.org/0000-0002-8676-6546">Tao Ni</a> <br>
  <a href="https://scholar.google.com/citations?user=4YOIYGwAAAAJ&hl=en">Junbo Chen</a> •
  <a href="https://xzhou.me">Xiaowei Zhou</a> •
  <a href="http://www.cad.zju.edu.cn/home/bao/">Hujun Bao</a> •
  <a href="https://csse.szu.edu.cn/staff/ruizhenhu/">Ruizhen Hu</a> •
  <a href="https://pengsida.net/">Sida Peng</a>
</p>

</div>

https://github.com/user-attachments/assets/08a775d7-0e6c-49b7-a740-9d131e6122b2

## 📢 News

> **[2026-04]** 🎉 Paper, project page, code and dataset of Habitat-GS are released! Check it out!

## 🧭 What Is Habitat-GS and What Can Habitat-GS Do?

`Habitat-GS` is a non-intrusive extension of [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) for embodied navigation tasks in Gaussian Splatting scenes. It keeps [Habitat](https://aihabitat.org)'s standard scene dataset abstraction, NavMesh/pathfinding, agent control, and [Habitat-Lab]((https://github.com/facebookresearch/habitat-lab)) integration, while extending the rendering backbone to support [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), and incorporating a dynamic gaussian avatar module to drive humanoid gaussian avatars.

In practice, Habitat-GS can:

- render photo-realistic GS scenes with Habitat RGB and depth sensors;
- support driving dynamic Gaussian avatars from [GaussianAvatar](https://github.com/aipixel/GaussianAvatar) and [AnimatableGaussians](https://github.com/lizhe00/AnimatableGaussians) in simulation environments;
- plug into Habitat-Lab for training and evaluation with the same scene dataset format.

Compared with traditional mesh-based simulators, Habitat-GS can achieve photo-realistic rendering and render high-fidelity gaussian avatars with high efficiency. By introducing Gaussian Splatting to embodied simulators, we hope our work can facilitate future embodied AI research.

## 🛠️ Install Habitat-GS

### 🧪 Create the environment

```bash
conda create -n habitat-gs python=3.12 cmake=3.27
conda activate habitat-gs

# IMPORTANT: Install CUDA-compatible torch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 📦 Install Habitat-GS

```bash
git clone https://github.com/zju3dv/habitat-gs.git
cd habitat-gs

# Recommended: CUDA on, Bullet off
HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=OFF pip install -e .
```

If you also need Bullet physics (e.g. manipulate mesh objects in a 3DGS scene), install with:

```bash
HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=ON pip install -e .
```

### 🤖 Install Habitat-Lab (optional but recommended)

Habitat-GS can be used standalone for rendering and scene inspection, but Habitat-Lab is typically used along with Habitat-GS for navigation task definition, training, and evaluation.

```bash
git clone https://github.com/facebookresearch/habitat-lab.git
```

**IMPORTANT**: Before installing Habitat-Lab into the same environment, update its NumPy pin to avoid conflicts with Habitat-GS:

- edit `habitat-lab/habitat-lab/requirements.txt`
- change `numpy==1.26.4` to `numpy>=2.0.0,<2.4`

Then install:

```bash
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```

## 📦 Download GS Asset

Please refer to our [🤗 huggingface dataset](https://huggingface.co/datasets/RukawaY/gs_scenes) for more details. We provide five categories of assets:

|   | Category | Size | Required For |
|---|----------|------|-------------|
| 1 | **GS Scenes** | ~7.2 GB | Everything — core scene assets |
| 2 | **Gaussian Avatars** | ~3.1 GB | Dynamic avatar simulation |
| 3 | **Habitat-Lab Nav Data** | ~11 MB | PointNav / ImageNav / ObjectNav training & evaluation |
| 4 | **StreamVLN Data** | ~50 GB | VLN training & evaluation ([StreamVLN](https://github.com/InternRobotics/StreamVLN)) |
| 5 | **Uni-NaVid Data** | ~21 GB | VLN training & evaluation ([Uni-NaVid](https://github.com/jzhzhang/Uni-NaVid)) |

## 🚀 Run Habitat-GS

### 🧱 Prepare GS Assets

Habitat-GS requires two categories of assets: GS scenes and GS avatars.

<details>
<summary>Click to expand: GS scene assets</summary>

For a static GS scene without avatars, you only need:

- a 3DGS render asset;
- a Habitat-format `.navmesh` file for navigation. It defines the walkable area for agents.

**IMPORTANT**: Habitat-GS recognizes GS stage assets by suffix. This means your scene file MUST end with `.gs.ply` or `.3dgs.ply`.

If you already have a high-quality NavMesh, you can use it directly. If not, we recommended the following pipeline to generate one from your GS scene:

1. convert the GS scene to collision mesh with [3DGS-to-PC](https://github.com/Lewis-Stuart-11/3DGS-to-PC) or other methods;
2. generate a Habitat NavMesh from that collision mesh with `tools_gs/generate_navmesh.py`.

Minimal example:

```bash
conda activate habitat-gs
python tools_gs/generate_navmesh.py \
  --input /path/to/scene_collision_mesh.ply \
  --output /path/to/scene.navmesh
```

</details>

<details>
<summary>Click to expand: GS avatar assets</summary>

Every gaussian avatar needs two assets:

- `canonical_gs.npz`: canonical gaussians exported from either [GaussianAvatar](https://github.com/aipixel/GaussianAvatar) or [AnimatableGaussians](https://github.com/lizhe00/AnimatableGaussians);
- `driver.pkl`: scene-specific motion driver generated on the NavMesh using [GAMMA](https://github.com/yz-cnsdqz/GAMMA-release) method.

1.1. Export canonical gaussians from GaussianAvatar

Run this in the `GaussianAvatar` conda environment after installing the upstream repo:

```bash
python tools_gs/export_gaussian_avatar_to_canonical.py \
  --posmap /path/to/query_posemap_.npz \
  --lbs-map /path/to/lbs_map_.npy \
  --joint-mat /path/to/smpl_cano_joint_mat.pth \
  --net-ckpt /path/to/net.pth \
  --out /path/to/canonical_gs.npz \
  --ga-root /path/to/GaussianAvatar
```

1.2. Export canonical gaussians from AnimatableGaussians

Similarly, run this in the `AnimatableGaussians` conda environment after installing the upstream repo:

```bash
python tools_gs/export_animatable_to_canonical.py \
  --config /path/to/config.yaml \
  --ckpt /path/to/checkpoints \
  --out /path/to/canonical_gs.npz \
  --anim-root /path/to/AnimatableGaussians \
  --smpl-model-path /path/to/AnimatableGaussians/smpl_files/smplx
```

2. Generate the motion driver on a scene NavMesh

`driver.pkl` depends on the target scene because it is generated on the scene NavMesh. Run the following command in the `GAMMA` environment after installing GAMMA and make sure `habitat_sim` is importable in that environment. We provide two modes for generating the driver trajectory:

Auto-sample a path by target length:

```bash
python tools_gs/generate_trajectory.py \
  --navmesh /path/to/scene.navmesh \
  --output /path/to/driver.pkl \
  --path-length 6.0 \
  --smpl-model-path /path/to/smpl_files/smplx \
  --gamma-root /path/to/GAMMA-release
```

Specify start/end/several optional via points explicitly:

```bash
python tools_gs/generate_trajectory.py \
  --navmesh /path/to/scene.navmesh \
  --output /path/to/driver.pkl \
  --start 0.0 0.0 0.0 \
  --via 1.0 0.0 -1.0 \
  --end 2.0 0.0 -2.0 \
  --smpl-model-path /path/to/smpl_files/smplx \
  --gamma-root /path/to/GAMMA-release
```

The generated `driver.pkl` contains precomputed `joint_mats` for rendering. Avatar rendering uses explicit Gaussians + CUDA LBS without neural forward pass at runtime. The `.pkl` also contains precomputed `proxy_capsules` used for NavMesh-level dynamic obstacle handling, guaranteeing agent cannot pass through gaussian avatars.

</details>

### 🗂️ Organize Scene Dataset

Habitat-GS follows Habitat's standard dataset hierarchy:

`scene_dataset_config.json` → `scene_instance.json` → `stage_config.json`

<details>
<summary>Click to expand: recommended dataset layout</summary>

```unicode
playroom/
├── playroom.scene_dataset_config.json
├── configs/
│   ├── scenes/
│   │   └── playroom.scene_instance.json
│   └── stages/
│       └── playroom_stage.stage_config.json
├── stages/
│   └── playroom.gs.ply
├── navmeshes/
│   └── playroom.navmesh
└── avatars/
    └── actor01/
        ├── canonical_gs.npz
        ├── driver.pkl
        └── smplx/
```

`playroom.scene_dataset_config.json`

```json
{
  "stages": {
    "paths": {
      ".json": ["configs/stages"]
    }
  },
  "scene_instances": {
    "paths": {
      ".json": ["configs/scenes"]
    }
  },
  "navmesh_instances": {
    "playroom_navmesh": "navmeshes/playroom.navmesh"
  }
}
```

`configs/stages/playroom_stage.stage_config.json`

```json
{
  "render_asset": "../../stages/playroom.gs.ply"
}
```

`configs/scenes/playroom.scene_instance.json`

```json
{
  "stage_instance": {
    "template_name": "playroom_stage"
  },
  "navmesh_instance": "playroom_navmesh",
  "time_max": 20.0,
  "time_loop": true,
  "gaussian_avatars": [
    {
      "name": "actor01",
      "canonical_gaussians": "../../avatars/actor01/canonical_gs.npz",
      "driver": "../../avatars/actor01/driver.pkl",
      "smpl_model_path": "../../avatars/actor01/smplx",
      "smpl_type": "smplx",
      "scale": 1.0,
      "offset_y": 1.0,
      "time_begin": 0.0,
      "time_end": 20.0
    }
  ]
}
```

Notes:

- If you only want a static GS scene, simply omit `gaussian_avatars`.
- If your GS exporter already stores normalized quaternions and you observe spikes or blur in rendering results, add `"norm_quaternion": false` to the stage config.
- Time related fields explanation:
  - `time_max`: maximum simulation time in seconds. max(time_end) of all avatars by default. 0 if no avatars in scene.
  - `time_loop`: whether to loop the simulation time. True by default.

  For each avatar, you can also specify:
  - `time_begin`: the simulation time when the avatar appears. 0 by default.
  - `time_end`: the simulation time when the avatar disappears. 0.025 * num_frames by default. If time_end > time_begin + 0.025 * num_frames, the avatar will be static at the destination in remaining time.

</details>

### 🖥️ Run Interactive Viewer

`examples/gaussian_viewer.py` is an interactive RGB/depth viewer for Habitat-GS scenes. A display is required to run the viewer (for example, a local desktop session, X11 session, or VNC session). We provide two modes to run it:

Quickly preview a GS scene:

```bash
python examples/gaussian_viewer.py --input /path/to/playroom.gs.ply
```

Run a full Habitat scene dataset with GS stage + NavMesh + gaussian avatars:

```bash
python examples/gaussian_viewer.py \
  --dataset /path/to/playroom.scene_dataset_config.json \
  --scene playroom
```

<details>
<summary>Click to expand: useful flags</summary>

- `--width` / `--height` to change window size.
- `--time` to start from a specific Gaussian time;
- `--time-rate` to change playback speed;
- `--enable-physics` to enable physics simulation if you need physics interaction with mesh objects in a GS scene. Requires building with Bullet.

</details>

<details>
<summary>Click to expand: useful viewer controls</summary>

- `W/S`: Move forward/backward
- `A/D`: Move left/right
- `Z/X`: Move up/down
- `Arrow keys`: Rotate view
- `TAB`: switch between RGB and depth
- `SPACE`: play/pause Gaussian time
- `H`: print help
- `[` / `]`: scrub backward/forward
- `N`: toggle NavMesh visualization
- `ESC`: exit viewer

</details>

## 🦞 HabitatAgent

`HabitatAgent` is an LLM-powered agent system built on top of Habitat-GS, enabling
natural-language navigation, MCP tool integration, and interactive scene exploration
via a terminal chat interface.

Key features: TUI chat, 16 MCP bridge tools, autonomous nav loops, scene-graph
query, SPL evaluation, rerun live visualization, third-person camera with
optional visual robot mesh, multi-client support (Claude Code, Codex, OpenClaw).

👉 **[Video Demo → Project Page](https://zju3dv.github.io/habitat-gs/#agent)**

👉 **[Full documentation → docs/habitatagent.md](docs/habitatagent.md)**

```bash
# Quick start (TUI + bridge)
pip install -r requirements-agent.txt
python tools/habitat_agent.py

# With MCP server (for Claude Code / Codex integration)
python tools/habitat_agent.py --mcp
```

## 🏋️ Train/Eval Navigation Agents on Habitat-GS

### 🗺️ Point/Image/Object Goal Navigation on Habitat-Lab

We provide **one-click** training and evaluation pipelines for three navigation tasks on GS scenes using [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) with DDPPO:

| Task | Goal | Sensors | Actions |
|------|------|---------|---------|
| **PointNav** | GPS coordinates | RGB, Depth, GPS, Compass | move_forward, turn_left, turn_right, stop |
| **ImageNav** | Goal image | RGB, Depth, ImageGoal | move_forward, turn_left, turn_right, stop |
| **ObjectNav** | Object category (e.g. "bench") | RGB, Depth, GPS, Compass, ObjectGoal | move_forward, turn_left, turn_right, look_up, look_down, stop |

#### Prerequisites

- Habitat-GS and Habitat-Lab installed in the `habitat-gs` conda environment (see [Install](#-install-habitat-gs))
- GS data downloaded and placed under `data/scene_datasets/gs_scenes/` (see [Data Layout](#data-layout))

#### Data Layout

<details>
<summary>Click to expand: structure of our provided scene data and generated episodes:</summary>

```
data/scene_datasets/gs_scenes/
├── train.scene_dataset_config.json
├── val.scene_dataset_config.json
├── train/                           # 42 training scenes
│   ├── scene01/
│   │   ├── scene01.gs.ply          # GS render asset
│   │   ├── scene01.mesh.ply        # collision mesh (will not be used unless physics is enabled)
│   │   └── scene01.navmesh         # navigation mesh
│   ├── scene02/
│   │   └── ...
│   └── scene42/
├── val/                             # 5 evaluation scenes
│   ├── scene43/
│   ├── scene44/
│   ├── scene45/
│   ├── scene46/
│   └── scene47/
├── configs/                         # Hydra YAML configs (provided)
│   ├── ddppo_pointnav_gs_train.yaml
│   ├── ddppo_pointnav_gs_eval.yaml
│   ├── ddppo_imagenav_gs_train.yaml
│   ├── ddppo_imagenav_gs_eval.yaml
│   ├── ddppo_objectnav_gs_train.yaml
│   └── ddppo_objectnav_gs_eval.yaml
└── episodes/                        # generated by scripts below
    ├── pointnav/{train,val}/
    ├── imagenav/{train,val}/
    └── objectnav/{train,val}/
```

</details>

#### Step 1: Generate Episodes

Episode data must be generated before training and evaluating. We have provided 1000 episodes for each training scene and 100 episodes for each evaluation scene in our released dataset, but you can also generate your own episodes with the following commands:

```bash
conda activate habitat-gs

# PointNav episodes
python scripts_gs/generate_pointnav_episodes.py

# ImageNav episodes
python scripts_gs/generate_imagenav_episodes.py

# ObjectNav episodes (uses SAM + CLIP, see details below)
python scripts_gs/generate_objectnav_episodes.py
```

<details>
<summary>Click to expand: ObjectNav episode generation details</summary>

ObjectNav uses **SAM (Segment Anything)** + **CLIP (zero-shot classification)** to automatically detect and classify objects in GS scenes, then generates navigation episodes to those objects.

**Required model checkpoints:**

| Model | Path | Download |
|-------|------|----------|
| SAM ViT-B | `~/.cache/sam_checkpoints/sam_vit_b_01ec64.pth` | [GitHub](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| CLIP ViT-B-32 | `~/.cache/clip_models/vit_b_32_laion400m.pt` | [GitHub](https://github.com/mlfoundations/open_clip/releases) |

**Object categories** (outdoor-focused): car, bench, tree, street lamp, traffic sign, fire hydrant, trash can, bicycle, potted plant, barrier, statue, chair.

</details>

#### Step 2: Train

```bash
# PointNav (default 5e8 steps)
bash scripts_gs/train_pointnav.sh --output output/pointnav

# ImageNav (default 2.5e9 steps)
bash scripts_gs/train_imagenav.sh --output output/imagenav

# ObjectNav (default 2.5e9 steps)
bash scripts_gs/train_objectnav.sh --output output/objectnav
```

<details>
<summary>Click to expand: training options</summary>

All training scripts accept the same options:

```
--output DIR             Output directory for checkpoints and tensorboard (required)
--num-envs N             Number of parallel environments per GPU (default: 4)
--num-gpus N             Number of GPUs for DDPPO (default: 1)
--total-steps N          Total training steps
--num-ckpts N            Number of checkpoints to save (default: 100)
--pretrained-ckpt PATH   Fine-tune from an existing .pth checkpoint
                         (this sets ddppo.pretrained=True; critic is re-initialised)
```

Extra arguments are forwarded as Hydra overrides. Example with multi-GPU:

```bash
bash scripts_gs/train_objectnav.sh \
    --output output/objectnav \
    --num-envs 8 \
    --num-gpus 4
```

Fine-tuning from a previously trained checkpoint:

```bash
bash scripts_gs/train_pointnav.sh \
    --output output/pointnav_ft \
    --pretrained-ckpt output/pointnav/checkpoints/ckpt.99.pth
```

By default this loads the **whole** policy (encoder + RNN + actor head) and re-initialises the critic head — appropriate when continuing on the same task or transferring to a closely related one. To customise the load behaviour, append Hydra overrides, e.g.:

```bash
# keep the trained critic
bash scripts_gs/train_pointnav.sh \
    --output output/pointnav_ft \
    --pretrained-ckpt output/pointnav/checkpoints/ckpt.99.pth \
    habitat_baselines.rl.ddppo.reset_critic=False

# load only the visual encoder backbone (e.g. transfer from PointNav to ImageNav)
bash scripts_gs/train_imagenav.sh \
    --output output/imagenav_ft \
    --pretrained-ckpt output/pointnav/checkpoints/ckpt.99.pth \
    habitat_baselines.rl.ddppo.pretrained=False \
    habitat_baselines.rl.ddppo.pretrained_encoder=True \
    habitat_baselines.rl.ddppo.train_encoder=False    # freeze the encoder
```

> Note: optimizer state, step counter and seeds are **reset** — this is fine-tuning, not resume. To resume an interrupted run, just re-launch with the same `--output` directory; habitat-baselines auto-detects `.resume_state.pth` and continues seamlessly.

Output structure:

```
output/objectnav/
├── checkpoints/    # .pth checkpoint files
├── tb/             # TensorBoard logs
└── train.log       # training log
```

</details>

#### Step 3: Evaluate

```bash
# PointNav
bash scripts_gs/eval_pointnav.sh --ckpt output/pointnav/checkpoints/ckpt.0.pth

# ImageNav
bash scripts_gs/eval_imagenav.sh --ckpt output/imagenav/checkpoints/ckpt.0.pth

# ObjectNav
bash scripts_gs/eval_objectnav.sh --ckpt output/objectnav/checkpoints/ckpt.0.pth
```

<details>
<summary>Click to expand: evaluation options</summary>

```
--ckpt PATH         Path to a .pth checkpoint file or a directory of checkpoints (required)
--num-envs N        Number of parallel environments (default: 1)
--video-dir DIR     Directory to save evaluation rollout videos (optional)
```

Pass `--ckpt` a directory to evaluate all checkpoints in it sequentially.

</details>

### 🗣️ Vision-and-Language Navigation with StreamVLN

We also provide **one-click** training and evaluation pipelines for [StreamVLN](https://github.com/InternRobotics/StreamVLN) (a SOTA VLM-based VLN agent built on LLaVA-Video-7B-Qwen2 + SigLIP) on GS scenes. Unlike PointNav/ImageNav/ObjectNav which use Habitat-Lab + DDPPO, StreamVLN is trained via supervised fine-tuning of a vision-language model on demonstration trajectories.

| Task | Goal | Sensors | Actions | Backbone |
|------|------|---------|---------|----------|
| **VLN-R2R** | Natural-language instruction (e.g. "walk past the table and stop near the window") | RGB, GPS, Compass | move_forward, turn_left, turn_right, stop | LLaVA-Video-7B-Qwen2 + SigLIP |

#### Prerequisites

- A **separate** `habitat-gs-streamvln` conda environment (see `Step 1` below). StreamVLN pins specific package versions (`transformers==4.45.1`, `accelerate==0.28.0`, etc.) that conflict with the main `habitat-gs` env.

- StreamVLN cloned as a sibling of `habitat-gs/`:

  ```bash
  cd /path/to/parent
  git clone https://github.com/InternRobotics/StreamVLN.git
  ```

- GS data downloaded and placed under `data/scene_datasets/gs_scenes/` (same layout as the section above)

#### Data Layout

<details>
<summary>Click to expand: VLN-specific files added on top of the base dataset layout:</summary>

```
data/scene_datasets/gs_scenes/
├── configs/
│   └── vln_gs_eval.yaml             # habitat config for VLN evaluation (provided)
├── episodes/
│   └── vln/                         # generated by generate_vln_episodes.py
│       ├── train/train.json.gz      # 42 scenes × 200 episodes = 8,400 train
│       └── val/val.json.gz          # 5 scenes × 50 episodes = 250 val
└── trajectory_data/
    └── vln/                         # generated by generate_vln_trajectories.py
        ├── annotations.json         # StreamVLN-format action sequences
        └── images/{scene}_gs_{ep_id}/rgb/*.jpg   # rendered RGB frames
```

</details>

#### Step 1: One-Time Setup

`setup_vln.sh` creates the `habitat-gs-streamvln` conda environment cloned from `habitat-gs`, patches the StreamVLN repo for compatibility with Habitat-GS, installs StreamVLN Python dependencies, and downloads the LLaVA-Video-7B-Qwen2 (~15GB) and SigLIP (~3.3GB) checkpoints into `StreamVLN/checkpoints/`.

```bash
bash scripts_gs/setup_vln.sh
```

<details>
<summary>Click to expand: what setup_vln.sh actually does</summary>

The script applies `scripts_gs/streamvln_compat.patch` to the StreamVLN clone (4 files, +42/−13 lines) so that:

- `streamvln/habitat_extensions/measures.py` works with habitat-lab 0.3.3 (which removed `try_cv2_import`)
- `streamvln/streamvln_train.py` honors a `--vision_tower` CLI override for local model paths, fixes `low_cpu_mem_usage` for quantized loading, removes duplicate quantization kwargs, and adds a tokenizer-loading fallback for merged LoRA checkpoints
- `streamvln/streamvln_eval.py` explicitly loads the SigLIP vision tower (fixes `delay_load` issue) and auto-selects `flash_attention_2` with `eager` fallback
- `llava/model/multimodal_encoder/siglip_encoder.py` passes `low_cpu_mem_usage=True` for `device_map`-based loading

The script is idempotent — re-running `setup_vln.sh` is safe. Available flags:

```
--skip-env          Skip creating the conda environment
--skip-patch        Skip applying the compat patch
--skip-download     Skip downloading model checkpoints
--skip-deps         Skip installing Python dependencies
--hf-token TOKEN    HuggingFace token for gated models
```

</details>

#### Step 2: Generate Episodes and Trajectories

VLN needs both **episodes** (start/goal + natural-language instruction) and **trajectory data** (rendered RGB frames + ground-truth action sequences for SFT). We provide both in the released dataset, but you can also re-generate them:

```bash
conda activate habitat-gs-streamvln

# 1. Generate VLN episodes (samples paths on the navmesh, renders waypoints with GS,
#    and queries a VLM to produce the instruction text). Outputs R2RVLN-v1 format.
python scripts_gs/generate_vln_episodes.py

# 2. Generate StreamVLN trajectory data by replaying each episode with a greedy
#    path follower that records (RGB frame, action) pairs.
python scripts_gs/generate_vln_trajectories.py
```

<details>
<summary>Click to expand: episode/trajectory generation details</summary>

`generate_vln_episodes.py` produces 200 episodes per training scene and 50 per evaluation scene by default, in the standard R2RVLN-v1 format consumed by habitat-lab's `vln_r2r` task. Instructions are generated by querying an OpenAI-compatible VLM endpoint with multi-view renderings along the path. Configure the endpoint via `OPENAI_BASE_URL` + `OPENAI_API_KEY` environment variables, or pass `--api-config /path/to/config.json`.

`generate_vln_trajectories.py` runs a greedy heading-based path follower (`forward_step=0.25m`, `turn_angle=15°`, `success_distance=0.25m` for the final waypoint) on each episode and records:

- **annotations.json** — one entry per episode with the instruction, the action sequence (`-1`=initial, `0`=stop, `1`=forward, `2`=turn-left, `3`=turn-right), and per-step poses
- **images/{scene}\_gs\_{ep_id}/rgb/*.jpg** — RGB frame at each step, rendered through the GS pipeline

The script supports `--resume` to skip already-completed scenes, which is useful if generation is interrupted.

</details>

#### Step 3: Train

By default, the training script performs **standard full fine-tune** (vision tower + projector + LLM, matching StreamVLN official config). This requires **≥80 GB VRAM per GPU** (A100 80GB recommended) with the default DeepSpeed ZeRO-2 config. For consumer GPUs (RTX 3090/4090), add `--lora` flag to enable memory-efficient LoRA training.

```bash
# ── Standard full fine-tune ──

# Stage-1: SFT on demonstration trajectories
bash scripts_gs/train_vln.sh --output output/vln_stage1 --stage stage-one

# DAgger: retrain with DAgger-collected data
bash scripts_gs/train_vln.sh --output output/vln_dagger --stage dagger \
    --ckpt output/vln_stage1/checkpoint-XXX

# Stage-2: co-training with auxiliary QA data
bash scripts_gs/train_vln.sh --output output/vln_stage2 --stage stage-two \
    --ckpt output/vln_dagger/checkpoint-XXX


# ── LoRA mode ──
bash scripts_gs/train_vln.sh --output output/vln_stage1 --stage stage-one --lora
bash scripts_gs/train_vln.sh --output output/vln_dagger --stage dagger \
    --ckpt output/vln_stage1 --lora
```

<details>
<summary>Click to expand: training options</summary>

```
--output DIR        Output directory for checkpoints (required)
--stage STAGE       Training stage: stage-one | dagger | stage-two (default: stage-one)
--num-gpus N        Number of GPUs (default: 1)
--ckpt PATH         Base checkpoint (default: local LLaVA-Video-7B-Qwen2 for stage-one)
--epochs N          Number of epochs (default: 1)
--batch-size N      Per-device batch size (default: 2)
--grad-accum N      Gradient accumulation steps (default: 2)
--lr RATE           Learning rate (default: 2e-5)
--num-frames N      Frames per sample (default: 32)
--lora              Enable LoRA mode (see below)
```

**Standard mode (default):** full fine-tune of the entire model (~8 GB trainable parameters) with `anyres_max_9` image tiling, 32 frames, 32K context, and `torch.compile`. Multi-GPU training uses DeepSpeed ZeRO-2. Requires **≥80 GB VRAM per GPU** (A100 80GB recommended).

**LoRA mode (`--lora`):** freezes the LLM backbone, trains only the MM projector + LoRA adapters (`r=64, alpha=128`, ~17 MB trainable), reduces frames to 4 and context to 2K. Fits on a **single 24GB RTX 4090**. When chaining stages with `--lora` (e.g. stage-one → dagger), the script auto-merges the previous LoRA checkpoint before applying new adapters.

</details>

#### Step 4: Evaluate

```bash
bash scripts_gs/eval_vln.sh --ckpt output/vln_stage1/checkpoint-XXX
```

<details>
<summary>Click to expand: evaluation options</summary>

```
--ckpt PATH         Path to a trained StreamVLN checkpoint (required)
--output DIR        Output directory for results (default: results/vln/<ckpt>_<split>)
--num-gpus N        Number of GPUs for parallel rollout (default: 1)
--split SPLIT       Evaluation split: train | val (default: val)
--num-frames N      Frames per sample (default: 32)
--save-video        Save visualization videos
```

The evaluator uses `data/scene_datasets/gs_scenes/configs/vln_gs_eval.yaml` (RGB+Depth at 640x480, hfov=79°, `forward_step=0.25m`, `turn_angle=15°`, `success_distance=3.0m`, `max_episode_steps=500`) and reports the standard VLN metrics: Success, SPL, Oracle Success, Distance-to-Goal, and Oracle Navigation Error.

</details>

### ✈️ Vision-and-Language Navigation with Uni-NaVid

We also support **one-click** training and evaluation pipeline for [Uni-NaVid](https://github.com/jzhzhang/Uni-NaVid) (RSS 2025), a unified video-based vision-language-action model that handles multiple embodied navigation tasks (VLN, ObjectNav, EQA, etc.) with a single model. Uni-NaVid is built on Vicuna-7B + EVA-ViT-G with online token merging for efficient streaming inference.

| Task | Goal | Sensors | Actions | Backbone |
|------|------|---------|---------|----------|
| **VLN-R2R** | Natural-language instruction | RGB (120° HFOV) | move_forward (0.25m), turn_left/right (30°), stop | Vicuna-7B + EVA-ViT-G |

#### Prerequisites

- A **separate** `habitat-gs-uni-navid` conda environment (see `Step 1` below).

- Uni-NaVid repo cloned as a sibling of `habitat-gs/`:

  ```bash
  cd /path/to/parent
  git clone https://github.com/jzhzhang/Uni-NaVid.git
  ```

- GS data downloaded and placed under `data/scene_datasets/gs_scenes/` (same layout as other tasks)

#### Data Layout

<details>
<summary>Click to expand: Uni-NaVid-specific files added on top of the base dataset layout:</summary>

```
data/scene_datasets/gs_scenes/
├── configs/
│   └── vln_uninavid_gs_eval.yaml    # habitat config for Uni-NaVid eval (provided)
├── episodes/
│   └── vln/                         # shared with StreamVLN
│       ├── train/train.json.gz      # 42 scenes × 200 episodes = 8,400 train
│       └── val/val.json.gz          # 5 scenes × 50 episodes = 250 val
└── trajectory_data/
    └── uninavid/                    # generated by generate_uninavid_trajectories.py
        ├── nav_gs_train.json        # Uni-NaVid conversation-format annotations
        ├── nav_gs_val.json
        └── nav_videos/              # .mp4 trajectory videos
            ├── scene01_gs_000000.mp4
            └── ...
```

</details>

#### Step 1: One-Time Setup

`setup_uninavid.sh` creates the `habitat-gs-uni-navid` conda environment cloned from `habitat-gs`, installs Uni-NaVid's Python dependencies, and downloads model checkpoints (EVA-ViT-G ~3.5GB, Vicuna-7B ~13GB, Uni-NaVid pretrained ~14GB).

```bash
bash scripts_gs/setup_uninavid.sh
```

<details>
<summary>Click to expand: setup options</summary>

```
--skip-env          Skip creating the conda environment
--skip-deps         Skip installing Python dependencies
--skip-download     Skip downloading model checkpoints
--proxy URL         HTTP proxy for downloads
```

</details>

#### Step 2: Generate Trajectory Data

Uni-NaVid is trained on video trajectories in a conversation format. This script replays each VLN episode with a greedy path follower and records RGB frames as `.mp4` videos + action annotations in Uni-NaVid's conversation JSON format, which are both included in our released dataset.

```bash
conda activate habitat-gs-uni-navid
python scripts_gs/generate_uninavid_trajectories.py
```

<details>
<summary>Click to expand: trajectory generation details</summary>

The greedy controller uses `forward_step=0.25m`, `turn_angle=30°`, `HFOV=120°`. Each trajectory video is encoded at 10 fps. The output JSON uses Uni-NaVid's conversation format with `NAV_ID` prefix and `NAVIGATION_IDENTIFIER` string to trigger navigation-specific token processing during training.

</details>

#### Step 3: Train

Two-stage training is supported as standard Uni-NaVid. Requires **≥80 GB VRAM per GPU** (A100 80GB recommended).
- **stage-1**: Fine-tune from Vicuna-7B (training from scratch, requires large dataset)
- **stage-2**: Fine-tune from pre-trained Uni-NaVid checkpoint (recommended)

```bash
conda activate habitat-gs-uni-navid

# Recommended: fine-tune from pre-trained Uni-NaVid
bash scripts_gs/train_uninavid.sh --output output/uninavid_gs --stage stage-2

# Or train from scratch with Vicuna-7B
bash scripts_gs/train_uninavid.sh --output output/uninavid_gs --stage stage-1
```

<details>
<summary>Click to expand: training options</summary>

```
--output DIR         Output directory for checkpoints (required)
--stage STAGE        Training stage: stage-1|stage-2 (default: stage-2)
--num-gpus N         Number of GPUs (default: 1)
--ckpt PATH          Base checkpoint path (auto-selected per stage)
--epochs N           Number of epochs (default: 1)
--batch-size N       Per-device batch size (default: 8)
--grad-accum N       Gradient accumulation steps (default: 2)
--lr RATE            Learning rate (default: 1e-5)
```

</details>

#### Step 4: Evaluate

Evaluation runs online in a `habitat.Env` with the VLN-v0 task, matching the [NaVid-VLN-CE](https://github.com/jzhzhang/NaVid-VLN-CE) evaluation pattern. 

```bash
conda activate habitat-gs-uni-navid
bash scripts_gs/eval_uninavid.sh --ckpt output/uninavid_gs/<checkpoint>
```

<details>
<summary>Click to expand: evaluation options</summary>

```
--ckpt PATH         Path to trained Uni-NaVid checkpoint (required)
--output DIR        Output directory for results (default: results/uninavid/<ckpt>_<split>)
--num-gpus N        Number of GPUs for parallel evaluation (default: 1)
--split SPLIT       Evaluation split: train|val (default: val)
--save-video        Save evaluation rollout videos
```

The evaluator reports: Success Rate (SR), SPL, Oracle Success (OSR), and Distance-to-Goal (DTG).

</details>

## 📚 Citation

Coming soon.
