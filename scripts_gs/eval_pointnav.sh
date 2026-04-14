#!/usr/bin/env bash
#
# One-click DDPPO PointNav evaluation on Gaussian Splatting scenes.
#
# Usage:
#   bash scripts_gs/eval_pointnav.sh --ckpt /path/to/checkpoint.pth
#   bash scripts_gs/eval_pointnav.sh --ckpt /path/to/checkpoints_dir
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
CKPT_PATH=""
NUM_ENVS=1
VIDEO_DIR=""
EXTRA_ARGS=()

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)       CKPT_PATH="$2";    shift 2;;
        --num-envs)   NUM_ENVS="$2";     shift 2;;
        --video-dir)  VIDEO_DIR="$2";    shift 2;;
        *)            EXTRA_ARGS+=("$1"); shift;;
    esac
done

if [[ -z "$CKPT_PATH" ]]; then
    echo "Usage: bash scripts_gs/eval_pointnav.sh --ckpt /path/to/checkpoint [options]"
    echo ""
    echo "Options:"
    echo "  --ckpt PATH         Path to checkpoint .pth file or directory (required)"
    echo "  --num-envs N        Number of parallel environments (default: 1)"
    echo "  --video-dir DIR     Directory to save evaluation videos (optional)"
    echo ""
    echo "Extra arguments are forwarded as Hydra overrides."
    exit 1
fi

# ── Resolve paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Verify data ──────────────────────────────────────────────────────
if [[ ! -d "data/scene_datasets/gs_scenes/episodes/pointnav/val/content" ]]; then
    echo "ERROR: Validation episode data not found."
    echo "Run:  python scripts_gs/generate_pointnav_episodes.py"
    exit 1
fi

CKPT_PATH="$(realpath "$CKPT_PATH")"
if [[ ! -e "$CKPT_PATH" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

echo "=========================================="
echo " DDPPO PointNav Evaluation (GS Scenes)"
echo "=========================================="
echo "  Project root : $PROJECT_ROOT"
echo "  Checkpoint   : $CKPT_PATH"
echo "  Num envs     : $NUM_ENVS"
[[ -n "$VIDEO_DIR" ]] && echo "  Video dir    : $VIDEO_DIR"
echo "=========================================="
echo ""

# ── Hydra overrides ──────────────────────────────────────────────────
OVERRIDES=(
    "--config-name=ddppo_pointnav_gs_eval"
    "habitat_baselines.num_environments=$NUM_ENVS"
    "habitat_baselines.eval_ckpt_path_dir=$CKPT_PATH"
)

if [[ -n "$VIDEO_DIR" ]]; then
    mkdir -p "$VIDEO_DIR"
    OVERRIDES+=(
        "habitat_baselines.video_dir=$VIDEO_DIR"
        "habitat_baselines.eval.video_option=[disk]"
    )
fi

exec python -u scripts_gs/run_pointnav.py \
    "${OVERRIDES[@]}" \
    "${EXTRA_ARGS[@]}"
