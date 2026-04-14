#!/usr/bin/env bash
#
# One-click DDPPO ImageNav evaluation on Gaussian Splatting scenes.
#
# Usage:
#   bash scripts_gs/eval_imagenav.sh --ckpt /path/to/checkpoint.pth
#
set -euo pipefail

CKPT_PATH=""
NUM_ENVS=1
VIDEO_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)       CKPT_PATH="$2";    shift 2;;
        --num-envs)   NUM_ENVS="$2";     shift 2;;
        --video-dir)  VIDEO_DIR="$2";    shift 2;;
        *)            EXTRA_ARGS+=("$1"); shift;;
    esac
done

if [[ -z "$CKPT_PATH" ]]; then
    echo "Usage: bash scripts_gs/eval_imagenav.sh --ckpt /path/to/checkpoint [options]"
    echo ""
    echo "Options:"
    echo "  --ckpt PATH         Path to checkpoint .pth file or directory (required)"
    echo "  --num-envs N        Number of parallel environments (default: 1)"
    echo "  --video-dir DIR     Directory to save evaluation videos (optional)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -d "data/scene_datasets/gs_scenes/episodes/imagenav/val/content" ]]; then
    echo "ERROR: ImageNav validation episode data not found."
    echo "Run:  python scripts_gs/generate_imagenav_episodes.py"
    exit 1
fi

CKPT_PATH="$(realpath "$CKPT_PATH")"
if [[ ! -e "$CKPT_PATH" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

echo "=========================================="
echo " DDPPO ImageNav Evaluation (GS Scenes)"
echo "=========================================="
echo "  Project root : $PROJECT_ROOT"
echo "  Checkpoint   : $CKPT_PATH"
echo "  Num envs     : $NUM_ENVS"
[[ -n "$VIDEO_DIR" ]] && echo "  Video dir    : $VIDEO_DIR"
echo "=========================================="
echo ""

OVERRIDES=(
    "--config-name=ddppo_imagenav_gs_eval"
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

exec python -u scripts_gs/run_imagenav.py \
    "${OVERRIDES[@]}" \
    "${EXTRA_ARGS[@]}"
