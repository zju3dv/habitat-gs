#!/usr/bin/env bash
#
# One-click DDPPO ImageNav training on Gaussian Splatting scenes.
#
# Usage:
#   bash scripts_gs/train_imagenav.sh --output /path/to/output
#   bash scripts_gs/train_imagenav.sh --output /path/to/output --num-envs 8 --num-gpus 2
#
set -euo pipefail

OUTPUT_DIR=""
NUM_ENVS=4
NUM_GPUS=1
TOTAL_STEPS="2.5e9"
NUM_CHECKPOINTS=100
PRETRAINED_CKPT=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)          OUTPUT_DIR="$2";        shift 2;;
        --num-envs)        NUM_ENVS="$2";          shift 2;;
        --num-gpus)        NUM_GPUS="$2";          shift 2;;
        --total-steps)     TOTAL_STEPS="$2";       shift 2;;
        --num-ckpts)       NUM_CHECKPOINTS="$2";   shift 2;;
        --pretrained-ckpt) PRETRAINED_CKPT="$2";   shift 2;;
        *)                 EXTRA_ARGS+=("$1");     shift;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash scripts_gs/train_imagenav.sh --output /path/to/output [options]"
    echo ""
    echo "Options:"
    echo "  --output DIR             Output directory for checkpoints and tensorboard (required)"
    echo "  --num-envs N             Number of parallel environments per GPU (default: 4)"
    echo "  --num-gpus N             Number of GPUs for DDPPO (default: 1)"
    echo "  --total-steps N          Total training steps (default: 2.5e9)"
    echo "  --num-ckpts N            Number of checkpoints to save (default: 100)"
    echo "  --pretrained-ckpt PATH   Fine-tune from an existing .pth checkpoint"
    echo "                           (this sets ddppo.pretrained=True; critic is re-initialised)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -d "data/scene_datasets/gs_scenes/episodes/imagenav/train/content" ]]; then
    echo "ERROR: ImageNav training episode data not found."
    echo "Run:  python scripts_gs/generate_imagenav_episodes.py"
    exit 1
fi

if [[ -n "$PRETRAINED_CKPT" ]]; then
    PRETRAINED_CKPT="$(realpath "$PRETRAINED_CKPT")"
    if [[ ! -f "$PRETRAINED_CKPT" ]]; then
        echo "ERROR: --pretrained-ckpt file not found: $PRETRAINED_CKPT"
        exit 1
    fi
    # Adapt the ckpt's state_dict keys for habitat-baselines'
    # pretrained_weights load path (see scripts_gs/_adapt_pretrained_ckpt.py)
    PRETRAINED_CKPT_ADAPTED="$(mktemp -t habitat_ft_ckpt_XXXXXX.pth)"
    trap 'rm -f "$PRETRAINED_CKPT_ADAPTED"' EXIT
    python scripts_gs/_adapt_pretrained_ckpt.py \
        "$PRETRAINED_CKPT" "$PRETRAINED_CKPT_ADAPTED"
    PRETRAINED_CKPT="$PRETRAINED_CKPT_ADAPTED"
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo " DDPPO ImageNav Training (GS Scenes)"
echo "=========================================="
echo "  Project root : $PROJECT_ROOT"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Num GPUs     : $NUM_GPUS"
echo "  Num envs/GPU : $NUM_ENVS"
echo "  Total steps  : $TOTAL_STEPS"
echo "  Checkpoints  : $NUM_CHECKPOINTS"
[[ -n "$PRETRAINED_CKPT" ]] && echo "  Pretrained   : $PRETRAINED_CKPT"
echo "=========================================="
echo ""

OVERRIDES=(
    "habitat_baselines.num_environments=$NUM_ENVS"
    "habitat_baselines.checkpoint_folder=$OUTPUT_DIR/checkpoints"
    "habitat_baselines.tensorboard_dir=$OUTPUT_DIR/tb"
    "habitat_baselines.log_file=$OUTPUT_DIR/train.log"
    "habitat_baselines.total_num_steps=$TOTAL_STEPS"
    "habitat_baselines.num_checkpoints=$NUM_CHECKPOINTS"
)

if [[ -n "$PRETRAINED_CKPT" ]]; then
    OVERRIDES+=(
        "habitat_baselines.rl.ddppo.pretrained=True"
        "habitat_baselines.rl.ddppo.pretrained_weights=$PRETRAINED_CKPT"
    )
fi

if [[ $NUM_GPUS -gt 1 ]]; then
    python -m torch.distributed.launch \
        --use_env \
        --nproc_per_node "$NUM_GPUS" \
        scripts_gs/run_imagenav.py \
        "${OVERRIDES[@]}" \
        "${EXTRA_ARGS[@]}"
else
    python -u scripts_gs/run_imagenav.py \
        "${OVERRIDES[@]}" \
        "${EXTRA_ARGS[@]}"
fi
