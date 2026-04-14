#!/usr/bin/env bash
#
# One-click Uni-NaVid evaluation on Gaussian Splatting scenes.
#
# Uses habitat.Env with the VLN-v0 task (online evaluation, matching
# the NaVid-VLN-CE pattern). Multi-GPU via episode splitting.
#
# Usage:
#   bash scripts_gs/eval_uninavid.sh --ckpt /path/to/checkpoint
#   bash scripts_gs/eval_uninavid.sh --ckpt /path/to/checkpoint --num-gpus 4
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
CKPT_PATH=""
OUTPUT_DIR=""
NUM_GPUS=1
EVAL_SPLIT="val"
EXP_SAVE="data"
EXTRA_ARGS=()

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)           CKPT_PATH="$2";    shift 2;;
        --output)         OUTPUT_DIR="$2";    shift 2;;
        --num-gpus)       NUM_GPUS="$2";      shift 2;;
        --split)          EVAL_SPLIT="$2";    shift 2;;
        --save-video)     EXP_SAVE="video-data"; shift;;
        *)                EXTRA_ARGS+=("$1"); shift;;
    esac
done

if [[ -z "$CKPT_PATH" ]]; then
    echo "Usage: bash scripts_gs/eval_uninavid.sh --ckpt PATH [options]"
    echo ""
    echo "Options:"
    echo "  --ckpt PATH      Path to trained Uni-NaVid checkpoint (required)"
    echo "  --output DIR     Output directory for results (default: auto)"
    echo "  --num-gpus N     Number of GPUs for parallel evaluation (default: 1)"
    echo "  --split SPLIT    Evaluation split: train|val (default: val)"
    echo "  --save-video     Save evaluation rollout videos"
    exit 1
fi

# ── Resolve paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CKPT_PATH="$(realpath "$CKPT_PATH")"
if [[ ! -e "$CKPT_PATH" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

# ── Verify episode data ─────────────────────────────────────────────
EPISODE_FILE="$PROJECT_ROOT/data/scene_datasets/gs_scenes/episodes/vln/$EVAL_SPLIT/$EVAL_SPLIT.json.gz"
if [[ ! -f "$EPISODE_FILE" ]]; then
    echo "ERROR: Episode data not found: $EPISODE_FILE"
    echo "Run:  python scripts_gs/generate_vln_episodes.py"
    exit 1
fi

# ── Default output directory ─────────────────────────────────────────
if [[ -z "$OUTPUT_DIR" ]]; then
    CKPT_NAME="$(basename "$CKPT_PATH")"
    OUTPUT_DIR="$PROJECT_ROOT/results/uninavid/${CKPT_NAME}_${EVAL_SPLIT}"
fi
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo " Uni-NaVid Evaluation (GS Scenes)"
echo "=========================================="
echo "  Project root : $PROJECT_ROOT"
echo "  Checkpoint   : $CKPT_PATH"
echo "  Split        : $EVAL_SPLIT"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Num GPUs     : $NUM_GPUS"
echo "=========================================="
echo ""

# ── Launch evaluation ────────────────────────────────────────────────
cd "$PROJECT_ROOT"

EVAL_SCRIPT="$SCRIPT_DIR/eval_uninavid_gs.py"

if [[ $NUM_GPUS -gt 1 ]]; then
    # Multi-GPU: split episodes across GPUs (NaVid-VLN-CE pattern)
    PIDS=()
    for IDX in $(seq 0 $((NUM_GPUS-1))); do
        echo "  Launching worker $IDX on GPU $((IDX % NUM_GPUS)) ..."
        CUDA_VISIBLE_DEVICES=$((IDX % NUM_GPUS)) python "$EVAL_SCRIPT" \
            --model-path "$CKPT_PATH" \
            --eval-split "$EVAL_SPLIT" \
            --result-path "$OUTPUT_DIR" \
            --split-num "$NUM_GPUS" \
            --split-id "$IDX" \
            --exp-save "$EXP_SAVE" \
            "${EXTRA_ARGS[@]}" &
        PIDS+=($!)
    done

    echo "  Waiting for all workers to finish ..."
    FAIL=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAIL=$((FAIL+1))
    done

    if [[ $FAIL -gt 0 ]]; then
        echo "WARNING: $FAIL worker(s) failed"
    fi

    # Aggregate results
    echo ""
    echo "── Aggregating results ──"
    python "$EVAL_SCRIPT" \
        --model-path "$CKPT_PATH" \
        --result-path "$OUTPUT_DIR" \
        --aggregate-only
else
    # Single GPU
    exec python -u "$EVAL_SCRIPT" \
        --model-path "$CKPT_PATH" \
        --eval-split "$EVAL_SPLIT" \
        --result-path "$OUTPUT_DIR" \
        --split-num 1 \
        --split-id 0 \
        --exp-save "$EXP_SAVE" \
        "${EXTRA_ARGS[@]}"
fi
