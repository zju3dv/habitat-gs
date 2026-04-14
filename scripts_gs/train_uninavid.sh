#!/usr/bin/env bash
#
# One-click Uni-NaVid training on Gaussian Splatting scenes.
#
# Two-stage training (matching standard Uni-NaVid):
#   stage-1: Fine-tune from Vicuna-7B (requires sufficient data)
#   stage-2: Fine-tune from pre-trained Uni-NaVid checkpoint (recommended)
#
# Usage:
#   bash scripts_gs/train_uninavid.sh --output output/uninavid --stage stage-2
#   bash scripts_gs/train_uninavid.sh --output output/uninavid --stage stage-1 --num-gpus 4
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
OUTPUT_DIR=""
STAGE="stage-2"
NUM_GPUS=1
CKPT_PATH=""
EPOCHS=1
BATCH_SIZE=8
GRAD_ACCUM=2
LR="1e-5"
VIDEO_FPS=10
COMPRESS_TYPE="grid:2"
MODEL_MAX_LENGTH=2048
EXTRA_ARGS=()

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)         OUTPUT_DIR="$2";       shift 2;;
        --stage)          STAGE="$2";            shift 2;;
        --num-gpus)       NUM_GPUS="$2";         shift 2;;
        --ckpt)           CKPT_PATH="$2";        shift 2;;
        --epochs)         EPOCHS="$2";           shift 2;;
        --batch-size)     BATCH_SIZE="$2";       shift 2;;
        --lr)             LR="$2";               shift 2;;
        --grad-accum)     GRAD_ACCUM="$2";       shift 2;;
        *)                EXTRA_ARGS+=("$1");    shift;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash scripts_gs/train_uninavid.sh --output DIR [options]"
    echo ""
    echo "Stages:"
    echo "  stage-1   Fine-tune from Vicuna-7B (training from scratch)"
    echo "  stage-2   Fine-tune from pre-trained Uni-NaVid (recommended, default)"
    echo ""
    echo "Options:"
    echo "  --output DIR           Output directory for checkpoints (required)"
    echo "  --stage STAGE          Training stage: stage-1|stage-2 (default: stage-2)"
    echo "  --num-gpus N           Number of GPUs (default: 1)"
    echo "  --ckpt PATH            Base checkpoint path (auto-selected per stage)"
    echo "  --epochs N             Number of epochs (default: 1)"
    echo "  --batch-size N         Per-device batch size (default: 8)"
    echo "  --grad-accum N         Gradient accumulation steps (default: 2)"
    echo "  --lr RATE              Learning rate (default: 1e-5)"
    echo ""
    echo "VRAM estimate (batch_size=8, ZeRO-2, bf16, gradient_checkpointing):"
    echo "  >=80GB per GPU  (A100 80GB recommended)"
    echo "  For smaller GPUs, reduce --batch-size and/or use zero2_offload.json"
    exit 1
fi

if [[ "$STAGE" != "stage-1" && "$STAGE" != "stage-2" ]]; then
    echo "ERROR: --stage must be one of: stage-1, stage-2"
    exit 1
fi

# ── Resolve paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UNINAVID_ROOT="$(cd "$PROJECT_ROOT/../Uni-NaVid" && pwd)"

TRAJ_DATA="$PROJECT_ROOT/data/scene_datasets/gs_scenes/trajectory_data/uninavid"
DEEPSPEED_CFG="$UNINAVID_ROOT/scripts/zero2.json"
MODEL_ZOO="$UNINAVID_ROOT/model_zoo"

# ── Verify data ─────────────────────────────────────────────────────
if [[ ! -f "$TRAJ_DATA/nav_gs_train.json" ]]; then
    echo "ERROR: Training data not found at $TRAJ_DATA/nav_gs_train.json"
    echo "Run:  python scripts_gs/generate_uninavid_trajectories.py"
    exit 1
fi

if [[ ! -f "$MODEL_ZOO/eva_vit_g.pth" ]]; then
    echo "ERROR: EVA-ViT-G not found at $MODEL_ZOO/eva_vit_g.pth"
    echo "Run:  bash scripts_gs/setup_uninavid.sh"
    exit 1
fi

# ── Default checkpoint per stage ─────────────────────────────────────
if [[ -z "$CKPT_PATH" ]]; then
    case "$STAGE" in
        stage-1)
            CKPT_PATH="$MODEL_ZOO/vicuna-7b-v1.5"
            if [[ ! -d "$CKPT_PATH" ]]; then
                echo "ERROR: Vicuna-7B not found at $CKPT_PATH"
                echo "Run:  bash scripts_gs/setup_uninavid.sh"
                exit 1
            fi
            ;;
        stage-2)
            CKPT_PATH="$MODEL_ZOO/uninavid-7b-full-224-video-fps-1-grid-2"
            if [[ ! -d "$CKPT_PATH" ]]; then
                echo "ERROR: Uni-NaVid pretrained checkpoint not found at $CKPT_PATH"
                echo "Run:  bash scripts_gs/setup_uninavid.sh"
                exit 1
            fi
            ;;
    esac
fi

CKPT_PATH="$(realpath "$CKPT_PATH")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_DIR"

# ── Fix hardcoded paths in checkpoint config ────────────────────────
# Pretrained checkpoints may contain absolute paths from the original
# training machine. Patch mm_vision_tower and image_processor to match
# the current deployment paths so that from_pretrained() can find them.
if [[ -f "$CKPT_PATH/config.json" ]]; then
    _cfg="$CKPT_PATH/config.json"
    _need_fix=false
    if grep -q '"mm_vision_tower"' "$_cfg" && ! grep -q "\"mm_vision_tower\": \"$MODEL_ZOO/eva_vit_g.pth\"" "$_cfg"; then
        _need_fix=true
    fi
    if [[ "$_need_fix" == "true" ]]; then
        python3 -c "
import json, sys
cfg_path = '$_cfg'
with open(cfg_path) as f: cfg = json.load(f)
changed = False
if cfg.get('mm_vision_tower','') != '$MODEL_ZOO/eva_vit_g.pth':
    cfg['mm_vision_tower'] = '$MODEL_ZOO/eva_vit_g.pth'; changed = True
if 'image_processor' in cfg and cfg['image_processor'] != '$UNINAVID_ROOT/uninavid/processor/clip-patch14-224':
    cfg['image_processor'] = '$UNINAVID_ROOT/uninavid/processor/clip-patch14-224'; changed = True
if changed:
    with open(cfg_path, 'w') as f: json.dump(cfg, f, indent=2)
    print('  Patched config.json: mm_vision_tower / image_processor updated')
"
    fi
fi

echo "=========================================="
echo " Uni-NaVid Training (GS Scenes)"
echo "=========================================="
echo "  Stage        : $STAGE"
echo "  Checkpoint   : $CKPT_PATH"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Num GPUs     : $NUM_GPUS"
echo "  Epochs       : $EPOCHS"
echo "  Batch size   : $BATCH_SIZE"
echo "  Grad accum   : $GRAD_ACCUM"
echo "  Learning rate: $LR"
echo "  Traj data    : $TRAJ_DATA"
echo "=========================================="
echo ""

# ── Build training command ───────────────────────────────────────────
cd "$UNINAVID_ROOT"

# Consumer RTX cards (30xx/40xx/50xx) need P2P disabled; professional cards (A6000 etc.) do not
GPU_NAME="$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | head -1)"
if [[ "$GPU_NAME" == *"GeForce"* ]] || [[ "$GPU_NAME" =~ RTX\ [2-9][0-9]{3} ]]; then
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
fi

TRAIN_ARGS=(
    --model_name_or_path "$CKPT_PATH"
    --version imgsp_v1
    --data_path "$TRAJ_DATA/nav_gs_train.json"
    --image_folder "$TRAJ_DATA"
    --video_folder "$TRAJ_DATA"
    --vision_tower "$MODEL_ZOO/eva_vit_g.pth"
    --image_processor "$UNINAVID_ROOT/uninavid/processor/clip-patch14-224"
    --tune_vision_encoder False
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --group_by_modality_length True
    --video_fps "$VIDEO_FPS"
    --compress_type "$COMPRESS_TYPE"
    --bf16 True
    --output_dir "$OUTPUT_DIR"
    --num_train_epochs "$EPOCHS"
    --per_device_train_batch_size "$BATCH_SIZE"
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --evaluation_strategy "no"
    --save_strategy "steps"
    --save_steps 8000
    --save_total_limit 1
    --learning_rate "$LR"
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --tf32 True
    --model_max_length "$MODEL_MAX_LENGTH"
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --lazy_preprocess True
    --report_to none
)

# Add DeepSpeed (ZeRO-2, matching original Uni-NaVid)
if [[ -f "$DEEPSPEED_CFG" ]]; then
    TRAIN_ARGS=(--deepspeed "$DEEPSPEED_CFG" "${TRAIN_ARGS[@]}")
fi

# ── Launch training ──────────────────────────────────────────────────
if [[ $NUM_GPUS -gt 1 ]]; then
    exec deepspeed --num_gpus="$NUM_GPUS" \
        uninavid/train/train_mem.py \
        "${TRAIN_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
else
    # Single GPU: use deepspeed launcher if deepspeed config is active
    if echo "${TRAIN_ARGS[@]}" | grep -q "\-\-deepspeed"; then
        exec deepspeed --num_gpus=1 \
            uninavid/train/train_mem.py \
            "${TRAIN_ARGS[@]}" \
            "${EXTRA_ARGS[@]}"
    else
        exec python -u uninavid/train/train_mem.py \
            "${TRAIN_ARGS[@]}" \
            "${EXTRA_ARGS[@]}"
    fi
fi
