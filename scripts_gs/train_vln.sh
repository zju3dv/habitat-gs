#!/usr/bin/env bash
#
# One-click StreamVLN training on Gaussian Splatting scenes.
#
# Usage:
#   bash scripts_gs/train_vln.sh --output /path/to/output --stage stage-one
#   bash scripts_gs/train_vln.sh --output /path/to/output --stage stage-one --lora
#   bash scripts_gs/train_vln.sh --output /path/to/output --stage dagger --ckpt /path/to/stage1/ckpt
#   bash scripts_gs/train_vln.sh --output /path/to/output --stage stage-two --ckpt /path/to/dagger/ckpt
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
OUTPUT_DIR=""
STAGE="stage-one"
NUM_GPUS=1
CKPT_PATH=""
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=2
LR="2e-5"
NUM_FRAMES=32
NUM_HISTORY=8
NUM_FUTURE_STEPS=4
USE_LORA=false
EXTRA_ARGS=()

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)       OUTPUT_DIR="$2";       shift 2;;
        --stage)        STAGE="$2";            shift 2;;
        --num-gpus)     NUM_GPUS="$2";         shift 2;;
        --ckpt)         CKPT_PATH="$2";        shift 2;;
        --epochs)       EPOCHS="$2";           shift 2;;
        --batch-size)   BATCH_SIZE="$2";       shift 2;;
        --grad-accum)   GRAD_ACCUM="$2";       shift 2;;
        --lr)           LR="$2";               shift 2;;
        --num-frames)   NUM_FRAMES="$2";       shift 2;;
        --lora)         USE_LORA=true;         shift;;
        *)              EXTRA_ARGS+=("$1");    shift;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash scripts_gs/train_vln.sh --output DIR --stage STAGE [options]"
    echo ""
    echo "Stages:"
    echo "  stage-one   Fine-tune LLaVA-Video on trajectory data (default)"
    echo "  dagger      DAgger: collect new data with current model + retrain"
    echo "  stage-two   Co-training with additional QA data"
    echo ""
    echo "Options:"
    echo "  --output DIR         Output directory for checkpoints (required)"
    echo "  --stage STAGE        Training stage: stage-one|dagger|stage-two (default: stage-one)"
    echo "  --num-gpus N         Number of GPUs (default: 1)"
    echo "  --ckpt PATH          Base checkpoint (default: LLaVA-Video-7B-Qwen2 for stage-one)"
    echo "  --epochs N           Number of epochs (default: 1)"
    echo "  --batch-size N       Per-device batch size (default: 2)"
    echo "  --grad-accum N       Gradient accumulation steps (default: 2)"
    echo "  --lr RATE            Learning rate (default: 2e-5)"
    echo "  --num-frames N       Frames per sample (default: 32)"
    echo "  --lora               Enable LoRA training (freeze backbone, reduced frames/context;"
    echo "                       fits on a single 24GB GPU. Without this flag, standard full"
    echo "                       fine-tune is used, which requires >=40GB VRAM per GPU)"
    echo ""
    echo "Extra arguments are forwarded to streamvln_train.py."
    exit 1
fi

if [[ "$STAGE" != "stage-one" && "$STAGE" != "dagger" && "$STAGE" != "stage-two" ]]; then
    echo "ERROR: --stage must be one of: stage-one, dagger, stage-two"
    exit 1
fi

# ── Resolve paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STREAMVLN_ROOT="$(cd "$PROJECT_ROOT/../StreamVLN" && pwd)"

TRAJ_DATA="$PROJECT_ROOT/data/scene_datasets/gs_scenes/trajectory_data/vln"
DEEPSPEED_CFG="$STREAMVLN_ROOT/scripts/zero2.json"

# ── Verify data and model checkpoints ────────────────────────────────
if [[ ! -f "$TRAJ_DATA/annotations.json" ]]; then
    echo "ERROR: Trajectory data not found at $TRAJ_DATA/annotations.json"
    echo "Run:  python scripts_gs/generate_vln_trajectories.py"
    exit 1
fi

if [[ ! -d "$STREAMVLN_ROOT/checkpoints/siglip-so400m-patch14-384" ]]; then
    echo "ERROR: SigLIP vision tower not found at $STREAMVLN_ROOT/checkpoints/siglip-so400m-patch14-384"
    echo "Run:  bash scripts_gs/setup_vln.sh"
    exit 1
fi

# ── Default checkpoint per stage ─────────────────────────────────────
if [[ -z "$CKPT_PATH" ]]; then
    case "$STAGE" in
        stage-one)
            # Use local checkpoint if available, else HuggingFace ID
            if [[ -d "$STREAMVLN_ROOT/checkpoints/LLaVA-Video-7B-Qwen2" ]]; then
                CKPT_PATH="$STREAMVLN_ROOT/checkpoints/LLaVA-Video-7B-Qwen2"
            else
                CKPT_PATH="lmms-lab/LLaVA-Video-7B-Qwen2"
            fi
            ;;
        dagger|stage-two)
            echo "ERROR: --ckpt is required for stage '$STAGE'"
            exit 1
            ;;
    esac
fi

# Resolve checkpoint to absolute path (since we cd later)
if [[ -d "$CKPT_PATH" || -f "$CKPT_PATH" ]]; then
    CKPT_PATH="$(realpath "$CKPT_PATH")"
fi
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_DIR"

# ── Auto-merge LoRA checkpoint for continuation stages ──────────────
# LoRA training produces adapter_model.bin. DAgger/stage-two continuation
# with --lora needs a full (merged) model so new LoRA adapters can be applied.
if [[ -f "$CKPT_PATH/adapter_model.bin" && "$USE_LORA" == "true" ]]; then
    MERGED_DIR="${CKPT_PATH}_merged"
    if [[ ! -f "$MERGED_DIR/config.json" ]]; then
        echo "── Merging LoRA checkpoint into base model ──"
        echo "  LoRA ckpt : $CKPT_PATH"
        echo "  Merged out: $MERGED_DIR"
        PYTHONPATH="$STREAMVLN_ROOT:$STREAMVLN_ROOT/streamvln:${PYTHONPATH:-}" python -c "
import torch, os, shutil, json
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
from peft import PeftModel
from transformers import AutoTokenizer, AutoConfig
from streamvln.model.stream_video_vln import StreamVLNForCausalLM
siglip = '$STREAMVLN_ROOT/checkpoints/siglip-so400m-patch14-384'
default_base = '$STREAMVLN_ROOT/checkpoints/LLaVA-Video-7B-Qwen2'
ckpt_path, merged_dir = '$CKPT_PATH', '$MERGED_DIR'
# Read actual base model from adapter_config (supports chained LoRA stages)
ac = os.path.join(ckpt_path, 'adapter_config.json')
base_id = json.load(open(ac)).get('base_model_name_or_path', default_base) if os.path.exists(ac) else default_base
if not os.path.isdir(base_id): base_id = default_base
print(f'  Base model: {base_id}')
cfg = AutoConfig.from_pretrained(base_id)
if hasattr(cfg, 'mm_vision_tower'): cfg.mm_vision_tower = siglip
print('  Loading base model...')
model = StreamVLNForCausalLM.from_pretrained(base_id, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation='eager')
nlt = os.path.join(ckpt_path, 'non_lora_trainables.bin')
if os.path.exists(nlt):
    d = torch.load(nlt, map_location='cpu')
    model.load_state_dict({k.replace('base_model.model.',''):v for k,v in d.items()}, strict=False)
    print('  Loaded non-LoRA trainables')
print('  Loading LoRA adapter...')
model = PeftModel.from_pretrained(model, ckpt_path)
print('  Merging...')
model = model.merge_and_unload()
print('  Saving merged model...')
model.save_pretrained(merged_dir)
AutoTokenizer.from_pretrained(base_id).save_pretrained(merged_dir)
for f in ['config.json','generation_config.json']:
    s = os.path.join(ckpt_path, f)
    if os.path.exists(s): shutil.copy2(s, merged_dir)
print('  Done.')
"
        if [[ $? -ne 0 ]]; then
            echo "ERROR: LoRA merge failed"
            exit 1
        fi
    else
        echo "  Using existing merged checkpoint: $MERGED_DIR"
    fi
    CKPT_PATH="$MERGED_DIR"
fi

echo "=========================================="
echo " StreamVLN Training (GS Scenes)"
echo "=========================================="
echo "  Stage        : $STAGE"
echo "  Project root : $PROJECT_ROOT"
echo "  StreamVLN    : $STREAMVLN_ROOT"
echo "  Checkpoint   : $CKPT_PATH"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Num GPUs     : $NUM_GPUS"
echo "  Epochs       : $EPOCHS"
echo "  Batch size   : $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Num frames   : $NUM_FRAMES"
echo "  LoRA         : $USE_LORA"
echo "  Traj data    : $TRAJ_DATA"
echo "=========================================="
echo ""

# ── Build training command ───────────────────────────────────────────
cd "$STREAMVLN_ROOT"
export PYTHONPATH="$STREAMVLN_ROOT:$STREAMVLN_ROOT/streamvln:${PYTHONPATH:-}"
# Consumer RTX cards (30xx/40xx/50xx) need P2P disabled; professional cards (A6000 etc.) do not
GPU_NAME="$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | head -1)"
if [[ "$GPU_NAME" == *"GeForce"* ]] || [[ "$GPU_NAME" =~ RTX\ [2-9][0-9]{3} ]]; then
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
fi
# Prevent HF network calls (all models should be local)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Common arguments
TRAIN_ARGS=(
    --model_name_or_path "$CKPT_PATH"
    --version qwen_1_5
    --video_folder "$TRAJ_DATA"
    --num_history "$NUM_HISTORY"
    --num_future_steps "$NUM_FUTURE_STEPS"
    --num_frames "$NUM_FRAMES"
    --vision_tower "$STREAMVLN_ROOT/checkpoints/siglip-so400m-patch14-384"
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --bf16 True
    --output_dir "$OUTPUT_DIR"
    --num_train_epochs "$EPOCHS"
    --per_device_train_batch_size "$BATCH_SIZE"
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --evaluation_strategy "no"
    --save_strategy "epoch"
    --save_total_limit 1
    --learning_rate "$LR"
    --mm_vision_tower_lr 5e-6
    --weight_decay 0.
    --warmup_ratio 0.075
    --lr_scheduler_type "cosine_with_min_lr"
    --lr_scheduler_kwargs '{"min_lr": 1.85e-05}'
    --logging_steps 10
    --tf32 True
    --model_max_length 32768
    --gradient_checkpointing True
    --dataloader_num_workers 8
    --lazy_preprocess True
    --dataloader_drop_last True
    --report_to none
)

# Stage-specific modifications
case "$STAGE" in
    stage-one)
        TRAIN_ARGS+=(--data_augmentation True)
        RUN_NAME="streamvln_gs_stage1"
        ;;
    dagger)
        # DAgger uses data from both original trajectories and DAgger-collected data
        DAGGER_DATA="$PROJECT_ROOT/data/scene_datasets/gs_scenes/trajectory_data/vln_dagger"
        if [[ -d "$DAGGER_DATA" && -f "$DAGGER_DATA/annotations.json" ]]; then
            TRAIN_ARGS=(${TRAIN_ARGS[@]/--video_folder*/})
            TRAIN_ARGS+=(--video_folder "$TRAJ_DATA,$DAGGER_DATA")
        fi
        TRAIN_ARGS+=(--data_augmentation True)
        RUN_NAME="streamvln_gs_dagger"
        ;;
    stage-two)
        # Stage-2 co-training with QA datasets (if available)
        TRAIN_ARGS+=(--data_augmentation True)
        TRAIN_ARGS+=(--multi_task_training False)
        RUN_NAME="streamvln_gs_stage2"
        ;;
esac

TRAIN_ARGS+=(--run_name "$RUN_NAME")

# ── LoRA vs standard full fine-tune ─────────────────────────────────
if [[ "$USE_LORA" == "true" ]]; then
    # LoRA mode: freeze backbone, train only projector + LoRA adapters,
    # reduced frames/context to fit on a single 24GB GPU.
    TRAIN_ARGS+=(
        --attn_implementation eager
        --freeze_backbone True
        --tune_mm_mlp_adapter True
        --mm_tunable_parts "mm_mlp_adapter"
        --lora_enable True --lora_r 64 --lora_alpha 128
        --num_frames 4
        --num_history 2
        --model_max_length 2048
    )
    echo "  Mode: LoRA (adapter-only, 4 frames, 2K context)"
else
    # Standard full fine-tune matching StreamVLN official config
    TRAIN_ARGS+=(
        --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model"
        --image_aspect_ratio anyres_max_9
        --image_grid_pinpoints "(1x1),...,(6x6)"
        --torch_compile True
        --torch_compile_backend "inductor"
    )
    echo "  Mode: Standard full fine-tune (torch_compile)"
fi

# Add DeepSpeed for multi-GPU training only
if [[ $NUM_GPUS -gt 1 && -f "$DEEPSPEED_CFG" ]]; then
    TRAIN_ARGS=(--deepspeed "$DEEPSPEED_CFG" "${TRAIN_ARGS[@]}")
fi

# ── Launch training ──────────────────────────────────────────────────
if [[ $NUM_GPUS -gt 1 ]]; then
    exec torchrun --nproc_per_node="$NUM_GPUS" \
        streamvln/streamvln_train.py \
        "${TRAIN_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
else
    exec python -u streamvln/streamvln_train.py \
        "${TRAIN_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
fi
