#!/usr/bin/env bash
#
# One-click StreamVLN evaluation on Gaussian Splatting scenes.
#
# Usage:
#   bash scripts_gs/eval_vln.sh --ckpt /path/to/model_checkpoint
#   bash scripts_gs/eval_vln.sh --ckpt /path/to/model_checkpoint --save-video
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
CKPT_PATH=""
OUTPUT_DIR=""
NUM_GPUS=1
EVAL_SPLIT="val"
NUM_FRAMES=32
NUM_HISTORY=8
NUM_FUTURE_STEPS=4
SAVE_VIDEO="False"
EXTRA_ARGS=()

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)           CKPT_PATH="$2";          shift 2;;
        --output)         OUTPUT_DIR="$2";          shift 2;;
        --num-gpus)       NUM_GPUS="$2";            shift 2;;
        --split)          EVAL_SPLIT="$2";          shift 2;;
        --num-frames)     NUM_FRAMES="$2";          shift 2;;
        --save-video)     SAVE_VIDEO="True";        shift;;
        *)                EXTRA_ARGS+=("$1");       shift;;
    esac
done

if [[ -z "$CKPT_PATH" ]]; then
    echo "Usage: bash scripts_gs/eval_vln.sh --ckpt /path/to/checkpoint [options]"
    echo ""
    echo "Options:"
    echo "  --ckpt PATH          Path to trained StreamVLN checkpoint (required)"
    echo "  --output DIR         Output directory for results (default: auto)"
    echo "  --num-gpus N         Number of GPUs (default: 1)"
    echo "  --split SPLIT        Evaluation split: train|val (default: val)"
    echo "  --num-frames N       Frames per sample (default: 32)"
    echo "  --save-video         Save visualization videos"
    echo ""
    echo "Extra arguments are forwarded to streamvln_eval.py."
    exit 1
fi

# ── Resolve paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STREAMVLN_ROOT="$(cd "$PROJECT_ROOT/../StreamVLN" && pwd)"

CKPT_PATH="$(realpath "$CKPT_PATH")"
if [[ ! -e "$CKPT_PATH" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

# ── Verify model checkpoints ────────────────────────────────────────
if [[ ! -d "$STREAMVLN_ROOT/checkpoints/siglip-so400m-patch14-384" ]]; then
    echo "ERROR: SigLIP vision tower not found at $STREAMVLN_ROOT/checkpoints/siglip-so400m-patch14-384"
    echo "Run:  bash scripts_gs/setup_vln.sh"
    exit 1
fi

# ── Auto-merge LoRA checkpoint if needed ─────────────────────────────
# Single-GPU training produces LoRA checkpoints; eval needs a full model.
if [[ -f "$CKPT_PATH/adapter_model.bin" ]]; then
    MERGED_DIR="${CKPT_PATH}_merged"
    if [[ ! -f "$MERGED_DIR/config.json" ]]; then
        echo "── Merging LoRA checkpoint for evaluation ──"
        export PYTHONPATH="$STREAMVLN_ROOT:$STREAMVLN_ROOT/streamvln:${PYTHONPATH:-}"
        TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -c "
import torch, os, shutil, json
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
from peft import PeftModel
from transformers import AutoTokenizer, AutoConfig
from streamvln.model.stream_video_vln import StreamVLNForCausalLM
siglip = '$STREAMVLN_ROOT/checkpoints/siglip-so400m-patch14-384'
default_base = '$STREAMVLN_ROOT/checkpoints/LLaVA-Video-7B-Qwen2'
ckpt_path, merged_dir = '$CKPT_PATH', '$MERGED_DIR'
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
print('  Loading LoRA + merging...')
model = PeftModel.from_pretrained(model, ckpt_path).merge_and_unload()
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
    OUTPUT_DIR="$PROJECT_ROOT/results/vln/${CKPT_NAME}_${EVAL_SPLIT}"
fi
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo " StreamVLN Evaluation (GS Scenes)"
echo "=========================================="
echo "  Project root : $PROJECT_ROOT"
echo "  StreamVLN    : $STREAMVLN_ROOT"
echo "  Checkpoint   : $CKPT_PATH"
echo "  Split        : $EVAL_SPLIT"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Num GPUs     : $NUM_GPUS"
echo "  Save video   : $SAVE_VIDEO"
echo "=========================================="
echo ""

# ── Launch evaluation ────────────────────────────────────────────────
cd "$PROJECT_ROOT"  # CWD = habitat-gs so scene paths resolve

EVAL_ARGS=(
    --model_path "$CKPT_PATH"
    --habitat_config_path "$PROJECT_ROOT/data/scene_datasets/gs_scenes/configs/vln_gs_eval.yaml"
    --eval_split "$EVAL_SPLIT"
    --output_path "$OUTPUT_DIR"
    --num_frames "$NUM_FRAMES"
    --num_future_steps "$NUM_FUTURE_STEPS"
    --num_history "$NUM_HISTORY"
    --world_size "$NUM_GPUS"
)

# --save_video is a flag (store_true) in streamvln_eval.py
if [[ "$SAVE_VIDEO" == "True" ]]; then
    EVAL_ARGS+=(--save_video)
fi

# Add StreamVLN to Python path
export PYTHONPATH="$STREAMVLN_ROOT:$STREAMVLN_ROOT/streamvln:${PYTHONPATH:-}"

if [[ $NUM_GPUS -gt 1 ]]; then
    exec torchrun --nproc_per_node="$NUM_GPUS" \
        "$STREAMVLN_ROOT/streamvln/streamvln_eval.py" \
        "${EVAL_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
else
    exec python -u \
        "$STREAMVLN_ROOT/streamvln/streamvln_eval.py" \
        "${EVAL_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
fi
