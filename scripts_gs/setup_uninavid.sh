#!/usr/bin/env bash
#
# One-time setup for Uni-NaVid on Gaussian Splatting scenes.
#
# What this script does:
#   1. Creates conda env habitat-gs-uni-navid (cloned from habitat-gs)
#   2. Installs Uni-NaVid Python dependencies
#   3. Downloads required model checkpoints (EVA-ViT-G, Vicuna-7B, Uni-NaVid)
#
# Prerequisites:
#   - habitat-gs conda env already set up
#   - Uni-NaVid repo cloned at ../Uni-NaVid (relative to habitat-gs/)
#
# Usage:
#   bash scripts_gs/setup_uninavid.sh
#   bash scripts_gs/setup_uninavid.sh --skip-download
#   bash scripts_gs/setup_uninavid.sh --skip-env
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
SKIP_ENV=false
SKIP_DEPS=false
SKIP_DOWNLOAD=false
PROXY=""

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-env)       SKIP_ENV=true;      shift;;
        --skip-deps)      SKIP_DEPS=true;     shift;;
        --skip-download)  SKIP_DOWNLOAD=true; shift;;
        --proxy)          PROXY="$2";         shift 2;;
        *)                shift;;
    esac
done

# ── Resolve paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UNINAVID_ROOT="$(cd "$PROJECT_ROOT/../Uni-NaVid" && pwd 2>/dev/null)" || {
    echo "ERROR: Uni-NaVid repo not found at $PROJECT_ROOT/../Uni-NaVid"
    exit 1
}

echo "=========================================="
echo " Uni-NaVid Setup for GS Scenes"
echo "=========================================="
echo "  habitat-gs : $PROJECT_ROOT"
echo "  Uni-NaVid  : $UNINAVID_ROOT"
echo "=========================================="
echo ""

# ══════════════════════════════════════════════════════════════════════
#  Step 1: Create conda environment
# ══════════════════════════════════════════════════════════════════════
ENV_NAME="habitat-gs-uni-navid"

if [[ "$SKIP_ENV" == "false" ]]; then
    echo "── Step 1: Creating conda env '$ENV_NAME' ──"
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "  Environment '$ENV_NAME' already exists. Skipping clone."
    else
        echo "  Cloning habitat-gs -> $ENV_NAME ..."
        conda create -n "$ENV_NAME" --clone habitat-gs -y 2>&1 | tail -3
        echo "  Done."
    fi
    echo ""
else
    echo "── Step 1: Skipped (--skip-env) ──"
    echo ""
fi

# Activate the environment for subsequent steps
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "  Active env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo ""

# ══════════════════════════════════════════════════════════════════════
#  Step 2: Install Python dependencies
# ══════════════════════════════════════════════════════════════════════
if [[ "$SKIP_DEPS" == "false" ]]; then
    echo "── Step 2: Installing Python dependencies ──"

    # Core Uni-NaVid dependencies (compatible with torch 2.5.x / Python 3.12)
    pip install \
        "transformers>=4.38.0,<4.46" \
        "accelerate>=0.28.0" \
        "peft>=0.8.0" \
        "deepspeed>=0.13.0" \
        "bitsandbytes>=0.42.0" \
        decord einops "timm>=0.9.0" fairscale \
        "sentencepiece>=0.1.99" \
        "scikit-learn>=1.2.0" \
        gradio wandb \
        2>&1 | tail -5

    # Install flash-attn (optional, falls back to eager attention)
    echo "  Installing flash-attn (may take a few minutes) ..."
    pip install flash-attn --no-build-isolation 2>&1 | tail -3 || \
        echo "  WARNING: flash-attn installation failed. Will use eager attention."

    # Install Uni-NaVid package in editable mode
    echo "  Installing Uni-NaVid package ..."
    cd "$UNINAVID_ROOT"
    pip install -e . --no-deps 2>&1 | tail -3

    echo "  Dependencies installed."
    echo ""
else
    echo "── Step 2: Skipped (--skip-deps) ──"
    echo ""
fi

# ══════════════════════════════════════════════════════════════════════
#  Step 3: Download model checkpoints
# ══════════════════════════════════════════════════════════════════════
if [[ "$SKIP_DOWNLOAD" == "false" ]]; then
    echo "── Step 3: Downloading model checkpoints ──"

    MODEL_ZOO="$UNINAVID_ROOT/model_zoo"
    mkdir -p "$MODEL_ZOO"

    # Set proxy if provided
    if [[ -n "$PROXY" ]]; then
        export http_proxy="$PROXY"
        export https_proxy="$PROXY"
    fi

    # 1. EVA-ViT-G vision encoder (~3.5GB)
    if [[ -f "$MODEL_ZOO/eva_vit_g.pth" ]]; then
        echo "  EVA-ViT-G: already downloaded"
    else
        echo "  Downloading EVA-ViT-G (~3.5GB) ..."
        wget -q --show-progress -O "$MODEL_ZOO/eva_vit_g.pth" \
            "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
        echo "  EVA-ViT-G: done"
    fi

    # 2. Vicuna-7B v1.5 (~13GB)
    if [[ -f "$MODEL_ZOO/vicuna-7b-v1.5/config.json" ]]; then
        echo "  Vicuna-7B-v1.5: already downloaded"
    else
        echo "  Downloading Vicuna-7B-v1.5 (~13GB) ..."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('lmsys/vicuna-7b-v1.5',
    local_dir='$MODEL_ZOO/vicuna-7b-v1.5', max_workers=2)
print('  Vicuna-7B-v1.5: done')
"
    fi

    # 3. Uni-NaVid pretrained checkpoint (~14GB)
    if [[ -f "$MODEL_ZOO/uninavid-7b-full-224-video-fps-1-grid-2/config.json" ]]; then
        echo "  Uni-NaVid pretrained: already downloaded"
    else
        echo "  Downloading Uni-NaVid pretrained checkpoint (~14GB) ..."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('Jzzhang/Uni-NaVid',
    local_dir='$MODEL_ZOO/_uninavid_hf_tmp', max_workers=2,
    allow_patterns='uninavid-7b-full-224-video-fps-1-grid-2/*')
import shutil, os
src = '$MODEL_ZOO/_uninavid_hf_tmp/uninavid-7b-full-224-video-fps-1-grid-2'
dst = '$MODEL_ZOO/uninavid-7b-full-224-video-fps-1-grid-2'
if os.path.isdir(src):
    shutil.move(src, dst)
    shutil.rmtree('$MODEL_ZOO/_uninavid_hf_tmp', ignore_errors=True)
print('  Uni-NaVid pretrained: done')
"
    fi

    echo ""
else
    echo "── Step 3: Skipped (--skip-download) ──"
    echo ""
fi

# ══════════════════════════════════════════════════════════════════════
#  Verify
# ══════════════════════════════════════════════════════════════════════
echo "── Verification ──"

ERRORS=0

# Check conda env
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  [OK] Conda env '$ENV_NAME'"
else
    echo "  [FAIL] Conda env '$ENV_NAME' not found"
    ERRORS=$((ERRORS+1))
fi

# Check key packages
for pkg in transformers peft accelerate deepspeed decord einops; do
    if python -c "import $pkg" 2>/dev/null; then
        echo "  [OK] $pkg"
    else
        echo "  [FAIL] $pkg not importable"
        ERRORS=$((ERRORS+1))
    fi
done

# Check uninavid package
if python -c "from uninavid.model.builder import load_pretrained_model" 2>/dev/null; then
    echo "  [OK] uninavid package"
else
    echo "  [FAIL] uninavid package not importable"
    ERRORS=$((ERRORS+1))
fi

# Check model weights
MODEL_ZOO="$UNINAVID_ROOT/model_zoo"
if [[ -f "$MODEL_ZOO/eva_vit_g.pth" ]]; then
    echo "  [OK] EVA-ViT-G checkpoint"
else
    echo "  [WARN] EVA-ViT-G not found (needed for training from scratch)"
fi

if [[ -f "$MODEL_ZOO/vicuna-7b-v1.5/config.json" ]]; then
    echo "  [OK] Vicuna-7B-v1.5"
else
    echo "  [WARN] Vicuna-7B-v1.5 not found (needed for stage-1 training)"
fi

if [[ -f "$MODEL_ZOO/uninavid-7b-full-224-video-fps-1-grid-2/config.json" ]]; then
    echo "  [OK] Uni-NaVid pretrained checkpoint"
else
    echo "  [WARN] Uni-NaVid pretrained not found (needed for stage-2 fine-tuning)"
fi

# Check episodes
if [[ -f "$PROJECT_ROOT/data/scene_datasets/gs_scenes/episodes/vln/train/train.json.gz" ]]; then
    echo "  [OK] VLN episodes (train)"
else
    echo "  [WARN] VLN episodes not generated yet"
fi

echo ""
if [[ $ERRORS -eq 0 ]]; then
    echo "Setup complete! Next steps:"
    echo "  1. conda activate $ENV_NAME"
    echo "  2. python scripts_gs/generate_uninavid_trajectories.py"
    echo "  3. bash scripts_gs/train_uninavid.sh --output output/uninavid --stage stage-2"
    echo "  4. bash scripts_gs/eval_uninavid.sh --ckpt output/uninavid/<checkpoint>"
else
    echo "Setup completed with $ERRORS error(s). Please fix the issues above."
    exit 1
fi
