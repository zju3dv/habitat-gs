#!/usr/bin/env bash
#
# One-time setup for StreamVLN on Gaussian Splatting scenes.
#
# What this script does:
#   1. Creates conda env habitat-gs-streamvln (cloned from habitat-gs)
#   2. Applies compatibility patches to the StreamVLN repo
#      (habitat 0.3.3 compat, vision_tower override, quantization fixes)
#   3. Installs Python dependencies
#   4. Downloads required model checkpoints (LLaVA-Video-7B-Qwen2, SigLIP)
#
# Prerequisites:
#   - habitat-gs conda env already set up
#   - StreamVLN repo cloned at ../StreamVLN (relative to habitat-gs/)
#
# Usage:
#   bash scripts_gs/setup_vln.sh
#   bash scripts_gs/setup_vln.sh --skip-download   # skip model downloads
#   bash scripts_gs/setup_vln.sh --skip-patch       # skip code patches
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
SKIP_ENV=false
SKIP_PATCH=false
SKIP_DOWNLOAD=false
SKIP_DEPS=false
HF_TOKEN=""
PROXY=""

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-env)       SKIP_ENV=true;      shift;;
        --skip-patch)     SKIP_PATCH=true;    shift;;
        --skip-download)  SKIP_DOWNLOAD=true; shift;;
        --skip-deps)      SKIP_DEPS=true;     shift;;
        --hf-token)       HF_TOKEN="$2";      shift 2;;
        --proxy)          PROXY="$2";          shift 2;;
        *)                shift;;
    esac
done

# ── Resolve paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STREAMVLN_ROOT="$(cd "$PROJECT_ROOT/../StreamVLN" && pwd 2>/dev/null)" || {
    echo "ERROR: StreamVLN repo not found at $PROJECT_ROOT/../StreamVLN"
    echo "Please clone it:  git clone <streamvln-repo-url> $PROJECT_ROOT/../StreamVLN"
    exit 1
}

PATCH_FILE="$SCRIPT_DIR/streamvln_compat.patch"

echo "=========================================="
echo " StreamVLN Setup for GS Scenes"
echo "=========================================="
echo "  habitat-gs   : $PROJECT_ROOT"
echo "  StreamVLN    : $STREAMVLN_ROOT"
echo "=========================================="
echo ""

# ══════════════════════════════════════════════════════════════════════
#  Step 0: Create conda environment
# ══════════════════════════════════════════════════════════════════════
ENV_NAME="habitat-gs-streamvln"

if [[ "$SKIP_ENV" == "false" ]]; then
    echo "── Step 0: Creating conda env '$ENV_NAME' ──"
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "  Environment '$ENV_NAME' already exists. Skipping clone."
    else
        echo "  Cloning habitat-gs -> $ENV_NAME ..."
        conda create -n "$ENV_NAME" --clone habitat-gs -y 2>&1 | tail -3
        echo "  Done."
    fi
    echo ""
else
    echo "── Step 0: Skipped (--skip-env) ──"
    echo ""
fi

# Activate the environment for subsequent steps
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "  Active env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo ""

# ══════════════════════════════════════════════════════════════════════
#  Step 1: Apply compatibility patches
# ══════════════════════════════════════════════════════════════════════
if [[ "$SKIP_PATCH" == "false" ]]; then
    echo "── Step 1: Applying compatibility patches ──"

    if [[ ! -f "$PATCH_FILE" ]]; then
        echo "ERROR: Patch file not found: $PATCH_FILE"
        exit 1
    fi

    cd "$STREAMVLN_ROOT"

    # Check if patches are already applied
    if git diff --quiet 2>/dev/null; then
        echo "  Applying streamvln_compat.patch ..."
        if git apply --check "$PATCH_FILE" 2>/dev/null; then
            git apply "$PATCH_FILE"
            echo "  Patches applied successfully."
        else
            echo "  Patches appear to be already applied or conflict. Trying force ..."
            git apply --3way "$PATCH_FILE" 2>/dev/null || \
                echo "  WARNING: Could not auto-apply. Patches may already be in place."
        fi
    else
        echo "  StreamVLN repo has uncommitted changes. Checking if patches needed ..."
        # Check each critical fix
        if grep -q "try_cv2_import" streamvln/habitat_extensions/measures.py 2>/dev/null; then
            echo "  Applying measures.py patch (habitat 0.3.3 compat) ..."
            sed -i 's/from habitat.core.utils import try_cv2_import//' \
                streamvln/habitat_extensions/measures.py
            sed -i 's/cv2 = try_cv2_import()/try:\n    import cv2\nexcept ImportError:\n    cv2 = None/' \
                streamvln/habitat_extensions/measures.py
        fi
        echo "  Skipping full patch (repo already modified). Verify manually if needed."
    fi

    echo "  Patch summary:"
    echo "    - measures.py: habitat 0.3.3 compatibility (try_cv2_import removed)"
    echo "    - streamvln_train.py: --vision_tower CLI override, quantization fixes"
    echo "    - siglip_encoder.py: low_cpu_mem_usage for quantized loading"
    echo ""
else
    echo "── Step 1: Skipped (--skip-patch) ──"
    echo ""
fi

# ══════════════════════════════════════════════════════════════════════
#  Step 2: Install Python dependencies
# ══════════════════════════════════════════════════════════════════════
if [[ "$SKIP_DEPS" == "false" ]]; then
    echo "── Step 2: Installing Python dependencies ──"
    pip install transformers==4.45.1 accelerate==0.28.0 \
        deepspeed peft bitsandbytes openai \
        tyro trl einops einops-exts timm decord sentencepiece mpi4py \
        2>&1 | tail -3
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

    CKPT_DIR="$STREAMVLN_ROOT/checkpoints"
    mkdir -p "$CKPT_DIR"

    # Set proxy if provided or auto-detect
    if [[ -n "$PROXY" ]]; then
        export http_proxy="$PROXY"
        export https_proxy="$PROXY"
    fi

    # Download LLaVA-Video-7B-Qwen2 (base model, ~15GB)
    if [[ -f "$CKPT_DIR/LLaVA-Video-7B-Qwen2/config.json" ]]; then
        echo "  LLaVA-Video-7B-Qwen2: already downloaded"
    else
        echo "  Downloading LLaVA-Video-7B-Qwen2 (~15GB) ..."
        python -c "
import os
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
from huggingface_hub import snapshot_download
snapshot_download('lmms-lab/LLaVA-Video-7B-Qwen2',
    local_dir='$CKPT_DIR/LLaVA-Video-7B-Qwen2',
    max_workers=1, token='$HF_TOKEN' or None)
print('  LLaVA-Video-7B-Qwen2: done')
"
    fi

    # Download SigLIP vision tower (~3.3GB)
    if [[ -f "$CKPT_DIR/siglip-so400m-patch14-384/model.safetensors" ]]; then
        echo "  SigLIP vision tower: already downloaded"
    else
        echo "  Downloading SigLIP vision tower (~3.3GB) ..."
        python -c "
import os
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
from huggingface_hub import snapshot_download
snapshot_download('google/siglip-so400m-patch14-384',
    local_dir='$CKPT_DIR/siglip-so400m-patch14-384',
    max_workers=1, token='$HF_TOKEN' or None)
print('  SigLIP: done')
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

# Check patches
cd "$STREAMVLN_ROOT"
if grep -q "try_cv2_import" streamvln/habitat_extensions/measures.py 2>/dev/null; then
    echo "  [FAIL] measures.py patch not applied"
    ERRORS=$((ERRORS+1))
else
    echo "  [OK] measures.py patched"
fi

if grep -q "overwrite_config\[\"mm_vision_tower\"\]" streamvln/streamvln_train.py 2>/dev/null; then
    echo "  [OK] streamvln_train.py patched (vision_tower override)"
else
    echo "  [FAIL] streamvln_train.py vision_tower override missing"
    ERRORS=$((ERRORS+1))
fi

# Check models
if [[ -f "$STREAMVLN_ROOT/checkpoints/LLaVA-Video-7B-Qwen2/config.json" ]]; then
    echo "  [OK] LLaVA-Video-7B-Qwen2 checkpoint"
else
    echo "  [FAIL] LLaVA-Video-7B-Qwen2 not found"
    ERRORS=$((ERRORS+1))
fi

if [[ -f "$STREAMVLN_ROOT/checkpoints/siglip-so400m-patch14-384/model.safetensors" ]]; then
    echo "  [OK] SigLIP vision tower"
else
    echo "  [FAIL] SigLIP model not found"
    ERRORS=$((ERRORS+1))
fi

# Check episodes
if [[ -f "$PROJECT_ROOT/data/scene_datasets/gs_scenes/episodes/vln/train/train.json.gz" ]]; then
    echo "  [OK] VLN episodes (train)"
else
    echo "  [WARN] VLN episodes not generated yet (run generate_vln_episodes.py)"
fi

# Check trajectory data
if [[ -f "$PROJECT_ROOT/data/scene_datasets/gs_scenes/trajectory_data/vln/annotations.json" ]]; then
    echo "  [OK] Trajectory data"
else
    echo "  [WARN] Trajectory data not generated yet (run generate_vln_trajectories.py)"
fi

echo ""
if [[ $ERRORS -eq 0 ]]; then
    echo "Setup complete! You can now run:"
    echo "  bash scripts_gs/train_vln.sh --output <dir> --stage stage-one"
    echo "  bash scripts_gs/eval_vln.sh --ckpt <checkpoint>"
else
    echo "Setup completed with $ERRORS error(s). Please fix the issues above."
    exit 1
fi
