#!/usr/bin/env bash
set -e

BLENDER_BIN="/home/yanci/3D/blender-5.1.2-linux-x64/blender"

if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  bash tools/process_fbx.sh <input_fbx_relative_path> <clip_name>"
  echo ""
  echo "Example:"
  echo "  bash tools/process_fbx.sh assets/fbx/Walking.fbx walk"
  exit 1
fi

INPUT_FBX="$1"
CLIP_NAME="$2"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RAW_DIR="${PROJECT_ROOT}/assets/baked_raw"
OUT_DIR="${PROJECT_ROOT}/assets/humans"

RAW_NPZ="${RAW_DIR}/${CLIP_NAME}_raw.npz"
OUT_NPZ="${OUT_DIR}/${CLIP_NAME}_meters_inplace.npz"

mkdir -p "${RAW_DIR}"
mkdir -p "${OUT_DIR}"

echo "Project root: ${PROJECT_ROOT}"
echo "Blender:      ${BLENDER_BIN}"
echo "Input FBX:    ${PROJECT_ROOT}/${INPUT_FBX}"
echo "Raw NPZ:      ${RAW_NPZ}"
echo "Output NPZ:   ${OUT_NPZ}"
echo ""

if [ ! -f "${BLENDER_BIN}" ]; then
  echo "ERROR: Blender not found: ${BLENDER_BIN}"
  exit 1
fi

if [ ! -f "${PROJECT_ROOT}/${INPUT_FBX}" ]; then
  echo "ERROR: FBX not found: ${PROJECT_ROOT}/${INPUT_FBX}"
  exit 1
fi

echo "Step 1: bake FBX to raw NPZ"
"${BLENDER_BIN}" --background --python "${PROJECT_ROOT}/tools/bake_fbx_to_npz.py" -- \
  --fbx "${PROJECT_ROOT}/${INPUT_FBX}" \
  --out "${RAW_NPZ}" \
  --fps 30

echo ""
echo "Step 2: normalize to meters + remove root motion"
python3 "${PROJECT_ROOT}/tools/normalize_baked_human.py" \
  --in_npz "${RAW_NPZ}" \
  --out_npz "${OUT_NPZ}" \
  --scale 0.01 \
  --up_axis y \
  --remove_root_motion \
  --ground_align

echo ""
echo "Step 3: check output"
python3 "${PROJECT_ROOT}/tools/check_baked_human.py" \
  --npz "${OUT_NPZ}" \
  --up_axis y

echo ""
echo "Done."
echo "Final output:"
echo "  ${OUT_NPZ}"
