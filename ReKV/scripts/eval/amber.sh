#!/usr/bin/env bash
# Evaluate local Qwen2.5-VL-7B + ReKV on AMBER generative.
set -euo pipefail

ROOT="/root/autodl-tmp/Method/ReKV"
VLESS_ROOT="/root/autodl-tmp/Method/VLessHallu/VLessHallu"
AMBER_ROOT="/root/autodl-tmp/Method/dataset/AMBER"
CONFIG="${CONFIG:-${ROOT}/configs/local_qwen25.toml}"
VARIANT="${VARIANT:-rekv}"
export AMBER_LIMIT="${AMBER_LIMIT:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vlesshallu

export PYTHONPATH="${ROOT}:${VLESS_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "${ROOT}"
python -u run_benchmark.py --config "${CONFIG}" run \
  --variant "${VARIANT}" \
  --benchmark amber \
  --split generation

echo
echo "Score with:"
echo "  bash ${VLESS_ROOT}/scripts/eval/score_amber.sh ${AMBER_ROOT}/results/ReKV/Qwen2.5-VL-7B/amber_generative/amber_generative_${VARIANT}.json g"
