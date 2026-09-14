#!/usr/bin/env bash
# Evaluate local Qwen2.5-VL-7B + VLessHallu on the AMBER generative split.
set -euo pipefail

ROOT="/root/autodl-tmp/Method/VLessHallu/VLessHallu"
MODEL_PATH="/root/autodl-tmp/model/Qwen2.5-VL-7B"
AMBER_ROOT="/root/autodl-tmp/Method/dataset/AMBER"
CONFIG="${CONFIG:-${ROOT}/configs/local_qwen25.toml}"

source /root/miniconda3/etc/profile.d/conda.sh
if conda env list | awk '{print $1}' | grep -qx vlesshallu; then
  conda activate vlesshallu
else
  echo "conda env vlesshallu not found; falling back to mmshiftkv"
  conda activate mmshiftkv
fi

cd "${ROOT}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
PY_SITE="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export LD_LIBRARY_PATH="${PY_SITE}/torch/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# VLessHallu method variant: baseline | kvsmooth | mmshift | myopia_score | prunehal | core4_no_rb
VARIANT="${VARIANT:-core4_no_rb}"
# Optional: AMBER_LIMIT=2 for a smoke run. Empty / 0 = all 1004 generative samples.
export AMBER_LIMIT="${AMBER_LIMIT:-}"

PROJECT="${PROJECT:-VLessHallu}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
TASK="amber_generative"
OUT_DIR="${AMBER_ROOT}/results/${PROJECT}/${MODEL_NAME}/${TASK}"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/${TASK}_${VARIANT}.log"

if [[ ! -f "${ROOT}/resources.lock.json" ]]; then
  python -u run_benchmark.py --config "${CONFIG}" prepare-local
fi

python -u run_benchmark.py --config "${CONFIG}" run \
  --variant "${VARIANT}" \
  --benchmark amber \
  --split generation \
  2>&1 | tee "${LOG}"

echo
echo "Generation finished."
echo "Shared responses: ${OUT_DIR}/${TASK}_${VARIANT}.json"
echo "Score with:"
echo "  bash ${ROOT}/scripts/eval/score_amber.sh ${OUT_DIR}/${TASK}_${VARIANT}.json g"
