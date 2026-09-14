#!/usr/bin/env bash
# Evaluate Qwen2.5-VL-7B + MM-ShiftKV + Recycling Bin on local AMBER.
set -euo pipefail

RE_ROOT="/root/autodl-tmp/Method/MM-ShiftKV-Re"
MMSHIFT_ROOT="/root/autodl-tmp/Method/MM-ShiftKV-main"
MODEL_PATH="/root/autodl-tmp/model/Qwen2.5-VL-7B"
AMBER_ROOT="/root/autodl-tmp/Method/dataset/AMBER"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mmshiftkv

cd "${MMSHIFT_ROOT}"

export PYTHONPATH="${RE_ROOT}:${MMSHIFT_ROOT}:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export AMBER_ROOT
export AMBER_IMAGE_DIR="${AMBER_ROOT}/image"

TASK="${TASK:-amber_generative}"
METHOD="${METHOD:-shiftkv_re}"
BUDGET="${BUDGET:-64}"
BIN_SIZE="${BIN_SIZE:-20}"
RATIO="${RATIO:-0.1}"
MASK_RATIO="${MASK_RATIO:-0.1}"
LIMIT="${LIMIT:-}"
PORT="${PORT:-54324}"
PROJECT="${PROJECT:-MM-ShiftKV-Re}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"

export METHOD BUDGET BIN_SIZE RATIO MASK_RATIO PROJECT MODEL_NAME

OUT_DIR="${AMBER_ROOT}/results/${PROJECT}/${MODEL_NAME}/${TASK}"
mkdir -p "${OUT_DIR}"
export AMBER_OUTPUT="${OUT_DIR}/${TASK}_${METHOD}_${BUDGET}_b${BIN_SIZE}.json"

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS+=(--limit "${LIMIT}")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 -u -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port "${PORT}" \
    -m mmshift_re.run_eval \
    --model qwen2_5_vl \
    --model_args "pretrained=${MODEL_PATH},use_flash_attention_2=True,device_map=cuda:0" \
    --tasks "${TASK}" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "qwen25vl_${METHOD}_b${BIN_SIZE}" \
    --output_path "${OUT_DIR}/logs/" \
    --gen_kwargs temperature=0 \
    --verbosity=DEBUG \
    "${LIMIT_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/${TASK}_${METHOD}_${BUDGET}_b${BIN_SIZE}.log"

echo
echo "Generation finished. Responses: ${AMBER_OUTPUT}"
echo "Score with:"
echo "  bash ${MMSHIFT_ROOT}/scripts/eval/score_amber.sh ${AMBER_OUTPUT} g"
