#!/usr/bin/env bash
# Score MM-ShiftKV AMBER responses with the official AMBER inference.py.
set -euo pipefail

AMBER_ROOT="/root/autodl-tmp/Method/dataset/AMBER"
RESPONSE_FILE="${1:-}"
EVAL_TYPE="${2:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mmshiftkv

if [[ -z "${RESPONSE_FILE}" ]]; then
  echo "Usage: bash scripts/eval/score_amber.sh <response.json> [evaluation_type]"
  echo "  evaluation_type: a (all), g (generative), d (discriminative), de/da/dr"
  echo "Example:"
  echo "  bash scripts/eval/score_amber.sh ${AMBER_ROOT}/results/MM-ShiftKV/Qwen2.5-VL-7B/amber_generative/amber_generative_shiftkv_64_0.1.json g"
  exit 1
fi

if [[ ! -f "${RESPONSE_FILE}" ]]; then
  echo "Response file not found: ${RESPONSE_FILE}"
  exit 1
fi

# Infer evaluation type from filename when not given.
if [[ -z "${EVAL_TYPE}" ]]; then
  base="$(basename "${RESPONSE_FILE}")"
  if [[ "${base}" == *"generative"* ]]; then
    EVAL_TYPE="g"
  elif [[ "${base}" == *"discriminative"* ]]; then
    EVAL_TYPE="d"
  else
    EVAL_TYPE="a"
  fi
fi

RESPONSE_FILE="$(python -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "${RESPONSE_FILE}")"

cd "${AMBER_ROOT}"
python inference.py \
    --inference_data "${RESPONSE_FILE}" \
    --evaluation_type "${EVAL_TYPE}" \
    --annotation data/annotations.json \
    --word_association data/relation.json \
    --safe_words data/safe_words.txt \
    --metrics data/metrics.txt
