#!/usr/bin/env bash
# Score VLessHallu AMBER responses with the official inference.py.
set -euo pipefail

AMBER_ROOT="/root/autodl-tmp/Method/dataset/AMBER"
RESPONSE_FILE="${1:-}"
EVAL_TYPE="${2:-g}"

source /root/miniconda3/etc/profile.d/conda.sh
if conda env list | grep -qE '^vlesshallu[[:space:]]'; then
  conda activate vlesshallu
else
  conda activate mmshiftkv
fi

if [[ -z "${RESPONSE_FILE}" ]]; then
  echo "Usage: bash scripts/eval/score_amber.sh <response.json> [evaluation_type]"
  echo "  evaluation_type: a (all), g (generative), d (discriminative)"
  exit 1
fi

if [[ ! -f "${RESPONSE_FILE}" ]]; then
  echo "Response file not found: ${RESPONSE_FILE}"
  exit 1
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
