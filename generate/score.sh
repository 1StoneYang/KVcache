#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT="${1:-${ROOT}/outputs/amber_baseline/responses.json}"
RESULT="$(realpath "$RESULT")"
cd /root/autodl-tmp/Method/dataset/AMBER
exec /root/miniconda3/envs/mmshiftkv/bin/python inference.py \
  --inference_data "$RESULT" --evaluation_type g \
  --annotation data/annotations.json --word_association data/relation.json \
  --safe_words data/safe_words.txt --metrics data/metrics.txt
