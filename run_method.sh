#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VLESS_ROOT="/root/autodl-tmp/Method/VLessHallu/VLessHallu"
MMSHIFT_ROOT="/root/autodl-tmp/Method/MM-ShiftKV-main"
REKV_ROOT="/root/autodl-tmp/Method/ReKV"

usage() {
  echo "Usage:"
  echo "  bash /root/autodl-tmp/Method/run_method.sh vlesshallu [variant] [split] [config] [benchmark]"
  echo "  bash /root/autodl-tmp/Method/run_method.sh mmshiftkv [method] [limit] [budget] [task]"
  echo "  bash /root/autodl-tmp/Method/run_method.sh mmshiftkv-re [limit] [budget] [bin_size] [task]"
  echo "  bash /root/autodl-tmp/Method/run_method.sh rekv [limit] [config]"
  echo
  echo "mmshiftkv always runs MM-ShiftKV-main (lmms-eval + flash-attn). Do not use"
  echo "  run_method.sh vlesshallu mmshift  — that is VLessHallu's reimplementation."
  echo "mmshiftkv-re = original MM-ShiftKV + Myopia Recycling Bin (B=20 by default)."
  echo
  echo "Examples:"
  echo "  bash /root/autodl-tmp/Method/run_method.sh mmshiftkv"
  echo "  bash /root/autodl-tmp/Method/run_method.sh mmshiftkv shiftkv 1 64 amber_generative"
  echo "  bash /root/autodl-tmp/Method/run_method.sh mmshiftkv-re 1"
  echo "  bash /root/autodl-tmp/Method/run_method.sh mmshiftkv-re 200 64 20"
  echo "  bash /root/autodl-tmp/Method/run_method.sh vlesshallu baseline generation configs/local_qwen25_4090.toml"
  echo "  bash /root/autodl-tmp/Method/run_method.sh rekv 1"
}

family="${1:-}"
case "${family}" in
  vlesshallu)
    variant="${2:-core4_no_rb}"
    split="${3:-smoke}"
    config="${4:-configs/local_qwen25.toml}"
    benchmark="${5:-amber}"
    cd "${VLESS_ROOT}"
    exec /root/miniconda3/envs/vlesshallu/bin/python -u run_benchmark.py \
      --config "${config}" run --variant "${variant}" --benchmark "${benchmark}" --split "${split}"
    ;;
  rekv)
    limit="${2:-}"
    config="${3:-${REKV_ROOT}/configs/local_qwen25.toml}"
    export AMBER_LIMIT="${limit}"
    export PYTHONPATH="${REKV_ROOT}:${VLESS_ROOT}:${PYTHONPATH:-}"
    cd "${REKV_ROOT}"
    exec /root/miniconda3/envs/vlesshallu/bin/python -u run_benchmark.py \
      --config "${config}" run --variant rekv --benchmark amber --split generation
    ;;
  mmshiftkv)
    method="${2:-shiftkv}"
    limit="${3:-}"
    budget="${4:-64}"
    task="${5:-amber_generative}"
    cd "${MMSHIFT_ROOT}"
    exec env METHOD="${method}" LIMIT="${limit}" BUDGET="${budget}" TASK="${task}" \
      bash scripts/eval/amber_qwen.sh
    ;;
  mmshiftkv-re|mmshiftkv_re)
    limit="${2:-}"
    budget="${3:-64}"
    bin_size="${4:-20}"
    task="${5:-amber_generative}"
    exec env LIMIT="${limit}" BUDGET="${budget}" BIN_SIZE="${bin_size}" TASK="${task}" \
      bash /root/autodl-tmp/Method/MM-ShiftKV-Re/scripts/eval/amber_qwen.sh
    ;;
  *)
    usage
    exit 2
    ;;
esac
