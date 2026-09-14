#!/usr/bin/env bash
# Shared environment for MM-ShiftKV evaluation.
# Usage: source scripts/env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mmshiftkv

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export MMSHIFT_ROOT="/root/autodl-tmp/Method/MM-ShiftKV-main"
export MODEL_PATH="/root/autodl-tmp/model/Qwen2.5-VL-7B"
export AMBER_ROOT="/root/autodl-tmp/Method/dataset/AMBER"
export AMBER_IMAGE_DIR="${AMBER_ROOT}/image"

cd "${MMSHIFT_ROOT}"
echo "mmshiftkv env ready"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  AMBER_ROOT=${AMBER_ROOT}"
echo "  GPU=$(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"
