#!/usr/bin/env bash
# Official lmms-eval benchmarks with the local Qwen2.5-VL-7B checkpoint.
# This machine has 1x RTX 4090; use amber_qwen.sh for the local AMBER dataset.
set -euo pipefail

ROOT="/root/autodl-tmp/Method/MM-ShiftKV-main"
MODEL_PATH="/root/autodl-tmp/model/Qwen2.5-VL-7B"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mmshiftkv
cd "${ROOT}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ratios=(0.1)
methods=("shiftkv") # "keydiff" "streamingllm" "expectedAttention" "sparsemm_query" "snapkv"
budgets=(64) # 128 256 512
tasks=("ocrbench") # "textvqa" "ocrbench" "docvqa" "chartqa" "textcaps" "mmmu_pro" "pope" "ok_vqa" "ST-VQA" "flicker30k"

mask_ratio=0.1

for task in "${tasks[@]}"; do
    for budget in "${budgets[@]}"; do
        for ratio in "${ratios[@]}"; do
            for method in "${methods[@]}"; do

                export METHOD=${method}
                export BUDGET=${budget}
                export RATIO=${ratio}
                export MASK_RATIO=${mask_ratio}

                mkdir -p ./results/${task}/llama_resultsfull/

                python3 -u -m accelerate.commands.launch \
                    --num_processes=1 \
                    --main_process_port 54323 \
                    -m lmms_eval \
                    --model qwen2_5_vl \
                    --model_args pretrained="${MODEL_PATH},use_flash_attention_2=True,device_map=cuda:0" \
                    --tasks ${task} \
                    --batch_size 1 \
                    --log_samples \
                    --log_samples_suffix qwen25vl_${method} \
                    --output_path ./logs/ \
                    --gen_kwargs temperature=0 \
                    --verbosity=DEBUG 2>&1 | tee \
                    ./results/${task}/llama_resultsfull/${task}_${method}_${budget}_${ratio}.log

            done
        done
    done
done
