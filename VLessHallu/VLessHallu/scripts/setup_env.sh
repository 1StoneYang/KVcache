#!/usr/bin/env bash
# Create the vlesshallu conda env for local Qwen2.5-VL-7B + AMBER.
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -qE '^vlesshallu[[:space:]]'; then
  conda create -n vlesshallu python=3.11 -y
fi
conda activate vlesshallu

python -m pip install -U pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
  transformers==4.50.0 \
  accelerate==1.12.0 \
  huggingface-hub \
  safetensors \
  numpy \
  pillow \
  nltk \
  spacy \
  tqdm \
  qwen-vl-utils==0.0.14

ROOT="/root/autodl-tmp/Method/VLessHallu/VLessHallu"
python -m pip install -e "${ROOT}" --no-deps

python -m spacy download en_core_web_lg || true
python - <<'PY'
import nltk
for pkg in (
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "wordnet",
    "omw-1.4",
):
    nltk.download(pkg, quiet=True)
PY

cd "${ROOT}"
python -u run_benchmark.py --config configs/local_qwen25.toml prepare-local
python -u run_benchmark.py --config configs/local_qwen25.toml doctor
echo "vlesshallu environment is ready."
