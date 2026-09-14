#!/usr/bin/env bash
# Install AMBER scoring extras (spaCy large model + NLTK corpora).
# Generation does not need these; only official AMBER metric scoring does.
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mmshiftkv

echo "Installing spaCy model en_core_web_lg ..."
python -m spacy download en_core_web_lg

echo "Downloading NLTK data ..."
python - <<'PY'
import nltk
for p in [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "wordnet",
    "omw-1.4",
]:
    print("download", p, flush=True)
    nltk.download(p)
print("done")
PY

python -c "import spacy; spacy.load('en_core_web_lg'); print('spaCy lg OK')"
echo "AMBER scorer extras installed."
