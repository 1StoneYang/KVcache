# Standalone Qwen2.5-VL baseline

Uses the original Hugging Face model.generate() with full KV cache.
No VLessHallu or MM-ShiftKV algorithm imports or patches.
The existing mmshiftkv Python environment supplies dependencies only.

Defaults: local Qwen2.5-VL-7B, BF16 (no FP8), FlashAttention 2,
greedy decoding, 512 new tokens, max_pixels=1605632.
Questions come unchanged from AMBER query_generative.json (1004 images).

## Full AMBER generation

    cd /root/autodl-tmp/Method/generate
    bash run.sh

Repeat the same command to resume. Completed samples are skipped.
Use tmux/screen for long runs over SSH.

## Small checks

    bash run.sh --limit 2 --output outputs/smoke
    bash run.sh --ids 147 --output outputs/check147

To use the VLessHallu prompt explicitly:

    bash run.sh --prompt 'Please describe the image in detail.' --output outputs/detailed_prompt

## Outputs

Default directory:
/root/autodl-tmp/Method/generate/outputs/amber_baseline/

- responses.json: official AMBER id/response format.
- records.json: responses, prompts, token counts, latency and peak GPU memory.
- manifest.json: settings, script/query fingerprints and library versions.
- last_error.json: last failed attempt, if any.

Results are saved atomically after each sample. Changed settings or code are
rejected on resume; choose a different --output directory for a new experiment.
The process lock is released by the OS even after interruption.

## Official AMBER scoring

After completing generation:

    bash score.sh

Or specify a result file:

    bash score.sh /root/autodl-tmp/Method/generate/outputs/detailed_prompt/responses.json

Scoring uses AMBER inference.py directly. Partial-sample scores are not a full
AMBER benchmark. This project does not run any KV pruning or smoothing method.
