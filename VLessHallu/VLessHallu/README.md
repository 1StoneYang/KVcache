# VLessHallu

Standalone Qwen3-VL KV-cache hallucination benchmark. It does not import
CursorLLM and does not use vLLM, TGI, SGLang, or another serving engine.

## Runtime

The benchmark loads `Qwen/Qwen3-VL-8B-Instruct` directly with Transformers and
runs an owned, batch-size-one greedy decoding loop. Text attention is forced to
the eager implementation so every layer's attention probabilities are
available while the cache is still mutable.

- Activations: BF16.
- Weights: TorchAO dynamic-activation/per-tensor FP8 by default.
- Hardware: one RTX 5090.
- Generation: greedy, `max_new_tokens=512`.
- Speculation: disabled. Cache scoring, KVSmooth, and possible physical
  compaction happen after every token, so batched n-gram verification would
  change the method's causal execution order.

The cache tensor remains rectangular because all KV heads retain the same
visual-token count. Each layer and KV head has its own gather indices, so their
retained visual positions may differ physically. Original positions are
tracked separately; text and generated tokens are never selected for removal.

## Variants

- `baseline`: local eager greedy decoding with an untouched cache.
- `kvsmooth`: baseline plus post-attention smoothing of each new generated
  token's K/V in layers 3 through 34.
- `mmshift`: full prefill, proxy scoring, then a prefill-budget visual gather.
- `myopia_score`: cumulative and text-guided visual scoring without DAS or a
  recycling bin, followed by a prefill-budget gather.
- `prunehal`: full prefill; first-step reference, forced step-2 pruning, then
  half-layer dynamic votes.
- `core4_no_rb`: full prefill with MM-ShiftKV and Myopia reference scores;
  PruneHal controls when to prune; decode/shift/text scores select tokens; and
  KVSmooth runs after the current layer's attention.

Recycling Bin, DeCo, and all other external methods are excluded.

## Installation

```bash
python -m pip install -e '.[runtime,eval,test]'
```

## Commands

```bash
python run_benchmark.py doctor
python run_benchmark.py prepare
python run_benchmark.py self-test

python run_benchmark.py run --variant baseline --split smoke
python run_benchmark.py run --variant core4_no_rb --split smoke
python run_benchmark.py sweep
python run_benchmark.py final
```

`prepare` downloads and pins the model, COCO 2014 validation resources, CHAIR,
and the official AMBER generation set. It writes `resources.lock.json` and
`data/splits/coco2014_val.json`.

The CHAIR test split is not exposed by `run`. Only `final`, after a successful
dev sweep has written `configs/locked_selection.json`, can run test and AMBER.

## Image and prefill memory limits

All shipped configurations limit images to 1,605,632 pixels, matching the local
MM-ShiftKV image cap. An explicit `model.max_pixels` overrides the runtime default.
The runtime keeps eager attention for method statistics, but consumes each layer's
prefill attention in its hook instead of retaining every quadratic attention matrix.
Only a copied last-row logits tensor is carried from prefill into decode.
Implementation revision 3 separates these runs from earlier configurations.

## Run Artifacts

Every run is isolated by a hash of the public configuration, resource lock,
variant, benchmark, and split:

```text
runs/<hash>/
  manifest.json
  captions.jsonl
  pruning.jsonl
  metrics.json
  runtime.json
```

Caption and pruning ledgers resume by sample ID. A caption embeds its pruning
events so an interruption between the two ledger appends is repaired on the
next invocation. `runtime.json` reports generation time, output-token
throughput, and peak allocated CUDA memory.

## Sweep And Final Gate

The CHAIR dev search is sequential rather than Cartesian:

1. `lambda_ref`: `0.5`, `0.7`, `0.9`.
2. `(r,t)`: `(0.7,2)`, `(0.8,3)`, `(0.9,4)`.
3. The four fusion tuples in `configs/default.toml`.

Candidates must keep F1 within one percentage point of baseline and average
caption length at least 90% of baseline. The winner minimizes the
min-max-normalized mean of CHAIRs and CHAIRi. Final success requires lower
CHAIRs and CHAIRi, F1 within one point, at least two improved AMBER
hallucination metrics, and Cover within two points.

## Tests

```bash
python -m pytest -q
python run_benchmark.py self-test
```
