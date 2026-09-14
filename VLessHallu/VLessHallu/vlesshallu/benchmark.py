from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import torch

from .artifacts import RunLock, RunStore, atomic_write_json, read_json
from .config import VARIANTS, clone_config, config_hash, project_path, public_config
from .datasets import load_amber_samples, load_coco_samples
from .evaluation import choose_sweep_winner, evaluate_amber, evaluate_chair, final_success
from .resources import load_resource_lock
from .runtime import TransformersRuntime


def run_variant(
    config: Mapping[str, Any],
    *,
    variant: str,
    benchmark: str,
    split: str,
) -> dict[str, Any]:
    if split == "test":
        raise ValueError("test is sealed until final; use the final command")
    if benchmark == "amber":
        return _execute(
            config,
            variant=variant,
            benchmark="amber",
            split="generation",
            allow_test=True,
        )
    return _execute(config, variant=variant, benchmark=benchmark, split=split)


def sweep(config: Mapping[str, Any]) -> dict[str, Any]:
    resource_lock = load_resource_lock(config)
    baseline_run = _execute(config, variant="baseline", benchmark="chair", split="dev")
    baseline_metrics = baseline_run["metrics"]
    candidates: list[dict[str, Any]] = []
    working = clone_config(config)

    lambda_stage = [
        (_with_kvsmooth(working, float(value)), value)
        for value in config["sweep"]["lambdas"]
    ]
    stage_candidates = _run_stage("lambda", lambda_stage, candidates)
    winner = choose_sweep_winner(baseline_metrics, stage_candidates)
    working = clone_config(winner["config"])

    prune_stage = [
        (_with_prunehal(working, float(pair[0]), int(pair[1])), pair)
        for pair in config["sweep"]["prunehal_pairs"]
    ]
    stage_candidates = _run_stage("prunehal", prune_stage, candidates)
    winner = choose_sweep_winner(baseline_metrics, stage_candidates)
    working = clone_config(winner["config"])

    weight_stage = [
        (_with_weights(working, tuple(map(float, weights))), weights)
        for weights in config["sweep"]["fusion_weights"]
    ]
    stage_candidates = _run_stage("fusion", weight_stage, candidates)
    winner = choose_sweep_winner(baseline_metrics, stage_candidates)

    lock_payload = {
        "schema_version": 1,
        "resource_lock_hash": resource_lock["lock_hash"],
        "baseline_run_hash": baseline_run["run_hash"],
        "winner_run_hash": winner["run_hash"],
        "selection_score": winner["selection_score"],
        "locked_config": public_config(winner["config"]),
        "baseline_metrics": baseline_metrics,
        "winner_metrics": winner["metrics"],
        "stage_runs": [
            {
                "stage": candidate["stage"],
                "value": candidate["value"],
                "run_hash": candidate["run_hash"],
                "metrics": candidate["metrics"],
            }
            for candidate in candidates
        ],
    }
    lock_path = project_path(config, config["paths"]["selection_lock"])
    atomic_write_json(lock_path, lock_payload)
    return lock_payload


def _run_stage(
    stage_name: str,
    stage_configs: Iterable[tuple[Mapping[str, Any], Any]],
    all_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stage_candidates = []
    for stage_config, value in stage_configs:
        result = _execute(
            stage_config, variant="core4_no_rb", benchmark="chair", split="dev"
        )
        candidate = {
            "stage": stage_name,
            "value": value,
            "config": stage_config,
            "run_hash": result["run_hash"],
            "metrics": result["metrics"],
        }
        stage_candidates.append(candidate)
        all_candidates.append(candidate)
    return stage_candidates


def final(config: Mapping[str, Any]) -> dict[str, Any]:
    selection_path = project_path(config, config["paths"]["selection_lock"])
    if not selection_path.is_file():
        raise FileNotFoundError("configuration is not locked; run sweep before final")
    selection = read_json(selection_path)
    resource_lock = load_resource_lock(config)
    if selection["resource_lock_hash"] != resource_lock["lock_hash"]:
        raise ValueError("selection lock belongs to a different resource lock")
    locked = clone_config(selection["locked_config"])
    locked["_config_path"] = config["_config_path"]
    locked["_project_root"] = config["_project_root"]

    chair_baseline = _execute(
        config, variant="baseline", benchmark="chair", split="test", allow_test=True
    )
    chair_core = _execute(
        locked, variant="core4_no_rb", benchmark="chair", split="test", allow_test=True
    )
    amber_baseline = _execute(
        config, variant="baseline", benchmark="amber", split="generation", allow_test=True
    )
    amber_core = _execute(
        locked,
        variant="core4_no_rb",
        benchmark="amber",
        split="generation",
        allow_test=True,
    )
    outcome = final_success(
        chair_baseline["metrics"],
        chair_core["metrics"],
        amber_baseline["metrics"],
        amber_core["metrics"],
    )
    report = {
        **outcome,
        "chair": {"baseline": chair_baseline, "core4_no_rb": chair_core},
        "amber": {"baseline": amber_baseline, "core4_no_rb": amber_core},
    }
    report_path = project_path(config, config["paths"]["run_root"]) / "final_report.json"
    atomic_write_json(report_path, report)
    return report


def _execute(
    config: Mapping[str, Any],
    *,
    variant: str,
    benchmark: str,
    split: str,
    allow_test: bool = False,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if benchmark not in {"chair", "amber"}:
        raise ValueError(f"unknown benchmark: {benchmark}")
    if benchmark == "chair" and split not in {"smoke", "dev", "test"}:
        raise ValueError(f"unknown CHAIR split: {split}")
    if benchmark == "amber" and split != "generation":
        raise ValueError(f"unknown AMBER split: {split}")
    if split == "test" and not allow_test:
        raise ValueError("test is sealed until final")

    resource_lock = load_resource_lock(config)
    run_hash = config_hash(
        config,
        variant=variant,
        benchmark=benchmark,
        split=split,
        resource_lock_hash=resource_lock["lock_hash"],
    )
    manifest = {
        "schema_version": 1,
        "run_hash": run_hash,
        "variant": variant,
        "benchmark": benchmark,
        "split": split,
        "resource_lock_hash": resource_lock["lock_hash"],
        "config": public_config(config),
    }
    run_root = project_path(config, config["paths"]["run_root"])
    store = RunStore.create(run_root, run_hash, manifest)
    samples = (
        load_coco_samples(config, split)
        if benchmark == "chair"
        else load_amber_samples(config)
    )

    with RunLock(store.path):
        _repair_pruning_ledger(store)
        pending = store.pending(samples)
        total = len(samples)
        completed = total - len(pending)
        print(
            f"Progress: {completed}/{total} completed; {len(pending)} remaining; "
            f"run={run_hash}",
            flush=True,
        )
        runtime = TransformersRuntime(config, resource_lock, variant) if pending else None
        for index, sample in enumerate(pending, start=completed + 1):
            assert runtime is not None
            print(
                f"[{index}/{total}] processing sample_id={sample['sample_id']}...",
                flush=True,
            )
            try:
                result = runtime.generate(sample)
            except torch.OutOfMemoryError as error:
                report_path = _record_oom(
                    store,
                    index=index,
                    total=total,
                    sample=sample,
                    error=error,
                )
                image_path = sample.get("image_path", "unknown")
                print(
                    f"[{index}/{total}] CUDA OOM | sample_id={sample['sample_id']} | "
                    f"image={image_path} | report={report_path}",
                    flush=True,
                )
                raise RuntimeError(
                    f"CUDA OOM at sample {index}/{total}, "
                    f"sample_id={sample['sample_id']}, image={image_path}. "
                    f"See {report_path}"
                ) from error
            store.captions.append(
                {
                    "sample_id": sample["sample_id"],
                    "image_id": sample["image_id"],
                    "caption": result.caption,
                    "input_token_count": result.input_token_count,
                    "visual_token_count": result.visual_token_count,
                    "output_token_count": result.output_token_count,
                    "elapsed_seconds": result.elapsed_seconds,
                    "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
                    "method_trace": result.method_trace,
                    "pruning_events": result.pruning_events,
                }
            )
            store.pruning.append(
                {"sample_id": sample["sample_id"], "events": result.pruning_events}
            )
            print(
                f"[{index}/{total}] done | input_tokens={result.input_token_count} | "
                f"visual_tokens={result.visual_token_count} | "
                f"output_tokens={result.output_token_count} | "
                f"time={result.elapsed_seconds:.2f}s | "
                f"peak={result.peak_gpu_memory_bytes / (1024**3):.2f} GiB",
                flush=True,
            )

    captions = store.captions.records()
    expected_ids = {str(sample["sample_id"]) for sample in samples}
    actual_ids = {str(record["sample_id"]) for record in captions}
    pruning_ids = {str(record["sample_id"]) for record in store.pruning.records()}
    if actual_ids != expected_ids or pruning_ids != expected_ids:
        raise RuntimeError(
            "run is incomplete: "
            f"captions={len(actual_ids)}/{len(expected_ids)}, "
            f"pruning_logs={len(pruning_ids)}/{len(expected_ids)}"
        )
    metrics = (
        evaluate_chair(config, captions, store.path)
        if benchmark == "chair"
        else evaluate_amber(config, captions, store.path)
    )
    elapsed = sum(float(record["elapsed_seconds"]) for record in captions)
    output_tokens = sum(int(record["output_token_count"]) for record in captions)
    runtime_metrics = {
        "sample_count": len(captions),
        "generation_seconds": elapsed,
        "output_tokens": output_tokens,
        "throughput_tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
        "peak_gpu_memory_bytes": max(
            int(record["peak_gpu_memory_bytes"]) for record in captions
        ),
    }
    store.write_metrics(metrics)
    store.write_runtime(runtime_metrics)
    if benchmark == "amber":
        from .evaluation import export_shared_amber_responses

        export_shared_amber_responses(config, captions, variant)
    return {
        "run_hash": run_hash,
        "run_dir": str(store.path),
        "metrics": metrics,
        "runtime": runtime_metrics,
    }


def _record_oom(
    store: RunStore,
    *,
    index: int,
    total: int,
    sample: Mapping[str, Any],
    error: BaseException,
) -> str:
    """Append one CUDA OOM event to a human-readable run document."""
    report_path = store.path / "oom_report.md"
    is_new = not report_path.exists()
    with report_path.open("a", encoding="utf-8") as handle:
        if is_new:
            handle.write("# CUDA OOM Report\n\n")
            handle.write(
                "This file records samples that could not complete because of "
                "GPU memory exhaustion. Failed samples are not counted as completed.\n\n"
            )
        handle.write(f"## Sample {index}/{total}\n\n")
        handle.write(f"- Time (UTC): {datetime.now(timezone.utc).isoformat()}\n")
        handle.write(f"- Sample ID: `{sample.get('sample_id', 'unknown')}`\n")
        handle.write(f"- Image ID: `{sample.get('image_id', 'unknown')}`\n")
        handle.write(f"- Image path: `{sample.get('image_path', 'unknown')}`\n")
        handle.write(f"- Error: `{str(error).replace(chr(10), ' ')}`\n\n")
    return str(report_path)


def _repair_pruning_ledger(store: RunStore) -> None:
    """Complete a caption-first two-file append interrupted between ledgers."""
    for caption in store.captions.records():
        sample_id = caption["sample_id"]
        if store.pruning.contains(sample_id):
            continue
        if "pruning_events" not in caption:
            raise RuntimeError(
                f"caption {sample_id} has no recoverable per-sample pruning log"
            )
        store.pruning.append(
            {"sample_id": sample_id, "events": caption["pruning_events"]}
        )


def _with_kvsmooth(config: Mapping[str, Any], value: float) -> dict[str, Any]:
    result = clone_config(config)
    result["kvsmooth"]["lambda_ref"] = value
    return result


def _with_prunehal(config: Mapping[str, Any], ratio: float, count: int) -> dict[str, Any]:
    result = clone_config(config)
    result["prunehal"]["retention_ratio"] = ratio
    result["prunehal"]["max_prunes"] = count
    return result


def _with_weights(
    config: Mapping[str, Any], weights: tuple[float, float, float]
) -> dict[str, Any]:
    result = clone_config(config)
    result["fusion"].update(
        {"decode": weights[0], "shift": weights[1], "text": weights[2]}
    )
    return result
