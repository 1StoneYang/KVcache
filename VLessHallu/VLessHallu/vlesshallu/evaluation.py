from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import atomic_write_json, read_json
from .config import project_path


def evaluate_chair(
    config: Mapping[str, Any], captions: Iterable[Mapping[str, Any]], run_dir: str | Path
) -> dict[str, float]:
    run_path = Path(run_dir)
    records = list(captions)
    input_path = run_path / "chair_input.json"
    output_path = run_path / "chair_details.json"
    atomic_write_json(
        input_path,
        [
            {"image_id": int(record["image_id"]), "caption": str(record["caption"])}
            for record in records
        ],
    )
    data_root = project_path(config, config["paths"]["data_root"])
    chair_root = data_root / "external" / "chair"
    command = [
        str(config["evaluation"]["chair_python"]),
        str(chair_root / "chair.py"),
        "--cap_file",
        str(input_path),
        "--image_id_key",
        "image_id",
        "--caption_key",
        "caption",
        "--cache",
        str(chair_root / "chair.pkl"),
        "--coco_path",
        str(data_root / "coco" / "annotations"),
        "--save_path",
        str(output_path),
    ]
    subprocess.run(command, check=True, cwd=chair_root)
    raw = read_json(output_path)["overall_metrics"]
    chair_s = 100 * float(raw["CHAIRs"])
    chair_i = 100 * float(raw["CHAIRi"])
    precision = 100 - chair_i
    recall = 100 * float(raw["Recall"])
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    average_length = sum(len(str(record["caption"]).split()) for record in records) / len(records)
    return {
        "CHAIRs": chair_s,
        "CHAIRi": chair_i,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "average_length": average_length,
        "sample_count": float(len(records)),
    }


def evaluate_amber(
    config: Mapping[str, Any], captions: Iterable[Mapping[str, Any]], run_dir: str | Path
) -> dict[str, float]:
    run_path = Path(run_dir)
    records = list(captions)
    if not records:
        metrics = {"CHAIR": None, "Cover": None, "Hal": None, "Cog": None, "partial": True}
        (run_path / "amber_partial.txt").write_text(
            "skipped AMBER scoring because there are no captions\n",
            encoding="utf-8",
        )
        return metrics
    input_path = run_path / "amber_input.json"
    atomic_write_json(
        input_path,
        [
            {"id": int(record["sample_id"]), "response": str(record["caption"])}
            for record in sorted(records, key=lambda item: int(item["sample_id"]))
        ],
    )
    data_root = project_path(config, config["paths"]["data_root"])
    amber_root = Path(config["paths"]["amber_root"]) if config.get("paths", {}).get("amber_root") else data_root / "external" / "amber"
    configured_python = str(config["evaluation"]["amber_python"])
    amber_python = sys.executable if configured_python == "python" else configured_python
    command = [
        amber_python,
        "inference.py",
        "--inference_data",
        str(input_path),
        "--evaluation_type",
        "g",
    ]
    result = subprocess.run(
        command, check=True, cwd=amber_root, text=True, capture_output=True
    )
    metrics = {}
    for name in ("CHAIR", "Cover", "Hal", "Cog"):
        match = re.search(rf"^{name}:\s*([0-9]+(?:\.[0-9]+)?)", result.stdout, re.MULTILINE)
        if match is None:
            raise ValueError(f"could not parse AMBER {name} from output:\n{result.stdout}")
        metrics[name] = float(match.group(1))
    metrics["partial"] = len(records) != 1004
    metrics["sample_count"] = float(len(records))
    (run_path / "amber_stdout.txt").write_text(result.stdout, encoding="utf-8")
    return metrics


def export_shared_amber_responses(
    config: Mapping[str, Any],
    captions: Iterable[Mapping[str, Any]],
    variant: str,
) -> Path | None:
    """Copy {id, response} JSON into the shared AMBER results tree."""
    paths = config.get("paths", {})
    amber_root = paths.get("amber_root")
    if not amber_root:
        return None
    model_name = Path(
        str(config["model"].get("local_path") or config["model"]["id"])
    ).name
    project = str(paths.get("project_name") or "VLessHallu")
    destination_dir = (
        Path(amber_root) / "results" / project / model_name / "amber_generative"
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {"id": int(record["sample_id"]), "response": str(record["caption"])}
        for record in sorted(captions, key=lambda item: int(item["sample_id"]))
    ]
    destination = destination_dir / f"amber_generative_{variant}.json"
    atomic_write_json(destination, payload)
    return destination


def choose_sweep_winner(
    baseline: Mapping[str, float], candidates: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    candidates = list(candidates)
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["metrics"]["F1"]) >= float(baseline["F1"]) - 1.0
        and float(candidate["metrics"]["average_length"])
        >= 0.90 * float(baseline["average_length"])
    ]
    if not eligible:
        raise ValueError("no sweep configuration satisfies the F1 and length constraints")
    chair_s_values = [float(item["metrics"]["CHAIRs"]) for item in eligible]
    chair_i_values = [float(item["metrics"]["CHAIRi"]) for item in eligible]
    for item in eligible:
        normalized_s = _minmax(float(item["metrics"]["CHAIRs"]), chair_s_values)
        normalized_i = _minmax(float(item["metrics"]["CHAIRi"]), chair_i_values)
        item["selection_score"] = (normalized_s + normalized_i) / 2
    return min(
        eligible,
        key=lambda item: (
            float(item["selection_score"]),
            float(item["metrics"]["CHAIRs"]),
            float(item["metrics"]["CHAIRi"]),
            -float(item["metrics"]["F1"]),
            str(item.get("run_hash", "")),
        ),
    )


def final_success(
    chair_baseline: Mapping[str, float],
    chair_core: Mapping[str, float],
    amber_baseline: Mapping[str, float],
    amber_core: Mapping[str, float],
) -> dict[str, Any]:
    chair_conditions = {
        "CHAIRs_lower": float(chair_core["CHAIRs"]) < float(chair_baseline["CHAIRs"]),
        "CHAIRi_lower": float(chair_core["CHAIRi"]) < float(chair_baseline["CHAIRi"]),
        "F1_within_1pp": float(chair_core["F1"]) >= float(chair_baseline["F1"]) - 1.0,
    }
    hallucination_improvements = sum(
        float(amber_core[name]) < float(amber_baseline[name])
        for name in ("CHAIR", "Hal", "Cog")
    )
    amber_conditions = {
        "two_hallucination_metrics_improve": hallucination_improvements >= 2,
        "Cover_within_2pp": float(amber_core["Cover"])
        >= float(amber_baseline["Cover"]) - 2.0,
    }
    conditions = {**chair_conditions, **amber_conditions}
    return {"success": all(conditions.values()), "conditions": conditions}


def _minmax(value: float, values: list[float]) -> float:
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper):
        return 0.0
    return (value - lower) / (upper - lower)
