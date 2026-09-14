from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

from .artifacts import atomic_write_json
from .config import project_path


def build_coco_splits(
    instances_path: str | Path,
    destination: str | Path,
    *,
    seed: int,
    smoke_size: int,
    dev_size: int,
    test_size: int,
) -> dict[str, Any]:
    with Path(instances_path).open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    image_ids = sorted(int(image["id"]) for image in annotations["images"])
    required = smoke_size + dev_size + test_size
    if len(image_ids) < required:
        raise ValueError(f"COCO val has {len(image_ids)} images, need {required}")
    generator = random.Random(seed)
    generator.shuffle(image_ids)
    smoke_end = smoke_size
    dev_end = smoke_end + dev_size
    test_end = dev_end + test_size
    payload = {
        "schema_version": 1,
        "source": str(Path(instances_path).resolve()),
        "seed": seed,
        "smoke": image_ids[:smoke_end],
        "dev": image_ids[smoke_end:dev_end],
        "test": image_ids[dev_end:test_end],
    }
    _assert_disjoint_splits(payload)
    atomic_write_json(destination, payload)
    return payload


def load_coco_samples(config: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in {"smoke", "dev", "test"}:
        raise ValueError(f"unknown COCO split: {split}")
    data_root = project_path(config, config["paths"]["data_root"])
    split_path = data_root / "splits" / "coco2014_val.json"
    with split_path.open(encoding="utf-8") as handle:
        splits = json.load(handle)
    _assert_disjoint_splits(splits)
    image_root = data_root / "coco" / "val2014"
    samples = []
    for image_id in splits[split]:
        image_path = image_root / f"COCO_val2014_{int(image_id):012d}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        samples.append(
            {
                "sample_id": int(image_id),
                "image_id": int(image_id),
                "image_path": str(image_path),
                "prompt": config["decoding"]["prompt"],
            }
        )
    return samples


def load_amber_samples(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = config.get("paths", {})
    if paths.get("amber_root"):
        amber_root = Path(paths["amber_root"])
        query_path = amber_root / "query_generative.json"
        if not query_path.is_file():
            query_path = amber_root / "data" / "query" / "query_generative.json"
        image_root = Path(paths.get("amber_images") or (amber_root / "image"))
    else:
        data_root = project_path(config, config["paths"]["data_root"])
        amber_root = data_root / "external" / "amber"
        query_path = amber_root / "data" / "query" / "query_generative.json"
        image_root = _find_amber_image_root(data_root / "amber_images")
    with query_path.open(encoding="utf-8") as handle:
        queries = json.load(handle)
    limit = int(os.environ.get("AMBER_LIMIT") or config.get("dataset", {}).get("amber_limit") or 0)
    if limit > 0:
        queries = queries[:limit]
    elif len(queries) != 1004:
        raise ValueError(f"AMBER generation set must contain 1004 rows, found {len(queries)}")
    if paths.get("amber_images") or paths.get("amber_root"):
        image_root = Path(image_root)
        if not (image_root / "AMBER_1.jpg").is_file():
            image_root = _find_amber_image_root(image_root)
    samples = []
    for query in queries:
        sample_id = int(query["id"])
        image_path = image_root / f"AMBER_{sample_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        prompt = query.get("query") or query.get("question") or query.get("prompt")
        if not prompt:
            prompt = config["decoding"]["prompt"]
        samples.append(
            {
                "sample_id": sample_id,
                "image_id": sample_id,
                "image_path": str(image_path),
                "prompt": str(prompt),
            }
        )
    return samples


def _find_amber_image_root(root: Path) -> Path:
    direct = root / "AMBER_1.jpg"
    if direct.is_file():
        return root
    matches = list(root.rglob("AMBER_1.jpg"))
    if len(matches) != 1:
        raise FileNotFoundError(f"could not uniquely locate AMBER_1.jpg below {root}")
    return matches[0].parent


def _assert_disjoint_splits(payload: Mapping[str, Any]) -> None:
    split_sets = {name: set(map(int, payload[name])) for name in ("smoke", "dev", "test")}
    if split_sets["smoke"] & split_sets["dev"]:
        raise ValueError("smoke and dev overlap")
    if split_sets["smoke"] & split_sets["test"]:
        raise ValueError("smoke and test overlap")
    if split_sets["dev"] & split_sets["test"]:
        raise ValueError("dev and test overlap")
