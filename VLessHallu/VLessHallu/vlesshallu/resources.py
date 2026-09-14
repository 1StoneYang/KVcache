from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping

from .artifacts import atomic_write_json, read_json, sha256_file, sha256_json
from .config import project_path
from .datasets import build_coco_splits
from .http_download import download_zip as _download
from .nlp_resources import ensure_nlp_resources, nlp_resource_checks


def doctor(config: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [_gpu_check()]
    checks.extend(_dependency_checks(config))
    checks.extend(_data_checks(config))
    return {
        "ok": all(check["ok"] for check in checks if check.get("required", True)),
        "checks": checks,
    }


def prepare(config: Mapping[str, Any]) -> dict[str, Any]:
    data_root = project_path(config, config["paths"]["data_root"])
    data_root.mkdir(parents=True, exist_ok=True)
    sources = config["dataset"]
    evaluation_nlp = ensure_nlp_resources()

    coco_root = data_root / "coco"
    coco_root.mkdir(parents=True, exist_ok=True)
    images_archive = _download(sources["coco_images_url"], coco_root / "val2014.zip")
    annotations_archive = _download(
        sources["coco_annotations_url"], coco_root / "annotations_trainval2014.zip"
    )
    _extract_zip(images_archive, coco_root)
    _extract_zip(annotations_archive, coco_root)

    external = data_root / "external"
    chair_root = _checkout(
        sources["chair_repo"], sources["chair_revision"], external / "chair"
    )
    amber_root = _checkout(
        sources["amber_repo"], sources["amber_revision"], external / "amber"
    )

    amber_archive = data_root / "amber_images.zip"
    if not _amber_images_ready(data_root / "amber_images"):
        _download_google_drive(sources["amber_drive_id"], amber_archive)
        _extract_zip(amber_archive, data_root / "amber_images")

    split_path = data_root / "splits" / "coco2014_val.json"
    build_coco_splits(
        coco_root / "annotations" / "instances_val2014.json",
        split_path,
        seed=int(sources["split_seed"]),
        smoke_size=int(sources["smoke_size"]),
        dev_size=int(sources["dev_size"]),
        test_size=int(sources["test_size"]),
    )

    model_path = _download_model(config, data_root / "models" / "qwen3-vl-8b-instruct")
    lock = {
        "schema_version": 1,
        "model": {
            "id": config["model"]["id"],
            "revision": config["model"]["revision"],
            "path": str(model_path.resolve()),
        },
        "coco": {
            "images_url": sources["coco_images_url"],
            "images_sha256": sha256_file(images_archive),
            "annotations_url": sources["coco_annotations_url"],
            "annotations_sha256": sha256_file(annotations_archive),
            "split_sha256": sha256_file(split_path),
        },
        "chair": {
            "repo": sources["chair_repo"],
            "revision": _git_head(chair_root),
        },
        "amber": {
            "repo": sources["amber_repo"],
            "revision": _git_head(amber_root),
            "drive_id": sources["amber_drive_id"],
            "images_sha256": (
                sha256_file(amber_archive) if amber_archive.is_file() else None
            ),
        },
        "evaluation_nlp": evaluation_nlp,
        "runtime": {
            "backend": config["model"]["backend"],
            "quantization": config["model"]["quantization"],
            "python": sys.version.split()[0],
            "torch": _version("torch"),
            "torchvision": _version("torchvision"),
            "torchao": _version("torchao"),
            "transformers": _version("transformers"),
            "accelerate": _version("accelerate"),
            "nltk": _version("nltk"),
            "patternfork_nosql": _version("patternfork-nosql"),
            "spacy": _version("spacy"),
        },
    }
    lock["lock_hash"] = sha256_json(lock)
    lock_path = project_path(config, config["paths"]["lock_file"])
    atomic_write_json(lock_path, lock)
    return lock


def prepare_local(config: Mapping[str, Any]) -> dict[str, Any]:
    """Wire the existing local Qwen2.5-VL checkpoint and AMBER dataset."""
    model_path = Path(
        config["model"].get("local_path")
        or config["paths"].get("model_path")
        or "/root/autodl-tmp/model/Qwen2.5-VL-7B"
    )
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"local model config missing: {model_path}")
    amber_root = Path(
        config["paths"].get("amber_root") or "/root/autodl-tmp/Method/dataset/AMBER"
    )
    query = amber_root / "query_generative.json"
    if not query.is_file():
        query = amber_root / "data" / "query" / "query_generative.json"
    if not query.is_file():
        raise FileNotFoundError(f"AMBER query file missing under {amber_root}")
    images = Path(config["paths"].get("amber_images") or (amber_root / "image"))
    if not (images / "AMBER_1.jpg").is_file():
        raise FileNotFoundError(f"AMBER images missing under {images}")

    try:
        evaluation_nlp = ensure_nlp_resources()
    except Exception:
        evaluation_nlp = None

    lock = {
        "schema_version": 1,
        "model": {
            "id": config["model"]["id"],
            "revision": config["model"].get("revision") or "local",
            "path": str(model_path.resolve()),
        },
        "amber": {
            "root": str(amber_root.resolve()),
            "images": str(images.resolve()),
            "query": str(query.resolve()),
        },
        "runtime": {
            "backend": config["model"]["backend"],
            "quantization": config["model"]["quantization"],
            "python": sys.version.split()[0],
            "torch": _version("torch"),
            "transformers": _version("transformers"),
        },
    }
    if evaluation_nlp is not None:
        lock["evaluation_nlp"] = evaluation_nlp
    lock["lock_hash"] = sha256_json(lock)
    lock_path = project_path(config, config["paths"]["lock_file"])
    atomic_write_json(lock_path, lock)
    return lock


def load_resource_lock(
    config: Mapping[str, Any], *, required: bool = True
) -> dict[str, Any] | None:
    path = project_path(config, config["paths"]["lock_file"])
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"resource lock missing; run prepare first: {path}")
        return None
    lock = read_json(path)
    claimed = lock.get("lock_hash")
    check = dict(lock)
    check.pop("lock_hash", None)
    if claimed != sha256_json(check):
        raise ValueError(f"resource lock hash is invalid: {path}")
    if lock.get("runtime", {}).get("backend") != config["model"]["backend"]:
        raise ValueError("resource lock was created for a different runtime backend")
    return lock


def _gpu_check() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, check=True, text=True, capture_output=True).stdout
        rows = [row.strip() for row in output.splitlines() if row.strip()]
        if len(rows) != 1:
            return _check("gpu", False, f"expected one GPU, found {len(rows)}")
        name, memory, capability = [part.strip() for part in rows[0].rsplit(",", 2)]
        ok = float(memory) >= 20000
        return _check(
            "gpu",
            ok,
            f"{name}; {float(memory) / 1024:.1f} GiB; compute {capability}",
            required=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as error:
        return _check("gpu", False, str(error))


def _dependency_checks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    requirements = [
        ("python", None, sys.version_info >= (3, 10) and sys.version_info < (3, 13), sys.version.split()[0]),
        ("torch", "torch", importlib.util.find_spec("torch") is not None, _version("torch")),
        (
            "torchvision",
            "torchvision",
            importlib.util.find_spec("torchvision") is not None,
            _version("torchvision"),
        ),
        (
            "transformers",
            "transformers",
            importlib.util.find_spec("transformers") is not None,
            _version("transformers"),
        ),
        (
            "huggingface_hub",
            "huggingface-hub",
            importlib.util.find_spec("huggingface_hub") is not None,
            _version("huggingface-hub"),
        ),
        (
            "accelerate",
            "accelerate",
            importlib.util.find_spec("accelerate") is not None,
            _version("accelerate"),
        ),
        ("nltk", "nltk", importlib.util.find_spec("nltk") is not None, _version("nltk")),
        ("spacy", "spacy", importlib.util.find_spec("spacy") is not None, _version("spacy")),
        ("tqdm", "tqdm", importlib.util.find_spec("tqdm") is not None, _version("tqdm")),
    ]
    local = bool(config["model"].get("local_path") or config.get("paths", {}).get("amber_root"))
    if not local:
        requirements.extend(
            [
                ("gdown", "gdown", importlib.util.find_spec("gdown") is not None, _version("gdown")),
                (
                    "pattern",
                    "patternfork-nosql",
                    importlib.util.find_spec("pattern") is not None,
                    _version("patternfork-nosql"),
                ),
            ]
        )
    if config["model"]["quantization"] == "torchao_fp8_per_tensor":
        requirements.append(
            (
                "torchao",
                "torchao",
                importlib.util.find_spec("torchao") is not None,
                _version("torchao"),
            )
        )
    return [_check(name, ok, detail) for name, _, ok, detail in requirements]


def _data_checks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = config.get("paths", {})
    if config["model"].get("local_path") or paths.get("amber_root"):
        model_path = Path(
            config["model"].get("local_path")
            or "/root/autodl-tmp/model/Qwen2.5-VL-7B"
        )
        amber_root = Path(paths.get("amber_root") or "/root/autodl-tmp/Method/dataset/AMBER")
        images = Path(paths.get("amber_images") or (amber_root / "image"))
        query = amber_root / "query_generative.json"
        if not query.is_file():
            query = amber_root / "data" / "query" / "query_generative.json"
        checks = [
            _check("resource_lock", project_path(config, config["paths"]["lock_file"]).exists(), str(project_path(config, config["paths"]["lock_file"]))),
            _check("model", (model_path / "config.json").is_file(), str(model_path)),
            _check("amber_query", query.is_file(), str(query)),
            _check("amber_images", (images / "AMBER_1.jpg").is_file(), str(images)),
            _check("amber_scorer", (amber_root / "inference.py").is_file(), str(amber_root / "inference.py")),
        ]
        checks.extend(nlp_resource_checks())
        return checks
    data_root = project_path(config, config["paths"]["data_root"])
    paths = {
        "resource_lock": project_path(config, config["paths"]["lock_file"]),
        "model": data_root / "models" / "qwen3-vl-8b-instruct" / "config.json",
        "coco_images": data_root / "coco" / "val2014",
        "coco_annotations": data_root / "coco" / "annotations" / "instances_val2014.json",
        "coco_split": data_root / "splits" / "coco2014_val.json",
        "chair": data_root / "external" / "chair" / "chair.py",
        "amber": data_root / "external" / "amber" / "inference.py",
    }
    checks = [_check(name, path.exists(), str(path)) for name, path in paths.items()]
    checks.append(
        _check(
            "amber_images",
            _amber_images_ready(data_root / "amber_images"),
            str(data_root / "amber_images"),
        )
    )
    checks.extend(nlp_resource_checks())
    return checks


def _check(name: str, ok: bool, detail: str | None, *, required: bool = True) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "required": required, "detail": detail}


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _download_google_drive(file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "gdown", file_id, "-O", str(destination)]
    subprocess.run(command, check=True)


def _extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        root = destination.resolve()
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def _checkout(repo: str, revision: str, destination: Path) -> Path:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", repo, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "fetch", "origin", revision], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)
    if _git_head(destination) != revision:
        raise RuntimeError(f"failed to lock {repo} at {revision}")
    return destination


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _download_model(config: Mapping[str, Any], destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    result = snapshot_download(
        repo_id=config["model"]["id"],
        revision=config["model"]["revision"],
        local_dir=destination,
    )
    model_config = json.loads((Path(result) / "config.json").read_text(encoding="utf-8"))
    text_config = model_config["text_config"]
    observed = (
        int(text_config["num_hidden_layers"]),
        int(text_config["num_attention_heads"]),
        int(text_config["num_key_value_heads"]),
    )
    if observed != (36, 32, 8):
        raise ValueError(f"unexpected Qwen3-VL text architecture: {observed}")
    return Path(result)


def _amber_images_ready(root: Path) -> bool:
    if (root / "AMBER_1.jpg").is_file() and (root / "AMBER_1004.jpg").is_file():
        return True
    return bool(list(root.rglob("AMBER_1.jpg"))) and bool(
        list(root.rglob("AMBER_1004.jpg"))
    )
