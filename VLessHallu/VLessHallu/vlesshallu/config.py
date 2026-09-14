from __future__ import annotations

import copy
import hashlib
import json
import math
import tomllib
from pathlib import Path
from typing import Any, Mapping


VARIANTS = (
    "baseline",
    "kvsmooth",
    "mmshift",
    "prunehal",
    "myopia_score",
    "core4_no_rb",
    "rekv",
)


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    validate_config(config)
    return config


def clone_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(config))


def project_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(str(config["_project_root"])) / path


def config_hash(
    config: Mapping[str, Any],
    *,
    variant: str,
    benchmark: str,
    split: str,
    resource_lock_hash: str | None,
) -> str:
    payload = {
        "config": _public_config(config),
        "variant": variant,
        "benchmark": benchmark,
        "split": split,
        "resource_lock_hash": resource_lock_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return _public_config(config)


def _public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if not key.startswith("_")
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")

    model = _section(config, "model")
    allowed_ids = {
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-7B",
    }
    if model.get("id") not in allowed_ids and not model.get("local_path"):
        raise ConfigError(
            "model.id must be Qwen3-VL-8B-Instruct or Qwen2.5-VL-7B, or set model.local_path"
        )
    if model.get("backend") != "transformers_eager":
        raise ConfigError("model.backend must be transformers_eager")
    if model.get("attn_implementation") != "eager":
        raise ConfigError("attention must be eager so method statistics are observable")
    if model.get("device") != "cuda":
        raise ConfigError("the single-GPU protocol requires model.device=cuda")
    if model.get("dtype") != "bfloat16":
        raise ConfigError("the local runtime uses bfloat16 activations")
    if model.get("quantization") not in {"torchao_fp8_per_tensor", "bf16"}:
        raise ConfigError("unsupported local quantization")
    if "max_pixels" in model and int(model["max_pixels"]) <= 0:
        raise ConfigError("model.max_pixels must be positive")

    decoding = _section(config, "decoding")
    if decoding.get("strategy") != "greedy":
        raise ConfigError("CHAIR/AMBER generation must use greedy decoding")
    if int(decoding.get("max_new_tokens", 0)) != 512:
        raise ConfigError("max_new_tokens must be 512")
    if decoding.get("speculative_method") != "none":
        raise ConfigError(
            "speculation must be disabled because cache mutation occurs after every token"
        )

    mmshift = _section(config, "mmshift")
    n = int(mmshift.get("num_proxies", 0))
    groups = int(mmshift.get("num_groups", 0))
    if n <= 0 or groups <= 0 or n % groups:
        raise ConfigError("mmshift.num_proxies must be divisible by num_groups")
    if float(mmshift.get("variance_gamma", 0)) <= 1:
        raise ConfigError("mmshift.variance_gamma must be greater than 1")
    _open_unit(mmshift.get("mass_threshold"), "mmshift.mass_threshold")
    if int(mmshift.get("cache_budget", 0)) <= 0:
        raise ConfigError("mmshift.cache_budget must be positive")

    myopia = _section(config, "myopia")
    _closed_unit(myopia.get("text_alpha"), "myopia.text_alpha")
    if int(myopia.get("cache_budget", 0)) <= 0:
        raise ConfigError("myopia.cache_budget must be positive")

    prunehal = _section(config, "prunehal")
    _open_unit(prunehal.get("retention_ratio"), "prunehal.retention_ratio")
    vote_fraction = float(prunehal.get("vote_fraction", 0))
    if not 0 < vote_fraction <= 1:
        raise ConfigError("prunehal.vote_fraction must be in (0, 1]")
    if int(prunehal.get("force_step", 0)) != 2:
        raise ConfigError("PruneHal must force its first prune at decoding step 2")

    fusion = _section(config, "fusion")
    weights = [float(fusion.get(name, -1)) for name in ("decode", "shift", "text")]
    if any(weight < 0 for weight in weights) or not math.isclose(sum(weights), 1.0):
        raise ConfigError("fusion weights must be non-negative and sum to 1")

    kvsmooth = _section(config, "kvsmooth")
    first_layer = int(kvsmooth.get("first_layer", -1))
    last_layer = int(kvsmooth.get("last_layer", -1))
    max_layers = 28 if "Qwen2.5" in str(model.get("id", "")) or "Qwen2.5" in str(model.get("local_path", "")) else 36
    if not 0 <= first_layer <= last_layer < max_layers:
        raise ConfigError(f"KVSmooth layer interval must be within 0..{max_layers - 1}")
    if int(kvsmooth.get("fifo_size", 0)) <= 1:
        raise ConfigError("kvsmooth.fifo_size must be greater than 1")
    _closed_unit(kvsmooth.get("lambda_ref"), "kvsmooth.lambda_ref")

    if "rekv" in config:
        rekv = _section(config, "rekv")
        if int(rekv.get("active_budget", 0)) <= 0 or int(rekv.get("bin_budget", 0)) <= 0:
            raise ConfigError("rekv active_budget and bin_budget must be positive")
        if int(rekv.get("window", 0)) <= 0:
            raise ConfigError("rekv.window must be positive")
        weights = [
            float(rekv.get("future_weight", -1)),
            float(rekv.get("current_weight", -1)),
            float(rekv.get("ema_weight", -1)),
        ]
        if any(weight < 0 for weight in weights) or not math.isclose(sum(weights), 1.0):
            raise ConfigError("rekv fusion weights must be non-negative and sum to 1")

    dataset = _section(config, "dataset")
    sizes = [int(dataset.get(f"{name}_size", 0)) for name in ("smoke", "dev", "test")]
    if sizes != [8, 100, 500]:
        raise ConfigError("COCO split sizes must be smoke=8, dev=100, test=500")

    selection = _section(config, "selection")
    if float(selection.get("f1_tolerance_pp", -1)) != 1.0:
        raise ConfigError("selection F1 tolerance must be 1 percentage point")
    if float(selection.get("minimum_length_ratio", -1)) != 0.90:
        raise ConfigError("selection minimum length ratio must be 0.90")


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"missing [{name}] section")
    return value


def _open_unit(value: Any, name: str) -> None:
    numeric = float(value)
    if not 0 < numeric < 1:
        raise ConfigError(f"{name} must be in (0, 1)")


def _closed_unit(value: Any, name: str) -> None:
    numeric = float(value)
    if not 0 <= numeric <= 1:
        raise ConfigError(f"{name} must be in [0, 1]")
