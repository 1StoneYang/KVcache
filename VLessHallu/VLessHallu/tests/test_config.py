from pathlib import Path

from vlesshallu.config import config_hash, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_and_hash_are_stable():
    config = load_config(ROOT / "configs" / "default.toml")
    first = config_hash(
        config,
        variant="baseline",
        benchmark="chair",
        split="smoke",
        resource_lock_hash="abc",
    )
    second = config_hash(
        config,
        variant="baseline",
        benchmark="chair",
        split="smoke",
        resource_lock_hash="abc",
    )
    assert first == second
    assert len(first) == 16
