from __future__ import annotations

import traceback
from typing import Any, Callable

import torch

from .methods import (
    DynamicPruneState,
    KVSmoothState,
    gqa_to_kv,
    prune_visual_cache,
    visual_mask,
)


def run_self_tests() -> dict[str, Any]:
    tests: list[tuple[str, Callable[[], None]]] = [
        ("gqa_aggregation", _test_gqa),
        ("visual_mask", _test_visual_mask),
        ("dynamic_pruning", _test_dynamic_pruning),
        ("kvsmooth", _test_kvsmooth),
        ("cache_positions", _test_cache_positions),
    ]
    results = []
    for name, function in tests:
        try:
            function()
            results.append({"name": name, "ok": True})
        except Exception as error:  # Self-test must report every component.
            results.append(
                {
                    "name": name,
                    "ok": False,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
    return {"ok": all(result["ok"] for result in results), "tests": results}


def _test_gqa() -> None:
    attention = torch.arange(32, dtype=torch.float32).view(32, 1)
    observed = gqa_to_kv(attention, 8).squeeze(-1)
    expected = torch.tensor([1.5, 5.5, 9.5, 13.5, 17.5, 21.5, 25.5, 29.5])
    torch.testing.assert_close(observed, expected)


def _test_visual_mask() -> None:
    observed = visual_mask(12, [(2, 5), (8, 10)])
    expected = torch.tensor(
        [False, False, True, True, True, False, False, False, True, True, False, False]
    )
    if not torch.equal(observed, expected):
        raise AssertionError("visual mask did not preserve half-open spans")
    try:
        visual_mask(12, [(2, 6), (5, 8)])
    except ValueError:
        return
    raise AssertionError("overlapping visual spans were accepted")


def _test_dynamic_pruning() -> None:
    state = DynamicPruneState(retention_ratio=0.9, max_prunes=4, force_step=2)
    triggered, _ = state.observe(torch.ones(36))
    if triggered:
        raise AssertionError("step 1 must only establish the PruneHal reference")
    triggered, _ = state.observe(torch.ones(36))
    if not triggered or state.prune_count != 1:
        raise AssertionError("step 2 must force pruning")
    low = torch.cat((torch.full((18,), 0.1), torch.ones(18)))
    triggered, votes = state.observe(low)
    if not triggered or int(votes.sum()) != 18:
        raise AssertionError("half-layer vote did not trigger pruning")
    if state.next_visual_count(100) != 90:
        raise AssertionError("retention ratio produced the wrong cache size")


def _test_kvsmooth() -> None:
    state = KVSmoothState(fifo_size=15, lambda_ref=0.7)
    key = torch.tensor([2.0, 4.0])
    value = torch.tensor([6.0, 8.0])
    previous_key = torch.tensor([0.0, 2.0])
    previous_value = torch.tensor([2.0, 4.0])
    attention = torch.tensor([[0.5, 0.5]])
    smooth_key, smooth_value, coefficient, entropy = state.smooth(
        3, key, value, previous_key, previous_value, attention
    )
    if not 0.5 - 1e-9 <= coefficient <= 0.9 + 1e-9 or entropy <= 0:
        raise AssertionError("KVSmooth coefficient clipping or entropy failed")
    torch.testing.assert_close(
        smooth_key, (1 - coefficient) * key + coefficient * previous_key
    )
    torch.testing.assert_close(
        smooth_value, (1 - coefficient) * value + coefficient * previous_value
    )
    untouched = state.smooth(2, key, value, previous_key, previous_value, attention)
    if not torch.equal(untouched[0], key) or not torch.equal(untouched[1], value):
        raise AssertionError("KVSmooth modified a layer outside 3..34")


def _test_cache_positions() -> None:
    layers, heads, tokens, dim = 2, 8, 10, 4
    keys = torch.arange(layers * heads * tokens * dim).reshape(layers, heads, tokens, dim)
    values = -keys
    mask = visual_mask(tokens, [(2, 6)])
    scores = torch.zeros(layers, heads, tokens)
    for layer in range(layers):
        for head in range(heads):
            scores[layer, head, 2 + ((layer + head) % 4)] = 10
            scores[layer, head, 2 + ((layer + head + 1) % 4)] = 9
    pruned = prune_visual_cache(keys, values, mask, scores, keep_count=2)
    if pruned.keys.shape != (layers, heads, 8, dim):
        raise AssertionError(f"unexpected pruned shape: {pruned.keys.shape}")
    if not torch.equal(pruned.values, -pruned.keys):
        raise AssertionError("K/V selections diverged")
