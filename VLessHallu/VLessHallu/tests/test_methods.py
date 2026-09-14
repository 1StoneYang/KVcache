from __future__ import annotations

import pytest
import torch

from vlesshallu.methods import (
    DynamicPruneState,
    KVSmoothState,
    fuse_scores,
    gqa_to_kv,
    prune_visual_cache,
    select_visual_positions,
    visual_mask,
)


def test_gqa_aggregation() -> None:
    attention = torch.arange(2 * 8 * 5, dtype=torch.float32).reshape(2, 8, 5)
    reduced = gqa_to_kv(attention, num_kv_heads=2)
    expected = attention.reshape(2, 2, 4, 5).mean(dim=2)
    torch.testing.assert_close(reduced, expected)


def test_visual_mask_is_strict() -> None:
    mask = visual_mask(8, [(2, 5), (6, 7)])
    assert mask.tolist() == [False, False, True, True, True, False, True, False]


def test_equal_per_head_selection() -> None:
    scores = torch.tensor([[0.0, 3.0, 2.0, 1.0], [4.0, 1.0, 2.0, 3.0]])
    image_mask = torch.ones(4, dtype=torch.bool)
    kept = select_visual_positions(scores, image_mask, keep_count=2)
    assert kept.shape == (2, 2)
    assert kept[0].tolist() == [1, 2]
    assert kept[1].tolist() == [0, 3]


def test_cache_pruning_preserves_nonvisual_positions() -> None:
    keys = torch.arange(2 * 2 * 8, dtype=torch.float32).reshape(2, 2, 8, 1)
    values = keys + 100
    image_mask = visual_mask(8, [(2, 6)])
    scores = torch.zeros(2, 2, 8)
    scores[0, 0, [2, 5]] = torch.tensor([2.0, 1.0])
    scores[0, 1, [3, 4]] = torch.tensor([2.0, 1.0])
    scores[1] = scores[0]
    gathered = prune_visual_cache(keys, values, image_mask, scores, keep_count=2)
    assert gathered.keys.shape == (2, 2, 6, 1)
    assert gathered.original_positions[0, 0].tolist() == [0, 1, 2, 5, 6, 7]
    assert gathered.original_positions[0, 1].tolist() == [0, 1, 3, 4, 6, 7]


def test_dynamic_pruning_forces_step_two_then_votes() -> None:
    state = DynamicPruneState(retention_ratio=0.9, max_prunes=4)
    triggered, _ = state.observe(torch.ones(36))
    assert triggered is False
    triggered, _ = state.observe(torch.ones(36))
    assert triggered is True
    masses = torch.cat((torch.zeros(18), torch.ones(18)))
    triggered, votes = state.observe(masses)
    assert triggered is True
    assert int(votes.sum()) == 18


def test_fusion_uses_configured_weights() -> None:
    decode = torch.tensor([[1.0, 2.0]])
    shift = torch.tensor([[4.0, 2.0]])
    text = torch.tensor([[1.0, 5.0]])
    image_mask = torch.ones(2, dtype=torch.bool)
    fused = fuse_scores(
        decode, shift, text, image_mask, weights=(0.5, 0.25, 0.25)
    )
    torch.testing.assert_close(fused.sum(dim=-1), torch.ones(1))
    assert fused[0, 1] > fused[0, 0]


def test_kvsmooth_layer_bounds_fifo_and_clip() -> None:
    state = KVSmoothState(first_layer=3, last_layer=34, fifo_size=2, lambda_ref=0.7)
    assert state.includes(2) is False
    assert state.includes(3) is True
    assert state.includes(34) is True
    assert state.includes(35) is False
    coefficient, entropy = state.coefficient(3, torch.tensor([[0.5, 0.5]]))
    assert coefficient == pytest.approx(0.5)
    assert entropy > 0
    state.coefficient(3, torch.tensor([[0.9, 0.1]]))
    state.coefficient(3, torch.tensor([[0.6, 0.4]]))
    assert len(state.queues[3]) == 2


def test_kvsmooth_updates_only_current_kv() -> None:
    state = KVSmoothState()
    key = torch.tensor([5.0, 7.0])
    value = torch.tensor([9.0, 11.0])
    previous_key = torch.tensor([1.0, 3.0])
    previous_value = torch.tensor([5.0, 7.0])
    output_key, output_value, coefficient, _ = state.smooth(
        3,
        key,
        value,
        previous_key,
        previous_value,
        torch.tensor([[0.5, 0.5]]),
    )
    torch.testing.assert_close(
        output_key, (1 - coefficient) * key + coefficient * previous_key
    )
    torch.testing.assert_close(
        output_value, (1 - coefficient) * value + coefficient * previous_value
    )
