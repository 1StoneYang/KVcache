from __future__ import annotations

from types import SimpleNamespace

import torch

from vlesshallu.headwise_cache import HeadwiseCacheState
from vlesshallu.methods import visual_mask


def _fake_cache() -> SimpleNamespace:
    layers = []
    for layer_idx in range(2):
        keys = torch.arange(2 * 8, dtype=torch.float32).reshape(1, 2, 8, 1)
        keys = keys + 100 * layer_idx
        layers.append(SimpleNamespace(keys=keys, values=keys + 1000))
    return SimpleNamespace(layers=layers)


def test_headwise_cache_compacts_different_positions_with_equal_counts() -> None:
    cache = _fake_cache()
    state = HeadwiseCacheState.from_prefill(cache, visual_mask(8, [(2, 6)]))
    scores = torch.zeros(2, 2, 8)
    scores[:, 0, [2, 5]] = torch.tensor([2.0, 1.0])
    scores[:, 1, [3, 4]] = torch.tensor([2.0, 1.0])
    result = state.prune(scores, keep_count=2)
    assert result.before_visual_count == 4
    assert result.after_visual_count == 2
    assert state.image_mask.tolist() == [False, False, True, True, False, False]
    assert state.original_positions[0, 0].tolist() == [0, 1, 2, 5, 6, 7]
    assert state.original_positions[0, 1].tolist() == [0, 1, 3, 4, 6, 7]
    assert cache.layers[0].keys.shape == (1, 2, 6, 1)


def test_headwise_cache_tracks_generated_tokens_across_prunes() -> None:
    cache = _fake_cache()
    state = HeadwiseCacheState.from_prefill(cache, visual_mask(8, [(2, 6)]))
    for layer in cache.layers:
        layer.keys = torch.cat((layer.keys, layer.keys[:, :, -1:, :]), dim=2)
        layer.values = torch.cat((layer.values, layer.values[:, :, -1:, :]), dim=2)
    state.append_nonvisual(8)
    scores = torch.zeros(2, 2, 9)
    scores[..., 2] = 1.0
    state.prune(scores, keep_count=1)
    assert state.original_positions[..., -1].eq(8).all()
    assert state.image_mask[-1].item() is False
