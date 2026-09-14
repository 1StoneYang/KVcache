from types import SimpleNamespace

import torch

from rekv.modules.budget_control import decide_budget
from rekv.modules.decode_monitor import observe_decode
from rekv.modules.delete_restore import fuse_token_scores, shadow_attention
from rekv.modules.prefill_predict import split_visual_ranks
from rekv.modules.two_level_cache import TwoLevelCache, shared_visual_ranks
from vlesshallu.methods import visual_mask


def test_split_visual_ranks():
    ranked = torch.arange(10).view(1, 1, 10).expand(2, 2, 10)
    active, recycling, dropped = split_visual_ranks(
        ranked, active_budget=4, bin_budget=3
    )
    assert active.shape[-1] == 4
    assert recycling.shape[-1] == 3
    assert dropped.shape[-1] == 3


def test_decode_monitor_mass_and_entropy():
    attention = torch.zeros(4, 8)
    attention[:, 2:6] = 0.2
    attention[:, 6:] = 0.1
    mask = visual_mask(8, [(2, 6)])
    observation = observe_decode([attention, attention], mask, probe_layers=[0, 1])
    assert 0.79 < observation.visual_mass < 0.81
    assert observation.normalized_entropy > 0


def test_budget_restore_on_high_entropy():
    decision = decide_budget(
        current_active=32,
        bin_count=16,
        visual_mass=0.10,
        normalized_entropy=0.90,
        mass_low=0.18,
        mass_high=0.40,
        entropy_low=0.35,
        entropy_high=0.80,
        step=8,
        min_active=16,
        max_active=128,
    )
    assert decision.action == "restore"
    assert decision.target_active == 40


def test_budget_evict_on_confident_low_mass():
    decision = decide_budget(
        current_active=32,
        bin_count=16,
        visual_mass=0.10,
        normalized_entropy=0.20,
        mass_low=0.18,
        mass_high=0.40,
        entropy_low=0.35,
        entropy_high=0.80,
        step=8,
        min_active=16,
        max_active=128,
    )
    assert decision.action == "evict"
    assert decision.target_active == 24


def test_fuse_and_shadow_shapes():
    mask = visual_mask(6, [(1, 5)])
    scores = fuse_token_scores(
        torch.ones(2, 2, 6),
            torch.arange(24, dtype=torch.float32).view(2, 2, 6),
        torch.zeros(2, 2, 6),
        mask,
        future_weight=0.4,
        current_weight=0.4,
        ema_weight=0.2,
    )
    assert scores.shape == (2, 2, 6)
    shadow = shadow_attention(torch.randn(2, 4, 8), torch.randn(2, 2, 3, 8))
    assert shadow.shape == (2, 2, 3)
    # Qwen2.5 M-RoPE leftover: stacked decode queries can be [L,1,Q,1,D]
    leftover = torch.randn(2, 1, 4, 1, 8)
    assert shadow_attention(leftover, torch.randn(2, 2, 3, 8)).shape == (2, 2, 3)


def _fake_cache(tokens: int = 8):
    layers = []
    for layer_idx in range(2):
        keys = torch.arange(2 * tokens, dtype=torch.float32).reshape(1, 2, tokens, 1)
        keys = keys + 100 * layer_idx
        layers.append(SimpleNamespace(keys=keys, values=keys + 1000))
    return SimpleNamespace(layers=layers)


def test_two_level_init_and_restore():
    cache = _fake_cache(8)
    mask = visual_mask(8, [(2, 6)])
    shift = torch.zeros(2, 2, 8)
    shift[..., 2] = 4
    shift[..., 3] = 3
    shift[..., 4] = 2
    shift[..., 5] = 1
    store = TwoLevelCache.from_prefill(
        cache, mask, shift, active_budget=2, bin_budget=2
    )
    assert store.active_visual == 2
    assert store.bin_count == 2
    restored = store.restore_from_bin(torch.tensor([[[0.1, 0.9], [0.1, 0.9]], [[0.2, 0.8], [0.2, 0.8]]]), 1)
    assert restored["moved"] == 1
    assert store.active_visual == 3
    assert store.bin_count == 1
    _ = shared_visual_ranks(shift, mask)
