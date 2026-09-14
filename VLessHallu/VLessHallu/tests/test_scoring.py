from __future__ import annotations

import torch

from vlesshallu.methods import future_proxy_positions, mmshift_scores, myopia_scores


def test_mmshift_future_position_schedule() -> None:
    positions = future_proxy_positions(100, num_proxies=8, num_groups=2)
    assert positions.tolist() == [102, 104, 106, 108, 102, 104, 106, 108]


def test_mmshift_scores_are_per_kv_head() -> None:
    proxy_queries = torch.arange(4 * 4 * 2, dtype=torch.float32).reshape(4, 4, 2)
    prompt_keys = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    last_query = torch.ones(4, 2)
    scores = mmshift_scores(
        proxy_queries,
        prompt_keys,
        last_query,
        num_groups=2,
        mass_threshold=0.95,
        anchor=1.0,
    )
    assert scores.shape == (2, 3)
    assert torch.all(scores >= 0)


def test_myopia_uses_cumulative_and_text_guided_visual_attention() -> None:
    attention = torch.zeros(4, 6, 6)
    attention[:, :, 1] = 0.25
    attention[:, :, 2] = 0.75
    attention[:, 4:, 1] = 0.9
    attention[:, 4:, 2] = 0.1
    image_mask = torch.tensor([False, True, True, False, False, False])
    text_mask = torch.tensor([False, False, False, True, True, True])
    scores = myopia_scores(
        attention,
        image_mask,
        text_mask,
        num_kv_heads=2,
        text_alpha=0.5,
    )
    assert scores.shape == (2, 6)
    torch.testing.assert_close(scores.sum(dim=-1), torch.ones(2))
    assert torch.all(scores[:, ~image_mask] == 0)
