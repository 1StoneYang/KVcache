from __future__ import annotations

import torch

from mmshift_re.recycle import recycle_deleted_visual


def test_topb_then_cosine_merge_keeps_b_tokens():
    keys = torch.tensor(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
            [0.7, 0.7],
        ]
    )
    values = keys.clone()
    scores = torch.tensor([0.1, 0.9, 0.8, 0.2, 0.05])
    kept = torch.tensor([True, False, False, False, False])
    visual = torch.tensor([False, True, True, True, True])

    bin_k, bin_v, bin_pos = recycle_deleted_visual(
        keys, values, scores, kept, visual, bin_size=2
    )

    assert bin_k is not None and bin_v is not None and bin_pos is not None
    assert bin_k.shape[0] == 2
    assert bin_v.shape == bin_k.shape
    # Highest dropped visual scores are positions 1 (0.9) and 2 (0.8).
    assert set(bin_pos.tolist()) == {1, 2}
    # Position 3 is close to the [0, 1] representative and should be merged in.
    merged = bin_k[bin_pos.tolist().index(2)]
    assert merged[1] > merged[0]


def test_fewer_deleted_than_b_keeps_all():
    keys = torch.eye(3)
    values = keys.clone()
    scores = torch.tensor([0.3, 0.2, 0.1])
    kept = torch.tensor([True, False, False])
    visual = torch.ones(3, dtype=torch.bool)

    bin_k, _bin_v, bin_pos = recycle_deleted_visual(
        keys, values, scores, kept, visual, bin_size=20
    )
    assert bin_k.shape[0] == 2
    assert set(bin_pos.tolist()) == {1, 2}


def test_no_visual_deleted_returns_none():
    keys = torch.ones(4, 2)
    values = keys.clone()
    scores = torch.arange(4.0)
    kept = torch.tensor([True, True, False, False])
    visual = torch.tensor([True, True, False, False])

    bin_k, bin_v, bin_pos = recycle_deleted_visual(
        keys, values, scores, kept, visual, bin_size=20
    )
    assert bin_k is None and bin_v is None and bin_pos is None
