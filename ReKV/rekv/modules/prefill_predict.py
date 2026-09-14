"""Module 1: Prefill future-importance prediction.

Reuse MM-ShiftKV's future Query Proxy scores P_i. This module only ranks
and splits those scores. Proxy construction stays in the eager attention hook.
"""

from __future__ import annotations

import torch


Tensor = torch.Tensor


def rank_visual_tokens(shift_scores: Tensor, image_mask: Tensor) -> Tensor:
    """Return visual-token ranks, descending by P_i, shape [L, H, V]."""
    visual = image_mask.nonzero(as_tuple=False).squeeze(-1)
    if visual.numel() == 0:
        raise ValueError("prefill image mask has no visual tokens")
    visual_scores = shift_scores.index_select(-1, visual)
    return visual[visual_scores.argsort(dim=-1, descending=True, stable=True)]


def split_visual_ranks(
    ranked: Tensor,
    *,
    active_budget: int,
    bin_budget: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Split ranked visual positions into active / recycling-bin / dropped."""
    visual_count = ranked.shape[-1]
    active_n = max(1, min(int(active_budget), visual_count))
    leftover = visual_count - active_n
    bin_n = max(0, min(int(bin_budget), leftover))
    active = ranked[..., :active_n]
    recycling = ranked[..., active_n : active_n + bin_n]
    dropped = ranked[..., active_n + bin_n :]
    return active, recycling, dropped
