"""Myopia-style Recycling Bin for visual KV dropped by MM-ShiftKV."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def recycle_deleted_visual(
    keys: torch.Tensor,
    values: torch.Tensor,
    scores: torch.Tensor,
    kept_mask: torch.Tensor,
    visual_mask: torch.Tensor,
    bin_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Compress deleted visual KV into B representative tokens.

    ``K_b = TopB(K_p)``. Remaining ``K_e`` tokens are merged into the most
    similar bin token by cosine similarity of keys. Values follow the same
    assignment and are count-weighted averaged.

    Args:
        keys: ``[S, D]`` RoPE-applied keys for one KV head.
        values: ``[S, D]`` values for one KV head.
        scores: ``[S]`` MM-ShiftKV importance scores.
        kept_mask: ``[S]`` tokens already kept by MM-ShiftKV.
        visual_mask: ``[S]`` visual-token mask.
        bin_size: ``B``, number of Recycling Bin representatives.
    """
    if bin_size <= 0:
        return None, None, None

    deleted = visual_mask.bool() & ~kept_mask.bool()
    deleted_idx = deleted.nonzero(as_tuple=False).reshape(-1)
    if deleted_idx.numel() == 0:
        return None, None, None

    dropped_keys = keys.index_select(0, deleted_idx)
    dropped_values = values.index_select(0, deleted_idx)
    dropped_scores = scores.index_select(0, deleted_idx).float()

    actual_b = min(int(bin_size), deleted_idx.numel())
    topb = torch.topk(dropped_scores, k=actual_b, largest=True, sorted=True).indices
    bin_keys = dropped_keys.index_select(0, topb).clone()
    bin_values = dropped_values.index_select(0, topb).clone()
    bin_positions = deleted_idx.index_select(0, topb)

    if deleted_idx.numel() == actual_b:
        return bin_keys, bin_values, bin_positions

    rest_mask = torch.ones(deleted_idx.numel(), dtype=torch.bool, device=keys.device)
    rest_mask[topb] = False
    rest_keys = dropped_keys[rest_mask]
    rest_values = dropped_values[rest_mask]
    bin_keys, bin_values = _merge_by_cosine(bin_keys, bin_values, rest_keys, rest_values)
    return bin_keys, bin_values, bin_positions


def _merge_by_cosine(
    bin_keys: torch.Tensor,
    bin_values: torch.Tensor,
    rest_keys: torch.Tensor,
    rest_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count-weighted average of rest tokens into their nearest bin token."""
    if rest_keys.numel() == 0:
        return bin_keys, bin_values

    rest_norm = F.normalize(rest_keys.float(), dim=-1)
    bin_norm = F.normalize(bin_keys.float(), dim=-1)
    assign = (rest_norm @ bin_norm.transpose(0, 1)).argmax(dim=-1)

    bin_count, head_dim = bin_keys.shape
    counts = torch.ones(bin_count, 1, device=bin_keys.device, dtype=bin_keys.dtype)
    sum_keys = bin_keys.to(dtype=bin_keys.dtype)
    sum_values = bin_values.to(dtype=bin_values.dtype)
    expand_assign = assign.unsqueeze(-1).expand(-1, head_dim)
    sum_keys = sum_keys.scatter_add(0, expand_assign, rest_keys.to(sum_keys.dtype))
    sum_values = sum_values.scatter_add(0, expand_assign, rest_values.to(sum_values.dtype))
    ones = torch.ones(rest_keys.shape[0], 1, device=bin_keys.device, dtype=counts.dtype)
    counts = counts.scatter_add(0, assign.unsqueeze(-1), ones)
    return sum_keys / counts, sum_values / counts
