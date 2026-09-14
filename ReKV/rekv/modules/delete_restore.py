"""Module 5: Fused scores, eviction into the bin, and shadow restore."""

from __future__ import annotations

import math

import torch

from vlesshallu.methods import masked_l1_normalize


Tensor = torch.Tensor


def fuse_token_scores(
    future_scores: Tensor,
    current_attention: Tensor,
    ema_scores: Tensor,
    image_mask: Tensor,
    *,
    future_weight: float,
    current_weight: float,
    ema_weight: float,
) -> Tensor:
    """Fuse future, current, and EMA scores into S[t,i]."""
    weights = (float(future_weight), float(current_weight), float(ema_weight))
    if any(weight < 0 for weight in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("score fusion weights must be non-negative and sum to 1")
    if future_scores.shape != current_attention.shape or future_scores.shape != ema_scores.shape:
        raise ValueError("fused score tensors must share the same shape")
    components = [
        masked_l1_normalize(future_scores, image_mask),
        masked_l1_normalize(current_attention, image_mask),
        masked_l1_normalize(ema_scores, image_mask),
    ]
    return sum(weight * value for weight, value in zip(weights, components, strict=True))


def _as_layer_queries(queries: Tensor) -> Tensor:
    """Accept Qwen2.5 M-RoPE leftovers and collapse to ``[L, query_heads, dim]``."""
    while queries.ndim > 3:
        singleton = [dim for dim in range(1, queries.ndim - 1) if queries.shape[dim] == 1]
        if not singleton:
            queries = queries.reshape(queries.shape[0], -1, queries.shape[-1])
            break
        queries = queries.squeeze(singleton[0])
    if queries.ndim == 2:
        queries = queries.unsqueeze(0)
    if queries.ndim != 3:
        raise ValueError(f"shadow queries must be [L,Q,D], got {tuple(queries.shape)}")
    return queries


def _as_bin_keys(bin_keys: Tensor) -> Tensor:
    if bin_keys.ndim == 5 and bin_keys.shape[1] == 1:
        bin_keys = bin_keys.squeeze(1)
    if bin_keys.ndim != 4:
        raise ValueError(f"shadow keys must be [L,H,B,D], got {tuple(bin_keys.shape)}")
    return bin_keys


def shadow_attention(queries: Tensor, bin_keys: Tensor) -> Tensor:
    """Score Recycling Bin keys with the current decode query qK^T.

    ``queries`` is ``[L, query_heads, head_dim]``.
    ``bin_keys`` is ``[L, kv_heads, bin_tokens, head_dim]``.
    Returns ``[L, kv_heads, bin_tokens]``.
    """
    queries = _as_layer_queries(queries)
    bin_keys = _as_bin_keys(bin_keys)
    if queries.shape[0] != bin_keys.shape[0]:
        raise ValueError("query and bin layer counts differ")
    kv_heads = bin_keys.shape[1]
    query_heads = queries.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    group = query_heads // kv_heads
    head_dim = queries.shape[-1]
    grouped = []
    for layer_idx in range(queries.shape[0]):
        query = queries[layer_idx].float().view(kv_heads, group, head_dim)
        logits = torch.einsum("hgd,htd->hgt", query, bin_keys[layer_idx].float())
        probabilities = logits.mul(head_dim**-0.5).softmax(dim=-1).mean(dim=1)
        grouped.append(probabilities)
    return torch.stack(grouped)


def update_ema(previous: Tensor, current: Tensor, decay: float) -> Tensor:
    if not 0 <= decay < 1:
        raise ValueError("ema decay must be in [0, 1)")
    return decay * previous + (1.0 - decay) * current.float()
