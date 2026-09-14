from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch


Tensor = torch.Tensor


def visual_mask(length: int, spans: Iterable[tuple[int, int]], *, device=None) -> Tensor:
    """Build a strict half-open visual-token mask."""
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    previous_end = 0
    for start, end in sorted(spans):
        if start < previous_end or start < 0 or end <= start or end > length:
            raise ValueError(f"invalid or overlapping visual span [{start}, {end})")
        mask[start:end] = True
        previous_end = end
    if not mask.any():
        raise ValueError("at least one visual token is required")
    return mask


def gqa_to_kv(attention: Tensor, num_kv_heads: int, *, reduction: str = "mean") -> Tensor:
    """Aggregate consecutive GQA query heads into their shared KV heads.

    ``attention`` is shaped ``[..., query_heads, sequence]``.
    """
    query_heads = attention.shape[-2]
    if query_heads % num_kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    grouped = attention.reshape(
        *attention.shape[:-2],
        num_kv_heads,
        query_heads // num_kv_heads,
        attention.shape[-1],
    )
    if reduction == "mean":
        return grouped.mean(dim=-2)
    if reduction == "sum":
        return grouped.sum(dim=-2)
    if reduction == "max":
        return grouped.max(dim=-2).values
    raise ValueError(f"unsupported GQA reduction: {reduction}")


def masked_l1_normalize(scores: Tensor, mask: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Normalize non-negative per-head scores over visual positions only."""
    if scores.shape[-1] != mask.numel():
        raise ValueError("score length and mask length differ")
    if not mask.any():
        raise ValueError("visual mask is empty")
    clean = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    clean = clean.clamp_min(0) * mask.to(clean.dtype)
    denominator = clean.sum(dim=-1, keepdim=True)
    uniform = mask.to(clean.dtype) / mask.sum().to(clean.dtype)
    normalized = clean / denominator.clamp_min(eps)
    return torch.where(denominator > eps, normalized, uniform.expand_as(clean))


def future_proxy_positions(
    last_position: int,
    num_proxies: int,
    num_groups: int,
    *,
    device=None,
) -> Tensor:
    """Return the grouped future-position schedule used by MM-ShiftKV."""
    if num_proxies <= 0 or num_groups <= 0 or num_proxies % num_groups:
        raise ValueError("num_proxies must be divisible by num_groups")
    group_size = num_proxies // num_groups
    offsets = torch.arange(1, group_size + 1, device=device) * num_groups
    return last_position + offsets.repeat(num_groups)


def sample_query_proxies(
    hidden_states: Tensor,
    q_weight: Tensor,
    *,
    num_query_heads: int,
    head_dim: int,
    num_proxies: int,
    gamma: float,
    generator: torch.Generator | None = None,
    q_bias: Tensor | None = None,
    postprocess: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Sample paper-faithful feature-wise Gaussian hidden states and project Q."""
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden_size]")
    if gamma <= 1:
        raise ValueError("variance expansion gamma must be greater than 1")
    mean = hidden_states.float().mean(dim=0)
    std = hidden_states.float().std(dim=0, unbiased=False)
    noise = torch.randn(
        (num_proxies, hidden_states.shape[-1]),
        generator=generator,
        device=hidden_states.device,
        dtype=torch.float32,
    )
    sampled = mean + noise * (gamma * std)
    projected = torch.nn.functional.linear(sampled.to(q_weight.dtype), q_weight, q_bias)
    projected = projected.reshape(num_proxies, num_query_heads, head_dim).transpose(0, 1)
    if postprocess is not None:
        projected = postprocess(projected)
    return projected


def mmshift_scores(
    proxy_queries: Tensor,
    prompt_keys: Tensor,
    last_query: Tensor,
    *,
    num_groups: int = 32,
    mass_threshold: float = 0.95,
    anchor: float = 1.0,
) -> Tensor:
    """Compute group-vote MM-ShiftKV scores for each KV head and token.

    Args:
        proxy_queries: ``[query_heads, N, head_dim]`` after future-position RoPE.
        prompt_keys: ``[kv_heads, tokens, head_dim]`` with prefill RoPE applied.
        last_query: ``[query_heads, head_dim]`` for the final real prefill query.
    """
    if not 0 < mass_threshold < 1:
        raise ValueError("mass_threshold must be in (0, 1)")
    query_heads, num_proxies, head_dim = proxy_queries.shape
    kv_heads, tokens, key_dim = prompt_keys.shape
    if head_dim != key_dim or query_heads % kv_heads:
        raise ValueError("incompatible query/key head shapes")
    if num_proxies % num_groups:
        raise ValueError("proxy count must be divisible by group count")

    queries_per_kv = query_heads // kv_heads
    queries = proxy_queries.reshape(kv_heads, queries_per_kv, num_proxies, head_dim)
    logits = torch.einsum("hqnd,htd->hqnt", queries.float(), prompt_keys.float())
    probabilities = logits.mul(head_dim**-0.5).softmax(dim=-1).mean(dim=1)
    group_size = num_proxies // num_groups
    grouped_mass = probabilities.reshape(kv_heads, num_groups, group_size, tokens).sum(dim=2)

    sorted_mass, sorted_indices = grouped_mass.sort(dim=-1, descending=True, stable=True)
    cumulative = sorted_mass.cumsum(dim=-1)
    thresholds = mass_threshold * cumulative[..., -1:]
    selected_count = torch.searchsorted(
        cumulative.contiguous(), thresholds.contiguous(), right=False
    ).squeeze(-1) + 1

    ranks = torch.arange(tokens, device=prompt_keys.device)
    chosen = ranks.view(1, 1, -1) < selected_count.unsqueeze(-1)
    votes = torch.zeros_like(grouped_mass)
    votes.scatter_add_(-1, sorted_indices, chosen.to(votes.dtype))
    votes = votes.sum(dim=1)

    anchor_queries = last_query.reshape(kv_heads, queries_per_kv, head_dim)
    anchor_logits = torch.einsum(
        "hqd,htd->hqt", anchor_queries.float(), prompt_keys.float()
    )
    anchor_attention = anchor_logits.mul(head_dim**-0.5).softmax(dim=-1).mean(dim=1)
    return votes + float(anchor) * anchor_attention


def myopia_scores(
    prefill_attention: Tensor,
    image_mask: Tensor,
    text_mask: Tensor,
    *,
    num_kv_heads: int,
    text_alpha: float = 0.5,
) -> Tensor:
    """Combine Myopia's cumulative visual and text-guided scores, without DAS.

    ``prefill_attention`` is ``[query_heads, query_tokens, key_tokens]``.
    """
    if prefill_attention.ndim != 3:
        raise ValueError("prefill attention must be [heads, queries, keys]")
    if prefill_attention.shape[-2:] != (image_mask.numel(), image_mask.numel()):
        raise ValueError("prefill attention and token masks disagree")
    if text_mask.shape != image_mask.shape or torch.any(text_mask & image_mask):
        raise ValueError("text and image masks must be disjoint and aligned")
    if not 0 <= text_alpha <= 1:
        raise ValueError("text_alpha must be in [0, 1]")

    cumulative = prefill_attention.float().sum(dim=-2)
    if text_mask.any():
        guided = prefill_attention[:, text_mask, :].float().sum(dim=-2)
    else:
        guided = torch.zeros_like(cumulative)
    cumulative = gqa_to_kv(cumulative, num_kv_heads)
    guided = gqa_to_kv(guided, num_kv_heads)
    cumulative = masked_l1_normalize(cumulative, image_mask)
    guided = masked_l1_normalize(guided, image_mask)
    return (1 - text_alpha) * cumulative + text_alpha * guided


def decode_visual_scores(
    decode_attention: Tensor, image_mask: Tensor, *, num_kv_heads: int
) -> Tensor:
    """Aggregate a decoding query's GQA attention into per-KV-head scores."""
    if decode_attention.ndim == 3 and decode_attention.shape[-2] == 1:
        decode_attention = decode_attention.squeeze(-2)
    if decode_attention.ndim != 2:
        raise ValueError("decode attention must be [query_heads, key_tokens]")
    return masked_l1_normalize(gqa_to_kv(decode_attention, num_kv_heads), image_mask)


def fuse_scores(
    decode: Tensor,
    shift: Tensor,
    text: Tensor,
    image_mask: Tensor,
    *,
    weights: tuple[float, float, float] = (0.50, 0.25, 0.25),
) -> Tensor:
    if decode.shape != shift.shape or decode.shape != text.shape:
        raise ValueError("all score components must have identical shapes")
    if any(weight < 0 for weight in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("fusion weights must be non-negative and sum to 1")
    components = [masked_l1_normalize(value, image_mask) for value in (decode, shift, text)]
    return sum(weight * value for weight, value in zip(weights, components, strict=True))


def select_visual_positions(scores: Tensor, image_mask: Tensor, keep_count: int) -> Tensor:
    """Select an equal visual-token count independently for every KV head."""
    visual_positions = image_mask.nonzero(as_tuple=False).squeeze(-1)
    if keep_count <= 0 or keep_count > visual_positions.numel():
        raise ValueError("keep_count must be within the current visual-token count")
    visual_scores = scores.index_select(-1, visual_positions)
    selected_local = torch.topk(
        visual_scores, keep_count, dim=-1, largest=True, sorted=False
    ).indices
    selected = visual_positions[selected_local]
    return selected.sort(dim=-1).values


@dataclass(frozen=True)
class PrunedCache:
    keys: Tensor
    values: Tensor
    original_positions: Tensor
    visual_count: int


def prune_visual_cache(
    keys: Tensor,
    values: Tensor,
    image_mask: Tensor,
    scores: Tensor,
    keep_count: int,
) -> PrunedCache:
    """Physically gather a cache while preserving every non-visual token.

    K/V shapes are ``[layers, kv_heads, tokens, head_dim]``. Scores are
    ``[layers, kv_heads, tokens]``. Visual selections may differ by layer/head,
    but every cache row has the same resulting length.
    """
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("K/V must share [layers, kv_heads, tokens, head_dim]")
    if scores.shape != keys.shape[:-1] or image_mask.numel() != keys.shape[-2]:
        raise ValueError("cache, score, and mask shapes disagree")
    selected = torch.stack(
        [
            select_visual_positions(scores[layer], image_mask, keep_count)
            for layer in range(keys.shape[0])
        ]
    )
    nonvisual = (~image_mask).nonzero(as_tuple=False).squeeze(-1)
    nonvisual = nonvisual.view(1, 1, -1).expand(keys.shape[0], keys.shape[1], -1)
    positions = torch.cat((selected, nonvisual), dim=-1).sort(dim=-1).values
    gather_index = positions.unsqueeze(-1).expand(*positions.shape, keys.shape[-1])
    pruned_keys = keys.gather(dim=2, index=gather_index)
    pruned_values = values.gather(dim=2, index=gather_index)
    assert_cache_position_invariants(positions, image_mask, keep_count)
    return PrunedCache(pruned_keys, pruned_values, positions, keep_count)


def assert_cache_position_invariants(
    original_positions: Tensor, image_mask: Tensor, visual_count: int
) -> None:
    if original_positions.ndim != 3:
        raise AssertionError("positions must be [layers, kv_heads, tokens]")
    if not torch.all(original_positions[..., 1:] > original_positions[..., :-1]):
        raise AssertionError("cache positions must remain strictly increasing")
    expected_nonvisual = (~image_mask).nonzero(as_tuple=False).squeeze(-1)
    for row in original_positions.reshape(-1, original_positions.shape[-1]):
        kept_visual = image_mask[row].sum().item()
        if kept_visual != visual_count:
            raise AssertionError("every KV head must retain the same visual count")
        actual_nonvisual = row[~image_mask[row]]
        if not torch.equal(actual_nonvisual, expected_nonvisual):
            raise AssertionError("text and generated tokens may not be pruned or reordered")


@dataclass
class DynamicPruneState:
    retention_ratio: float = 0.9
    max_prunes: int = 4
    force_step: int = 2
    vote_fraction: float = 0.5
    step: int = 0
    prune_count: int = 0
    reference: Tensor | None = None
    previous: Tensor | None = None

    def observe(self, layer_visual_mass: Tensor) -> tuple[bool, Tensor]:
        """Observe per-layer visual mass and return (triggered, layer votes)."""
        if layer_visual_mass.ndim != 1:
            raise ValueError("layer visual mass must be one-dimensional")
        current = layer_visual_mass.detach().float()
        self.step += 1
        if self.reference is None:
            self.reference = current.clone()
            self.previous = current.clone()
            return False, torch.zeros_like(current, dtype=torch.bool)
        votes = current < math.sqrt(self.retention_ratio) * self.reference
        required = math.ceil(current.numel() * self.vote_fraction)
        triggered = self.prune_count < self.max_prunes and (
            self.step == self.force_step or int(votes.sum()) >= required
        )
        if triggered:
            self.prune_count += 1
            self.reference = (self.previous if self.previous is not None else current).clone()
        self.previous = current.clone()
        return triggered, votes

    def next_visual_count(self, current_count: int) -> int:
        if current_count <= 0:
            raise ValueError("current visual count must be positive")
        return max(1, math.floor(current_count * self.retention_ratio))


@dataclass
class KVSmoothState:
    first_layer: int = 3
    last_layer: int = 34
    fifo_size: int = 15
    lambda_ref: float = 0.7
    clip_radius: float = 0.2
    queues: dict[int, deque[float]] = field(default_factory=dict)

    def includes(self, layer: int) -> bool:
        return self.first_layer <= layer <= self.last_layer

    def coefficient(self, layer: int, attention_row: Tensor) -> tuple[float, float]:
        """Return (clipped lambda, row entropy) and update the layer FIFO."""
        if not self.includes(layer):
            return 0.0, 0.0
        probabilities = attention_row.float().clamp_min(1e-12)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        entropy = float((-(probabilities * probabilities.log()).sum(dim=-1)).mean())
        queue = self.queues.setdefault(layer, deque(maxlen=self.fifo_size))
        queue.append(entropy)
        rank = sum(value < entropy for value in queue)
        raw = rank / self.fifo_size
        lower = max(0.0, self.lambda_ref - self.clip_radius)
        upper = min(1.0, self.lambda_ref + self.clip_radius)
        return min(upper, max(lower, raw)), entropy

    def smooth(
        self,
        layer: int,
        key: Tensor,
        value: Tensor,
        previous_key: Tensor,
        previous_value: Tensor,
        attention_row: Tensor,
    ) -> tuple[Tensor, Tensor, float, float]:
        coefficient, entropy = self.coefficient(layer, attention_row)
        if not self.includes(layer):
            return key, value, coefficient, entropy
        smoothed_key = (1 - coefficient) * key + coefficient * previous_key
        smoothed_value = (1 - coefficient) * value + coefficient * previous_value
        return smoothed_key, smoothed_value, coefficient, entropy
