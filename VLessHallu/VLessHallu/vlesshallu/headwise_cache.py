from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .methods import select_visual_positions


Tensor = torch.Tensor


class _KeyCacheLayer:
    """Adapter so Transformers 4.50 DynamicCache looks like cache.layers[i]."""

    def __init__(self, cache: Any, index: int) -> None:
        self._cache = cache
        self._index = index

    @property
    def keys(self) -> Tensor:
        return self._cache.key_cache[self._index]

    @keys.setter
    def keys(self, value: Tensor) -> None:
        self._cache.key_cache[self._index] = value

    @property
    def values(self) -> Tensor:
        return self._cache.value_cache[self._index]

    @values.setter
    def values(self, value: Tensor) -> None:
        self._cache.value_cache[self._index] = value


def cache_layers(cache: Any) -> list[Any]:
    layers = getattr(cache, "layers", None)
    if layers:
        return list(layers)
    key_cache = getattr(cache, "key_cache", None)
    if key_cache:
        return [_KeyCacheLayer(cache, index) for index in range(len(key_cache))]
    raise ValueError("cache has no initialized layers")


@dataclass(frozen=True)
class CachePruneResult:
    before_visual_count: int
    after_visual_count: int
    selected_original_visual_positions: Tensor


@dataclass
class HeadwiseCacheState:
    """Metadata and in-place compaction for a single-sample DynamicCache.

    Transformers stores K/V as ``[batch, kv_heads, tokens, head_dim]``. Equal
    retained counts let every head keep the same tensor shape while choosing
    different visual positions.
    """

    cache: Any
    image_mask: Tensor
    original_positions: Tensor
    shift_scores: Tensor
    text_scores: Tensor

    @classmethod
    def from_prefill(
        cls,
        cache: Any,
        image_mask: Tensor,
        *,
        shift_scores: Tensor | None = None,
        text_scores: Tensor | None = None,
    ) -> "HeadwiseCacheState":
        if image_mask.ndim != 1 or not image_mask.any():
            raise ValueError("image_mask must be one-dimensional and non-empty")
        layers = cache_layers(cache)
        layer_count = len(layers)
        first = layers[0].keys
        if first.ndim != 4 or first.shape[0] != 1:
            raise ValueError("only batch-size-one [1, heads, tokens, dim] caches are supported")
        kv_heads = first.shape[1]
        sequence_length = first.shape[-2]
        if sequence_length != image_mask.numel():
            raise ValueError("cache and image mask lengths differ")
        for layer in layers:
            if layer.keys.shape[:3] != (1, kv_heads, sequence_length):
                raise ValueError("all cache layers must share head and sequence dimensions")
            if layer.keys.shape != layer.values.shape:
                raise ValueError("cache K/V shapes differ")

        score_shape = (layer_count, kv_heads, sequence_length)
        device = first.device
        shift = _validated_scores(shift_scores, score_shape, device)
        text = _validated_scores(text_scores, score_shape, device)
        positions = torch.arange(sequence_length, device=device).view(1, 1, -1)
        positions = positions.expand(layer_count, kv_heads, -1).clone()
        return cls(
            cache=cache,
            image_mask=image_mask.to(device=device, dtype=torch.bool),
            original_positions=positions,
            shift_scores=shift,
            text_scores=text,
        )

    @property
    def visual_count(self) -> int:
        return int(self.image_mask.sum())

    @property
    def sequence_length(self) -> int:
        return self.image_mask.numel()

    def append_nonvisual(self, original_position: int) -> None:
        expected = self.sequence_length + 1
        for layer in cache_layers(self.cache):
            if layer.keys.shape[-2] != expected:
                raise AssertionError("cache did not append exactly one generated token")
        device = self.image_mask.device
        self.image_mask = torch.cat(
            (self.image_mask, torch.zeros(1, dtype=torch.bool, device=device))
        )
        layer_count, kv_heads, _ = self.original_positions.shape
        appended_position = torch.full(
            (layer_count, kv_heads, 1),
            int(original_position),
            dtype=self.original_positions.dtype,
            device=device,
        )
        self.original_positions = torch.cat(
            (self.original_positions, appended_position), dim=-1
        )
        zeros = torch.zeros(
            (layer_count, kv_heads, 1),
            dtype=self.shift_scores.dtype,
            device=device,
        )
        self.shift_scores = torch.cat((self.shift_scores, zeros), dim=-1)
        self.text_scores = torch.cat((self.text_scores, zeros.clone()), dim=-1)

    def prune(self, scores: Tensor, keep_count: int) -> CachePruneResult:
        if scores.shape != self.original_positions.shape:
            raise ValueError("pruning scores and tracked cache shape differ")
        before = self.visual_count
        if keep_count <= 0 or keep_count > before:
            raise ValueError("keep_count must be within the current visual-token count")

        selected = torch.stack(
            [
                select_visual_positions(scores[layer], self.image_mask, keep_count)
                for layer in range(scores.shape[0])
            ]
        )
        nonvisual = (~self.image_mask).nonzero(as_tuple=False).squeeze(-1)
        nonvisual = nonvisual.view(1, 1, -1).expand(
            scores.shape[0], scores.shape[1], -1
        )
        indices = torch.cat((selected, nonvisual), dim=-1).sort(dim=-1).values

        row_masks = self.image_mask[indices]
        new_image_mask = row_masks[0, 0]
        if not torch.all(row_masks == new_image_mask):
            raise AssertionError("head-wise compaction produced inconsistent visual slots")

        for layer_idx, layer in enumerate(cache_layers(self.cache)):
            layer_indices = indices[layer_idx]
            gather_index = layer_indices.unsqueeze(0).unsqueeze(-1).expand(
                1, layer_indices.shape[0], layer_indices.shape[1], layer.keys.shape[-1]
            )
            layer.keys = layer.keys.gather(dim=2, index=gather_index)
            layer.values = layer.values.gather(dim=2, index=gather_index)

        self.original_positions = self.original_positions.gather(dim=-1, index=indices)
        self.shift_scores = self.shift_scores.gather(dim=-1, index=indices)
        self.text_scores = self.text_scores.gather(dim=-1, index=indices)
        self.image_mask = new_image_mask
        selected_global = self.original_positions[..., self.image_mask]
        self.assert_invariants()
        return CachePruneResult(
            before_visual_count=before,
            after_visual_count=keep_count,
            selected_original_visual_positions=selected_global,
        )

    def assert_invariants(self) -> None:
        expected_length = self.sequence_length
        expected_shape = self.original_positions.shape
        if self.shift_scores.shape != expected_shape or self.text_scores.shape != expected_shape:
            raise AssertionError("tracked score shapes diverged")
        if not torch.all(
            self.original_positions[..., 1:] > self.original_positions[..., :-1]
        ):
            raise AssertionError("original cache positions are not strictly increasing")
        for layer in cache_layers(self.cache):
            if layer.keys.shape[-2] != expected_length or layer.values.shape[-2] != expected_length:
                raise AssertionError("physical cache and metadata lengths diverged")
        per_row_visual = self.image_mask.sum().expand(expected_shape[:2])
        if not torch.all(per_row_visual == self.visual_count):
            raise AssertionError("KV heads do not share a retained visual count")


def _validated_scores(
    scores: Tensor | None,
    shape: tuple[int, int, int],
    device: torch.device,
) -> Tensor:
    if scores is None:
        return torch.zeros(shape, dtype=torch.float32, device=device)
    if tuple(scores.shape) != shape:
        raise ValueError(f"expected score shape {shape}, got {tuple(scores.shape)}")
    return scores.detach().float().to(device)
