"""Module 2: Active Cache + Recycling Bin. No Myopia-style KV merge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vlesshallu.headwise_cache import HeadwiseCacheState, cache_layers
from vlesshallu.methods import select_visual_positions

from .prefill_predict import split_visual_ranks


Tensor = torch.Tensor


@dataclass
class RecyclingBin:
    keys: Tensor
    values: Tensor
    original_positions: Tensor
    shift_scores: Tensor
    ema_scores: Tensor

    @property
    def count(self) -> int:
        return 0 if self.keys.numel() == 0 else int(self.keys.shape[-2])

    def empty(self) -> bool:
        return self.count == 0


def shared_visual_ranks(shift_scores: Tensor, image_mask: Tensor) -> Tensor:
    """Rank visual tokens once, then broadcast to every layer/head."""
    visual = image_mask.nonzero(as_tuple=False).squeeze(-1)
    if visual.numel() == 0:
        raise ValueError("prefill image mask has no visual tokens")
    scores = shift_scores.index_select(-1, visual).float().mean(dim=(0, 1))
    ranked = visual[scores.argsort(dim=-1, descending=True, stable=True)]
    layers, heads, _ = shift_scores.shape
    return ranked.view(1, 1, -1).expand(layers, heads, -1).contiguous()


def _empty_bin(layers: int, heads: int, dim: int, device, dtype) -> RecyclingBin:
    tokens = torch.empty(layers, heads, 0, dim, device=device, dtype=dtype)
    meta = torch.empty(layers, heads, 0, device=device, dtype=torch.float32)
    pos = torch.empty(layers, heads, 0, device=device, dtype=torch.long)
    return RecyclingBin(tokens, tokens.clone(), pos, meta, meta.clone())


def _gather_cache(cache: Any, positions: Tensor) -> tuple[Tensor, Tensor]:
    keys = []
    values = []
    for layer_idx, layer in enumerate(cache_layers(cache)):
        gather = positions[layer_idx].unsqueeze(0).unsqueeze(-1).expand(
            1, positions.shape[1], positions.shape[2], layer.keys.shape[-1]
        )
        keys.append(layer.keys.gather(dim=2, index=gather)[0])
        values.append(layer.values.gather(dim=2, index=gather)[0])
    return torch.stack(keys), torch.stack(values)


def _prune_indices(scores: Tensor, image_mask: Tensor, keep_count: int) -> Tensor:
    selected = torch.stack(
        [select_visual_positions(scores[layer], image_mask, keep_count) for layer in range(scores.shape[0])]
    )
    nonvisual = (~image_mask).nonzero(as_tuple=False).squeeze(-1)
    nonvisual = nonvisual.view(1, 1, -1).expand(scores.shape[0], scores.shape[1], -1)
    return torch.cat((selected, nonvisual), dim=-1).sort(dim=-1).values


class TwoLevelCache:
    """High-score visual KV stay active; mid-score KV are parked intact."""

    def __init__(
        self,
        tracker: HeadwiseCacheState,
        recycling: RecyclingBin,
        ema_scores: Tensor,
    ) -> None:
        self.tracker = tracker
        self.recycling = recycling
        self.ema_scores = ema_scores

    @classmethod
    def from_prefill(
        cls,
        cache: Any,
        image_mask: Tensor,
        shift_scores: Tensor,
        *,
        active_budget: int,
        bin_budget: int,
    ) -> "TwoLevelCache":
        tracker = HeadwiseCacheState.from_prefill(
            cache, image_mask, shift_scores=shift_scores
        )
        ranked = shared_visual_ranks(shift_scores, tracker.image_mask)
        _active, recycling_pos, _dropped = split_visual_ranks(
            ranked, active_budget=active_budget, bin_budget=bin_budget
        )
        layers = cache_layers(cache)
        first = layers[0].keys
        if recycling_pos.shape[-1]:
            bin_keys, bin_values = _gather_cache(cache, recycling_pos)
            recycling = RecyclingBin(
                keys=bin_keys,
                values=bin_values,
                original_positions=recycling_pos,
                shift_scores=shift_scores.gather(dim=-1, index=recycling_pos),
                ema_scores=torch.zeros(
                    recycling_pos.shape, device=shift_scores.device, dtype=torch.float32
                ),
            )
        else:
            recycling = _empty_bin(
                len(layers), first.shape[1], first.shape[-1], first.device, first.dtype
            )
        keep = min(int(active_budget), tracker.visual_count)
        shared_scores = shift_scores.mean(dim=(0, 1), keepdim=True).expand_as(shift_scores)
        if keep < tracker.visual_count:
            indices = _prune_indices(shared_scores, tracker.image_mask, keep)
            tracker.prune(shared_scores, keep)
            ema = shift_scores.gather(dim=-1, index=indices)
        else:
            ema = tracker.shift_scores.clone()
        return cls(tracker, recycling, ema)

    @property
    def active_visual(self) -> int:
        return self.tracker.visual_count

    @property
    def bin_count(self) -> int:
        return self.recycling.count

    def append_generated(self, original_position: int) -> None:
        self.tracker.append_nonvisual(original_position)
        zeros = torch.zeros(
            (*self.ema_scores.shape[:2], 1),
            dtype=self.ema_scores.dtype,
            device=self.ema_scores.device,
        )
        self.ema_scores = torch.cat((self.ema_scores, zeros), dim=-1)

    def update_ema(self, decode_scores: Tensor, decay: float) -> None:
        from .delete_restore import update_ema

        if decode_scores.shape != self.ema_scores.shape:
            raise ValueError("EMA and decode score shapes differ")
        self.ema_scores = update_ema(self.ema_scores, decode_scores, decay)

    def evict_to_bin(self, scores: Tensor, new_active: int) -> dict[str, int]:
        if new_active >= self.active_visual:
            return {"moved": 0, "active": self.active_visual, "bin": self.bin_count}
        if new_active <= 0:
            raise ValueError("Active Cache must keep at least one visual token")
        shared = scores.mean(dim=(0, 1), keepdim=True).expand_as(scores)
        visual = self.tracker.image_mask.nonzero(as_tuple=False).squeeze(-1)
        keep = select_visual_positions(shared[0], self.tracker.image_mask, new_active)[0]
        evict = visual[~torch.isin(visual, keep)]
        scores = shared
        moved = int(evict.numel())
        if moved:
            positions = evict.view(1, 1, -1).expand(
                scores.shape[0], scores.shape[1], -1
            ).contiguous()
            keys, values = _gather_cache(self.tracker.cache, positions)
            self._append_bin(
                keys,
                values,
                self.tracker.original_positions.gather(dim=-1, index=positions),
                self.tracker.shift_scores.gather(dim=-1, index=positions),
                self.ema_scores.gather(dim=-1, index=positions),
            )
        indices = _prune_indices(scores, self.tracker.image_mask, new_active)
        self.ema_scores = self.ema_scores.gather(dim=-1, index=indices)
        pruned = self.tracker.prune(scores, new_active)
        return {"moved": moved, "active": pruned.after_visual_count, "bin": self.bin_count}

    def restore_from_bin(self, shadow_scores: Tensor, add_count: int) -> dict[str, int]:
        if add_count <= 0 or self.recycling.empty():
            return {"moved": 0, "active": self.active_visual, "bin": self.bin_count}
        take = min(int(add_count), self.bin_count)
        ranking = shadow_scores.float().mean(dim=(0, 1))
        chosen = torch.topk(ranking, take, largest=True, sorted=True).indices
        keys, values, original, shift, ema = self._take_bin_shared(chosen)
        self._insert_active(keys, values, original, shift, ema)
        return {"moved": take, "active": self.active_visual, "bin": self.bin_count}

    def _append_bin(
        self,
        keys: Tensor,
        values: Tensor,
        original: Tensor,
        shift: Tensor,
        ema: Tensor,
    ) -> None:
        if self.recycling.empty():
            self.recycling = RecyclingBin(keys, values, original, shift, ema)
            return
        self.recycling.keys = torch.cat((self.recycling.keys, keys), dim=-2)
        self.recycling.values = torch.cat((self.recycling.values, values), dim=-2)
        self.recycling.original_positions = torch.cat(
            (self.recycling.original_positions, original), dim=-1
        )
        self.recycling.shift_scores = torch.cat((self.recycling.shift_scores, shift), dim=-1)
        self.recycling.ema_scores = torch.cat((self.recycling.ema_scores, ema), dim=-1)

    def _take_bin_shared(
        self, chosen: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        index = chosen.view(1, 1, -1).expand(
            self.recycling.keys.shape[0], self.recycling.keys.shape[1], -1
        )
        gather_kv = index.unsqueeze(-1).expand(*index.shape, self.recycling.keys.shape[-1])
        keys = self.recycling.keys.gather(dim=-2, index=gather_kv)
        values = self.recycling.values.gather(dim=-2, index=gather_kv)
        original = self.recycling.original_positions.gather(dim=-1, index=index)
        shift = self.recycling.shift_scores.gather(dim=-1, index=index)
        ema = self.recycling.ema_scores.gather(dim=-1, index=index)
        keep = torch.ones(self.bin_count, dtype=torch.bool, device=chosen.device)
        keep[chosen] = False
        if not keep.any():
            self.recycling = _empty_bin(
                keys.shape[0], keys.shape[1], keys.shape[-1], keys.device, keys.dtype
            )
        else:
            self.recycling = RecyclingBin(
                keys=self.recycling.keys[:, :, keep],
                values=self.recycling.values[:, :, keep],
                original_positions=self.recycling.original_positions[:, :, keep],
                shift_scores=self.recycling.shift_scores[:, :, keep],
                ema_scores=self.recycling.ema_scores[:, :, keep],
            )
        return keys, values, original, shift, ema

    def _insert_active(
        self,
        keys: Tensor,
        values: Tensor,
        original: Tensor,
        shift: Tensor,
        ema: Tensor,
    ) -> None:
        layers = cache_layers(self.tracker.cache)
        added = original.shape[-1]
        extra_mask = torch.ones(added, dtype=torch.bool, device=keys.device)
        merged_mask = torch.cat((self.tracker.image_mask, extra_mask))
        new_pos = []
        new_shift = []
        new_ema = []
        ordered_mask = None
        for layer_idx, layer in enumerate(layers):
            merged_k = torch.cat((layer.keys[0], keys[layer_idx]), dim=-2)
            merged_v = torch.cat((layer.values[0], values[layer_idx]), dim=-2)
            merged_pos = torch.cat(
                (self.tracker.original_positions[layer_idx], original[layer_idx]), dim=-1
            )
            order = merged_pos.argsort(dim=-1, stable=True)
            gather_kv = order.unsqueeze(-1).expand(*order.shape, merged_k.shape[-1])
            layer.keys = merged_k.gather(dim=-2, index=gather_kv).unsqueeze(0)
            layer.values = merged_v.gather(dim=-2, index=gather_kv).unsqueeze(0)
            new_pos.append(merged_pos.gather(dim=-1, index=order))
            new_shift.append(
                torch.cat((self.tracker.shift_scores[layer_idx], shift[layer_idx]), dim=-1).gather(
                    dim=-1, index=order
                )
            )
            new_ema.append(
                torch.cat((self.ema_scores[layer_idx], ema[layer_idx]), dim=-1).gather(
                    dim=-1, index=order
                )
            )
            if ordered_mask is None:
                ordered_mask = merged_mask[order[0]]
        self.tracker.original_positions = torch.stack(new_pos)
        self.tracker.shift_scores = torch.stack(new_shift)
        self.ema_scores = torch.stack(new_ema)
        self.tracker.image_mask = ordered_mask
        self.tracker.text_scores = torch.zeros_like(self.tracker.shift_scores)
        self.tracker.assert_invariants()
