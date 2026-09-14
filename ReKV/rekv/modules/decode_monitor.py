"""Module 3: Decode-time visual dependence and entropy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class DecodeObservation:
    visual_mass: float
    visual_entropy: float
    normalized_entropy: float
    layer_mass: Tensor
    layer_entropy: Tensor


def visual_attention_mass(attention: Tensor, visual_mask: Tensor) -> Tensor:
    """Visual attention mass Mt, averaged over query heads."""
    if attention.ndim == 3:
        attention = attention[:, -1]
    weights = attention.float()[..., visual_mask]
    return weights.sum(dim=-1).mean()


def visual_attention_entropy(attention: Tensor, visual_mask: Tensor) -> Tensor:
    """Entropy of the visual-only attention distribution."""
    if attention.ndim == 3:
        attention = attention[:, -1]
    weights = attention.float()[..., visual_mask].clamp_min(1e-12)
    probabilities = weights / weights.sum(dim=-1, keepdim=True)
    return (-(probabilities * probabilities.log()).sum(dim=-1)).mean()


def observe_decode(
    layer_attention: list[Tensor] | Tensor,
    visual_mask: Tensor,
    *,
    probe_layers: list[int] | None = None,
) -> DecodeObservation:
    if isinstance(layer_attention, Tensor):
        attentions = [layer_attention[index] for index in range(layer_attention.shape[0])]
    else:
        attentions = list(layer_attention)
    if not attentions:
        raise ValueError("decode monitor received no layer attention")
    if probe_layers:
        selected = [attentions[index] for index in probe_layers if 0 <= index < len(attentions)]
        if not selected:
            raise ValueError("probe_layers did not match any decoder layer")
    else:
        selected = attentions
    masses = torch.stack([visual_attention_mass(item, visual_mask) for item in selected])
    entropies = torch.stack(
        [visual_attention_entropy(item, visual_mask) for item in selected]
    )
    visual_count = max(int(visual_mask.sum()), 2)
    mean_entropy = float(entropies.mean())
    return DecodeObservation(
        visual_mass=float(masses.mean()),
        visual_entropy=mean_entropy,
        normalized_entropy=mean_entropy / math.log(visual_count),
        layer_mass=masses,
        layer_entropy=entropies,
    )
