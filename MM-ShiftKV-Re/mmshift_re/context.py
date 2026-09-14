from __future__ import annotations

import torch

_visual_mask: torch.Tensor | None = None


def set_visual_mask(mask: torch.Tensor | None) -> None:
    global _visual_mask
    if mask is None:
        _visual_mask = None
        return
    _visual_mask = mask.detach().bool().reshape(-1).clone()


def get_visual_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    if _visual_mask is None or _visual_mask.numel() != seq_len:
        return torch.zeros(seq_len, dtype=torch.bool, device=device)
    return _visual_mask.to(device=device)
