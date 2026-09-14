"""Module 4: Small-step Active Cache budget updates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetDecision:
    target_active: int
    action: str
    reason: str


def decide_budget(
    *,
    current_active: int,
    bin_count: int,
    visual_mass: float,
    normalized_entropy: float,
    mass_low: float,
    mass_high: float,
    entropy_low: float,
    entropy_high: float,
    step: int,
    min_active: int,
    max_active: int,
) -> BudgetDecision:
    """Adjust Active KV count from Mt and visual entropy."""
    step = max(1, int(step))
    current = int(current_active)
    ceiling = min(int(max_active), current + bin_count)
    floor = max(1, int(min_active))

    if visual_mass < mass_low and normalized_entropy > entropy_high:
        target = min(ceiling, current + step)
        return BudgetDecision(target, "restore", "low_mass_high_entropy")
    if visual_mass > mass_high:
        target = min(ceiling, current + step)
        return BudgetDecision(
            target,
            "restore" if target > current else "hold",
            "high_visual_mass",
        )
    if visual_mass < mass_low and normalized_entropy < entropy_low:
        target = max(floor, current - step)
        return BudgetDecision(
            target,
            "evict" if target < current else "hold",
            "low_mass_low_entropy",
        )
    return BudgetDecision(current, "hold", "stable")
