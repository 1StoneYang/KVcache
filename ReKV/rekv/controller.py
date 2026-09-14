"""Glue the five ReKV modules onto one decode step."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .modules.budget_control import BudgetDecision, decide_budget
from .modules.decode_monitor import DecodeObservation, observe_decode
from .modules.delete_restore import fuse_token_scores, shadow_attention
from .modules.two_level_cache import TwoLevelCache


Tensor = torch.Tensor


class ReKVController:
    def __init__(self, config: Mapping[str, Any]) -> None:
        settings = config["rekv"]
        self.window = int(settings.get("window", 8))
        self.probe_layers = [int(item) for item in settings.get("probe_layers", [7, 14, 21])]
        self.ema_decay = float(settings.get("ema_decay", 0.8))
        self.future_weight = float(settings.get("future_weight", 0.4))
        self.current_weight = float(settings.get("current_weight", 0.4))
        self.ema_weight = float(settings.get("ema_weight", 0.2))
        self.mass_low = float(settings.get("mass_low", 0.18))
        self.mass_high = float(settings.get("mass_high", 0.40))
        self.entropy_low = float(settings.get("entropy_low", 0.35))
        self.entropy_high = float(settings.get("entropy_high", 0.80))
        self.budget_step = int(settings.get("budget_step", 8))
        self.min_active = int(settings.get("min_active", 16))
        self.max_active = int(settings.get("max_active", 128))
        self.active_budget = int(settings.get("active_budget", 64))
        self.bin_budget = int(settings.get("bin_budget", 64))
        self.store: TwoLevelCache | None = None
        self.events: list[dict[str, Any]] = []

    def initialize(self, cache: Any, image_mask: Tensor, shift_scores: Tensor) -> TwoLevelCache:
        self.store = TwoLevelCache.from_prefill(
            cache,
            image_mask,
            shift_scores,
            active_budget=self.active_budget,
            bin_budget=self.bin_budget,
        )
        self.events.append(
            {
                "step": 0,
                "trigger": "prefill_split",
                "active": self.store.active_visual,
                "bin": self.store.bin_count,
            }
        )
        return self.store

    def on_generated_token(self, original_position: int) -> None:
        assert self.store is not None
        self.store.append_generated(original_position)

    def maybe_adjust(
        self,
        *,
        step: int,
        decode_scores: Tensor,
        layer_attention: list[Tensor],
        last_queries: Tensor,
        image_mask: Tensor,
    ) -> DecodeObservation | None:
        assert self.store is not None
        self.store.update_ema(decode_scores, self.ema_decay)
        if step % self.window != 0:
            return None
        observation = observe_decode(
            layer_attention,
            image_mask,
            probe_layers=self.probe_layers,
        )
        decision = decide_budget(
            current_active=self.store.active_visual,
            bin_count=self.store.bin_count,
            visual_mass=observation.visual_mass,
            normalized_entropy=observation.normalized_entropy,
            mass_low=self.mass_low,
            mass_high=self.mass_high,
            entropy_low=self.entropy_low,
            entropy_high=self.entropy_high,
            step=self.budget_step,
            min_active=self.min_active,
            max_active=self.max_active,
        )
        applied = self._apply(decision, decode_scores, last_queries, image_mask)
        self.events.append(
            {
                "step": step,
                "trigger": decision.reason,
                "action": decision.action,
                "target_active": decision.target_active,
                "visual_mass": observation.visual_mass,
                "normalized_entropy": observation.normalized_entropy,
                **applied,
            }
        )
        return observation

    def _apply(
        self,
        decision: BudgetDecision,
        decode_scores: Tensor,
        last_queries: Tensor,
        image_mask: Tensor,
    ) -> dict[str, int]:
        assert self.store is not None
        if decision.action == "evict":
            scores = fuse_token_scores(
                self.store.tracker.shift_scores,
                decode_scores,
                self.store.ema_scores,
                image_mask,
                future_weight=self.future_weight,
                current_weight=self.current_weight,
                ema_weight=self.ema_weight,
            )
            return self.store.evict_to_bin(scores, decision.target_active)
        if decision.action == "restore":
            add = decision.target_active - self.store.active_visual
            if add <= 0 or self.store.recycling.empty():
                return {"moved": 0, "active": self.store.active_visual, "bin": self.store.bin_count}
            scores = shadow_attention(last_queries, self.store.recycling.keys)
            return self.store.restore_from_bin(scores, add)
        return {"moved": 0, "active": self.store.active_visual, "bin": self.store.bin_count}
