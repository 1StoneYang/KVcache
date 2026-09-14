"""Decode loop that runs the five ReKV modules after MM-ShiftKV prefill."""

from __future__ import annotations

from typing import Any

import torch

from .controller import ReKVController


def run_rekv_decode(
    runtime: Any,
    first_logits: torch.Tensor,
    cache: Any,
    session: Any,
    image_mask: torch.Tensor,
    prefill: Any,
    prompt_length: int,
    rope_delta: torch.Tensor,
) -> dict[str, Any]:
    if prefill.shift is None:
        raise RuntimeError("ReKV requires MM-ShiftKV prefill scores P_i")
    controller = ReKVController(runtime.config)
    store = controller.initialize(cache, image_mask, prefill.shift)

    eos_ids = runtime.model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = runtime.processor.tokenizer.eos_token_id
    eos_set = {int(eos_ids)} if isinstance(eos_ids, int) else set(map(int, eos_ids))
    max_tokens = int(runtime.config["decoding"]["max_new_tokens"])
    tokens: list[int] = []
    next_logits = first_logits

    while len(tokens) < max_tokens:
        token = int(torch.argmax(next_logits.float()).item())
        tokens.append(token)
        if token in eos_set or len(tokens) == max_tokens:
            break

        session.begin_decode(
            current_image_mask=store.tracker.image_mask,
            collect_dynamic=True,
            apply_kvsmooth=False,
            collect_queries=True,
        )
        absolute_position = prompt_length + len(tokens) - 1
        cache_position = torch.tensor([absolute_position], device=runtime.device)
        text_position = cache_position[0] + rope_delta[0, 0]
        decode_position_ids = text_position.view(1, 1, 1).expand(3, 1, 1)
        outputs = runtime._model_forward(
            input_ids=torch.tensor([[token]], device=runtime.device),
            position_ids=decode_position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
            logits_to_keep=1,
        )
        decode = session.finish_decode()
        controller.on_generated_token(absolute_position)
        if decode.scores is None or decode.last_queries is None or decode.layer_attention is None:
            raise RuntimeError("ReKV decode hook did not collect attention and queries")
        controller.maybe_adjust(
            step=len(tokens),
            decode_scores=decode.scores,
            layer_attention=decode.layer_attention,
            last_queries=decode.last_queries,
            image_mask=store.tracker.image_mask,
        )
        next_logits = outputs.logits[0, -1]

    return {
        "tokens": tokens,
        "pruning_events": controller.events,
        "final_visual_tokens": store.active_visual,
        "kvsmooth": {
            "step_count": 0,
            "layer_update_count": 0,
            "coefficient_mean": None,
            "entropy_mean": None,
            "coefficient_min": None,
            "coefficient_max": None,
            "entropy_min": None,
            "entropy_max": None,
        },
        "rekv": {
            "active_visual": store.active_visual,
            "bin_count": store.bin_count,
            "events": controller.events,
        },
    }
