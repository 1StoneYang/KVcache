from __future__ import annotations

import torch
from transformers import DynamicCache, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from vlesshallu.headwise_cache import HeadwiseCacheState
from vlesshallu.transformers_runtime import _AttentionSession


def _tiny_model() -> Qwen3VLForConditionalGeneration:
    text = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
    )
    vision = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=32,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    config = Qwen3VLConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    return Qwen3VLForConditionalGeneration(config).eval()


def test_attention_session_collects_prefill_and_decode_statistics() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    method_config = {
        "seed": 17,
        "mmshift": {
            "num_proxies": 4,
            "num_groups": 2,
            "variance_gamma": 10.0,
            "mass_threshold": 0.95,
            "anchor": 1.0,
        },
        "myopia": {"text_alpha": 0.5},
        "kvsmooth": {
            "first_layer": 0,
            "last_layer": 1,
            "fifo_size": 3,
            "lambda_ref": 0.7,
            "clip_radius": 0.2,
        },
    }
    runtime = type(
        "RuntimeStub",
        (),
        {"model": model, "config": method_config, "variant": "core4_no_rb"},
    )()
    input_ids = torch.tensor([[1, 122, 120, 120, 123, 7]])
    image_mask = input_ids[0].eq(120)
    text_mask = torch.tensor([False, False, False, False, False, True])
    position_ids = torch.arange(6).view(1, 1, 6).expand(3, 1, 6)
    cache_position = torch.arange(6)
    cache = DynamicCache(config=model.config)
    session = _AttentionSession(
        runtime=runtime,
        cache=cache,
        image_mask=image_mask,
        text_mask=text_mask,
        prefill_position_ids=position_ids,
        sample_id=3,
    )

    session.install()
    try:
        session.begin_prefill()
        model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        prefill = session.finish_prefill()
        assert prefill.shift.shape == (2, 2, 6)
        assert prefill.text.shape == (2, 2, 6)
        assert prefill.layer_visual_mass.shape == (2,)

        tracker = HeadwiseCacheState.from_prefill(
            cache,
            image_mask,
            shift_scores=prefill.shift,
            text_scores=prefill.text,
        )
        session.begin_decode(
            current_image_mask=tracker.image_mask,
            collect_dynamic=True,
            apply_kvsmooth=True,
        )
        decode_position_ids = torch.full((3, 1, 1), 6)
        model(
            input_ids=torch.tensor([[8]]),
            position_ids=decode_position_ids,
            cache_position=torch.tensor([6]),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        decode = session.finish_decode()
        tracker.append_nonvisual(6)
        assert decode.scores.shape == (2, 2, 7)
        assert decode.layer_visual_mass.shape == (2,)
        assert len(decode.kvsmooth) == 2
        pruned = tracker.prune(decode.scores, keep_count=1)
        assert pruned.after_visual_count == 1
        assert cache.get_seq_length() == 6
    finally:
        session.remove()
