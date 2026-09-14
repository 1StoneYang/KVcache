from types import SimpleNamespace

import pytest
import torch
from vlesshallu.transformers_runtime import _AttentionSession
from vlesshallu.headwise_cache import HeadwiseCacheState


def test_prefill_hook_consumes_attention_without_returning_matrix():
    collected = []
    session = SimpleNamespace(phase='prefill', _collect_prefill=lambda *args: collected.append(args))
    output = (torch.randn(1, 2, 4), torch.rand(1, 2, 2, 2), object())
    result = _AttentionSession._hook(session, 0)(None, (), {}, output)
    assert len(collected) == 1
    torch.testing.assert_close(collected[0][-1], output[1][0])
    assert result[0] is output[0] and result[2] is output[2]
    assert result[1] is None


def test_qwen25_prefill_statistics_survive_attention_release_and_decode():
    transformers = pytest.importorskip('transformers')
    if not hasattr(transformers, 'Qwen2_5_VLForConditionalGeneration'):
        pytest.skip('Qwen2.5-VL is unavailable')
    cfg = transformers.Qwen2_5_VLConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        rope_scaling={'rope_type': 'default', 'mrope_section': [2, 1, 1]},
        vision_config={'depth': 1, 'hidden_size': 16, 'intermediate_size': 32,
                       'num_heads': 2, 'patch_size': 2, 'spatial_merge_size': 1,
                       'temporal_patch_size': 1, 'out_hidden_size': 32,
                       'fullatt_block_indexes': [0]},
        image_token_id=120, video_token_id=121, vision_start_token_id=122,
        vision_end_token_id=123,
    )
    cfg._attn_implementation = 'eager'
    torch.manual_seed(7)
    model = transformers.Qwen2_5_VLForConditionalGeneration(cfg).eval()
    config = {'seed': 1, 'mmshift': {'num_proxies': 8, 'num_groups': 2,
              'variance_gamma': 10., 'mass_threshold': .95, 'anchor': 1.},
              'myopia': {'text_alpha': .5},
              'kvsmooth': {'first_layer': 0, 'last_layer': 1, 'fifo_size': 15,
                           'lambda_ref': .7, 'clip_radius': .2}}
    runtime = SimpleNamespace(model=model, config=config, variant='core4_no_rb',
                              _decoder=model.model, _text_config=cfg)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    mask = torch.tensor([False, True, True, True, False, False])
    pos = torch.arange(6).view(1, 1, 6).expand(3, 1, 6)
    cache = transformers.DynamicCache()
    expected = model(input_ids=ids, position_ids=pos, use_cache=False).logits
    session = _AttentionSession(runtime=runtime, cache=cache, image_mask=mask,
        text_mask=torch.tensor([False]*4+[True]*2), prefill_position_ids=pos, sample_id=1)
    session.install()
    try:
        session.begin_prefill()
        out = model(input_ids=ids, position_ids=pos, cache_position=torch.arange(6),
                    past_key_values=cache, use_cache=True, output_attentions=True)
        prefill = session.finish_prefill()
        torch.testing.assert_close(out.logits, expected)
        assert all(a is None for a in out.attentions)
        assert prefill.shift.shape == prefill.text.shape == (2, 2, 6)
        assert torch.isfinite(prefill.shift).all() and torch.isfinite(prefill.text).all()
        tracker = HeadwiseCacheState.from_prefill(cache, mask, shift_scores=prefill.shift)
        tracker.prune(prefill.shift, 2)
        session.begin_decode(current_image_mask=tracker.image_mask,
                             collect_dynamic=True, apply_kvsmooth=True)
        model(input_ids=torch.tensor([[7]]), position_ids=torch.full((3,1,1),6),
              cache_position=torch.tensor([6]), past_key_values=cache,
              use_cache=True, output_attentions=True)
        decode = session.finish_decode()
        tracker.append_nonvisual(6)
        assert decode.scores.shape[-1] == tracker.sequence_length
        assert len(decode.kvsmooth) == 2
    finally:
        session.remove()
