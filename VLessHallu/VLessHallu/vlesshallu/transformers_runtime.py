from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image

from .headwise_cache import HeadwiseCacheState, cache_layers
from .methods import (
    DynamicPruneState,
    KVSmoothState,
    decode_visual_scores,
    fuse_scores,
    future_proxy_positions,
    mmshift_scores,
    myopia_scores,
)


Tensor = torch.Tensor


@dataclass(frozen=True)
class GenerationResult:
    caption: str
    input_token_count: int
    visual_token_count: int
    output_token_count: int
    elapsed_seconds: float
    peak_gpu_memory_bytes: int
    pruning_events: list[dict[str, Any]]
    method_trace: dict[str, Any]


class TransformersRuntime:
    """Single-GPU Qwen3-VL runtime with an owned eager attention/cache loop."""

    def __init__(
        self,
        config: Mapping[str, Any],
        resource_lock: Mapping[str, Any],
        variant: str,
    ) -> None:
        from transformers import AutoProcessor

        self.config = config
        self.variant = variant
        self.device = torch.device(str(config["model"]["device"]))
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("the benchmark requires a CUDA GPU")

        model_path = str(Path(resource_lock["model"]["path"]))
        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(config["model"]["trust_remote_code"]),
        }
        max_pixels = config["model"].get("max_pixels", 1605632)
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = int(max_pixels)
        self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
        if max_pixels is not None and hasattr(self.processor, "image_processor"):
            # Transformers 4.50's Qwen2.5 processor accepts max_pixels at
            # construction but does not propagate it into preprocessing. The
            # image processor reads size['longest_edge'] instead.
            image_size = getattr(self.processor.image_processor, "size", None)
            if isinstance(image_size, dict):
                image_size["longest_edge"] = int(max_pixels)
        self.model = _load_vl_model(
            model_path,
            trust_remote_code=bool(config["model"]["trust_remote_code"]),
            attn_implementation=str(config["model"]["attn_implementation"]),
        ).to(self.device)
        self.model.eval()
        self._decoder = _text_decoder(self.model)
        self._text_config = getattr(self.model.config, "text_config", self.model.config)
        self._rope_owner = self.model if hasattr(self.model, "get_rope_index") else self.model.model
        self._forward_accepts_logits_to_keep = (
            "logits_to_keep" in inspect.signature(self.model.forward).parameters
        )
        self._apply_quantization()

    def _apply_quantization(self) -> None:
        quantization = str(self.config["model"]["quantization"])
        if quantization == "bf16":
            return
        if quantization != "torchao_fp8_per_tensor":
            raise ValueError(f"unsupported local quantization: {quantization}")
        try:
            from torchao.quantization import PerTensor, quantize_
        except ImportError as error:
            raise RuntimeError(
                "TorchAO is required for FP8 local inference; run doctor"
            ) from error
        try:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

            recipe = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
        except ImportError:
            from torchao.quantization import float8_dynamic_activation_float8_weight

            recipe = float8_dynamic_activation_float8_weight(granularity=PerTensor())
        quantize_(self.model, recipe, device=self.device)

    def generate(self, sample: Mapping[str, Any]) -> GenerationResult:
        with torch.inference_mode():
            return self._generate(sample)

    def _generate(self, sample: Mapping[str, Any]) -> GenerationResult:
        from transformers import DynamicCache

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()

        with Image.open(sample["image_path"]) as image:
            inputs = self._prepare_inputs(image.convert("RGB"), str(sample["prompt"]))
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        prompt_length = input_ids.shape[1]
        cache_position = torch.arange(prompt_length, device=self.device)
        position_ids, rope_delta = self._rope_owner.get_rope_index(
            input_ids,
            inputs.get("image_grid_thw"),
            None,
            attention_mask=attention_mask,
        )
        position_ids = position_ids.to(self.device)
        rope_delta = rope_delta.to(self.device)
        self.model.rope_deltas = rope_delta
        if hasattr(self._decoder, "rope_deltas"):
            self._decoder.rope_deltas = rope_delta

        image_mask = input_ids[0].eq(self.model.config.image_token_id)
        if not image_mask.any():
            raise RuntimeError("processor output contains no image tokens")
        text_mask = self._text_query_mask(input_ids[0], attention_mask)
        try:
            cache = DynamicCache(config=self.model.config)
        except TypeError:
            cache = DynamicCache()
        session = _AttentionSession(
            runtime=self,
            cache=cache,
            image_mask=image_mask,
            text_mask=text_mask,
            prefill_position_ids=position_ids,
            sample_id=int(sample["sample_id"]),
        )

        session.install()
        try:
            session.begin_prefill()
            outputs = self._model_forward(
                **inputs,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
                logits_to_keep=1,
            )
            prefill = session.finish_prefill()
            # Keep only the last logits row, not a view retaining all prompt logits.
            first_logits = outputs.logits[0, -1].clone()
            del outputs
            result = self._decode(
                first_logits,
                cache,
                session,
                image_mask,
                prefill,
                prompt_length,
                rope_delta,
            )
        finally:
            session.remove()

        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        caption = self.processor.decode(
            result["tokens"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        return GenerationResult(
            caption=caption,
            input_token_count=int(prompt_length),
            visual_token_count=int(image_mask.sum().item()),
            output_token_count=len(result["tokens"]),
            elapsed_seconds=elapsed,
            peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(self.device)),
            pruning_events=result["pruning_events"],
            method_trace={
                "runtime": "transformers_eager",
                "quantization": self.config["model"]["quantization"],
                "prompt_tokens": prompt_length,
                "generated_tokens": len(result["tokens"]),
                "initial_visual_tokens": int(image_mask.sum()),
                "final_visual_tokens": result["final_visual_tokens"],
                "kvsmooth": result["kvsmooth"],
                **({"rekv": result["rekv"]} if "rekv" in result else {}),
            },
        )

    def _prepare_inputs(self, image: Image.Image, prompt: str) -> dict[str, Tensor]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = self.processor(
            text=[text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )
        allowed = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        return {
            key: value.to(self.device)
            for key, value in batch.items()
            if key in allowed and isinstance(value, Tensor)
        }

    def _text_query_mask(
        self, input_ids: Tensor, attention_mask: Tensor | None
    ) -> Tensor:
        vision_end = input_ids.eq(self.model.config.vision_end_token_id).nonzero(
            as_tuple=False
        )
        start = int(vision_end[-1]) + 1 if vision_end.numel() else 0
        mask = torch.arange(input_ids.numel(), device=input_ids.device) >= start
        if attention_mask is not None:
            mask &= attention_mask[0].bool()
        special_ids = torch.tensor(
            self.processor.tokenizer.all_special_ids,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        if special_ids.numel():
            mask &= ~torch.isin(input_ids, special_ids)
        return mask

    def _model_forward(self, **kwargs: Any):
        if not self._forward_accepts_logits_to_keep:
            kwargs.pop("logits_to_keep", None)
        return self.model(**kwargs)

    def _decode(
        self,
        first_logits: Tensor,
        cache: Any,
        session: "_AttentionSession",
        image_mask: Tensor,
        prefill: "_PrefillScores",
        prompt_length: int,
        rope_delta: Tensor,
    ) -> dict[str, Any]:
        if self.variant == "rekv":
            try:
                from rekv.runtime_loop import run_rekv_decode
            except ImportError:
                import sys

                sys.path.insert(0, "/root/autodl-tmp/Method/ReKV")
                from rekv.runtime_loop import run_rekv_decode

            return run_rekv_decode(
                self,
                first_logits,
                cache,
                session,
                image_mask,
                prefill,
                prompt_length,
                rope_delta,
            )
        dynamic = self.variant in {"prunehal", "core4_no_rb"}
        static = self.variant in {"mmshift", "myopia_score"}
        smooth = self.variant in {"kvsmooth", "core4_no_rb"}
        tracker: HeadwiseCacheState | None = None
        pruning_events: list[dict[str, Any]] = []

        if dynamic or static:
            tracker = HeadwiseCacheState.from_prefill(
                cache,
                image_mask,
                shift_scores=prefill.shift,
                text_scores=prefill.text,
            )
        if static and tracker is not None:
            budget_key = "mmshift" if self.variant == "mmshift" else "myopia"
            keep_count = min(
                tracker.visual_count,
                int(self.config[budget_key]["cache_budget"]),
            )
            if keep_count < tracker.visual_count:
                scores = (
                    tracker.shift_scores
                    if self.variant == "mmshift"
                    else tracker.text_scores
                )
                pruned = tracker.prune(scores, keep_count)
                pruning_events.append(
                    _prune_event(
                        step=0,
                        trigger="prefill_budget",
                        votes=None,
                        layer_visual_mass=None,
                        pruned=pruned,
                    )
                )

        prune_state = None
        if dynamic:
            prune_config = self.config["prunehal"]
            prune_state = DynamicPruneState(
                retention_ratio=float(prune_config["retention_ratio"]),
                max_prunes=int(prune_config["max_prunes"]),
                force_step=int(prune_config["force_step"]),
                vote_fraction=float(prune_config["vote_fraction"]),
            )
            triggered, _ = prune_state.observe(prefill.layer_visual_mass)
            if triggered:
                raise AssertionError("prefill reference unexpectedly triggered pruning")

        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = self.processor.tokenizer.eos_token_id
        eos_set = {int(eos_ids)} if isinstance(eos_ids, int) else set(map(int, eos_ids))
        max_tokens = int(self.config["decoding"]["max_new_tokens"])
        tokens: list[int] = []
        next_logits = first_logits
        kvsmooth_summary: dict[str, Any] = {
            "step_count": 0,
            "layer_update_count": 0,
            "coefficient_sum": 0.0,
            "coefficient_min": None,
            "coefficient_max": None,
            "entropy_sum": 0.0,
            "entropy_min": None,
            "entropy_max": None,
        }

        while len(tokens) < max_tokens:
            token = int(torch.argmax(next_logits.float()).item())
            tokens.append(token)
            if token in eos_set or len(tokens) == max_tokens:
                break

            current_image_mask = tracker.image_mask if tracker is not None else image_mask
            session.begin_decode(
                current_image_mask=current_image_mask,
                collect_dynamic=dynamic,
                apply_kvsmooth=smooth,
            )
            absolute_position = prompt_length + len(tokens) - 1
            cache_position = torch.tensor([absolute_position], device=self.device)
            text_position = cache_position[0] + rope_delta[0, 0]
            decode_position_ids = text_position.view(1, 1, 1).expand(3, 1, 1)
            outputs = self._model_forward(
                input_ids=torch.tensor([[token]], device=self.device),
                position_ids=decode_position_ids,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
                logits_to_keep=1,
            )
            decode = session.finish_decode()
            if decode.kvsmooth:
                _update_kvsmooth_summary(kvsmooth_summary, decode.kvsmooth)
            if tracker is not None and dynamic:
                tracker.append_nonvisual(absolute_position)

            if dynamic:
                assert prune_state is not None and tracker is not None
                triggered, votes = prune_state.observe(decode.layer_visual_mass)
                if triggered:
                    keep_count = prune_state.next_visual_count(tracker.visual_count)
                    if self.variant == "prunehal":
                        scores = decode.scores
                    else:
                        weights = self.config["fusion"]
                        fusion_weights = (
                            float(weights["decode"]),
                            float(weights["shift"]),
                            float(weights["text"]),
                        )
                        scores = torch.stack(
                            [
                                fuse_scores(
                                    decode.scores[layer],
                                    tracker.shift_scores[layer],
                                    tracker.text_scores[layer],
                                    tracker.image_mask,
                                    weights=fusion_weights,
                                )
                                for layer in range(decode.scores.shape[0])
                            ]
                        )
                    pruned = tracker.prune(scores, keep_count)
                    pruning_events.append(
                        _prune_event(
                            step=prune_state.step,
                            trigger="forced" if prune_state.step == 2 else "layer_vote",
                            votes=votes,
                            layer_visual_mass=decode.layer_visual_mass,
                            pruned=pruned,
                        )
                    )

            next_logits = outputs.logits[0, -1]

        final_visual = tracker.visual_count if tracker is not None else int(image_mask.sum())
        return {
            "tokens": tokens,
            "pruning_events": pruning_events,
            "final_visual_tokens": final_visual,
            "kvsmooth": _finish_kvsmooth_summary(kvsmooth_summary),
        }


@dataclass(frozen=True)
class _PrefillScores:
    shift: Tensor | None
    text: Tensor | None
    layer_visual_mass: Tensor | None


@dataclass(frozen=True)
class _DecodeScores:
    scores: Tensor | None
    layer_visual_mass: Tensor | None
    kvsmooth: list[dict[str, float]]
    last_queries: Tensor | None = None
    layer_attention: list[Tensor] | None = None


class _AttentionSession:
    def __init__(
        self,
        *,
        runtime: TransformersRuntime,
        cache: Any,
        image_mask: Tensor,
        text_mask: Tensor,
        prefill_position_ids: Tensor,
        sample_id: int,
    ) -> None:
        self.runtime = runtime
        self.model = runtime.model
        self.config = runtime.config
        self.variant = runtime.variant
        self.cache = cache
        self.image_mask = image_mask
        self.text_mask = text_mask
        self.prefill_position_ids = prefill_position_ids
        self.sample_id = sample_id
        self.phase = "off"
        self.current_image_mask = image_mask
        self.collect_dynamic = False
        self.collect_queries = False
        self.apply_kvsmooth = False
        self.last_queries: list[Tensor | None] = []
        self.layer_attention: list[Tensor | None] = []
        self.handles = []
        self.shift: list[Tensor | None] = []
        self.text: list[Tensor | None] = []
        self.prefill_mass: list[Tensor | None] = []
        self.decode_scores: list[Tensor | None] = []
        self.decode_mass: list[Tensor | None] = []
        self.smoothing: list[dict[str, float]] = []
        self._decoder = (
            getattr(runtime, "_decoder", None)
            or getattr(runtime.model, "language_model", None)
            or runtime.model.model
        )
        self._text_config = (
            getattr(runtime, "_text_config", None)
            or getattr(runtime.model.config, "text_config", None)
            or runtime.model.config
        )
        smooth_config = self.config["kvsmooth"]
        self.kvsmooth = KVSmoothState(
            first_layer=int(smooth_config["first_layer"]),
            last_layer=int(smooth_config["last_layer"]),
            fifo_size=int(smooth_config["fifo_size"]),
            lambda_ref=float(smooth_config["lambda_ref"]),
            clip_radius=float(smooth_config["clip_radius"]),
        )

    def install(self) -> None:
        for layer_idx, decoder_layer in enumerate(self._decoder.layers):
            handle = decoder_layer.self_attn.register_forward_hook(
                self._hook(layer_idx), with_kwargs=True
            )
            self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def begin_prefill(self) -> None:
        layer_count = len(self._decoder.layers)
        self.phase = "prefill"
        self.shift = [None] * layer_count
        self.text = [None] * layer_count
        self.prefill_mass = [None] * layer_count

    def finish_prefill(self) -> _PrefillScores:
        self.phase = "off"
        shift = _stack_optional(self.shift, self.variant in {"mmshift", "core4_no_rb", "rekv"})
        text = _stack_optional(self.text, self.variant in {"myopia_score", "core4_no_rb"})
        mass = _stack_optional(
            self.prefill_mass, self.variant in {"prunehal", "core4_no_rb"}
        )
        return _PrefillScores(shift=shift, text=text, layer_visual_mass=mass)

    def begin_decode(
        self,
        *,
        current_image_mask: Tensor,
        collect_dynamic: bool,
        apply_kvsmooth: bool,
        collect_queries: bool = False,
    ) -> None:
        layer_count = len(self._decoder.layers)
        self.phase = "decode"
        self.current_image_mask = current_image_mask
        self.collect_dynamic = collect_dynamic
        self.apply_kvsmooth = apply_kvsmooth
        self.collect_queries = collect_queries
        self.decode_scores = [None] * layer_count
        self.decode_mass = [None] * layer_count
        self.last_queries = [None] * layer_count
        self.layer_attention = [None] * layer_count
        self.smoothing = []

    def finish_decode(self) -> _DecodeScores:
        self.phase = "off"
        scores = _stack_optional(self.decode_scores, self.collect_dynamic)
        mass = _stack_optional(self.decode_mass, self.collect_dynamic)
        queries = _stack_optional(self.last_queries, self.collect_queries)
        attentions = None
        if self.collect_queries:
            if any(item is None for item in self.layer_attention):
                raise RuntimeError("attention hook did not collect every decoder layer")
            attentions = [item for item in self.layer_attention if item is not None]
        return _DecodeScores(
            scores=scores,
            layer_visual_mass=mass,
            kvsmooth=self.smoothing,
            last_queries=queries,
            layer_attention=attentions,
        )

    def _hook(self, layer_idx: int):
        def hook(module, args, kwargs, output):
            if self.phase == "off":
                return
            attention = output[1]
            if attention is None:
                raise RuntimeError("eager attention did not return attention probabilities")
            if self.phase == "prefill":
                self._collect_prefill(layer_idx, module, kwargs, attention[0])
                # Statistics are consumed here. Do not accumulate quadratic
                # prefill attention matrices in the model output across layers.
                return (output[0], None, *output[2:])
            elif self.phase == "decode":
                self._collect_decode(layer_idx, attention[0], module, kwargs)

        return hook

    def _collect_prefill(
        self,
        layer_idx: int,
        module: Any,
        kwargs: Mapping[str, Any],
        attention: Tensor,
    ) -> None:
        if self.variant in {"mmshift", "core4_no_rb", "rekv"}:
            self.shift[layer_idx] = self._mmshift(layer_idx, module, kwargs)
        if self.variant in {"myopia_score", "core4_no_rb"}:
            self.text[layer_idx] = myopia_scores(
                attention,
                self.image_mask,
                self.text_mask,
                num_kv_heads=int(self._text_config.num_key_value_heads),
                text_alpha=float(self.config["myopia"]["text_alpha"]),
            )
        if self.variant in {"prunehal", "core4_no_rb"}:
            self.prefill_mass[layer_idx] = (
                attention[:, -1, self.image_mask].float().mean()
            )

    def _mmshift(
        self, layer_idx: int, module: Any, kwargs: Mapping[str, Any]
    ) -> Tensor:
        hidden = kwargs.get("hidden_states")
        if hidden is None:
            raise RuntimeError("attention hook did not receive hidden_states")
        hidden = hidden[0]
        settings = self.config["mmshift"]
        num_proxies = int(settings["num_proxies"])
        num_groups = int(settings["num_groups"])
        generator = torch.Generator(device=hidden.device)
        generator.manual_seed(
            int(self.config["seed"]) + self.sample_id * 1009 + layer_idx
        )
        mean = hidden.float().mean(dim=0)
        std = hidden.float().std(dim=0, unbiased=False)
        noise = torch.randn(
            (num_proxies, hidden.shape[-1]),
            generator=generator,
            device=hidden.device,
            dtype=torch.float32,
        )
        sampled = (mean + float(settings["variance_gamma"]) * std * noise).to(
            hidden.dtype
        )
        query_heads = int(self._text_config.num_attention_heads)
        projected = module.q_proj(sampled).view(
            num_proxies, query_heads, module.head_dim
        )
        projected = _maybe_q_norm(module, projected).transpose(0, 1)

        last_position = int(self.prefill_position_ids.max())
        proxy_positions = future_proxy_positions(
            last_position,
            num_proxies,
            num_groups,
            device=hidden.device,
        )
        proxy_position_ids = proxy_positions.view(1, 1, -1).expand(3, 1, -1)
        cos, sin = self._decoder.rotary_emb(
            sampled.unsqueeze(0), proxy_position_ids
        )
        proxy_queries = _apply_rope(projected, cos[0], sin[0])
        # Qwen2.5-VL returns one RoPE result per M-RoPE axis even when all
        # three proxy position streams are identical. Qwen3-VL returns the
        # head-first tensor directly. Collapse the redundant M-RoPE axis so
        # MM-Shift always receives [query_heads, proxies, head_dim].
        if proxy_queries.ndim == 4:
            proxy_queries = proxy_queries[0]

        real_queries = module.q_proj(hidden[-1:]).view(1, query_heads, module.head_dim)
        real_queries = _maybe_q_norm(module, real_queries).transpose(0, 1)
        real_cos, real_sin = kwargs["position_embeddings"]
        if real_cos.ndim == 4:
            real_cos = real_cos[0]
            real_sin = real_sin[0]
        last_query = _apply_rope(
            real_queries, real_cos[..., -1:, :], real_sin[..., -1:, :]
        ).squeeze(1)
        if last_query.ndim == 3:
            last_query = last_query[0]
        prompt_keys = cache_layers(self.cache)[layer_idx].keys[0]
        return mmshift_scores(
            proxy_queries,
            prompt_keys,
            last_query,
            num_groups=num_groups,
            mass_threshold=float(settings["mass_threshold"]),
            anchor=float(settings["anchor"]),
        )

    def _collect_decode(
        self,
        layer_idx: int,
        attention: Tensor,
        module: Any | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        query_length = attention.shape[-2]
        extended_mask = torch.cat(
            (
                self.current_image_mask,
                torch.zeros(
                    query_length,
                    dtype=torch.bool,
                    device=self.current_image_mask.device,
                ),
            )
        )
        if getattr(self, "collect_queries", False):
            self.layer_attention[layer_idx] = attention[:, -1].detach()
            if module is not None and kwargs is not None:
                self.last_queries[layer_idx] = self._decode_query(module, kwargs)
        if self.collect_dynamic:
            self.decode_scores[layer_idx] = decode_visual_scores(
                attention,
                extended_mask,
                num_kv_heads=int(self._text_config.num_key_value_heads),
            )
            self.decode_mass[layer_idx] = (
                attention[:, -1, extended_mask].float().mean()
            )
        if self.apply_kvsmooth and self.kvsmooth.includes(layer_idx):
            layer = cache_layers(self.cache)[layer_idx]
            if layer.keys.shape[-2] < 2:
                raise AssertionError("KVSmooth requires a previous cache token")
            coefficient, entropy = self.kvsmooth.coefficient(layer_idx, attention)
            layer.keys[:, :, -1, :] = (
                (1 - coefficient) * layer.keys[:, :, -1, :]
                + coefficient * layer.keys[:, :, -2, :]
            )
            layer.values[:, :, -1, :] = (
                (1 - coefficient) * layer.values[:, :, -1, :]
                + coefficient * layer.values[:, :, -2, :]
            )
            self.smoothing.append(
                {
                    "layer": float(layer_idx),
                    "lambda": float(coefficient),
                    "entropy": float(entropy),
                }
            )

    def _decode_query(self, module: Any, kwargs: Mapping[str, Any]) -> Tensor:
        hidden = kwargs.get("hidden_states")
        if hidden is None:
            raise RuntimeError("decode hook did not receive hidden_states")
        hidden = hidden[0, -1:]
        query_heads = int(self._text_config.num_attention_heads)
        queries = module.q_proj(hidden).view(1, query_heads, module.head_dim)
        queries = _maybe_q_norm(module, queries).transpose(0, 1)
        cos, sin = kwargs["position_embeddings"]
        if cos.ndim == 4:
            cos = cos[0]
            sin = sin[0]
        queries = _apply_rope(queries, cos[..., -1:, :], sin[..., -1:, :])
        return queries.reshape(query_heads, module.head_dim)


def _apply_rope(states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = states.shape[-1] // 2
    rotated = torch.cat((-states[..., half:], states[..., :half]), dim=-1)
    return states * cos.unsqueeze(0) + rotated * sin.unsqueeze(0)


def _stack_optional(values: list[Tensor | None], required: bool) -> Tensor | None:
    if not required:
        return None
    if any(value is None for value in values):
        raise RuntimeError("attention hook did not collect every decoder layer")
    return torch.stack([value for value in values if value is not None])


def _update_kvsmooth_summary(
    summary: dict[str, Any], rows: list[dict[str, float]]
) -> None:
    summary["step_count"] += 1
    summary["layer_update_count"] += len(rows)
    for row in rows:
        coefficient = float(row["lambda"])
        entropy = float(row["entropy"])
        summary["coefficient_sum"] += coefficient
        summary["entropy_sum"] += entropy
        summary["coefficient_min"] = (
            coefficient
            if summary["coefficient_min"] is None
            else min(summary["coefficient_min"], coefficient)
        )
        summary["coefficient_max"] = (
            coefficient
            if summary["coefficient_max"] is None
            else max(summary["coefficient_max"], coefficient)
        )
        summary["entropy_min"] = (
            entropy
            if summary["entropy_min"] is None
            else min(summary["entropy_min"], entropy)
        )
        summary["entropy_max"] = (
            entropy
            if summary["entropy_max"] is None
            else max(summary["entropy_max"], entropy)
        )


def _finish_kvsmooth_summary(summary: dict[str, Any]) -> dict[str, Any]:
    result = dict(summary)
    updates = int(result["layer_update_count"])
    coefficient_sum = float(result.pop("coefficient_sum"))
    entropy_sum = float(result.pop("entropy_sum"))
    result["coefficient_mean"] = coefficient_sum / updates if updates else None
    result["entropy_mean"] = entropy_sum / updates if updates else None
    return result


def _prune_event(
    *,
    step: int,
    trigger: str,
    votes: Tensor | None,
    layer_visual_mass: Tensor | None,
    pruned: Any,
) -> dict[str, Any]:
    return {
        "decode_step": int(step),
        "trigger": trigger,
        "layer_votes": None if votes is None else votes.detach().cpu().tolist(),
        "layer_visual_mass": (
            None
            if layer_visual_mass is None
            else layer_visual_mass.detach().cpu().tolist()
        ),
        "before_visual_count": pruned.before_visual_count,
        "after_visual_count": pruned.after_visual_count,
        "selected_original_visual_positions": (
            pruned.selected_original_visual_positions.detach().cpu().tolist()
        ),
    }


def _maybe_q_norm(module: Any, tensor: Tensor) -> Tensor:
    if hasattr(module, "q_norm") and module.q_norm is not None:
        return module.q_norm(tensor)
    return tensor


def _text_decoder(model: Any) -> Any:
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "language_model"):
        return inner.language_model
    if inner is not None and hasattr(inner, "layers"):
        return inner
    raise RuntimeError("could not locate the text decoder layers")


def _load_vl_model(
    model_path: str,
    *,
    trust_remote_code: bool,
    attn_implementation: str,
) -> Any:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": attn_implementation,
        "trust_remote_code": trust_remote_code,
    }
    architectures = [str(item) for item in (getattr(config, "architectures", None) or [])]
    model_type = str(getattr(config, "model_type", ""))
    if model_type == "qwen2_5_vl" or any("Qwen2_5_VL" in item for item in architectures):
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    try:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    except ImportError:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(model_path, **kwargs)
