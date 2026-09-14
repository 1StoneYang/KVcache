"""ShiftKVCluster.update_kv with MM-ShiftKV selection plus Recycling Bin."""

from __future__ import annotations

import os

import torch

from mmshift.utils.mmshift_util import repeat_kv
from mmshift_re.context import get_visual_mask
from mmshift_re.recycle import recycle_deleted_visual


def bin_size() -> int:
    return max(int(os.getenv("BIN_SIZE", "20")), 0)


def update_kv_with_recycle(
    self,
    origin_key_states,
    query_states,
    origin_value_states,
    query_samples,
    groups_size,
    num_groups,
):
    """Keep original MM-ShiftKV Top-K, then recycle dropped visual KV."""
    key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
    device = key_states.device
    bsz, num_heads, _q_len, head_dim = query_states.shape

    origin_heads_key_states = torch.split(origin_key_states, 1, dim=1)
    origin_heads_value_states = torch.split(origin_value_states, 1, dim=1)

    heads_key_states = []
    heads_value_states = []
    assert bsz == 1
    k_lens = []
    klen_sum = 0
    max_seqlen_k = 0
    self.cu_klen = 0

    def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
        self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=device)
        self.klen_sum = klen_sum
        self.max_seqlen_k = max_seqlen_k
        self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
        self.cu_klen = self.cu_headlens - self.head_lens
        self.cu_klen = torch.cat(
            [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=device)],
            dim=0,
        )
        self.layer_qlens = torch.ones(
            num_heads // self.num_key_value_groups, dtype=torch.int32, device=device
        )
        self.qlen_sum = num_heads // self.num_key_value_groups
        self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
        self.cu_qlen = torch.cat(
            [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=device)],
            dim=0,
        )
        self.cu_offset = torch.arange(
            0,
            num_heads // self.num_key_value_groups + 1,
            dtype=torch.int32,
            device=device,
        )
        self.cu_head_offset = torch.arange(
            1,
            num_heads // self.num_key_value_groups + 1,
            dtype=torch.int32,
            device=device,
        )

    attn_score = self.calcul_attn_sore(key_states, query_samples)
    last_attn_score = self.calcul_attn_sore(key_states, query_states[..., -1:, :])

    score_bz, score_h, score_len, score_s = attn_score.shape
    attn_score = attn_score.view(
        score_bz,
        score_h,
        score_len // groups_size,
        groups_size,
        score_s,
    ).sum(dim=-2)
    attn_score = attn_score + last_attn_score
    attn_score = attn_score / (1.0 + groups_size)
    sorted_scores, sorted_indices = attn_score.sort(dim=-1, descending=True)
    cum = sorted_scores.to(torch.float32).cumsum(dim=-1)
    key_len = attn_score.shape[-1]
    thr = 0.95 * cum[..., -1]
    k = torch.searchsorted(cum, thr.unsqueeze(-1), right=True).squeeze(-1)
    k = k.clamp(min=0, max=key_len)

    batch = bsz
    group_count = num_groups
    kv_heads = attn_score.shape[1]
    token_count = key_len
    rows = batch * kv_heads * group_count
    selected_num = torch.zeros(
        (bsz, kv_heads, key_len), device=query_states.device, dtype=torch.float32
    )
    idx_f = sorted_indices.reshape(rows, token_count).contiguous().long()
    k_f = k.reshape(rows).contiguous()
    if (k_f > 0).any():
        range_l = torch.arange(token_count, device=idx_f.device).unsqueeze(0)
        mask_first_k = range_l < k_f.unsqueeze(1)
        sel = idx_f[mask_first_k]
        row_offsets = torch.arange(rows, device=idx_f.device) * token_count
        off = torch.repeat_interleave(row_offsets, k_f)
        flat_keys = sel + off
        counts = torch.bincount(flat_keys, minlength=rows * token_count).view(rows, token_count)
        counts = counts.view(batch, kv_heads, group_count, token_count).sum(dim=2).to(
            selected_num.dtype
        )
        selected_num += counts

    selected_num = selected_num + last_attn_score.squeeze(-2)
    _, sorted_indices = selected_num.sort(dim=-1, descending=True)
    final_sorted_indices = torch.split(sorted_indices, 1, dim=1)

    seq_len = origin_key_states.shape[-2]
    visual_mask = get_visual_mask(seq_len, device)
    recycle_b = bin_size()

    for head_idx in range(num_heads // self.num_key_value_groups):
        head_key = origin_heads_key_states[head_idx]
        head_value = origin_heads_value_states[head_idx]
        recent_key = head_key[:, :, -1:, :]
        recent_value = head_value[:, :, -1:, :]

        original_budget = self.head_adaptive_capacity[self.layer_idx][head_idx].item()
        additional_budget = self.window_size - 1
        prompt_len = head_key.shape[-2] - 1
        total_budget = min(max(original_budget + additional_budget, 0), prompt_len)

        ranked = final_sorted_indices[head_idx].reshape(-1)
        ranked = ranked[ranked < seq_len - 1]
        if ranked.numel() < total_budget:
            total_budget = int(ranked.numel())
        selected_token_indices = ranked[:total_budget].sort()[0]
        gather_index = selected_token_indices.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
        if total_budget == 0:
            selected_keys = head_key[:, :, :0, :]
            selected_values = head_value[:, :, :0, :]
        else:
            selected_keys = head_key[:, :, :-1, :].gather(dim=2, index=gather_index)
            selected_values = head_value[:, :, :-1, :].gather(dim=2, index=gather_index)

        final_keys = torch.cat([selected_keys, recent_key], dim=2)
        final_values = torch.cat([selected_values, recent_value], dim=2)
        kept_positions = torch.cat(
            [
                selected_token_indices.reshape(-1),
                torch.tensor([seq_len - 1], device=device, dtype=torch.long),
            ]
        )

        if recycle_b > 0 and visual_mask.any():
            kept_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            kept_mask[kept_positions] = True
            bin_keys, bin_values, bin_positions = recycle_deleted_visual(
                keys=head_key[0, 0],
                values=head_value[0, 0],
                scores=selected_num[0, head_idx],
                kept_mask=kept_mask,
                visual_mask=visual_mask,
                bin_size=recycle_b,
            )
            if bin_keys is not None:
                final_keys = torch.cat([final_keys, bin_keys.view(1, 1, -1, head_dim)], dim=2)
                final_values = torch.cat(
                    [final_values, bin_values.view(1, 1, -1, head_dim)], dim=2
                )
                kept_positions = torch.cat([kept_positions, bin_positions])
                order = kept_positions.argsort(dim=-1, stable=True)
                gather = order.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
                final_keys = final_keys.gather(dim=2, index=gather)
                final_values = final_values.gather(dim=2, index=gather)

        length = final_keys.shape[-2]
        k_lens.append(length)
        max_seqlen_k = max(max_seqlen_k, length)
        klen_sum += length
        heads_key_states.append(final_keys.view(-1, head_dim))
        heads_value_states.append(final_values.view(-1, head_dim))

    init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)
    heads_key_states = torch.cat(heads_key_states, dim=0)
    heads_value_states = torch.cat(heads_value_states, dim=0)
    return heads_key_states, heads_value_states
