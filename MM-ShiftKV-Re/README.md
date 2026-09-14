# MM-ShiftKV-Re

MM-ShiftKV 原流程不变：Prefill 后按未来 Query Proxy 分数，每层每个 KV head 保留预算内 Top-K。

随后只对 **没被 MM-ShiftKV 留下的视觉 KV** 做 Myopia 式 Recycling Bin：

1. \(K_p\)：被删视觉 KV
2. \(K_b=\mathrm{TopB}(K_p)\)，默认 `B=20`
3. \(K_e=K_p-K_b\) 按 key 的 cosine similarity 合并进最近的 Bin token（V 同步）
4. Decode 使用 \(K_{\mathrm{MM-ShiftKV}} \cup K_{\mathrm{Recycle}}\)

原仓库 `MM-ShiftKV-main` 不改。`BIN_SIZE` 可改。

```bash
# 1 张
bash /root/autodl-tmp/Method/run_method.sh mmshiftkv-re 1

# 200 张，B=20
bash /root/autodl-tmp/Method/run_method.sh mmshiftkv-re 200 64 20

# 改 B
BIN_SIZE=32 bash /root/autodl-tmp/Method/run_method.sh mmshiftkv-re 200 64 32
```

打分：

```bash
bash /root/autodl-tmp/Method/MM-ShiftKV-main/scripts/eval/score_amber.sh \
  /root/autodl-tmp/Method/dataset/AMBER/results/MM-ShiftKV-Re/Qwen2.5-VL-7B/amber_generative/amber_generative_shiftkv_re_64_b20.json g
```
