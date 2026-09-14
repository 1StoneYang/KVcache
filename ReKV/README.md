# ReKV

Prefill 用 MM-ShiftKV 预测未来重要性，Decode 用 Active Cache + Recycling Bin 做小步删除/恢复。第一版不合并 KV。

模块：

1. `rekv/modules/prefill_predict.py`
2. `rekv/modules/two_level_cache.py`
3. `rekv/modules/decode_monitor.py`
4. `rekv/modules/budget_control.py`
5. `rekv/modules/delete_restore.py`

量化与本地 VLessHallu / 4090 一致：BF16。官方 VLessHallu 默认是 TorchAO FP8，当前环境没有 `torchao`，因此不对齐官方 FP8。
