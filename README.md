# 方法运行说明

本地模型：`/root/autodl-tmp/model/Qwen2.5-VL-7B`  
AMBER 数据：`/root/autodl-tmp/Method/dataset/AMBER`  
统一入口：`/root/autodl-tmp/Method/run_method.sh`

打分一律用官方 `inference.py`（conda 环境 `mmshiftkv` 或 `vlesshallu` 均可，不要用系统 Python）。

```bash
SCORE=/root/autodl-tmp/Method/MM-ShiftKV-main/scripts/eval/score_amber.sh
RESULTS=/root/autodl-tmp/Method/dataset/AMBER/results
```

---

## 1. 原始模型（Full KV，不剪 cache）

和 MM-ShiftKV 同一条流水线：lmms-eval + FlashAttention + bf16。

```bash
METHOD=fullkv LIMIT=200 PROJECT=FullKV \
  bash /root/autodl-tmp/Method/MM-ShiftKV-main/scripts/eval/amber_qwen.sh
```

全量 1004 张把 `LIMIT=200` 去掉即可。

```bash
bash ${SCORE} \
  ${RESULTS}/FullKV/Qwen2.5-VL-7B/amber_generative/amber_generative_fullkv_64_0.1.json g
```

---

## 2. MM-ShiftKV（原仓库）

走 `MM-ShiftKV-main`，不要走 `vlesshallu mmshift`。

```bash
# 1 张
bash /root/autodl-tmp/Method/run_method.sh mmshiftkv shiftkv 1 64 amber_generative

# 200 张
bash /root/autodl-tmp/Method/run_method.sh mmshiftkv shiftkv 200 64 amber_generative

# 全量 1004
bash /root/autodl-tmp/Method/run_method.sh mmshiftkv
```

每个 KV head 保留约 64 个 token。改预算把第三个数字改掉，例如 `128`。

```bash
bash ${SCORE} \
  ${RESULTS}/MM-ShiftKV/Qwen2.5-VL-7B/amber_generative/amber_generative_shiftkv_64_0.1.json g
```

---

## 3. MM-ShiftKV-Re（原版 MM-ShiftKV + Recycling Bin）

Prefill 仍按 MM-ShiftKV Top-K；被删视觉 KV 压成 B 个代表后放回 cache。默认 `B=20`。

```bash
# 1 张
bash /root/autodl-tmp/Method/run_method.sh mmshiftkv-re 1

# 200 张，B=20
bash /root/autodl-tmp/Method/run_method.sh mmshiftkv-re 200 64 20

# 改 B
bash /root/autodl-tmp/Method/run_method.sh mmshiftkv-re 200 64 32
```

```bash
bash ${SCORE} \
  ${RESULTS}/MM-ShiftKV-Re/Qwen2.5-VL-7B/amber_generative/amber_generative_shiftkv_re_64_b20.json g
```

注意文件名是 `shiftkv_re_64_b20`，不是原版的 `shiftkv_64_0.1`。

---

## 4. VLessHallu

独立 eager 循环，conda `vlesshallu`。AMBER 必须用 `generation`，不要用默认的 `smoke`（那是 CHAIR 8 张）。

```bash
# baseline
AMBER_LIMIT=200 bash /root/autodl-tmp/Method/run_method.sh vlesshallu baseline generation configs/local_qwen25_4090.toml

# 其它 variant：kvsmooth / mmshift / myopia_score / prunehal / core4_no_rb
AMBER_LIMIT=200 bash /root/autodl-tmp/Method/run_method.sh vlesshallu mmshift generation configs/local_qwen25_4090.toml

# 全量 1004：去掉 AMBER_LIMIT
bash /root/autodl-tmp/Method/run_method.sh vlesshallu baseline generation configs/local_qwen25_4090.toml
```

```bash
bash ${SCORE} \
  ${RESULTS}/VLessHallu/Qwen2.5-VL-7B/amber_generative/amber_generative_baseline.json g
```

`vlesshallu mmshift` 是 VLessHallu 里的重实现，和 `MM-ShiftKV-main` 数字不能直接比。

---

## 5. ReKV

挂在 VLessHallu runtime 上，conda `vlesshallu`。

```bash
# 1 张
bash /root/autodl-tmp/Method/run_method.sh rekv 1

# 200 张
bash /root/autodl-tmp/Method/run_method.sh rekv 200

# 全量 1004
bash /root/autodl-tmp/Method/run_method.sh rekv
```

```bash
bash ${SCORE} \
  ${RESULTS}/ReKV/Qwen2.5-VL-7B/amber_generative/amber_generative_rekv.json g
```

参数在 `ReKV/configs/local_qwen25.toml` 的 `[rekv]`。

---

## 对照关系

| 方法 | 命令 | 环境 | 精度 | 结果目录 |
|---|---|---|---|---|
| 原始 Full KV | `METHOD=fullkv ... amber_qwen.sh` | `mmshiftkv` | bf16 + FA2 | `results/FullKV/` |
| MM-ShiftKV | `run_method.sh mmshiftkv` | `mmshiftkv` | bf16 + FA2 | `results/MM-ShiftKV/` |
| MM-ShiftKV-Re | `run_method.sh mmshiftkv-re` | `mmshiftkv` | bf16 + FA2 | `results/MM-ShiftKV-Re/` |
| VLessHallu | `run_method.sh vlesshallu ...` | `vlesshallu` | TorchAO FP8 + eager | `results/VLessHallu/` |
| ReKV | `run_method.sh rekv` | `vlesshallu` | TorchAO FP8 + eager | `results/ReKV/` |

要和 MM-ShiftKV 论文数字比，只用上表前三行。200 张可以出 CHAIR/Cover/Hal/Cog，但是子集分，不能和全量 1004 直接比。
