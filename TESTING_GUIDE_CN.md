# VLessHallu / MM-ShiftKV 测试指南

两种方法使用同一个本地模型：`/root/autodl-tmp/model/Qwen2.5-VL-7B`。

## 最简单的切换方式

只改统一命令的第一个参数：

```bash
# VLessHallu 主方法
bash /root/autodl-tmp/Method/run_method.sh vlesshallu core4_no_rb generation

# VLessHallu 基线
bash /root/autodl-tmp/Method/run_method.sh vlesshallu baseline generation

# ReKV（先测 1 个 AMBER 样本）
bash /root/autodl-tmp/Method/run_method.sh rekv 1

# MM-ShiftKV（先测 1 个 AMBER 样本）
bash /root/autodl-tmp/Method/run_method.sh mmshiftkv shiftkv 1 64 amber_generative
```

## VLessHallu 可切换项

第二个参数是 `variant`：`baseline`、`kvsmooth`、`mmshift`、`myopia_score`、`prunehal`、`core4_no_rb`。

第三个参数是数据划分。本地 AMBER 使用 `generation`；CHAIR 使用 `smoke` 或 `dev`。第五个参数可显式传 `amber` 或 `chair`，默认是 `amber`。正式 CHAIR test 由项目的 `final` 门控命令运行。

默认配置文件是：
`/root/autodl-tmp/Method/VLessHallu/VLessHallu/configs/local_qwen25.toml`

要调 VLessHallu 参数，复制该配置后修改对应区块：

- `[mmshift]`：代理数量、分组数、cache budget。
- `[myopia]`：文本分数权重和 cache budget。
- `[prunehal]`：保留率、剪枝次数和触发步。
- `[fusion]`：decode / shift / text 三种分数权重。
- `[kvsmooth]`：层范围、FIFO 大小、lambda 和裁剪半径。
- `[decoding]`：生成 token 数和提示词。

把自定义配置作为第四个参数传入：

```bash
bash /root/autodl-tmp/Method/run_method.sh vlesshallu core4_no_rb smoke configs/my_test.toml
```

## ReKV 可切换项

改 `/root/autodl-tmp/Method/ReKV/configs/local_qwen25.toml` 的 `[rekv]`：

- `active_budget` / `bin_budget`：Prefill 后 Active 与 Recycling Bin 的视觉 KV 数量。
- `window`：每隔多少个生成 token 监测一次。
- `probe_layers`：用来算视觉依赖 \(M_t\) 和 entropy 的层。
- `future_weight` / `current_weight` / `ema_weight`：\(S_{t,i}\) 融合权重。
- `mass_low` / `mass_high` / `entropy_low` / `entropy_high`：加减预算的阈值。
- `budget_step`：每次最多移动多少个视觉 KV。

完整 AMBER generative：

```bash
bash /root/autodl-tmp/Method/run_method.sh rekv
```

## MM-ShiftKV 可切换项

统一命令参数依次为：方法、样本数、KV budget、AMBER task。

```bash
bash /root/autodl-tmp/Method/run_method.sh mmshiftkv shiftkv 10 128 amber_generative
```

底层参数来自环境变量 `METHOD`、`LIMIT`、`BUDGET`、`TASK`。原始入口位于：
`/root/autodl-tmp/Method/MM-ShiftKV-main/scripts/eval/amber_qwen.sh`

## 环境自检

```bash
cd /root/autodl-tmp/Method/VLessHallu/VLessHallu
/root/miniconda3/envs/vlesshallu/bin/python run_benchmark.py --config configs/local_qwen25.toml doctor

cd /root/autodl-tmp/Method/MM-ShiftKV-main
source scripts/env.sh
python -c "import torch, flash_attn, mmshift, lmms_eval; print('MM-ShiftKV OK', torch.cuda.is_available())"
```
