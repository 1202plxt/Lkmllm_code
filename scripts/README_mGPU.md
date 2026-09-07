# TimeLens-8B 多卡纯 GT 探测、Top-K HeadLoRA 微调与评测

本文档对应三个脚本：

- `scripts/m_d_startend_gradient_head_attribution.py`：TimeLens-8B 纯 GT attention 对齐分数探测，不使用 video token mass 候选筛选。
- `scripts/m_heads_finetune_layer_lora_attn_align.py`：仅对 GT 排名 Top-K heads 做 masked LoRA 微调。
- `scripts/m_e_head_eval.py`：对微调后的完整模型或基线模型生成预测，评测 IoU / Recall。

三个阶段均保持每张 GPU 加载一份完整模型、分摊不同样本，不跨 GPU 分片。探测恢复稳定的旧输入路径：先按 FPS 抽取 PIL 帧并保留原始帧索引/FPS，再交给 `qwen_vl_utils.process_vision_info`。微调和评测仍把视频路径直接交给 Qwen utils。

## 1. 本次实验配置

以下是本 README 命令显式采用的配置，不等于脚本全部默认值。

| 参数 | 本次设置 |
|---|---|
| GPU 数量 | 8 |
| 探测候选样本 | 全部 GPU 合计 500 |
| 微调样本 | 全部 GPU 合计最多 5000 |
| epochs | 2 |
| learning rate | 1e-5 |
| 微调目标 | GT 对齐 Top-20 heads（不是整层） |
| LoRA targets | q_proj、v_proj、o_proj |
| LoRA rank / alpha / dropout | 8 / 16 / 0.02 |
| gradient accumulation / clip | 8 / 1.0 |
| FPS | 2 |
| 额外帧数上限 | 无（微调和评测传 max_frames=0） |
| min tokens / total tokens | 64 / 14336 |
| attention implementation | 探测 SDPA；微调和评测 flash_attention_2 |

必须设置 `--max-samples-per-folder 0`，才能让 `--max-samples 5000` 生效。5000 不是每卡样本数；加载满 5000 条时每卡每轮为 625 条。有效全局 batch 为 `8 × 1 × 8 = 64`（完整累积窗口）。

当前只优化时间戳数字 token 的 CE。微调从探测 JSON 的 `gt_alignment_top_heads` 读取前 `--top-k` 个 head，只更新这些 head 对应的 q/v/o masked LoRA；不再提供 `--target-layers` 整层覆盖模式。

所有命令均在服务器的 `Lkmllm_code` 目录下运行。续行缩进四个空格，反斜杠必须是行末最后一个字符。

## 2. 八卡纯 GT 对齐 Head 探测

探测模型使用 `../shared_models/TimeLens-8B`，并传入 `--timelens-model` 启用 TimeLens 官方单 user-turn prompt。

只计算时间戳 query 的 GT attention 对齐分数。不做 backward，不使用梯度归因、联合分数、层间归一化，也不先取 video token attention mass 前20%候选池。所有有效 heads 直接按 GT 对齐分数降序排名。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node 8 scripts/m_d_startend_gradient_head_attribution.py \
    --filtered-json ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl \
    --model-path ../shared_models/TimeLens-8B \
    --timelens-model \
    --video-dir ../Lkmllm_data/datasets/Train/timelens-100k \
    --output-dir ../Lkmllm_data/outputs/m_timelens_gt_only_500 \
    --max-samples 500 \
    --max-duration 0 \
    --top-k 30 \
    --fps 2 \
    --max-frames 0 \
    --min-tokens 64 \
    --total-tokens 14336
```

每个 rank 处理候选集合的 `rank::world_size` 切片。Rank 0 按有效样本数加权合并后，只保留最终文件：

```text
../Lkmllm_data/outputs/m_timelens_gt_only_500/video_only_head_attribution.json
```

文件包含 `gt_alignment_top_heads`、`gt_alignment_score_matrix` 和有效样本计数。为兼容旧工具，同一排名也写入 `video_only_top_heads`，但 `_meta.video_mass_candidate_filter=false` 表示没有 video mass 前筛选。

`--max-frames 0` 表示不额外限帧。探测先按 2 FPS 抽帧，再由 `qwen_vl_utils` 处理 pixel budget 和视觉 tensor。

## 3. 八卡微调：5000 样本 × 2 轮

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node 8 scripts/m_heads_finetune_layer_lora_attn_align.py \
    --attr-json ../Lkmllm_data/outputs/m_timelens_gt_only_500/video_only_head_attribution.json \
    --model-path ../shared_models/TimeLens-8B \
    --timelens-model \
    --anno-json ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl \
    --video-dir ../Lkmllm_data/datasets/Train/timelens-100k \
    --output-dir ../Lkmllm_data/checkpoints/timelens_gt_top20_ce_2ep \
    --top-k 20 \
    --adapt-targets q_proj v_proj o_proj \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.02 \
    --max-samples-per-folder 0 \
    --max-samples 5000 \
    --epochs 2 \
    --lr 1e-5 \
    --warmup-ratio 0.1 \
    --gradient-accumulation-steps 8 \
    --grad-clip 1.0 \
    --fps 2 \
    --max-frames 0 \
    --min-tokens 64 \
    --total-tokens 14336 \
    --attn-implementation flash_attention_2
```

## 4. 微调权重保存与恢复

训练和下文微调评测统一使用目录名 `timelens_gt_top20_ce_2ep`：

```text
../Lkmllm_data/checkpoints/timelens_gt_top20_ce_2ep/
├── lora_layer_checkpoint.pt   # 最新训练断点；完成后保留
├── lora_epoch_001.pt          # 第 1 轮轻量 checkpoint
├── lora_epoch_002.pt          # 第 2 轮轻量 checkpoint
├── lora_layer_adapter.pt      # 独立 masked-LoRA 参数与结构
├── training_metadata.json    # 训练参数及分布式信息
├── config.json               # 合并后完整模型配置
├── model-*.safetensors       # 合并后的模型权重
└── processor/tokenizer files
```

本目录对应 TimeLens-8B 上的纯 GT Top-20 HeadLoRA。不要从旧整层 LoRA、Qwen base 或 alignment-loss 实验目录恢复。

正常完成后，评测的 `--model-path` 指向整个保存目录，不是单独的 adapter 或某个 safetensors 分片。

### Warmup、CE-only 与按需合并

代码原本已有约 10% 的线性 warmup，现在开放 `--warmup-ratio 0.1`，也可用 `--warmup-steps N` 指定优化器更新步数（优先于比例，0 为禁用）。Warmup 后线性降低学习率。5000 样本、8 卡、2 轮、梯度累积 8 的计划为 158 次更新、16 次 warmup 更新；实际跳过样本会影响更新数。修改 warmup 后若想从头比较，应使用新的输出目录，不要自动恢复旧断点。进度条 loss 是 epoch 内累计均值，`lr` 显示当前学习率。

alignment loss 和相关 hook 已从多卡脚本删除，不再接受 `--align-weight`、`--align-top-n`、`--align-temperature`。当前只监督时间戳数字 token 的 CE。CE-only 对照请使用新的输出目录，不要自动恢复旧 alignment 实验断点。

每轮只保存 LoRA 参数、mask、优化器和调度器状态，以及原始模型路径/LoRA 配置，不包含完整基础模型。默认仅在全部训练结束时合并一次完整模型；添加 `--skip-final-merge` 可完全跳过自动合并，只保留轻量文件。

例如按需合并第 2 轮（只加载自己生成、可信的 checkpoint）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/merge_m_head_lora.py \
    --checkpoint ../Lkmllm_data/checkpoints/timelens_gt_top20_ce_2ep/lora_epoch_002.pt \
    --model-path ../shared_models/TimeLens-8B \
    --output-dir ../shared_models/timelens_gt_top20_ce_2ep \
    --device cuda:0
```

必须加载训练时同一份 TimeLens-8B 原始权重，不能加载已合并过 LoRA 的模型，否则会叠加更新。导出目录必须是新目录或空目录。TimeLens 主实验的评测命令保留 `--timelens-model`。

## 5. 八卡评测指令

评测入口用 `python`，脚本内部创建 8 个 GPU worker；不要外套 `torchrun`。

以下数据路径按你提供的运行配置保留：`Charades_sta`、`Activitynet`、`Qvhighlights`。这是服务器目录写法，不要按展示名称擅自修改大小写；本地没有这些测试集，未验证服务器路径是否存在。

所有命令显式评完整测试集（`--max-samples 0`），并使用相同的视频输入参数。微调、base 和 TimeLens 分开保存结果，且每个 `--split` 同时标记数据集和模型，避免覆盖或混淆。若要复现历史 500 样本实验，请将对比各组都改成 `--max-samples 500` 并使用独立输出目录。

### 5.1 TimeLens-8B：纯 GT Top-20 HeadLoRA 微调模型

#### Charades-STA-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
    --model-path ../Lkmllm_data/checkpoints/timelens_gt_top20_ce_2ep \
    --timelens-model \
    --anno-json ../Lkmllm_data/datasets/Test/Charades_sta/charades-timelens.json \
    --video-dir ../Lkmllm_data/datasets/Test/Charades_sta/charades \
    --output-dir ../Lkmllm_data/outputs/eval_results/timelens_gt_top20_ce_2ep \
    --split Charades_timelens_gt_top20_ce_2ep \
    --num-gpus 8 \
    --max-samples 0 \
    --fps 2 \
    --max-frames 0 \
    --min-tokens 64 \
    --total-tokens 14336 \
    --attn-implementation flash_attention_2
```

#### ActivityNet-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
    --model-path ../Lkmllm_data/checkpoints/timelens_gt_top20_ce_2ep \
    --timelens-model \
    --anno-json ../Lkmllm_data/datasets/Test/Activitynet/activitynet-timelens.json \
    --video-dir ../Lkmllm_data/datasets/Test/Activitynet/activitynet \
    --output-dir ../Lkmllm_data/outputs/eval_results/timelens_gt_top20_ce_2ep \
    --split ActivityNet_timelens_gt_top20_ce_2ep \
    --num-gpus 8 \
    --max-samples 0 \
    --fps 2 \
    --max-frames 0 \
    --min-tokens 64 \
    --total-tokens 14336 \
    --attn-implementation flash_attention_2
```

#### QVHighlights-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
    --model-path ../Lkmllm_data/checkpoints/timelens_gt_top20_ce_2ep \
    --timelens-model \
    --anno-json ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights-timelens.json \
    --video-dir ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights \
    --output-dir ../Lkmllm_data/outputs/eval_results/timelens_gt_top20_ce_2ep \
    --split QVHighlights_timelens_gt_top20_ce_2ep \
    --num-gpus 8 \
    --max-samples 0 \
    --fps 2 \
    --max-frames 0 \
    --min-tokens 64 \
    --total-tokens 14336 \
    --attn-implementation flash_attention_2
```

### 5.2 Qwen3-VL-8B-Instruct base 对照

#### Charades-STA-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
    --model-path ../shared_models/Qwen3-VL-8B-Instruct \
    --anno-json ../Lkmllm_data/datasets/Test/Charades_sta/charades-timelens.json \
    --video-dir ../Lkmllm_data/datasets/Test/Charades_sta/charades \
    --output-dir ../Lkmllm_data/outputs/eval_results/Qwen3VL8B_base_mGPU \
    --split Charades_Qwen3VL8B_base_mGPU \
    --num-gpus 8 \
    --max-samples 0 \
    --fps 2 \
    --max-frames 0 \
    --min-tokens 64 \
    --total-tokens 14336 \
    --attn-implementation flash_attention_2
```

#### ActivityNet-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
    --model-path ../shared_models/Qwen3-VL-8B-Instruct \
    --anno-json ../Lkmllm_data/datasets/Test/Activitynet/activitynet-timelens.json \
    --video-dir ../Lkmllm_data/datasets/Test/Activitynet/activitynet \
    --output-dir ../Lkmllm_data/outputs/eval_results/Qwen3VL8B_base_mGPU \
    --split ActivityNet_Qwen3VL8B_base_mGPU \
    --num-gpus 8 \
    --max-samples 0 \
    --fps 2 \
    --max-frames 0 \
    --min-tokens 64 \
    --total-tokens 14336 \
    --attn-implementation flash_attention_2
```

#### QVHighlights-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
    --model-path ../shared_models/Qwen3-VL-8B-Instruct \
    --anno-json ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights-timelens.json \
    --video-dir ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights \
    --output-dir ../Lkmllm_data/outputs/eval_results/Qwen3VL8B_base_mGPU \
    --split QVHighlights_Qwen3VL8B_base_mGPU \
    --num-gpus 8 \
    --max-samples 0 \
    --fps 2 \
    --max-frames 0 \
    --min-tokens 64 \
    --total-tokens 14336 \
    --attn-implementation flash_attention_2
```

### 5.3 TimeLens-8B 对照

原始 TimeLens-8B 和本次由它微调得到的模型都使用 `--timelens-model`；只有 Qwen3 base 对照不加该标记。

#### Charades-STA-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
    --model-path ../shared_models/TimeLens-8B \
    --anno-json ../Lkmllm_data/datasets/Test/Charades_sta/charades-timelens.json \
    --video-dir ../Lkmllm_data/datasets/Test/Charades_sta/charades \
    --output-dir ../Lkmllm_data/outputs/eval_results/TimeLens8B_mGPU \
    --split Charades_TimeLens8B_mGPU \
    --num-gpus 8 \
    --max-samples 0 \
    --fps 2 \
    --max-frames 0 \
    --min-tokens 64 \
    --total-tokens 14336 \
    --attn-implementation flash_attention_2 \
    --timelens-model
```

#### ActivityNet-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
    --model-path ../shared_models/TimeLens-8B \
    --anno-json ../Lkmllm_data/datasets/Test/Activitynet/activitynet-timelens.json \
    --video-dir ../Lkmllm_data/datasets/Test/Activitynet/activitynet \
    --output-dir ../Lkmllm_data/outputs/eval_results/TimeLens8B_mGPU \
    --split ActivityNet_TimeLens8B_mGPU \
    --num-gpus 8 \
    --max-samples 0 \
    --fps 2 \
    --max-frames 0 \
    --min-tokens 64 \
    --total-tokens 14336 \
    --attn-implementation flash_attention_2 \
    --timelens-model
```

#### QVHighlights-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
    --model-path ../shared_models/TimeLens-8B \
    --anno-json ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights-timelens.json \
    --video-dir ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights \
    --output-dir ../Lkmllm_data/outputs/eval_results/TimeLens8B_mGPU \
    --split QVHighlights_TimeLens8B_mGPU \
    --num-gpus 8 \
    --max-samples 0 \
    --fps 2 \
    --max-frames 0 \
    --min-tokens 64 \
    --total-tokens 14336 \
    --attn-implementation flash_attention_2 \
    --timelens-model
```

## 6. 历史实验结果（保留附件记录）

以下保留你提供的历史结果数值，不代表本次 GT-only、5000 样本 × 2 轮实验的结果。原表混合了参考值和本地 topk 实验记录，未在本次修改中重新核验来源。

原文引用：[TimeLens 官方项目](https://github.com/pkuhxy/Timelens) 和 [TimeLens-8B 模型说明](https://huggingface.co/TencentARC/TimeLens-8B) 。R 指标均为 R@1；第一、第三张表使用百分数，500 样本对照表使用 0–1 小数。

### Qwen3-VL-8B-Instruct base /topk

| 测试集 | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---:|---:|---:|---:|
| ActivityNet-base | 62.1 | 51.2 | 34.4 | 46.8 |
| Charades-base | 69.2 | 53.4 | 27.5 | 48.4 |
| Charades-TOPKLora | 73.3 | 57.95 | 30.1 | 50.8 |
| QVHighlights-base | 69.4 | 60.35 | 46.9 | 57.6 |
| QVHighlights-TOPKLora | 74.5 | 65.54 | 50.7 | 61.39 |

### 500样本base模型和topk对比结果

| 测试集 | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---:|---:|---:|---:|
|Qvhighlights_base|	0.5640|	0.4680|	0.3720|	0.4437|
|Qvhighlights_topk|	0.6500|	0.5540|	0.4380|	0.5366|
|Charades_base|	0.6340|	0.4780|	0.2560	|0.4438|
|Charades_topk|	0.6960|	0.5600|	0.2960|	0.4956|
|Activitynet_base|	0.458|	0.364|	0.244|	0.3546|
|Activitynet_topk	|0.536|	0.458|	0.336|	0.4294|


### TimeLens-8B

| 测试集 | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---:|---:|---:|---:|
| ActivityNet-TimeLens | 68.9 | 58.4 | 40.6 | 53.2 |
| Charades-TimeLens | 76.6 | 63.0 | 35.2 | 55.2 |
| QVHighlights-TimeLens | 80.2 | 71.6 | 55.5 | 65.5 |

官方结果用于完整测试集参考。只有评测协议、模型版本、输入参数和完整样本集合都一致时，才适合做严格横向比较。

## 7. 命令注意事项

- 续行符 `\` 后不能有空格，也不要写成 `\--model-path`。
- 探测、微调用 `torchrun`；评测用 `python scripts/m_e_head_eval.py` 并传入 `--num-gpus 8`。
- 微调 JSON 必须是 GT-only 输出，不能传旧的 combined/gradient JSON。
- 评测模型目录必须包含正常训练结束后保存的完整模型；中途 checkpoint 不是可直接评测的模型目录。
