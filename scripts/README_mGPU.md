# TimeLens 多卡 GT-only 探测、微调与评测

本文档对应三个脚本：

- `scripts/m_d_startend_gradient_head_attribution.py`：TimeLens-8B 纯 GT 对齐分数探测（直接复制 new 方法的独立实现，不 import new 脚本，不再使用梯度归因）。
- `scripts/m_heads_finetune_layer_lora_attn_align.py`：使用 GT-only JSON 选择 alignment head 的多卡微调。
- `scripts/m_e_head_eval.py`：对微调后的完整模型或基线模型生成预测，评测 IoU / Recall。

三个阶段均保持每张 GPU 加载一份完整模型、分摊不同样本，不跨 GPU 分片。微调同步梯度；探测和评测汇总各卡结果。评测不再读取 head JSON，也不以 GT 分数替代定位指标。

## 1. 本次实验配置

以下是本 README 命令显式采用的配置，不等于脚本全部默认值。

| 参数 | 本次设置 |
|---|---|
| GPU 数量 | 8 |
| 探测候选样本 | 全部 GPU 合计 500 |
| 微调样本 | 全部 GPU 合计最多 5000 |
| epochs | 3 |
| learning rate | 5e-5 |
| alignment weight | 0.02 |
| alignment top heads | 10 |
| target layers | 12-19 |
| LoRA targets | q_proj、v_proj、o_proj |
| LoRA rank / alpha / dropout | 8 / 16 / 0.02 |
| gradient accumulation / clip | 8 / 1.0 |
| FPS | 2 |
| 额外帧数上限 | 无（微调和评测传 max_frames=0） |
| min tokens / total tokens | 64 / 14336 |
| attention implementation | 探测 SDPA；微调和评测 flash_attention_2 |

必须设置 `--max-samples-per-folder 0`，才能让 `--max-samples 5000` 生效。5000 不是每卡样本数；加载满 5000 条时每卡每轮为 625 条。有效全局 batch 为 `8 × 1 × 8 = 64`（完整累积窗口）。

总损失为 `CE_loss + 0.02 × alignment_loss`。保留此前的层级 LoRA 配置：12–19 层全部 head 的 q/v/o 投影参与适配，alignment head 从 GT-only JSON 中筛选且限定在这些层内；不是只训练全模型 GT Top-K head。

所有命令均在服务器的 `Lkmllm_code` 目录下运行。续行缩进四个空格，反斜杠必须是行末最后一个字符。

## 2. 八卡纯 GT 对齐 Head 探测

探测模型使用 `../shared_models/TimeLens-8B`。脚本默认启用 TimeLens prompt 和 processor 设置（`padding_side="left"`、`do_resize=False`），不强制覆盖处理器的视觉缩放尺寸。微调起点仍沿用第 3 节的 Qwen3-VL-8B-Instruct，本次只切换探测模型。

只计算时间戳 query 的 GT attention mass，经视频 attention mass 和均匀基线归一化得到 GT 对齐分数。不做 backward，不使用梯度归因、联合筛选或层间归一化联合排名。主模型使用 SDPA，hook 单独计算时间戳 query 对应的 attention 行。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node 8 scripts/m_d_startend_gradient_head_attribution.py \
    --filtered-json ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl \
    --model-path ../shared_models/TimeLens-8B \
    --timelens-model \
    --video-dir ../Lkmllm_data/datasets/Train/timelens-100k \
    --output-dir ../Lkmllm_data/outputs/m_timelens_head_attr_gt_only_500 \
    --max-samples 2000 \
    --max-duration 0 \
    --top-k 30 \
    --fps 2 \
    --min-tokens 64 \
    --total-tokens 14336
```

每个 rank 处理候选集合的 `rank::world_size` 切片。Rank 0 按有效样本数加权合并后，只保留最终文件：

```text
../Lkmllm_data/outputs/m_timelens_head_attr_gt_only_500/video_only_head_attribution.json
```

文件包含 `video_only_top_heads`、`gt_alignment_score_matrix` 和有效样本计数。微调只需这一个 JSON。临时 rank 结果在合并成功后清理。

不再传 `--layers-per-batch` 或 `--max-frames 448`；新探测入口没有这两个参数。2 FPS 采样后不额外截帧，仍保留视觉 token 预算。

## 3. 八卡微调：5000 样本 × 3 轮

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node 8 scripts/m_heads_finetune_layer_lora_attn_align.py \
    --attr-json ../Lkmllm_data/outputs/m_timelens_head_attr_gt_only_500/video_only_head_attribution.json \
    --model-path ../shared_models/Qwen3-VL-8B-Instruct \
    --anno-json ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl \
    --video-dir ../Lkmllm_data/datasets/Train/timelens-100k \
    --output-dir ../Lkmllm_data/checkpoints/layer12_19_qvo_lora_r8_5k_ep3_align002_lr5e5_mGPU \
    --target-layers 16-23 \
    --adapt-targets q_proj v_proj o_proj \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-dropout 0.02 \
    --align-top-n 10 \
    --align-weight 0.02 \
    --align-temperature 1.0 \
    --max-samples-per-folder 0 \
    --max-samples 5000 \
    --epochs 3 \
    --lr 3e-5 \
    --gradient-accumulation-steps 8 \
    --grad-clip 1.0 \
    --fps 2 \
    --max-frames 0 \
    --min-tokens 64 \
    --total-tokens 14336 \
    --attn-implementation flash_attention_2
```

## 4. 微调权重保存与恢复

训练和下文微调评测统一使用同一个目录（与上一轮给出的 5000 样本命令一致）：

```text
../Lkmllm_data/checkpoints/layer12_19_qvo_lora_r8_5k_ep3_align002_lr5e5_mGPU/
├── lora_layer_checkpoint.pt   # 训练断点；正常完成后删除
├── lora_layer_adapter.pt      # 独立 masked-LoRA 参数与结构
├── training_metadata.json    # 训练参数及分布式信息
├── config.json               # 合并后完整模型配置
├── model-*.safetensors       # 合并后的模型权重
└── processor/tokenizer files
```

目录标签 `lr3e5` 对应命令中的 `--lr 3e-5`。这里更新的是命令中的输出路径，不会重命名磁盘上已有权重。不要用旧的 `align001` 目录恢复这次新实验；如需恢复本次实验，使用原参数和本次输出目录重启。

正常完成后，评测的 `--model-path` 指向整个保存目录，不是单独的 adapter 或某个 safetensors 分片。

## 5. 八卡评测指令

评测入口用 `python`，脚本内部创建 8 个 GPU worker；不要外套 `torchrun`。

以下数据路径按你提供的运行配置保留：`Charades_sta`、`Activitynet`、`Qvhighlights`。这是服务器目录写法，不要按展示名称擅自修改大小写；本地没有这些测试集，未验证服务器路径是否存在。

所有命令显式评完整测试集（`--max-samples 0`），并使用相同的视频输入参数。微调、base 和 TimeLens 分开保存结果，且每个 `--split` 同时标记数据集和模型，避免覆盖或混淆。若要复现历史 500 样本实验，请将对比各组都改成 `--max-samples 500` 并使用独立输出目录。

### 5.1 GT-only head 筛选后的微调模型

#### Charades-STA-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
    --model-path ../Lkmllm_data/checkpoints/layer12_19_qvo_lora_r8_5k_ep3_align002_lr5e5_mGPU \
    --anno-json ../Lkmllm_data/datasets/Test/Charades_sta/charades-timelens.json \
    --video-dir ../Lkmllm_data/datasets/Test/Charades_sta/charades \
    --output-dir ../Lkmllm_data/outputs/eval_results/layer12_19_qvo_lora_r8_5k_ep3_align002_lr5e5_mGPU \
    --split Charades_mGPU \
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
    --model-path ../Lkmllm_data/checkpoints/layer12_19_qvo_lora_r8_5k_ep3_align002_lr5e5_mGPU \
    --anno-json ../Lkmllm_data/datasets/Test/Activitynet/activitynet-timelens.json \
    --video-dir ../Lkmllm_data/datasets/Test/Activitynet/activitynet \
    --output-dir ../Lkmllm_data/outputs/eval_results/layer12_19_qvo_lora_r8_5k_ep3_align002_lr5e5_mGPU \
    --split ActivityNet_mGPU \
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
    --model-path ../Lkmllm_data/checkpoints/layer12_19_qvo_lora_r8_5k_ep3_align002_lr5e5_mGPU \
    --anno-json ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights-timelens.json \
    --video-dir ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights \
    --output-dir ../Lkmllm_data/outputs/eval_results/layer12_19_qvo_lora_r8_5k_ep3_align002_lr5e5_mGPU \
    --split QVHighlights_mGPU \
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

仅 TimeLens-8B 使用 `--timelens-model`；Qwen3 base 和本次微调模型不要加该标记。

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

以下保留你提供的历史结果数值，不代表本次 GT-only、5000 样本 × 3 轮实验的结果。原表混合了参考值和本地 topk 实验记录，未在本次修改中重新核验来源。

原文引用：[TimeLens 官方项目](https://github.com/pkuhxy/Timelens) 和 [TimeLens-8B 模型说明](https://huggingface.co/TencentARC/TimeLens-8B) 。R 指标均为 R@1；第一、第三张表使用百分数，500 样本对照表使用 0–1 小数。

### Qwen3-VL-8B-Instruct base /topk

| 测试集 | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---:|---:|---:|---:|
| ActivityNet-TimeLens | 62.1 | 51.2 | 34.4 | 46.8 |
| Charades-TimeLens | 69.2 | 53.4 | 27.5 | 48.4 |
|  Charades-TOPKLora | 74.7 | 59.3 | 31.0 | 51.9 |
| QVHighlights-TimeLens | 69.4 | 60.35 | 46.9 | 57.6 |

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
