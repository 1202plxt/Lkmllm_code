# TimeLens 多卡微调与评测

本文档对应以下两个脚本：

- `scripts/m_heads_finetune_layer_lora_attn_align.py`：多卡数据并行微调。
- `scripts/m_e_head_eval.py`：多卡数据并行评测。

多卡模式下，每张 GPU 都会加载一份完整的 Qwen3-VL-8B 模型，并分摊不同样本；模型不会跨 GPU 分片。训练时各进程同步梯度，评测时各进程独立生成，最后汇总预测和指标。

## 1. 当前默认实验配置

`m_` 微调与评测脚本的主要视频输入参数已保持一致：

| 参数 | 默认值 |
|---|---:|
| FPS | 2.0 |
| max frames | 0（不额外限制帧数） |
| min tokens | 64 |
| total tokens | 14336 |
| attention implementation | flash_attention_2 |

微调脚本的代码默认参数：

| 参数 | 默认值 |
|---|---:|
| target layers | 12-19 |
| LoRA targets | q_proj、v_proj、o_proj |
| LoRA rank / alpha | 8 / 16 |
| LoRA dropout | 0.02 |
| alignment top heads | 10 |
| learning rate | 5e-5 |
| epochs | 3 |
| gradient accumulation | 8 |
| 每个数据子集最多采样 | 800 |
| max samples | 4000（仅关闭分来源采样时生效） |

### 推荐的 8 卡正式实验规模

不建议第一轮实验直接使用全部 TimeLens-100K。全量数据并非不能训练，但当前 LoRA 只有约 137 万个可训练参数，而且还需要先判断 alignment 是否能改善严格定位指标；直接跑 10 万条 × 3 轮会放大计算成本，也不利于快速选择学习率、alignment 权重和最佳 epoch。

建议按以下顺序实验：

| 阶段 | 总样本数 | 每个来源 | epochs | learning rate | 用途 |
|---|---:|---:|---:|---:|---|
| 快速验证 | 4000 | 800 | 1 | 5e-5 | 验证代码、显存、loss 和 alignment |
| 主实验（推荐） | 20000 | 4000 | 2 | 3e-5 | 兼顾覆盖度、训练步数和成本 |
| 扩大实验 | 40000 | 8000 | 1–2 | 2e-5 | 仅在 2 万条完整测试集结果继续提升时使用 |
| 全量 TimeLens-100K | 约 100000 | 不限制 | 1 | 1e-5～2e-5 | 最后确认上限，不作为首轮配置 |

当前数据包含 5 个主要来源，因此推荐配置的 20000 条由每个来源最多抽取 4000 条构成。八卡、单卡 batch size 1、梯度累积 8 时，一轮约有 `20000 / 8 / 8 = 313` 次优化器更新，两轮约 626 次；当前 4000 条训练三轮只有约 189 次优化器更新，通常偏少。

注意：当 `--max-samples-per-folder` 大于 0 时，代码会先读取全部 annotation，再按来源分别截取，`--max-samples` 会被忽略。因此控制 2 万条应使用 `--max-samples-per-folder 4000`。只有设置 `--max-samples-per-folder 0` 后，`--max-samples` 才直接控制总数。

训练启动后，rank 0 会打印完整参数配置、实际目标层、目标 head、alignment head、world size 和有效全局 batch size。

## 2. 八卡 Head 探测

对应脚本：

```text
scripts/m_d_startend_gradient_head_attribution.py
```

Head 探测用于确定值得进行 LoRA 或 attention alignment 的关键 head，不需要遍历全部 TimeLens-100K。推荐固定探测 500 条训练样本：规模已经足够用于估计稳定的 head 排名，同时不会让需要反向传播注意力权重的归因过程耗时过长。

八卡模式由 `torchrun` 启动 8 个 rank。每个 rank 在自己的 GPU 上加载一份完整 base model，并处理同一个 500 条候选集合的 `rank::8` 切片，因此每卡约处理 62–63 条；完成后 rank 0 按各卡实际有效样本数加权合并矩阵并重新计算 Top-K。

```bash
torchrun --standalone --nproc_per_node 8 scripts/m_d_startend_gradient_head_attribution.py \
  --filtered-json ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl \
  --model-path ../shared_models/Qwen3-VL-8B-Instruct \
  --video-dir ../Lkmllm_data/datasets/Train/timelens-100k \
  --output-dir ../Lkmllm_data/outputs/startend_gradient_head_attr_mGPU \
  --max-samples 500 \
  --max-duration 0 \
  --top-k 30 \
  --fps 2 \
  --min-tokens 64 \
  --total-tokens 14336 \
  --layers-per-batch 2
```

这里的 `--max-samples 500` 是八卡合计 500 条，不是每卡 500 条。`--max-duration 0` 表示不按视频时长筛选，使候选集合与微调数据分布一致。视频输入参数与当前微调/评测保持一致：2 FPS、不额外限制帧数、`min_tokens=64`、`total_tokens=14336`。

归因算法需要读取 attention weights，所以探测必须使用 `eager` attention，不能使用微调时的 `flash_attention_2`。这是算法实现上的必要差异；FPS、采帧和视觉 token budget 仍然保持一致。由于 eager attention 和反向传播比普通评测更占显存，24 GB GPU 若出现 OOM，先把 `--layers-per-batch` 从 2 降为 1；如果仍然 OOM，只能降低 `--total-tokens`，但此时需要在结果元数据中注明探测输入预算与微调不同。

最终供微调脚本 `--attr-json` 使用的文件为：

```text
../Lkmllm_data/outputs/startend_gradient_head_attr_mGPU/startend_gradient_head_attribution.json
```

各 rank 的中间结果保存在：

```text
../Lkmllm_data/outputs/startend_gradient_head_attr_mGPU/_rank_outputs/rank_<N>/
```

最终 JSON 已经是八卡加权合并结果，不要把单个 rank 的 JSON 传给微调脚本。

## 3. 八卡微调

以下命令使用 8 张 GPU，并直接通过 `torchrun` 启动。反斜杠 `\` 必须是每行最后一个字符，后面不能有空格。

```bash
torchrun --standalone --nproc_per_node 8 scripts/m_heads_finetune_layer_lora_attn_align.py \
  --attr-json ../Lkmllm_data/outputs/startend_gradient_head_attr_mGPU/startend_gradient_head_attribution.json \
  --model-path ../shared_models/Qwen3-VL-8B-Instruct \
  --anno-json ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl \
  --video-dir ../Lkmllm_data/datasets/Train/timelens-100k \
  --output-dir ../Lkmllm_data/checkpoints/layer12_19_qvo_lora_r8_align001_mGPU \
  --align-weight 0.01 \
  --max-samples-per-folder 4000 \
  --epochs 2 \
  --lr 3e-5
```

推荐配置总训练样本数约为 20000，八卡时每张卡每轮约处理 2500 个样本。单卡 batch size 为 1、梯度累积为 8，因此有效全局 batch size 为：

```text
8 GPUs × 1 sample/GPU × 8 accumulation steps = 64
```

如果机器上还有其他 GPU，可明确限制使用 0-7 号卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node 8 scripts/m_heads_finetune_layer_lora_attn_align.py \
  --attr-json ../Lkmllm_data/outputs/startend_gradient_head_attr_mGPU/startend_gradient_head_attribution.json \
  --model-path ../shared_models/Qwen3-VL-8B-Instruct \
  --anno-json ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl \
  --video-dir ../Lkmllm_data/datasets/Train/timelens-100k \
  --output-dir ../Lkmllm_data/checkpoints/layer12_19_qvo_lora_r8_align001_mGPU \
  --align-weight 0.01 \
  --max-samples-per-folder 4000 \
  --epochs 2 \
  --lr 3e-5
```

## 4. 保存内容与断点恢复

训练过程中，主进程在输出目录中维护：

```text
layer12_19_qvo_lora_r8_align001_mGPU/
├── lora_layer_checkpoint.pt   # 训练中的断点，正常完成后删除
├── lora_layer_adapter.pt      # 独立的 masked-LoRA 参数和结构信息
├── config.json                # 合并后完整模型的 Hugging Face 配置
├── model-*.safetensors        # 已合并 LoRA 的完整模型权重
└── processor/tokenizer files  # 评测所需处理器文件
```

如果训练中断，使用完全相同的微调命令再次启动。脚本会从输出目录中的 `lora_layer_checkpoint.pt` 恢复。正常训练结束后，LoRA 会合并进基础模型并通过 `save_pretrained` 保存，所以评测时应直接把整个输出目录传给 `--model-path`，不需要手动加载 `lora_layer_adapter.pt`。

## 5. 八卡评测已保存的微调模型

以下命令直接评测上面保存的模型。`m_e_head_eval.py` 自己创建 8 个 GPU worker，因此评测入口使用 `python`，不要外套 `torchrun`。当前评测脚本若由 `torchrun --nproc_per_node 8` 启动，每个 rank 都会再次创建一整组 worker，导致重复推理和显存冲突。

### Charades-STA-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
  --model-path ../Lkmllm_data/checkpoints/layer12_19_qvo_lora_r8_align001_mGPU \
  --anno-json ../Lkmllm_data/datasets/Test/Charades_sta/charades-timelens.json \
  --video-dir ../Lkmllm_data/datasets/Test/Charades_sta/charades \
  --output-dir ../Lkmllm_data/outputs/eval_results \
  --split Charades_layer12_19_qvo_r8_align001_mGPU \
  --num-gpus 8
```
### ActivityNet-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
  --model-path ../Lkmllm_data/checkpoints/layer12_19_qvo_lora_r8_align001_mGPU \
  --anno-json ../Lkmllm_data/datasets/Test/Activitynet/activitynet-timelens.json \
  --video-dir ../Lkmllm_data/datasets/Test/Activitynet/activitynet \
  --output-dir ../Lkmllm_data/outputs/eval_results \
  --split ActivityNet_Qwen3VL8B_base_mGPU \
  --num-gpus 8
```

#### QVHighlights-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
  --model-path ../Lkmllm_data/checkpoints/layer12_19_qvo_lora_r8_align001_mGPU \
  --anno-json ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights-timelens.json \
  --video-dir ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights \
  --output-dir ../Lkmllm_data/outputs/eval_results \
  --split QVHighlights_Qwen3VL8B_base_mGPU \
  --num-gpus 8
```
### Qwen3-VL-8B base：三个 TimeLens 数据集

三个数据集都使用同一个 base 模型：

```text
../shared_models/Qwen3-VL-8B-Instruct
```

所有评测参数保持完全一致，仅更换 annotation、视频目录和 `--split`。下面的 ActivityNet/QVHighlights 路径采用 TimeLens-Bench 官方目录结构；Charades 使用当前项目中已经验证过的本地目录。

#### Charades-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
  --model-path ../shared_models/Qwen3-VL-8B-Instruct \
  --anno-json ../Lkmllm_data/datasets/Test/Charades_sta/charades-timelens.json \
  --video-dir ../Lkmllm_data/datasets/Test/Charades_sta/charades \
  --output-dir ../Lkmllm_data/outputs/eval_results \
  --split Charades_Qwen3VL8B_base_mGPU \
  --num-gpus 8
```

#### ActivityNet-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
  --model-path ../shared_models/Qwen3-VL-8B-Instruct \
  --anno-json ../Lkmllm_data/datasets/Test/Activitynet/activitynet-timelens.json \
  --video-dir ../Lkmllm_data/datasets/Test/Activitynet/activitynet \
  --output-dir ../Lkmllm_data/outputs/eval_results \
  --split ActivityNet_Qwen3VL8B_base_mGPU \
  --num-gpus 8
```

#### QVHighlights-TimeLens

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/m_e_head_eval.py \
  --model-path ../shared_models/Qwen3-VL-8B-Instruct \
  --anno-json ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights-timelens.json \
  --video-dir ../Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights \
  --output-dir ../Lkmllm_data/outputs/eval_results \
  --split QVHighlights_Qwen3VL8B_base_mGPU \
  --num-gpus 8
```


## 6. TimeLens 官方实验结果

下面是 [TimeLens 官方项目](https://github.com/pkuhxy/Timelens) 和 [TimeLens-8B 模型说明](https://huggingface.co/TencentARC/TimeLens-8B) 报告的 temporal grounding 结果。所有 R 指标均为 R@1，表中数值以百分数表示；脚本打印的 `0.512` 对应表中的 `51.2`。

### Qwen3-VL-8B-Instruct base

| 测试集 | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---:|---:|---:|---:|
| ActivityNet-TimeLens | 62.1 | 51.2 | 34.4 | 46.8 |
| Charades-TimeLens | 69.2 | 53.4 | 27.5 | 48.3 |
| QVHighlights-TimeLens | 74.2 | 64.6 | 49.3 | 59.4 |

### TimeLens-8B

| 测试集 | R@0.3 | R@0.5 | R@0.7 | mIoU |
|---|---:|---:|---:|---:|
| ActivityNet-TimeLens | 68.9 | 58.4 | 40.6 | 53.2 |
| Charades-TimeLens | 76.6 | 63.0 | 35.2 | 55.2 |
| QVHighlights-TimeLens | 80.2 | 71.6 | 55.5 | 65.5 |

官方结果用于完整测试集参考。只有评测协议、模型版本、输入参数和完整样本集合都一致时，才适合做严格横向比较。



## 7. 常见问题

### 参数被识别为缺失

如果出现：

```text
error: the following arguments are required: --model-path ...
bash: --model-path: command not found
```

通常是上一行的 `\` 后有空格，或参数写成了 `\--model-path`。正确格式是：

```bash
command \
  --argument value \
  --next-argument value
```

### `use_cache=True` 与 gradient checkpointing 提示

Transformers 会自动关闭训练时的 KV cache，以兼容 gradient checkpointing。这是正常提示，不是报错。
