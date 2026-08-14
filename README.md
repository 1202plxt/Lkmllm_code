# Lkmllm_code

## 项目简介

TimeLens 视频时序定位（Video Temporal Grounding）微调：基于 Qwen3-VL-8B-Instruct，先对 start/end 时间数字 token 做梯度归因选出关键 attention head，再用 Layer LoRA + Attention Alignment 辅助 loss 微调，让模型定位事件发生的起止时间段（秒）。

## 目录说明

**所有命令都从 `Lkmllm_code/` 目录执行**（`cd Lkmllm_code`）。代码内部以 `Lkmllm_code/` 为项目根（见 `src/project_paths.py`），数据与权重是它的兄弟目录，访问时用 `../` 前缀：

- `Lkmllm_code/`：代码与脚本（执行目录）
  - `scripts/`：训练 / 归因脚本
  - `src/project_paths.py`：路径解析（以 `Lkmllm_code/` 为根）
  - `tools/organize_timelens.py`：数据集整理
- `../Lkmllm_data/`：数据与产物
  - `datasets/`：数据集（`Train/`、`Test/`）
  - `checkpoints/`：训练权重与 LoRA 产物
  - `outputs/`：head 归因结果、评测输出
  - `logs/`、`visualizations/`、`cache/`、`pretrained/`
- `../shared_models/`：基础模型权重（Qwen3-VL-8B-Instruct）

三者关系：`Lkmllm_code/` 与 `Lkmllm_data/`、`shared_models/` 在工作区根（`LK_OPD/`）下平级。`project_paths.py` 以 `Lkmllm_code/` 为根，数据/模型取它的兄弟目录；因此即使 `shared_models/` 还没建，代码也能正常定位（不会再报 `Could not locate project root`），只有真正加载模型时才需要权重就位。

## 环境配置

```bash
conda create -n qwen3vl python=3.11 -y
conda activate qwen3vl
cd Lkmllm_code
pip install -r requirements.txt
```

特殊版本要求（不匹配会导致加载/训练失败）：

- **CUDA 12.4**、**PyTorch 2.6.0+cu124**（requirements.txt 已带官方 cu124 源）
- **transformers 5.14.1**、**qwen-vl-utils 0.0.14**、**decord 0.6.0**
- **flash-attn 2.7.4.post1**（可选，加速注意力）：必须装 `cu12torch2.6` 的 wheel 匹配 torch + Python 3.11：

  ```bash
  pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
  ```

  装不上就删掉命令里的 `--attn-implementation flash_attention_2`，脚本自动降级 sdpa（略慢但稳定）。
- 本项目**不依赖 vLLM / deepspeed**；多卡用 torch 自带的 `torchrun` + DDP（`accelerate` 已不再用于启动）。

## 数据和权重准备

**初始化目录结构**（在 `Lkmllm_code/` 下执行，一次性创建所有目录）：

```bash
mkdir -p ../Lkmllm_data/datasets/Train \
         ../Lkmllm_data/datasets/Test \
         ../Lkmllm_data/checkpoints \
         ../Lkmllm_data/outputs/startend_gradient_head_attr \
         ../Lkmllm_data/logs \
         ../Lkmllm_data/visualizations \
         ../Lkmllm_data/cache \
         ../Lkmllm_data/pretrained \
         ../shared_models
```

（`datasets/Train`、`datasets/Test` 下的具体子目录如 `timelens-100k`、`Charades_sta` 由下载/整理脚本自动生成，无需手动建。）

| 资源 | 放置路径 | 获取方式 | 用途 |
| --- | --- | --- | --- |
| TimeLens 训练集 | `../Lkmllm_data/datasets/Train/timelens-100k/` | ModelScope `StudyAI123123/timelens-100k` + `tools/organize_timelens.py` 整理 | 训练 |
| TimeLens 测试集 | `../Lkmllm_data/datasets/Test/`（`Charades_sta/`、`Activitynet/`、`Qvhighlights/`） | HuggingFace `TencentARC/TimeLens-Bench` | head 归因 / 评测 |
| 基础模型 | `../shared_models/Qwen3-VL-8B-Instruct` | ModelScope `Qwen/Qwen3-VL-8B-Instruct` | backbone |
| head 归因结果 | `../Lkmllm_data/outputs/startend_gradient_head_attr/startend_gradient_head_attribution.json` | 脚本 `d_startend_gradient_head_attribution.py` 生成 | 训练时 `--attr-json` |
| 微调产物 | `../Lkmllm_data/checkpoints/lora_layer/` | 训练生成 | 继续训练 / 推理 |

下载命令（在 `Lkmllm_code/` 下执行）：

```bash
pip install modelscope
pip install -U "huggingface_hub[cli]"

# 训练集（下载到工作区根 → 解压 → 整理到 ../Lkmllm_data）
modelscope download --dataset 'StudyAI123123/timelens-100k' --local_dir ../timelens-100k
(cd ../timelens-100k && unzip *.zip)
python tools/organize_timelens.py

# 测试集（Charades-STA / ActivityNet / QVHighlights，HuggingFace TimeLens-Bench）
hf download TencentARC/TimeLens-Bench --repo-type=dataset --local-dir ../Lkmllm_data/datasets/Test

# 基础模型
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir ../shared_models/Qwen3-VL-8B-Instruct
```

测试集下载后，`Test/` 下是 `TimeLens-Bench/`，内含 `video_shards/`（三个数据集的 `.tar.gz` 视频分片）+ 三个 `*-timelens.json` 标注。需**先解压视频分片**，再和对应 json 放到同一个数据集目录下（`TimeLens-Bench/` 和 `video_shards/` 都不要）：

```bash
(cd ../Lkmllm_data/datasets/Test && \
  mkdir -p Activitynet/activitynet Charades_sta/charades Qvhighlights/qvhighlights && \
  find TimeLens-Bench/video_shards/activitynet  -name "*.tar.gz" -exec tar -xzf {} -C Activitynet/activitynet  \; && \
  find TimeLens-Bench/video_shards/charades     -name "*.tar.gz" -exec tar -xzf {} -C Charades_sta/charades     \; && \
  find TimeLens-Bench/video_shards/qvhighlights -name "*.tar.gz" -exec tar -xzf {} -C Qvhighlights/qvhighlights \; && \
  mv TimeLens-Bench/activitynet-timelens.json  Activitynet/ && \
  mv TimeLens-Bench/charades-timelens.json     Charades_sta/ && \
  mv TimeLens-Bench/qvhighlights-timelens.json Qvhighlights/ && \
  rm -rf TimeLens-Bench)
```

> 解压后如果某个 `.tar.gz` 里还套了一层同名目录（如变成 `Activitynet/activitynet/activitynet/xxx.mp4`），给上面的 `tar -xzf {}` 加 `--strip-components=1` 再跑一次。若你的下载里 `video_shards/` 是平铺的 `*.tar.gz`（没有三个子目录），把三个 `find TimeLens-Bench/video_shards/activitynet ...` 路径改成 `find TimeLens-Bench/video_shards -path "*activitynet*" -name "*.tar.gz" ...`（charades / qvhighlights 同理）。

整理后的最终结构（每个子数据集 = 视频目录 + `*-timelens.json` 标注）：

```text
Test/
├── Activitynet/
│   ├── activitynet/              # 视频
│   └── activitynet-timelens.json
├── Charades_sta/
│   ├── charades/                 # 视频
│   └── charades-timelens.json
└── Qvhighlights/
    ├── qvhighlights/             # 视频
    └── qvhighlights-timelens.json
```

> 若 head 归因结果 json 已生成、但放在了 `Lkmllm_code/` 根目录（`startend_gradient_head_attribution.json`），把它移到正确位置（在 `Lkmllm_code/` 下执行）：

```bash
mkdir -p ../Lkmllm_data/outputs/startend_gradient_head_attr
cp startend_gradient_head_attribution.json ../Lkmllm_data/outputs/startend_gradient_head_attr/
```

## 训练

### 1. Head 归因（训练前，生成 `--attr-json`）

```bash
python scripts/d_startend_gradient_head_attribution.py \
  --filtered-json ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl \
  --model-path ../shared_models/Qwen3-VL-8B-Instruct \
  --video-dir ../Lkmllm_data/datasets/Train/timelens-100k \
  --output-dir ../Lkmllm_data/outputs/startend_gradient_head_attr \
  --max-samples 100 --top-k 30 \
  --fps 2 --min-tokens 64 --max-tokens 14336 \
  --layers-per-batch 6 --min-attn-ratio 1.0 \
  --grad-only-top-k 30 --attn-only-top-k 30
```

输出 `startend_gradient_head_attribution.json`（含 `top_k_heads`、`combined_score_matrix`，以及 grad-only / attn-only 两条独立排名列表）。若该文件已存在可跳过此步。

> **TimeLens 标准采样参数**：脚本已直接暴露官方参数，与 TimeLens 一一对应（[TimeLens-7B](https://huggingface.co/TencentARC/TimeLens-7B) / [TimeLens-8B](https://huggingface.co/TencentARC/TimeLens-8B)）——`--fps 2`（官方 FPS=2）、`--min-tokens 64`（官方 min_tokens=64）、`--max-tokens 14336`（官方 total_tokens=14336）。`--max-side` 默认 `None`，不再强制 resize，每帧分辨率/token 数完全交给 processor 按 `min_tokens`/`max_tokens` 预算自适应控制。归因默认跑训练集（`timelens-100k`），与微调数据同分布，选出的 head 更贴合后续训练。

> **显存 / 多卡**：脚本是多卡数据并行——`--num-gpus` 默认 0（自动检测所有可用 GPU），每张卡加载**一个完整模型副本**（`device_map={"": cuda:N}`，不做跨卡分片），样本均匀分配后各卡独立归因、主进程汇总。因此**每张卡需约 16–18 GiB 显存放 8B bf16 权重**（再加 attention 显存）。旧的 `--gpu-mem-gib` / `--cpu-mem-gib` 已不再使用。OOM 时：调低 `--max-tokens`（attention 显存随 seq_len 平方增长）、`--layers-per-batch`（36 层分批数），或 `--num-gpus 1` 退回单卡；`--min-attn-ratio` 是过滤基线，不影响显存。

### 2. LoRA 微调（多卡）

```bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
torchrun --nproc_per_node=${NUM_GPUS} \
  scripts/heads_finetune_layer_lora_attn_align.py \
  --attr-json  ../Lkmllm_data/outputs/startend_gradient_head_attr/startend_gradient_head_attribution.json \
  --model-path ../shared_models/Qwen3-VL-8B-Instruct \
  --anno-json  ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl \
  --video-dir  ../Lkmllm_data/datasets/Train/timelens-100k \
  --output-dir ../Lkmllm_data/checkpoints/lora_layer \
  --top-k 20 --align-top-n 20 --align-weight 0.1 --lr 1e-5 --epochs 10
```

## 评测

暂无独立 eval 脚本。时序定位指标（MAE / IoU）的辅助函数在 `scripts/c_time_utils.py`：

- `generate_prediction(model, processor, inputs)`：生成并解析起止时间
- `compute_mae(pred, gt)` / `compute_iou(pred, gt)`：计算指标

如需评测，基于这几个函数补一个 eval 脚本即可（TODO）。

## 输出说明

- **head 归因**：`../Lkmllm_data/outputs/startend_gradient_head_attr/startend_gradient_head_attribution.json`
- **LoRA 权重**：`../Lkmllm_data/checkpoints/lora_layer/lora_layer_adapter.pt` + `config.json`
- **合并后的完整模型**：`../Lkmllm_data/checkpoints/lora_layer/` 下的 `model.safetensors` 等（`save_pretrained` 写出，可直接加载做推理）
- **断点续训**：`lora_layer_checkpoint.pt`（训练中周期性保存，正常结束后删除）
- **跑成功的标志**：训练打印 `Epoch ... valid=... ce=... align=... total=...`，最后出现 `Merging LoRA...` 并在 output-dir 生成合并后的模型权重。

## 常见问题

- **缺数据**：`--anno-json` / `--video-dir` 指向不存在 → 确认已下载并整理 TimeLens 到 `../Lkmllm_data/datasets/Train/timelens-100k/`（含 `timelens-100k.jsonl` 和 5 个视频子目录）。
- **缺权重**：`--model-path` 目录里没有 `config.json` / `*.safetensors` → 用 modelscope 下载到 `../shared_models/Qwen3-VL-8B-Instruct`。
- **路径没设对**：所有 `--*` 相对路径都相对 `Lkmllm_code/`（执行目录）解析，不确定时直接写绝对路径。
- **CUDA / torch / flash-attn 不匹配**：flash-attn 装不上或 import 报错时，删掉 `--attn-implementation flash_attention_2` 降级 sdpa。
- **多卡没生效 / 报 NCCL 错误**：多卡必须用 `torchrun` 启动（不能 `python`）；NCCL 需要 NVIDIA GPU + Linux。
- **OOM**：调低 `--max-side` / `--fps`（attention 显存与 seq_len 平方相关）、`--layers-per-batch`（归因脚本）、`--max-frames` / `--total-tokens`（微调脚本）。
