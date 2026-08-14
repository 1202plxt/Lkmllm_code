"""
e_head_eval.py — HeadLoRA 微调模型的 Temporal Grounding 评估
                 (TimeLens prompt 对齐版，纯 generate 路径，多卡数据并行)

⚠️ 采样/编码参数必须和 heads_finetune.py 训练时完全一致，否则模型看到的
   "视频"和训练时不是一回事（帧数、每帧分辨率、时间戳节奏、attention 数值
   实现都会变），IoU 会直接掉到接近随机猜的水平。默认值全部对齐训练：

     训练默认: --fps 2.0 --max-frames 200 --min-tokens 64 --total-tokens 14336
              --attn-implementation flash_attention_2
     评测默认: 同上

   如果训练时用的是不同参数，评测时也要对应传同样的值。用 --model-path
   同时评测 base model (baseline)、top-K HeadLoRA、random-head 对照，
   三者的采样参数必须相同才有可比性。

相对上一版评测脚本改了这几处 (为解决 IoU 从训练时期望的 ~0.5 掉到 0.09 的问题):
  1. attn_implementation 默认改成 flash_attention_2，和训练一致；
     可以用 --attn-implementation eager 切回旧行为 (但没必要)。
  2. 去掉 processor.video_processor.size['shortest_edge'] = 128 这行硬编码，
     训练时没有做这一步，评测强行压到 128 会让视频细节大量丢失。
  3. build_inputs_timelens 里的 video_content 现在带上 min_pixels / total_pixels，
     和训练时 collate_fn_with_processor 走的是同一条 pixel-budget 路径；
     sample_fps 也不再硬编码 1.0，改用外部传入的 --fps 值。
  4. CLI 加了 --min-tokens / --total-tokens；--fps 和 --max-frames 默认值改成
     和训练默认一致 (2.0 / 200)。
  5. EvalDataset 里当 total_tokens > 0 时不再做 max-size resize (和训练一致，
     让 processor 内部按 pixel budget 处理，避免我们在外面提前缩放搞乱)。

用法:
    # 微调后 (top-K HeadLoRA 主实验)
    python scripts/e_head_eval.py \
        --model-path  ../Lkmllm_data/checkpoints/lora_layer \
        --anno-json   ../Lkmllm_data/datasets/Test/Charades_sta/charades-timelens.json \
        --video-dir   ../Lkmllm_data/datasets/Test/Charades_sta/charades \
        --output-dir  ../Lkmllm_data/outputs/eval_results \
        --split       charades_topk

    # base baseline (⚠️ 采样参数必须传一样的)
    python scripts/e_head_eval.py \
        --model-path  ../shared_models/Qwen3-VL-8B-Instruct \
        --anno-json   ../Lkmllm_data/datasets/Test/Charades_sta/charades-timelens.json \
        --video-dir   ../Lkmllm_data/datasets/Test/Charades_sta/charades \
        --output-dir  ../Lkmllm_data/outputs/eval_results \
        --split       charades_base

    # 随机 head 对照
    python scripts/e_head_eval.py \
        --model-path  ../Lkmllm_data/checkpoints/lora_head_random \
        --anno-json   ../Lkmllm_data/datasets/Test/Charades_sta/charades-timelens.json \
        --video-dir   ../Lkmllm_data/datasets/Test/Charades_sta/charades \
        --output-dir  ../Lkmllm_data/outputs/eval_results \
        --split       charades_random

多卡：加 --num-gpus N（默认 0 = 自动检测所有 GPU，数据并行）。
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]   # 项目根目录 = Lkmllm_code
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.project_paths import A_DATA_ROOT, ensure_directory, resolve_path, SHARED_MODELS_ROOT

# processor 内部按 patch 换算 pixel 预算：每个 patch 是 28*28 像素
PATCH_PIXELS = 28 * 28


# ═══════════════════════════════════════════════════════════════════════════════════
# TimeLens Prompt (与 heads_finetune.py 完全一致，改这里会破坏推理分布匹配)
# ═══════════════════════════════════════════════════════════════════════════════════

TIMELENS_SYSTEM = (
    "You are a video time analysis assistant. "
    "Given a video and a query, you must locate the exact time segment "
    "where the described event occurs."
)

TIMELENS_USER_TEMPLATE = (
    "You are given a video with multiple frames. The numbers before each video "
    "frame indicate its sampling timestamp (in seconds). Please find the visual "
    "event described by the sentence '{query}', determining its starting and "
    "ending times. The format should be: "
    "'The event happens in <start time> - <end time> seconds'."
)


# ═══════════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HeadLoRA 微调模型 Temporal Grounding 评估 (纯 generate 路径)"
    )
    p.add_argument("--model-path", required=True,
                   help="HeadLoRA 合并后的模型目录 (heads_finetune.py 里 save_pretrained "
                        "的输出) 或原始 base model 目录 (base baseline)")
    p.add_argument("--anno-json",  required=True)
    p.add_argument("--video-dir",  required=True)
    p.add_argument("--output-dir", default=str(A_DATA_ROOT / "outputs" / "eval_results"))
    p.add_argument("--split",      default="eval",
                   help="本次评估的标识名，写入输出文件名 "
                        "(例如 charades_topk / charades_base / charades_random 等)")

    # 采样控制 - 默认全部对齐训练 (heads_finetune.py 的默认 & 你实际用的值)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--fps",         type=float, default=2.0,
                   help="和训练 --fps 保持一致 (训练默认 2.0)")
    p.add_argument("--max-frames",  type=int, default=200,
                   help="和训练 --max-frames 保持一致 (训练默认 200)")
    p.add_argument("--max-size",    type=int, default=256,
                   help="仅当 --total-tokens<=0 时才用；训练默认走 total-tokens 路径，"
                        "这个值就用不上")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 表示 greedy decoding (确定性、可复现)；> 0 会走采样")

    # Token 预算 (和训练一致，是决定 IoU 的关键路径)
    p.add_argument("--min-tokens",   type=int, default=64,
                   help="和训练 --min-tokens 保持一致 (训练默认 64，TimeLens 标准)")
    p.add_argument("--total-tokens", type=int, default=14336,
                   help="和训练 --total-tokens 保持一致 (训练默认 14336，TimeLens 标准)。"
                        "≤0 时降级到旧的 max-size resize 路径 (不推荐)。")

    # attention 实现 - 默认对齐训练 (flash_attention_2)，
    # 上一版硬编码 eager 会引入不必要的数值差异
    p.add_argument("--attn-implementation", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"],
                   help="和训练 --attn-implementation 保持一致 (训练默认 flash_attention_2)")

    # 多卡数据并行
    p.add_argument("--num-gpus", type=int, default=0,
                   help="数据并行 GPU 数量。0 表示自动检测所有可用 GPU；"
                        "1 表示单卡模式。每张卡加载一个完整模型副本，"
                        "样本均分后各卡独立评测、主进程合并结果。")

    # 指标
    p.add_argument("--iou-thresholds", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7])
    return p


# ═══════════════════════════════════════════════════════════════════════════════════
# 视频采帧
# ═══════════════════════════════════════════════════════════════════════════════════

def sample_frames(video_path: Path, fps: float) -> list:
    """从视频中提取帧，返回 PIL Image 列表。"""
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        native_fps = vr.get_avg_fps()
        step = max(1, int(native_fps / fps))
        indices = list(range(0, len(vr), step))
        frames_t = vr.get_batch(indices)
        from PIL import Image
        return [Image.fromarray(frames_t[i].numpy()) for i in range(len(indices))]
    except ImportError:
        pass
    try:
        import cv2
        from PIL import Image
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{video_path}")
        native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        step = max(1, int(native_fps / fps))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = list(range(0, n_frames, step))
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames
    except ImportError:
        pass
    sys.exit("ERROR 请安装 decord 或 opencv-python")


# ═══════════════════════════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════════════════════════

def load_model_and_processor(model_dir: Path, gpu_mem_gib: float = 16.0,
                               cpu_mem_gib: float = 64.0,
                               attn_implementation: str = "flash_attention_2"):
    """加载模型 (base 或 HeadLoRA 合并后的 checkpoint)，多卡均衡分配。"""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True,
    )
    # ⚠️ 上一版这里硬编码 shortest_edge=128，会把视频细节压得远低于训练时；
    # 训练时 collate_fn_with_processor 里没有做这一步 —— 保持 processor 默认。

    # flash-attn 环境检查 (跟 heads_finetune.py 里的降级逻辑保持一致)
    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            print(f"  [attn] flash_attn={flash_attn.__version__}")
        except ImportError:
            print("  [attn] flash_attn 不可用，降级为 sdpa")
            attn_implementation = "sdpa"

    n_gpus = torch.cuda.device_count()
    max_memory = {}
    if n_gpus >= 1:
        per_gpu_limit = f"{gpu_mem_gib:g}GiB"
        for i in range(n_gpus):
            max_memory[i] = per_gpu_limit
        max_memory["cpu"] = f"{cpu_mem_gib:g}GiB"
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        print(f"  [model] {n_gpus} GPUs: {gpu_names}")
        print(f"  [model] max_memory={max_memory}")

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map="auto",
        max_memory=max_memory if max_memory else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    print(f"  [model] attn_implementation={attn_implementation}")
    n_layers = len(model.model.language_model.layers)

    from collections import Counter
    layer_devices = Counter()
    for layer in model.model.language_model.layers:
        layer_devices[str(layer.self_attn.o_proj.weight.device)] += 1
    print(f"  [model] {n_layers} layers 分布: {dict(layer_devices)}")

    return model, processor, n_layers


def load_model_and_processor_single_gpu(model_dir: Path, gpu_id: int,
                                        attn_implementation: str = "flash_attention_2"):
    """在指定单张 GPU 上加载完整模型（多卡数据并行时每个进程调用一次）。
    不用 device_map='auto' 的跨卡分片，而是整个模型放到一张卡上。"""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True,
    )
    # 不手动改 processor.video_processor.size（和训练 collate 保持一致）

    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            print(f"  [attn] flash_attn={flash_attn.__version__}")
        except ImportError:
            print("  [attn] flash_attn 不可用，降级为 sdpa")
            attn_implementation = "sdpa"

    device = torch.device(f"cuda:{gpu_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map={"": device},
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    print(f"  [GPU{gpu_id}] 模型加载完成 device={device} "
          f"attn={attn_implementation}")
    return model, processor, len(model.model.language_model.layers)


# ═══════════════════════════════════════════════════════════════════════════════════
# 输入构建 (TimeLens 格式，视频参数和 heads_finetune.py 的 collate_fn_with_processor
# 完全一致：sample_fps 用外部传入的 fps；开启 total-tokens 时走 pixel-budget 路径
# 而不是自己在外面 resize)。
#
# 这里 add_generation_prompt=True 是关键：让 chat_template 在 assistant 位置留出
# 生成起点，模型才知道该"开始回答了"，而不是在补全 user turn。
# ═══════════════════════════════════════════════════════════════════════════════════

def build_inputs_timelens(processor, frames: list, query: str, device,
                          sample_fps: float,
                          min_tokens: int, total_tokens: int
                          ) -> Optional[dict]:
    """构建 TimeLens 格式的模型输入 (只用于 generate)，frames 为空返回 None。"""
    from qwen_vl_utils import process_vision_info

    if not frames:
        return None

    user_text = TIMELENS_USER_TEMPLATE.format(query=query)

    # 和训练时 collate_fn_with_processor 里的 video_content 走同一条路径
    video_content = {"type": "video", "video": frames, "sample_fps": sample_fps}
    if total_tokens and total_tokens > 0:
        video_content["min_pixels"]   = min_tokens * PATCH_PIXELS
        video_content["total_pixels"] = total_tokens * PATCH_PIXELS

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": TIMELENS_SYSTEM}
        ]},
        {"role": "user", "content": [
            video_content,
            {"type": "text", "text": user_text},
        ]},
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True,
    )
    if video_inputs:
        video_tensors, video_metadatas = zip(*video_inputs)
        video_tensors, video_metadatas = list(video_tensors), list(video_metadatas)
    else:
        video_tensors, video_metadatas = video_inputs, None

    scalar_video_kwargs = {
        k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
        for k, v in video_kwargs.items()
    }

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_tensors,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        **scalar_video_kwargs,
    )
    return {k: v.to(device) if hasattr(v, "to") else v
            for k, v in inputs.items()}


# ═══════════════════════════════════════════════════════════════════════════════════
# 时间戳解析 & IoU
# ═══════════════════════════════════════════════════════════════════════════════════

# 模式 1: TimeLens 格式 "The event happens in X - Y seconds."
_RE_TIMELENS = re.compile(
    r'(?:event\s+happens\s+in|event\s+occurs\s+in|starts\s+at|begins\s+at)'
    r'\s+([\d.]+)\s*[-–—to]+\s*([\d.]+)\s*seconds?',
    re.IGNORECASE,
)

# 模式 2: "start: X, end: Y"
_RE_STANDARD = re.compile(
    r"start\s*[::：]\s*([\d.]+)[,，\s]+end\s*[::：]\s*([\d.]+)",
    re.IGNORECASE,
)

# 模式 3: 纯数字
_RE_NUMS = re.compile(r"[\d.]+")


def parse_pred(text: str, duration: float = 120.0) -> Tuple[float, float]:
    """
    解析模型输出中的时间戳。支持多种格式，因为 base model 未微调时可能吐出
    别的格式，微调后应该稳定输出 TimeLens 格式：
      1. "The event happens in X - Y seconds."  (TimeLens, 主格式)
      2. "start: X, end: Y"
      3. "from X to Y" / "X to Y seconds"
      4. 连续两个数字 (最后兜底，可能有误报)

    解析失败返回 (0.0, 0.0)。
    """
    t = text.strip()

    m = _RE_TIMELENS.search(t)
    if m:
        s, e = float(m.group(1)), float(m.group(2))
        if 0 <= s < e <= duration * 2:
            return s, e

    m = _RE_STANDARD.search(t)
    if m:
        s, e = float(m.group(1)), float(m.group(2))
        if 0 <= s < e <= duration * 2:
            return s, e

    m = re.search(r'from\s+([\d.]+)\s+to\s+([\d.]+)', t, re.IGNORECASE)
    if m:
        s, e = float(m.group(1)), float(m.group(2))
        if 0 <= s < e <= duration * 2:
            return s, e

    nums = _RE_NUMS.findall(t)
    if len(nums) >= 2:
        s, e = float(nums[0]), float(nums[1])
        if 0 <= s < e <= duration * 2:
            return s, e
        elif 0 <= e < s <= duration * 2:
            return e, s

    return 0.0, 0.0


def temporal_iou(ps: float, pe: float, gs: float, ge: float) -> float:
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = (pe - ps) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════════
# 推理：Generate 模式 (唯一路径)
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def infer_generate(model, processor, frames: list, query: str, device,
                   duration: float, sample_fps: float,
                   min_tokens: int, total_tokens: int,
                   max_new_tokens: int = 64, temperature: float = 0.0
                   ) -> Tuple[Tuple[float, float], str]:
    """
    让模型生成 TimeLens 格式的答案，解析时间戳。
    返回 ((start, end), raw_text)。
    """
    inputs = build_inputs_timelens(
        processor, frames, query, device,
        sample_fps=sample_fps,
        min_tokens=min_tokens, total_tokens=total_tokens,
    )
    if inputs is None:
        return (0.0, 0.0), ""

    gkw = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
    if temperature > 0:
        gkw["temperature"] = temperature

    gen_ids = model.generate(
        **inputs,
        pad_token_id=(processor.tokenizer.pad_token_id
                      or processor.tokenizer.eos_token_id),
        **gkw,
    )
    prompt_len = inputs["input_ids"].shape[1]
    text = processor.tokenizer.decode(
        gen_ids[0, prompt_len:], skip_special_tokens=True,
    )
    pred = parse_pred(text, duration)
    return pred, text


# ═══════════════════════════════════════════════════════════════════════════════════
# 指标累积器
# ═══════════════════════════════════════════════════════════════════════════════════

class Metrics:
    def __init__(self, thresholds: List[float]):
        self.th = thresholds
        self.ious: List[float] = []
        self.hits: Dict[float, int] = {t: 0 for t in thresholds}
        self.total = 0
        self.no_vid = 0
        self.fail = 0
        self.num_oom = 0

    def update(self, ps: float, pe: float, gs: float, ge: float,
               has_video: bool, parse_ok: bool):
        self.total += 1
        if not has_video:
            self.no_vid += 1
            return
        if not parse_ok:
            self.fail += 1
            self.ious.append(0.0)
            return
        iou = temporal_iou(ps, pe, gs, ge)
        self.ious.append(iou)
        for t in self.th:
            if iou >= t:
                self.hits[t] += 1

    def summary(self) -> dict:
        n = max(len(self.ious), 1)
        d = {
            "total": self.total,
            "evaluated": len(self.ious),
            "no_video": self.no_vid,
            "parse_fail": self.fail,
            "num_oom": self.num_oom,
            "mIoU": round(sum(self.ious) / n, 4),
        }
        for t in self.th:
            d[f"R@1_IoU{t}"] = round(self.hits[t] / n, 4)
        return d

    def print_table(self, title: str = "") -> dict:
        d = self.summary()
        n = d["evaluated"]
        print(f"\n{'='*64}")
        print(f"  {title}")
        print(f"  样本: {d['total']}  有效推理: {n}  "
              f"(无视频: {d['no_video']}  解析失败: {d['parse_fail']}"
              f"  OOM: {d['num_oom']})")
        print(f"  {'-'*46}")
        print(f"  {'指标':<22} {'值':>8}  {'命中/总数':>12}")
        print(f"  {'-'*46}")
        print(f"  {'mIoU':<22} {d['mIoU']:>8.4f}")
        for t in self.th:
            k = f"R@1_IoU{t}"
            print(f"  {k:<22} {d[k]:>8.4f}  {self.hits[t]:>6}/{n}")
        print(f"{'='*64}\n")
        return d

    @classmethod
    def merge(cls, metrics_list):
        """合并多个 worker 的 Metrics（多卡数据并行时主进程调用）。"""
        if not metrics_list:
            return cls([0.3, 0.5, 0.7])
        m = cls(metrics_list[0].th)
        for o in metrics_list:
            m.ious.extend(o.ious)
            m.total += o.total
            m.no_vid += o.no_vid
            m.fail += o.fail
            m.num_oom += o.num_oom
            for t in m.th:
                m.hits[t] += o.hits[t]
        return m


# ═══════════════════════════════════════════════════════════════════════════════════
# 样本加载 (兼容多种数据集格式，跟微调路径无关，原样保留)
# ═══════════════════════════════════════════════════════════════════════════════════

def resolve_video_path(video_dir: Path, s: dict) -> Optional[Path]:
    for ext in [".mp4", ".mkv", ".avi", ".webm"]:
        vpath = video_dir / f"{s['video_id']}{ext}"
        if vpath.exists():
            return vpath
    rel = s.get("video_rel_path", "")
    if rel:
        vpath = video_dir / rel
        if vpath.exists():
            return vpath
    return None


def load_samples(anno_json: Path, video_dir: Path,
                 max_samples: int = 500) -> list[dict]:
    """
    加载样本，自动适配多种 JSON 格式：
      A. timelens 统一格式 / Charades 格式: {"video_id": {"duration":..., "queries":[...]}}
      B. ActivityNet 原始格式: {video_id: {timestamps: [...], sentences: [...]}}
      C. 嵌套格式: {video_id: [{...}, {...}]}
    """
    raw = json.loads(anno_json.read_text(encoding="utf-8"))
    raw.pop("_meta", None)

    if "数据" in raw:
        raw = raw["数据"]

    samples = []
    for vid, info in raw.items():
        if not isinstance(info, dict):
            continue

        # 格式 B: ActivityNet {timestamps, sentences}
        if "timestamps" in info and "sentences" in info:
            anns = info["timestamps"]
            sents = info["sentences"]
            duration = info.get("duration", 0.0)
            for i, (span, sent) in enumerate(zip(anns, sents)):
                if len(span) < 2:
                    continue
                samples.append({
                    "video_id": vid,
                    "query": sent,
                    "duration": duration if duration > 0 else span[-1] + 10.0,
                    "gt_start": float(span[0]),
                    "gt_end": float(span[1]),
                })
            continue

        # 格式 A / C
        duration = info.get("duration", 0.0)
        queries = info.get("queries", [])

        if queries:
            for q in queries:
                if isinstance(q, dict):
                    query_text = q.get("q_pos", q.get("query", ""))
                    if "spans" in q and q["spans"]:
                        gt_start = float(q["spans"][0][0])
                        gt_end = float(q["spans"][0][1])
                    elif "span" in q:
                        span = q["span"]
                        if isinstance(span, (list, tuple)) and len(span) >= 2:
                            gt_start = float(span[0])
                            gt_end = float(span[1])
                        else:
                            continue
                    else:
                        continue
                elif isinstance(q, str):
                    query_text = q
                    spans = info.get("spans", [])
                    idx = queries.index(q)
                    if idx < len(spans) and len(spans[idx]) >= 2:
                        gt_start = float(spans[idx][0])
                        gt_end = float(spans[idx][1])
                    else:
                        continue
                else:
                    continue

                if not query_text:
                    continue

                samples.append({
                    "video_id": vid,
                    "query": query_text,
                    "duration": duration if duration > 0 else 120.0,
                    "gt_start": gt_start,
                    "gt_end": gt_end,
                })
        else:
            items = info if isinstance(info, list) else info.get("items", [])
            for item in items:
                samples.append({
                    "video_id": vid,
                    "query": item.get("sentence", item.get("query", "")),
                    "duration": item.get("duration", duration or 120.0),
                    "gt_start": float(item.get("gt_start", item.get("start", 0))),
                    "gt_end": float(item.get("gt_end", item.get("end", 0))),
                })

    if max_samples and len(samples) > max_samples:
        import random
        rng = random.Random(42)
        rng.shuffle(samples)
        samples = samples[:max_samples]

    print(f"  加载 {len(samples)} 条样本")
    if samples:
        print(f"  示例: [{samples[0]['video_id']}] "
              f"[{samples[0]['gt_start']:.1f}, {samples[0]['gt_end']:.1f}] "
              f"'{samples[0]['query'][:50]}'")

    return samples


# ═══════════════════════════════════════════════════════════════════════════════════
# Dataset
#
# 帧数上限和外部 resize 的策略要和训练一致 (heads_finetune.py 的 VideoDataset)：
#   - 如果 total_tokens > 0: 只做帧数 subsample，不在外面手动 resize，
#     交给 processor 内部按 min_pixels / total_pixels 处理
#   - 如果 total_tokens <= 0: 用 max_size 手动 resize (旧路径，不推荐)
# ═══════════════════════════════════════════════════════════════════════════════════

class EvalDataset(Dataset):
    def __init__(self, samples, video_dir: Path, fps, max_frames, max_size,
                 total_tokens: int):
        self.samples = samples
        self.video_dir = video_dir
        self.fps = fps
        self.max_frames = max_frames
        self.max_size = max_size
        self.total_tokens = total_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        vp = resolve_video_path(self.video_dir, s)
        frames = []
        if vp is not None:
            try:
                frames = sample_frames(vp, self.fps)
            except Exception as e:
                print(f"  [WARN] {vp.name}: {e}")
            if len(frames) > self.max_frames:
                ii = [int(i * len(frames) / self.max_frames)
                      for i in range(self.max_frames)]
                frames = [frames[i] for i in ii]
            # 只有关闭 pixel-budget 路径 (total_tokens<=0) 时才在外面 resize
            if self.total_tokens <= 0:
                out = []
                for fr in frames:
                    w, h = fr.size
                    if max(w, h) > self.max_size:
                        sc = self.max_size / max(w, h)
                        fr = fr.resize((int(w * sc), int(h * sc)))
                    out.append(fr)
                frames = out
        return {
            **{k: s[k] for k in ("video_id", "query", "gt_start", "gt_end", "duration")},
            "frames": frames,
            "has_video": len(frames) > 0,
            "video_path": str(vp) if vp else None,
        }


# ═══════════════════════════════════════════════════════════════════════════════════
# 评估循环（单卡主进程 / 多卡 worker 共用）
# ═══════════════════════════════════════════════════════════════════════════════════

def evaluate_dataset(model, processor, dataset, device, args, gpu_id=None):
    """在给定 model/processor/dataset 上跑完整评估，返回 (acc, records)。
    gpu_id=None 表示主进程（单卡）；否则为多卡 worker 的 gpu 编号。"""
    acc = Metrics(args.iou_thresholds)
    records: List[dict] = []

    desc = f"[{args.split}]" if gpu_id is None else f"[{args.split} gpu{gpu_id}]"
    pbar = tqdm(range(len(dataset)), desc=desc, leave=(gpu_id is None))

    for idx in pbar:
        item = dataset[idx]
        gs, ge = float(item["gt_start"]), float(item["gt_end"])
        duration = float(item["duration"])

        if not item["has_video"]:
            acc.update(0, 0, gs, ge, has_video=False, parse_ok=False)
            records.append({
                "video_id": item["video_id"], "query": item["query"],
                "gt_start": gs, "gt_end": ge,
                "pred_start": 0.0, "pred_end": 0.0,
                "iou": 0.0, "status": "no_video", "pred_raw": "",
            })
            continue

        ps, pe = 0.0, 0.0
        pred_raw = ""

        try:
            (ps, pe), pred_raw = infer_generate(
                model, processor, item["frames"], item["query"],
                device, duration,
                sample_fps=args.fps,
                min_tokens=args.min_tokens, total_tokens=args.total_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            parse_ok = not (pred_raw.strip() and ps == 0.0 and pe == 0.0)

        except torch.cuda.OutOfMemoryError:
            acc.num_oom += 1
            torch.cuda.empty_cache()
            gc.collect()
            records.append({
                "video_id": item["video_id"], "query": item["query"],
                "gt_start": gs, "gt_end": ge,
                "pred_start": 0.0, "pred_end": 0.0,
                "iou": 0.0, "status": "oom", "pred_raw": "",
            })
            continue
        except Exception as ex:
            records.append({
                "video_id": item["video_id"], "query": item["query"],
                "gt_start": gs, "gt_end": ge,
                "pred_start": 0.0, "pred_end": 0.0,
                "iou": 0.0, "status": f"error:{ex}", "pred_raw": "",
            })
            continue

        iou = temporal_iou(ps, pe, gs, ge)
        acc.update(ps, pe, gs, ge, has_video=True, parse_ok=parse_ok)

        status = "ok"
        if not parse_ok:
            status = "parse_fail"
        elif iou == 0.0:
            status = "zero_iou"

        records.append({
            "video_id": item["video_id"],
            "query": item["query"],
            "gt_start": gs,
            "gt_end": ge,
            "pred_start": round(ps, 2),
            "pred_end": round(pe, 2),
            "iou": round(iou, 4),
            "status": status,
            "pred_raw": pred_raw,
        })

        if (idx + 1) % 20 == 0:
            d = acc.summary()
            pbar.set_postfix(mIoU=f"{d['mIoU']:.3f}",
                             R05=f"{d.get('R@1_IoU0.5', 0):.3f}")

        if (idx + 1) % 50 == 0:
            if gpu_id is not None:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
            else:
                for gi in range(torch.cuda.device_count()):
                    with torch.cuda.device(gi):
                        torch.cuda.empty_cache()

    pbar.close()
    return acc, records


def _worker_entry(gpu_id: int, model_dir: Path, samples: list, video_dir: Path,
                  args, result_queue):
    """单 GPU worker：加载模型，跑分配给自己的样本子集，结果放入队列。"""
    torch.cuda.set_device(gpu_id)

    print(f"\n[GPU{gpu_id}] 加载模型: {model_dir}")
    model, processor, _ = load_model_and_processor_single_gpu(
        model_dir, gpu_id, attn_implementation=args.attn_implementation)
    device = torch.device(f"cuda:{gpu_id}")

    dataset = EvalDataset(samples, video_dir, args.fps,
                          args.max_frames, args.max_size,
                          total_tokens=args.total_tokens)
    print(f"[GPU{gpu_id}] 样本数: {len(dataset)}")

    acc, records = evaluate_dataset(model, processor, dataset, device, args,
                                    gpu_id=gpu_id)
    result_queue.put((gpu_id, acc, records))


# ═══════════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = ensure_directory(Path(args.output_dir))

    # ── 环境 ──
    total_gpus = torch.cuda.device_count()
    num_gpus = args.num_gpus if args.num_gpus > 0 else total_gpus
    num_gpus = min(num_gpus, total_gpus)
    if num_gpus == 0:
        sys.exit("ERROR: 没有可用的 GPU")
    print(f"CUDA: {torch.cuda.is_available()}  |  总 GPU: {total_gpus}  |  使用: {num_gpus}")
    for gi in range(total_gpus):
        print(f"    GPU{gi}: {torch.cuda.get_device_name(gi)}")

    # ── 打印对齐配置 ──
    print("\n[对齐训练的采样/编码参数]")
    print(f"  fps            = {args.fps}   (训练默认 2.0)")
    print(f"  max_frames     = {args.max_frames}   (训练默认 200)")
    print(f"  min_tokens     = {args.min_tokens}   (训练默认 64)")
    print(f"  total_tokens   = {args.total_tokens}   (训练默认 14336)")
    print(f"  attn_impl      = {args.attn_implementation}   (训练默认 flash_attention_2)")
    if args.total_tokens <= 0:
        print(f"  [WARN] total_tokens<=0，走 max-size={args.max_size} 的旧路径，"
              f"和训练默认路径不一致，IoU 可能会差")

    # ── 解析路径 ──
    model_dir = resolve_path(
        args.model_path,
        default=SHARED_MODELS_ROOT / "Qwen3-VL-8B-Instruct",
        must_exist=True,
    )
    anno_json = resolve_path(args.anno_json, must_exist=True)
    video_dir = resolve_path(args.video_dir, must_exist=True)
    print(f"\n模型: {model_dir}")
    print(f"标注: {anno_json}")
    print(f"视频: {video_dir}")

    # ── 加载数据 ──
    samples = load_samples(anno_json, video_dir, args.max_samples)
    if not samples:
        print("[ERROR] 未加载到样本")
        return 1

    # ═══════════════════════════════════════════════════════════════════════════
    # 评估：单卡或多卡数据并行
    # ═══════════════════════════════════════════════════════════════════════════
    t0 = time.time()

    if num_gpus == 1:
        print(f"\n{'─'*64}")
        print(f"  模式: generate (单卡 GPU 0)")
        print(f"{'─'*64}")
        model, processor, _ = load_model_and_processor_single_gpu(
            model_dir, 0, attn_implementation=args.attn_implementation)
        device = torch.device("cuda:0")
        dataset = EvalDataset(samples, video_dir, args.fps,
                              args.max_frames, args.max_size,
                              total_tokens=args.total_tokens)
        print(f"\n推理模式: generate  |  {len(dataset)} 条样本")
        acc, records = evaluate_dataset(model, processor, dataset, device, args)
    else:
        print(f"\n{'─'*64}")
        print(f"  模式: generate (多卡数据并行 ×{num_gpus})")
        print(f"{'─'*64}")
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        splits = [[] for _ in range(num_gpus)]
        for i, s in enumerate(samples):
            splits[i % num_gpus].append(s)
        for gi in range(num_gpus):
            print(f"  GPU{gi}: 分配 {len(splits[gi])} 个样本")

        result_queue = mp.Queue()
        processes = []
        for gi in range(num_gpus):
            if not splits[gi]:
                continue
            p = mp.Process(
                target=_worker_entry,
                args=(gi, model_dir, splits[gi], video_dir, args, result_queue),
            )
            p.start()
            processes.append(p)

        results = []
        for _ in processes:
            gpu_id, acc_w, recs = result_queue.get()
            results.append((gpu_id, acc_w, recs))
        for p in processes:
            p.join()

        acc = Metrics.merge([r[1] for r in results])
        records = [rec for r in sorted(results, key=lambda x: x[0]) for rec in r[2]]

    elapsed = time.time() - t0

    # ── 打印指标 ──
    tag = f"[{args.split}]  {anno_json.name}"
    metrics = acc.print_table(title=tag)

    # ── IoU 分布直方图 ──
    if acc.ious:
        bins = 10
        step = 1.0 / bins
        counts = [0] * bins
        for v in acc.ious:
            counts[min(int(v / step), bins - 1)] += 1
        total = len(acc.ious)
        print("  IoU 分布:")
        for i, c in enumerate(counts):
            lo, hi = i * step, (i + 1) * step
            bar = "█" * int(c / total * 36) if total > 0 else ""
            print(f"    [{lo:.1f},{hi:.1f}) |{bar:<36}| {c:4d} ({c/total*100:5.1f}%)")
        print()

    metrics.update({
        "model_path": str(model_dir),
        "anno_json": str(anno_json),
        "split": args.split,
        "elapsed_sec": round(elapsed, 1),
        "avg_sec_per_sample": round(elapsed / max(len(records), 1), 3),
        "num_gpus": num_gpus,
        "config": {
            "fps": args.fps, "max_frames": args.max_frames,
            "min_tokens": args.min_tokens, "total_tokens": args.total_tokens,
            "attn_implementation": args.attn_implementation,
        },
    })

    # ── 保存 ──
    sp = args.split
    (out_dir / f"metrics_{sp}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2))

    with open(out_dir / f"details_{sp}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fails = [r for r in records
             if r["iou"] == 0.0 and r["status"] not in ("no_video", "oom")]
    with open(out_dir / f"failures_{sp}.jsonl", "w", encoding="utf-8") as f:
        for r in fails:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  → metrics_{sp}.json  |  details_{sp}.jsonl  "
          f"|  failures_{sp}.jsonl ({len(fails)} 零 IoU)")

    print(f"\n输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
