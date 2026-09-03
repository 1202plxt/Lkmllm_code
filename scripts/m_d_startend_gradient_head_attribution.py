"""
m_d_startend_gradient_head_attribution.py
— 纯 GT attention head 的 torchrun 数据并行探测。

每个 rank 在一张 GPU 上加载一份完整模型，并处理全局样本的
rank::world_size 切片；该“一卡一模型、按 rank 分样本”结构保持不变。
探测只使用 SDPA 时间戳 query 对视频内部 GT 区间的 attention 富集分数，
不计算梯度归因、不 backward、不计算 combined。rank 0 按各 head 的有效
样本数合并分数，最终只保留 video_only_head_attribution.json。

输入默认 fps=2、min_tokens=64、total_tokens=14336，不设置额外帧数上限；
token budget 交由 Qwen 视频处理器分配。

八卡运行：

  torchrun --standalone --nproc_per_node 8 \
    scripts/m_d_startend_gradient_head_attribution.py \
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
    --top-k 30

最终结果：
  <output-dir>/video_only_head_attribution.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 必须在 torch 初始化 CUDA 上下文之前设置，缓解显存碎片化导致的
# "reserved but unallocated" OOM（PyTorch 官方建议）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]   # 项目根目录 = Lkmllm_code
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.project_paths import A_DATA_ROOT, ensure_directory, resolve_path, SHARED_MODELS_ROOT


# ═══════════════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════════════

NUM_LAYERS = 36
NUM_HEADS = 28
HEAD_DIM = 128  # hidden_size / num_heads = 3584 / 28


# ═══════════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="仅针对 start/end 时间数字 token 的梯度归因（target logit 目标，36层分批）"
    )
    p.add_argument("--filtered-json", type=str, default=None)
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--video-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str,
                   default=str(A_DATA_ROOT / "outputs" / "startend_gradient_head_attr"))

    # 采样控制
    p.add_argument("--max-samples", type=int, default=50,
                   help="从 jsonl 中最多加载的样本数（候选池大小）。"
                        "配合 --max-valid-samples 使用时应设大一些，"
                        "保证候选池里有足够多的短视频可以过滤。")
    p.add_argument("--max-valid-samples", type=int, default=0,
                   help="实际要跑满的有效样本数（通过时长过滤 + 成功处理的）。"
                        "0 表示不限制，跑完 --max-samples 加载的全部。"
                        "配合 --max-duration 使用：先把 --max-samples 设大"
                        "（如 5000），再用 --max-valid-samples 100 控制"
                        "'跑满 100 个 30s 以内的视频就停'。")
    p.add_argument("--fps", type=float, default=2.0,
                   help="视频采样帧率")
    p.add_argument("--max-frames", type=int, default=0,
                   help="每个视频最多保留的帧数，超出时覆盖完整时间轴均匀采样；"
                        "不足时全部保留。0 表示在 total-tokens>0 时根据 "
                        "2*total_tokens/min_tokens 自动计算。")
    p.add_argument("--max-side", type=int, default=224,
                   help="采帧后 resize 长边上限，控制 visual token 数量")
    p.add_argument("--min-tokens", type=int, default=64,
                   help="每帧最小视觉 token 预算；与微调/评估保持一致")
    p.add_argument("--total-tokens", type=int, default=3584,
                   help="整段视频视觉 token 总预算；>0 时禁用 max-side resize")
    p.add_argument("--max-duration", type=float, default=30.0,
                   help="只对 ffprobe 探测到时长 ≤ 此值（秒）的视频做 head 探测，"
                        "超过则跳过（ffprobe 失败也跳过）。设 0 关闭过滤，处理全部视频。")

    # 归因参数
    p.add_argument("--top-k", type=int, default=20,
                   help="最终输出的 Top-K 关键 Head 数量")
    p.add_argument("--min-attn-ratio", type=float, default=1.0,
                   help="attn_align（baseline_ratio，是随机基线的几倍）的"
                        "绝对下限。低于这个值的 head 说明它对 GT 区间的"
                        "关注还不如完全随机瞎分配，不管按层归一化后排第几，"
                        "combined 分数直接清零、不进 Top-K 候选。默认 1.0"
                        "（严格及格线：不低于随机水平）。设成 0 可关闭这道"
                        "过滤，退化回纯按层归一化排名。")
    p.add_argument("--grad-only-top-k", type=int, default=30,
                   help="额外输出一份纯粹按梯度归因分数（grad_score）单独"
                        "排名的 Top-K 列表，跟 GT 对齐分数完全无关——是跟"
                        "combined score、attn_only 都不同的第三条独立"
                        "候选筛选路径。默认 30，设成 0 可关闭这项输出。")
    p.add_argument("--attn-only-top-k", type=int, default=30,
                   help="额外输出一份纯粹按 GT 对齐分数（attn_align）单独"
                        "排名的 Top-K 列表，跟梯度归因完全无关——是跟"
                        "combined score、grad_only 都不同的第三条独立"
                        "候选筛选路径。默认 30，设成 0 可关闭这项输出。")
    p.add_argument("--layers-per-batch", type=int, default=6,
                   help="每批处理的层数。36 层默认按 6 层一批 → 6 批，"
                        "每批独立做一次 forward+backward，用于控制显存。"
                        "OOM 就调小（如 3 或 2），层数越少批数越多、"
                        "显存占用越低但耗时越长。")

    # 多卡显存控制
    p.add_argument("--gpu-mem-gib", type=float, default=16.0,
                   help="每张 GPU 分配给模型权重的显存上限（GiB）。")
    p.add_argument("--cpu-mem-gib", type=float, default=64.0,
                   help="device_map='auto' 允许溢出到 CPU 的显存上限（GiB）。")

    p.add_argument("--force-output-attentions", action="store_true",
                   help="强制给 model() 调用传 output_attentions=True。"
                        "默认不传，本脚本用 forward hook 自己拿 attn_weights。"
                        "如果你环境里 self_attn 是否返回 attn_weights 依赖这个"
                        "flag（不传就直接 None），遇到 attn_align 全零再打开。")

    return p


# ═══════════════════════════════════════════════════════════════════════════════════
# 视频采帧（与原版一致）
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
        indices = list(range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), step))
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


def uniform_subsample_frames(frames: list, max_frames: int) -> list:
    """覆盖完整时间轴均匀采样；未超过上限时保留全部帧。"""
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    indices = np.rint(
        np.linspace(0, len(frames) - 1, num=max_frames)
    ).astype(np.int64)
    return [frames[int(i)] for i in indices]


def resolve_max_frames(max_frames: int, total_tokens: int,
                       min_tokens: int) -> int:
    """根据 Qwen3-VL 的两帧 temporal merge 推导安全帧数上限。"""
    if max_frames > 0:
        return max_frames
    if total_tokens <= 0 or min_tokens <= 0:
        return 0
    derived = (2 * total_tokens) // min_tokens
    return max(2, derived - derived % 2)


def resize_frames(frames: list, max_side: int = 336) -> list:
    """限制帧的长边，控制 visual token 数量。"""
    from PIL import Image
    resized = []
    for f in frames:
        w, h = f.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            new_w = max(64, int(w * scale))
            new_h = max(64, int(h * scale))
            f = f.resize((new_w, new_h), Image.BILINEAR)
        resized.append(f)
    return resized


def get_duration_ffprobe(video_path) -> Optional[float]:
    """用 ffprobe 读取视频真实时长（秒）。失败返回 None。

    与 analyze_video_duration.py 的 get_duration_ffprobe 一致：直接从视频文件
    元数据拿 duration，而不是依赖 jsonl 里可能不准/缺失的 duration 字段。
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════════
# 模型加载（与原版一致）
# ═══════════════════════════════════════════════════════════════════════════════════

def load_model_and_processor(model_dir: Path, gpu_mem_gib: float = 21.0,
                              cpu_mem_gib: float = 64.0):
    """加载 Qwen3-VL 模型（eager 模式，必须获取 attention weights）。"""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True,
    )
    if hasattr(processor, 'video_processor'):
        processor.video_processor.size['shortest_edge'] = 128
    if hasattr(processor, 'image_processor'):
        processor.image_processor.size['shortest_edge'] = 128

    n_gpus = torch.cuda.device_count()
    max_memory = {}
    if n_gpus >= 1:
        per_gpu_limit = f"{gpu_mem_gib:g}GiB"
        for i in range(n_gpus):
            max_memory[i] = per_gpu_limit
        max_memory["cpu"] = f"{cpu_mem_gib:g}GiB"
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        print(f"  [model] 检测到 {n_gpus} 张 GPU: {gpu_names}")
        print(f"  [model] 均衡加载: max_memory={max_memory}")

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        attn_implementation="eager",  # ← eager 才能拿到 attention weights
        device_map="auto",
        max_memory=max_memory if max_memory else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    n_layers = len(model.model.language_model.layers)

    from collections import Counter
    layer_devices = Counter()
    for i, layer in enumerate(model.model.language_model.layers):
        layer_devices[str(layer.self_attn.o_proj.weight.device)] += 1
    print(f"  [model] 加载完成：{n_layers} layers, {NUM_HEADS} heads, {HEAD_DIM} head_dim")
    print(f"  [model] language_model.layers 分布: {dict(layer_devices)}")
    if n_gpus >= 1:
        for i in range(n_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"    GPU{i} 当前显存: allocated={allocated:.2f}GiB "
                  f"reserved={reserved:.2f}GiB")
    return model, processor, n_layers


def build_inputs(processor, frames: list, query: str, duration: float,
                 device, sample_fps: float,
                 gt_start: float = 0.0, gt_end: float = 0.0,
                 min_tokens: int = 64, total_tokens: int = 3584) -> dict:
    """构建模型输入。返回 processor 输出 dict + prompt_len + answer_text。"""
    from qwen_vl_utils import process_vision_info

    answer_text = _format_answer(query, duration, frames, sample_fps,
                                  gt_start=gt_start, gt_end=gt_end)
    user_text = (
        f"You are given a video with multiple frames. The numbers before each video "
        f"frame indicate its sampling timestamp (in seconds). Please find the visual "
        f"event described by the sentence '{query}', determining its starting and "
        f"ending times. The format should be: "
        f"'The event happens in <start time> - <end time> seconds'."
    )

    video_content = {"type": "video", "video": frames,
                     "sample_fps": sample_fps}
    if total_tokens > 0:
        # Qwen3-VL uses a 16x16 vision patch and a 2x2 spatial merge.
        patch_pixels = 32 * 32
        video_content["min_pixels"] = min_tokens * patch_pixels
        video_content["total_pixels"] = total_tokens * patch_pixels

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": "You are a video time analysis assistant."}
        ]},
        {"role": "user", "content": [
            video_content,
            {"type": "text", "text": user_text},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": answer_text}
        ]},
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
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

    return inputs, text_input, answer_text


def _format_answer(query: str, duration: float, frames, sample_fps: float,
                    gt_start: float = 0.0, gt_end: float = 0.0) -> str:
    """固定模板格式化答案，必须用真实 GT 时间（下面的 start/end token
    定位函数依赖这个固定字符串模板 "The event happens in {s} - {e} seconds."
    来做局部重新分词 + offset 匹配，模板改了要同步改
    locate_start_end_token_positions()。"""
    return f"The event happens in {gt_start:.1f} - {gt_end:.1f} seconds."


def get_token_positions(input_ids: list[int], processor) -> dict:
    """
    解析 Qwen3-VL token 序列中的关键区域（与原版 c_gradient_head_attribution.py
    完全一致，这里只是为了让本脚本独立可运行而复制过来）。
    """
    tok = processor.tokenizer
    vs_id = tok.convert_tokens_to_ids("<|vision_start|>")
    ve_id = tok.convert_tokens_to_ids("<|vision_end|>")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    assistant_id = tok.convert_tokens_to_ids("assistant")

    n = len(input_ids)

    video_segments: list[tuple[int, int]] = []
    cur_start = None
    for i, tid in enumerate(input_ids):
        if tid == vs_id:
            cur_start = i + 1
        elif tid == ve_id and cur_start is not None:
            video_segments.append((cur_start, i))
            cur_start = None

    video_mask = np.zeros(n, dtype=bool)
    for seg_s, seg_e in video_segments:
        video_mask[seg_s:seg_e] = True

    video_start = video_segments[0][0] if video_segments else 0
    video_end = video_segments[-1][1] if video_segments else 0

    text_query_start = video_end + 1 if video_segments else 0
    text_query_end = text_query_start
    for i in range(text_query_start, n):
        if input_ids[i] == im_end_id:
            text_query_end = i
            break
    if text_query_end <= text_query_start:
        text_query_end = text_query_start

    answer_start, answer_end = None, n
    for i in range(n - 1, -1, -1):
        if input_ids[i] == im_start_id:
            if i + 1 < n and input_ids[i + 1] == assistant_id:
                answer_start = i
                break
    if answer_start is not None:
        ans_content_start = answer_start
        for i in range(answer_start, min(answer_start + 5, n)):
            if input_ids[i] == assistant_id:
                ans_content_start = i + 1
                break
        for i in range(ans_content_start, n):
            if input_ids[i] == im_end_id:
                answer_end = i + 1
                break
    else:
        ans_content_start = 0

    text_query_mask = np.zeros(n, dtype=bool)
    text_query_mask[text_query_start:text_query_end] = True

    answer_mask = np.zeros(n, dtype=bool)
    if answer_start is not None:
        answer_mask[answer_start:answer_end] = True

    answer_content_mask = np.zeros(n, dtype=bool)
    content_end = answer_end
    if answer_end > 0 and answer_end <= n and input_ids[answer_end - 1] == im_end_id:
        content_end = answer_end - 1
    if answer_start is not None and content_end > ans_content_start:
        answer_content_mask[ans_content_start:content_end] = True

    n_query_text = text_query_mask.sum()
    n_video = video_mask.sum()
    n_answer_tokens = int(answer_content_mask.sum())

    return {
        "video_start": video_start or 0,
        "video_end": video_end or 0,
        "n_video_tokens": int(n_video),
        "n_video_segments": len(video_segments),
        "text_query_start": text_query_start,
        "text_query_end": text_query_end,
        "n_text_query_tokens": int(n_query_text),
        "answer_start": answer_start or 0,
        "answer_end": answer_end,
        "n_answer_tokens": n_answer_tokens,
        "video_mask": video_mask,
        "text_query_mask": text_query_mask,
        "answer_mask": answer_mask,
        "answer_content_mask": answer_content_mask,
        "input_ids": np.array(input_ids),
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# 新增：定位 start / end 数字 token（本脚本的核心区别之一）
# ═══════════════════════════════════════════════════════════════════════════════════

def locate_start_end_token_positions(input_ids: list[int], token_info: dict,
                                     processor, gt_start: float,
                                     gt_end: float) -> dict:
    """
    在 answer_content_mask 覆盖的 token 区间里，分别定位 start 时间数字
    和 end 时间数字对应的 token（可能各是多个 sub-token）。

    做法（直接基于 input_ids 里真实的那批 token 重建 offset，而不是
    单独对 answer_text 重新分词）：
      1. 取出 answer_content 区间的真实 token id 列表 ids_slice。
      2. 用 tokenizer 对 ids_slice[:k] 逐步累加 decode，得到每个 token
         对应的字符偏移（cumulative decode 的长度差）——这样重建出来的
         offset 直接来自"真实编码在序列里的 token"，不存在脱离上下文
         单独重新分词导致的边界不一致问题（之前用局部重新分词 + 字符
         串匹配，会因为分词器在不同上下文下的边界合并行为不一致，出现
         长度对不上的情况）。
      3. 在重建出的 decoded 全文里找 start_str / end_str 的字符区间，
         再映射回 token 下标。
    """
    tok = processor.tokenizer
    ans_content_mask = token_info["answer_content_mask"]
    ans_positions = np.where(ans_content_mask)[0]
    empty_result = {
        "start_positions": [], "end_positions": [],
        "target_mask": np.zeros(len(input_ids), dtype=bool),
    }
    if len(ans_positions) == 0:
        return empty_result

    ans_content_start = int(ans_positions[0])
    ans_content_end = int(ans_positions[-1]) + 1  # exclusive
    ids_slice = input_ids[ans_content_start:ans_content_end]
    if not ids_slice:
        return empty_result

    # ── 用累积 decode 重建每个 token 的字符区间 ──
    offsets = []
    prev_len = 0
    prev_text = ""
    for k in range(1, len(ids_slice) + 1):
        text = tok.decode(ids_slice[:k], skip_special_tokens=False)
        # decode 结果理论上应该以 prev_text 为前缀单调增长；极少数情况下
        # 分词器会在拼接时对最后一个 token 做上下文相关的微调导致不是
        # 严格前缀，这里退化为直接用新旧文本长度差，不强制校验前缀关系，
        # 避免因为这种边缘情况直接抛异常。
        new_len = len(text)
        offsets.append((prev_len, max(new_len, prev_len)))
        prev_len = max(new_len, prev_len)
        prev_text = text
    full_text = prev_text

    start_str = f"{gt_start:.1f}"
    end_str = f"{gt_end:.1f}"

    try:
        start_char_s = full_text.index(start_str)
        start_char_e = start_char_s + len(start_str)
        end_char_s = full_text.index(end_str, start_char_e)
        end_char_e = end_char_s + len(end_str)
    except ValueError:
        print(f"    [WARN] start/end token 定位：在重建文本"
              f"'{full_text}'里找不到 '{start_str}' / '{end_str}'，跳过该样本。")
        return empty_result

    local_start_idx = [i for i, (cs, ce) in enumerate(offsets)
                        if ce > start_char_s and cs < start_char_e]
    local_end_idx = [i for i, (cs, ce) in enumerate(offsets)
                      if ce > end_char_s and cs < end_char_e]

    if not local_start_idx or not local_end_idx:
        print("    [WARN] start/end token 定位：字符区间未命中任何 token，跳过该样本。")
        return empty_result

    global_start = [ans_content_start + i for i in local_start_idx]
    global_end = [ans_content_start + i for i in local_end_idx]

    n = len(input_ids)
    target_mask = np.zeros(n, dtype=bool)
    for p in global_start + global_end:
        if 0 <= p < n:
            target_mask[p] = True

    return {
        "start_positions": global_start,
        "end_positions": global_end,
        "target_mask": target_mask,
    }


def compute_gt_video_token_range(gt_start: float, gt_end: float,
                                 duration: float, n_video: int) -> Tuple[int, int]:
    """GT [start, end] 按时间比例映射到 video token 子区间（与原版一致）。"""
    ratio_s = max(0.0, gt_start / max(duration, 0.1))
    ratio_e = min(1.0, gt_end / max(duration, 0.1))
    gt_tok_s = int(ratio_s * n_video)
    gt_tok_e = max(gt_tok_s + 1, int(ratio_e * n_video))
    gt_tok_s = min(gt_tok_s, max(n_video - 1, 0))
    gt_tok_e = min(gt_tok_e, n_video)
    return gt_tok_s, gt_tok_e


def compute_target_logit_sum(logits, input_ids, positions: list[int]):
    """
    Teacher-forcing 下 start/end 数字 token 的 target logit 求和。

    对每个 position p（数字 token 在完整序列里的下标），预测它的是
    位置 p-1 的隐状态，所以取 logits[0, p-1, input_ids[0, p]] —— 也就是
    模型在预测这个 token 时，真实答案这个类别拿到的原始 logit 值。
    把 start/end 涉及的所有这类 token 的 logit 加总，得到一个标量，
    用于 backward。与 CE loss（对数概率 + 归一化 + 取负 + 平均）不同，
    这里直接用原始 logit，不做 softmax 归一化，更贴近"这个 head/神经元
    对这个具体类别的原始得分有多大贡献"这个问题。
    """
    total = None
    used = 0
    for p in positions:
        if p <= 0 or p >= logits.shape[1]:
            continue
        tgt_id = input_ids[0, p]
        val = logits[0, p - 1, tgt_id]
        total = val if total is None else total + val
        used += 1
    return total, used


# ═══════════════════════════════════════════════════════════════════════════════════
# 核心：梯度归因器（36 层分批 retain_grad）
# ═══════════════════════════════════════════════════════════════════════════════════

class StartEndHeadAttributor:
    """
    梯度归因器：每次只给一批层挂 hook，独立做一次 forward+backward。

    与原版 GradientHeadAttributor 的区别：
      - attach_hooks() 接受显式的 layer_indices 列表（一批），而不是
        hook_interval 抽样。
      - compute_head_scores_batch() 只用 start/end token 位置做 query，
        grad_score 和 attn_align 两部分都用这批位置，不再用全部 answer
        token。
    """

    def __init__(self, model, n_layers: int, num_heads: int, head_dim: int):
        self.model = model
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.fwd_data: Dict[int, dict] = {}
        self.grad_data: Dict[int, dict] = {}
        self.hooks = []

    def _make_forward_hook(self, layer_idx: int):
        def hook(module, args, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_output = output[0]   # (batch, seq, hidden)
                attn_weights = output[1]  # (batch, heads, seq, seq)

                if attn_weights is not None:
                    self.fwd_data[layer_idx] = {
                        "attn_weights": attn_weights.detach().float().cpu(),
                        "attn_output": attn_output.detach().float().cpu(),
                    }
                else:
                    self.fwd_data[layer_idx] = {
                        "attn_weights": None,
                        "attn_output": attn_output.detach().float().cpu(),
                    }

                def grad_hook(grad):
                    self.grad_data[layer_idx] = {
                        "grad_output": grad.detach().float().cpu(),
                    }
                attn_output.register_hook(grad_hook)

        return hook

    def attach_hooks(self, layer_indices: list[int]):
        """只给指定的这批层挂 hook。"""
        count = 0
        for i, layer in enumerate(self.model.model.language_model.layers):
            if i not in layer_indices:
                continue
            h = layer.self_attn.register_forward_hook(self._make_forward_hook(i))
            self.hooks.append(h)
            count += 1
        print(f"    [hooks] 本批已注册 {count}/{len(layer_indices)} 层 attention hook "
              f"(layers={layer_indices})")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.fwd_data.clear()
        self.grad_data.clear()

    def compute_head_scores_batch(self, layer_indices: list[int],
                                  target_positions: list[int],
                                  n_video: int, gt_tok_s: int,
                                  gt_tok_e: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        针对本批已经 hook 到数据的层，计算 grad_score 与 attn_align——
        query 位置统一用 start/end 数字 token（对这些 token 求均值）。

        grad_score(l,h) = mean_{p∈start∪end}( |fwd_h[p] · grad_h[p]| )

        attn_align(l,h) = mean_{p∈start∪end}( ratio_p )，其中：
          ratio_p = GT_mass_p / baseline_p
          GT_mass_p = sum_{k∈gt_tok_s:gt_tok_e}( attn[h, p, k] )   ← GT 区间
              总共拿到的注意力质量（sum，不是 mean）
          baseline_p = n_gt_tok / (p + 1)  ← 因果 attention 下，query token
              p 一共能看到 (p+1) 个 key（0..p）。如果完全不挑、按 token
              数量均匀分配，GT 这 n_gt_tok 个 key 理论上应该拿到这么多
              份额。ratio_p 就是"实际拿到的份额是均匀分配的几倍"。

          直接用 ratio 而不是原始 GT_mass 的原因：baseline_p 这个分母
          只取决于 query token p 的序列位置和 GT 区间大小，样本之间的
          视频长度、GT 时长占比差异很大，会导致这个分母在样本间大幅
          波动。如果直接对原始 GT_mass 做跨样本平均，长视频/小 GT 占比
          样本和短视频/大 GT 占比样本的数值不可比，均值会被这种上下文
          差异污染；换成 ratio 后再跨样本平均，是在"是随机水平的几倍"
          这个统一单位上比较，才是公平的。

        返回 {layer_idx: (head_grad[num_heads], head_attn[num_heads])}
        """
        results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        pos_arr = np.array(sorted(set(target_positions)), dtype=np.int64)
        if len(pos_arr) == 0:
            return results

        n_gt_tok = max(gt_tok_e - gt_tok_s, 1)

        for l in layer_indices:
            if l not in self.fwd_data or l not in self.grad_data:
                continue

            fwd = self.fwd_data[l]
            grad = self.grad_data[l]

            attn_w = fwd["attn_weights"]
            fwd_out = fwd.get("attn_output")   # (1, seq, hidden) CPU tensor
            grad_out = grad.get("grad_output")  # (1, seq, hidden) CPU tensor

            head_grad = np.zeros(self.num_heads, dtype=np.float32)
            head_attn = np.zeros(self.num_heads, dtype=np.float32)

            if attn_w is not None:
                attn_w_cpu = attn_w[0].numpy()  # (heads, seq, seq)
                n_heads_actual = min(attn_w_cpu.shape[0], self.num_heads)
                seq_len = attn_w_cpu.shape[1]
            else:
                attn_w_cpu = None
                n_heads_actual = self.num_heads
                seq_len = fwd_out.shape[1] if fwd_out is not None else 0

            valid_pos = pos_arr[pos_arr < seq_len] if seq_len > 0 else pos_arr
            # 每个 query token 自己的因果基线：n_gt_tok / (p + 1)
            baseline_per_pos = (n_gt_tok / (valid_pos.astype(np.float64) + 1.0)
                                if len(valid_pos) > 0 else np.array([]))

            for h in range(n_heads_actual):
                # ---- grad_score：Taylor 一阶近似，只在 start/end token 上算 ----
                if grad_out is not None and fwd_out is not None and len(valid_pos) > 0:
                    h_fwd = fwd_out[0, :, h * self.head_dim:(h + 1) * self.head_dim]
                    h_grad = grad_out[0, :, h * self.head_dim:(h + 1) * self.head_dim]
                    pos_fwd = h_fwd[valid_pos]
                    pos_grad = h_grad[valid_pos]
                    taylor = (pos_fwd * pos_grad).sum(dim=-1).abs().mean()
                    head_grad[h] = float(taylor)

                # ---- attn_align：query=start/end token，key=GT 视频区间，
                #      GT 区间总质量 sum 后除以该 query token 的因果基线 ----
                if attn_w_cpu is not None and seq_len >= n_video and len(valid_pos) > 0:
                    head_attn_mat = attn_w_cpu[h]  # (seq, seq)
                    video_slice = head_attn_mat[valid_pos, :n_video]
                    gt_mass = video_slice[:, gt_tok_s:gt_tok_e].sum(axis=1)  # (n_pos,)
                    ratio = gt_mass / np.maximum(baseline_per_pos, 1e-12)
                    head_attn[h] = float(ratio.mean())

            results[l] = (head_grad, head_attn)

        return results


def normalize_and_combine(grad_score: np.ndarray, attn_align: np.ndarray,
                          min_attn_ratio: float = 1.0) -> np.ndarray:
    """按层归一化后取 min，并加一道绝对下限过滤：

    按层 min-max 归一化只回答"在本层 28 个 head 里排第几"，不回答
    "这个值本身好不好"——如果某一层所有 head 的 attn_align（baseline_
    ratio）普遍都低于 1.0（也就是比完全随机瞎分配还更不关注 GT），
    归一化照样会把里面矬子里最高的那个拉到 1.0，跟真正在深层大幅
    超过随机基线（比如 3 倍）的 head 拿到同样的归一化满分，两者在
    combined 分数上就没有区分度了，会把"层内虚高但绝对值不及格"的
    假阳性头排进 Top-K。

    加的这道过滤：凡是原始 attn_align < min_attn_ratio（默认 1.0，
    也就是不如随机基线）的 head，不管它按层归一化后排第几，直接把
    combined 清零，从候选里剔除。
    """
    eps = 1e-9

    def _normalize_per_layer(matrix: np.ndarray) -> np.ndarray:
        normed = np.zeros_like(matrix)
        for l in range(matrix.shape[0]):
            row_max = matrix[l].max()
            if row_max > eps:
                normed[l] = matrix[l] / row_max
        return normed

    combined = np.minimum(_normalize_per_layer(grad_score), _normalize_per_layer(attn_align))
    combined[attn_align < min_attn_ratio] = 0.0
    return combined


def select_top_grad_only(mean_grad: np.ndarray, top_k: int = 30) -> List[dict]:
    """
    纯粹按梯度归因分数（grad_score）单独排名，跟 attn_align 完全无关——
    是跟 combined_score（要求两边都强）、attn_only（纯看 attn）并列的
    第三条独立候选筛选路径中的一条：只看"这个 head 的输出对 target
    logit 的梯度响应强不强"，不管它生成 start/end 数字时 attention 有
    没有真的看向 GT 视频区间。

    做法：先按层 min-max 归一化（纠正深度带来的数值尺度差异——层越浅
    梯度链式累积得越少，原始数值天然偏小，不归一化直接全局排名会系统性
    偏向深层），归一化后在全体 36×28 个 head 里统一排名，取前 top_k。

    这类 head 可能是"确实参与了决定最终答案数字、但注意力模式本身不
    一定直接盯着 GT 视频段"的 head——比如做数值计算/格式化、或者从别的
    head 已经聚合好的信息里做进一步处理，跟 attn_only 选出来的那批
    "负责读取视频信息"的 head 角色可能完全不同。
    """
    eps = 1e-9
    n_layers, n_heads = mean_grad.shape

    grad_norm = np.zeros_like(mean_grad)
    for l in range(n_layers):
        row_max = mean_grad[l].max()
        if row_max > eps:
            grad_norm[l] = mean_grad[l] / row_max

    flat_norm = grad_norm.ravel()
    order = np.argsort(-flat_norm)[:top_k]

    results: List[dict] = []
    for rank, fi in enumerate(order):
        l, h = divmod(int(fi), n_heads)
        results.append({
            "rank": rank + 1,
            "layer": int(l),
            "head": int(h),
            "grad_score": round(float(mean_grad[l, h]), 6),
            "grad_score_norm": round(float(grad_norm[l, h]), 6),
        })
    return results


def select_top_attn_only(mean_attn: np.ndarray, top_k: int = 30,
                         min_attn_ratio: float = 1.0) -> List[dict]:
    """
    纯粹按 GT 对齐分数（attn_align，即"是随机基线的几倍"）单独排名，
    跟 grad_score 完全无关——不是 combined score 那种要求两边都强，
    是跟 combined score、grad_only 并列的第三条独立候选筛选路径：
    只要求"这个 head 生成 start/end 数字时，attention 真的显著聚焦到了
    GT 视频区间"，不管它的梯度响应强不强。

    这类 head 可能是"参与信息读取但本身不是 loss 主要梯度路径"的
    head——比如只负责把视频信息搬运到某个中间表示、真正的决策/写入
    发生在别的 head，这种角色梯度归因未必能捕捉到，但纯粹的 attention
    对齐可以。跟 grad_only、combined score 的候选列表放在一起看，
    可以检查这三条路径选出的 head 有多少重叠、多少是各自独有的。

    过滤：跟其他两条路径一致，attn_align < min_attn_ratio（默认 1.0，
    不如随机基线）的 head 不进候选池。
    """
    n_layers, n_heads = mean_attn.shape
    candidates = [(l, h) for l in range(n_layers) for h in range(n_heads)
                 if mean_attn[l, h] >= min_attn_ratio]
    if not candidates:
        return []

    candidates_sorted = sorted(
        candidates, key=lambda lh: -mean_attn[lh[0], lh[1]],
    )

    results: List[dict] = []
    for rank, (l, h) in enumerate(candidates_sorted[:top_k]):
        results.append({
            "rank": rank + 1,
            "layer": int(l),
            "head": int(h),
            "attn_align": round(float(mean_attn[l, h]), 6),
        })
    return results


def freeze_non_essential_params(model) -> list[str]:
    """
    整个脚本从头到尾都不需要任何"权重"的梯度——我们只通过 hook 拿
    decoder attention 输出这个"激活值"的梯度，backward 不需要经过任何
    参数的 dW 计算。把 vision tower / embed_tokens / lm_head 的参数
    requires_grad 全部关掉：

      - vision tower 处理的帧数往往很多（本例 1568 个 video token），
        如果它的权重还 requires_grad=True，整个视觉编码器的前向都会被
        autograd 记录、为反传保留一份完整的激活值副本——这是比 decoder
        attention weights 更大的显存黑洞，而且每一批（batch）都要重新
        跑一次完整 forward，等于每批都要重复背负这份显存，是当前 OOM
        最主要的来源之一。关掉 requires_grad 之后，只要权重和输入
        pixel_values 都不需要梯度，PyTorch 根本不会为这部分建反传图，
        视觉编码器前向会退化成类似 inference 模式的显存占用。
      - decoder layers（model.model.language_model.layers）的权重保持
        requires_grad=True 不动：正是靠着"当前 batch 起，后续每一层
        权重都 requires_grad=True"，才能让梯度图从被冻结的前置层
        （FreezeLayersNoGrad 包住的 no_grad 层）之后重新连接起来，这是
        整个分批方案能生效的前提，绝对不能一起冻结。
    """
    frozen = []
    visual = None
    for name in ("visual", "vision_tower", "vision_model", "vision_encoder"):
        visual = getattr(model.model, name, None)
        if visual is not None:
            for p in visual.parameters():
                p.requires_grad_(False)
            frozen.append(f"model.{name}")
            break
    if visual is None:
        print("  [WARN] 没找到 vision tower 子模块（试过 visual/vision_tower/"
              "vision_model/vision_encoder），如果显存仍然紧张，请检查模型"
              "结构，手动把视觉编码器权重的 requires_grad 关掉。")

    lm = model.model.language_model
    if hasattr(lm, "embed_tokens"):
        for p in lm.embed_tokens.parameters():
            p.requires_grad_(False)
        frozen.append("language_model.embed_tokens")

    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad_(False)
        frozen.append("lm_head")

    print(f"  [model] 已冻结 requires_grad=False：{frozen}"
          f"（decoder layers 权重保持可训练，用于梯度定位）")
    return frozen


def make_layer_batches(n_layers: int, layers_per_batch: int) -> List[List[int]]:
    return [list(range(i, min(i + layers_per_batch, n_layers)))
            for i in range(0, n_layers, layers_per_batch)]


class FreezeLayersNoGrad:
    """
    临时把"当前批次之前"的那些层的 forward 包进 torch.no_grad()，执行完
    以后再还原，用来真正切断反向传播图（而不是只决定给哪些层挂 hook）。

    重要说明——这个优化只对"批次靠后"的情况有实质性省显存效果：
      - 对某个 batch，它后面（batch 结束层号 ~ 第 35 层）的所有层，
        无论如何都必须保持 requires_grad 参与正常前向，因为 target
        logit 的梯度要一路反传回这个 batch，这部分显存没法省。
      - 只有 batch 之前的那些层可以冻结进 no_grad，省掉它们的
        attention weights / 激活值在反向图里的驻留。
      - 所以 batch 覆盖的层号越靠前（比如第一批 layer 0-5），它后面
        剩下的 30 层依然要全部保留梯度，跟不分批时几乎一样费显存；
        batch 越靠后（比如最后一批 layer 30-35），前面能冻结掉的层
        越多，省显存效果越明显。这是反向传播本身的结构性限制，
        不是实现问题——如果连最前面几层的 batch 都 OOM，靠调小
        --layers-per-batch 是无法绕开的，需要改用更短的视频/更低的
        --max-side、--fps 来压 seq_len（attention weights 显存跟
        seq_len 是平方关系，收益最大）。
    """

    def __init__(self, layers, freeze_indices: list[int]):
        self.layers = layers
        self.freeze_indices = freeze_indices
        self._originals: dict[int, object] = {}

    def __enter__(self):
        import torch

        for i in self.freeze_indices:
            layer = self.layers[i]
            orig_forward = layer.forward
            self._originals[i] = orig_forward

            def wrapped(*args, _orig=orig_forward, **kwargs):
                with torch.no_grad():
                    return _orig(*args, **kwargs)

            layer.forward = wrapped
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, orig in self._originals.items():
            self.layers[i].forward = orig
        self._originals.clear()
        return False


# ═══════════════════════════════════════════════════════════════════════════════════
# 样本加载（与原版一致）
# ═══════════════════════════════════════════════════════════════════════════════════

def _resolve_video_path(video_dir: Path, s: dict) -> Optional[Path]:
    """按样本定位视频文件。优先用 video_rel_path（训练集视频在子目录下），
    找不到再退化为扁平目录下的 {video_id}.{mp4/mkv/avi/webm}。"""
    rel = s.get("video_rel_path")
    if rel:
        vp = video_dir / rel
        if vp.exists():
            return vp
    for ext in [".mp4", ".mkv", ".avi", ".webm"]:
        vp = video_dir / f"{s['video_id']}{ext}"
        if vp.exists():
            return vp
    return None


def load_samples(filtered_json: Path, video_dir: Path,
                 max_samples: int) -> list[dict]:
    """从标注加载样本，自动识别两种格式：

      - 训练集 JSONL（timelens-100k.jsonl）：每行一个对象
          {"video_path": "cosmo_cap/BVs52yd-RUQ.mp4",
           "events": [{"query": "...", "span": [s, e]}, ...]}
        视频按 video_rel_path 子目录存放，通常没有 duration 字段
        （duration 记 0，主循环里用 len(frames)/fps 兜底）。
      - 测试集 dict（filtered_samples.json）：
          {vid: {"duration":.., "spans":[...], "queries":[...]}}
        queries 为 dict 列表或字符串列表。
    """
    text = filtered_json.read_text(encoding="utf-8").strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(raw, dict):
        raw.pop("_meta", None)
        if "数据" in raw:
            raw = raw["数据"]

    sample_list: list[dict] = []

    if isinstance(raw, list):
        # ── 训练集 JSONL：video_path + events ──
        for info in raw:
            if not isinstance(info, dict):
                continue
            vp = info.get("video_path", "")
            if not vp:
                continue
            vid = Path(vp).stem
            video_rel = vp if vp.endswith(".mp4") else vp + ".mp4"
            duration = info.get("duration", info.get("video_duration", 0.0))
            for event in info.get("events", []):
                if len(sample_list) >= max_samples:
                    break
                query = event.get("query", "")
                spans = event.get("span", [])
                if not spans or not query:
                    continue
                span = spans[0] if isinstance(spans[0], (list, tuple)) else spans
                if len(span) < 2:
                    continue
                sample_list.append({
                    "video_id": vid,
                    "video_rel_path": video_rel,
                    "query": query,
                    "duration": float(duration or 0.0),
                    "gt_start": float(span[0]),
                    "gt_end": float(span[1]),
                })
            if len(sample_list) >= max_samples:
                break
    else:
        # ── 测试集 dict：vid -> {duration, spans, queries} ──
        for vid, info in raw.items():
            if not isinstance(info, dict):
                continue
            video_spans = info.get("spans", [])
            duration = info.get("duration", 0.0)
            queries = info.get("queries", [])

            for i, q in enumerate(queries):
                if len(sample_list) >= max_samples:
                    break

                gt_start, gt_end = None, None
                query_text = ""

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
                elif isinstance(q, str):
                    query_text = q
                else:
                    continue

                if not query_text:
                    continue

                if gt_start is None and isinstance(video_spans, list):
                    if i < len(video_spans):
                        span = video_spans[i]
                    elif video_spans:
                        span = video_spans[0]
                    else:
                        span = None
                    if isinstance(span, (list, tuple)) and len(span) >= 2:
                        gt_start = float(span[0])
                        gt_end = float(span[1])

                if gt_start is None:
                    continue

                sample_list.append({
                    "video_id": vid,
                    "video_rel_path": f"{vid}.mp4",
                    "query": query_text,
                    "duration": duration,
                    "gt_start": gt_start,
                    "gt_end": gt_end,
                })
            if len(sample_list) >= max_samples:
                break

    if not sample_list:
        sys.exit("ERROR 无有效样本")

    keep = [s for s in sample_list if _resolve_video_path(video_dir, s) is not None]
    if len(keep) < len(sample_list):
        print(f"  [WARN] {len(sample_list) - len(keep)}/{len(sample_list)} 个视频缺失，跳过")
        sample_list = keep

    return sample_list


# ═══════════════════════════════════════════════════════════════════════════════════
# 可视化（与原版一致）
# ═══════════════════════════════════════════════════════════════════════════════════

def save_heatmaps(combined: np.ndarray, grad_score: np.ndarray,
                  attn_align: np.ndarray, top_k: int,
                  n_samples: int, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = [
        ("Combined Score (Gradient x Attention, start/end tokens)", combined),
        ("Gradient Norm Score (start/end tokens)", grad_score),
        ("Attention Alignment Score (start/end tokens)", attn_align),
    ]

    for fig_title, matrix in titles:
        top_k_mask = np.zeros_like(matrix, dtype=bool)
        flat = matrix.ravel()
        for fi in np.argsort(flat)[::-1][:top_k]:
            l, h = divmod(int(fi), matrix.shape[1])
            top_k_mask[l, h] = True

        fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                                 gridspec_kw={"width_ratios": [3, 1]})
        ax = axes[0]
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        rows, cols = np.where(top_k_mask)
        ax.scatter(cols, rows, marker="*", color="blue", s=80, zorder=5,
                   label=f"Top-{top_k} heads")
        ax.set_xlabel("Head index", fontsize=11)
        ax.set_ylabel("Layer index", fontsize=11)
        ax.set_title(f"{fig_title}\n({n_samples} samples)", fontsize=12)
        ax.legend(fontsize=9)

        ax2 = axes[1]
        layer_max = matrix.max(axis=1)
        layer_mean = matrix.mean(axis=1)
        ax2.plot(layer_max, np.arange(len(layer_max)), marker="o", markersize=4,
                 label="Max score", color="crimson")
        ax2.plot(layer_mean, np.arange(len(layer_mean)), marker="s", markersize=3,
                 label="Mean score", color="steelblue")
        ax2.set_xlabel("Score", fontsize=10)
        ax2.set_ylabel("Layer index", fontsize=10)
        ax2.set_title("Score distribution per layer", fontsize=11)
        ax2.invert_yaxis()
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        slug = fig_title.lower().replace(" ", "_").replace("x", "x").replace("(", "").replace(")", "").replace("/", "_")
        out_path = output_dir / f"startend_gradient_head_{slug}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> 热力图：{out_path}")


def save_per_layer_detail(combined: np.ndarray, grad_score: np.ndarray,
                           attn_align: np.ndarray, top_k: int,
                           output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers, n_heads = combined.shape

    fig, axes = plt.subplots(3, 1, figsize=(24, 12), sharex=True)
    for ax_idx, (title, matrix) in enumerate([
        ("Combined Score", combined),
        ("Gradient Norm Score", grad_score),
        ("Attention Alignment", attn_align),
    ]):
        ax = axes[ax_idx]
        x = np.arange(n_layers)
        top1_vals = matrix.max(axis=1)
        top1_heads = matrix.argmax(axis=1)
        bars = ax.bar(x, top1_vals, color=plt.cm.YlOrRd(top1_vals / max(top1_vals.max(), 1e-9)))
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f"Per-layer best head: {title}", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        for i in range(n_layers):
            if top1_vals[i] > 0:
                ax.annotate(f"H{top1_heads[i]}", (i, top1_vals[i]),
                           textcoords="offset points", xytext=(0, 3),
                           fontsize=6, ha="center", rotation=90, color="darkred")

    axes[-1].set_xlabel("Layer index", fontsize=11)
    axes[-1].set_xticks(range(0, n_layers, 2))
    plt.tight_layout()
    out_path = output_dir / "startend_gradient_head_per_layer.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> 逐层详情图：{out_path}")


# ═══════════════════════════════════════════════════════════════════════════════════
# 显存清理工具（与原版一致）
# ═══════════════════════════════════════════════════════════════════════════════════

def deep_gpu_cleanup(n_gpus: int):
    import torch
    gc.collect()
    if n_gpus <= 0:
        return
    for gi in range(n_gpus):
        with torch.cuda.device(gi):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def print_gpu_memory(n_gpus: int, tag: str = ""):
    import torch
    parts = []
    for gi in range(n_gpus):
        allocated = torch.cuda.memory_allocated(gi) / 1024**3
        reserved = torch.cuda.memory_reserved(gi) / 1024**3
        parts.append(f"GPU{gi}: alloc={allocated:.2f}G/res={reserved:.2f}G")
    print(f"    [mem{(' ' + tag) if tag else ''}] " + "  ".join(parts))


# ═══════════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════════

def _run_local_attribution(argv=None):
    args = build_parser().parse_args(argv)
    import torch

    filtered_json = resolve_path(
        args.filtered_json or (
            A_DATA_ROOT / "outputs" / "pre_sample" / "filtered_samples.json"
        ),
        default=A_DATA_ROOT / "outputs" / "pre_sample" / "filtered_samples.json",
        must_exist=True,
    )
    model_dir = resolve_path(
        args.model_path or (SHARED_MODELS_ROOT / "Qwen3-VL-8B-Instruct"),
        default=SHARED_MODELS_ROOT / "Qwen3-VL-8B-Instruct",
        must_exist=True,
    )
    video_dir = resolve_path(
        args.video_dir or (A_DATA_ROOT / "datasets" / "Test" / "Charades_sta" / "charades"),
        default=A_DATA_ROOT / "datasets" / "Test" / "Charades_sta" / "charades",
        must_exist=True,
    )
    output_dir = ensure_directory(resolve_path(args.output_dir))

    n_gpus = torch.cuda.device_count()
    print(f"CUDA: {torch.cuda.is_available()}  |  GPU 数量: {n_gpus}")
    for gi in range(n_gpus):
        print(f"    GPU{gi}: {torch.cuda.get_device_name(gi)}")

    print(f"\n加载模型（eager 模式，均衡分配到 {n_gpus} 张 GPU）：{model_dir}")
    model, processor, n_layers = load_model_and_processor(
        model_dir, gpu_mem_gib=args.gpu_mem_gib, cpu_mem_gib=args.cpu_mem_gib,
    )
    device = model.device
    freeze_non_essential_params(model)

    sample_list = load_samples(filtered_json, video_dir, args.max_samples)
    print(f"\n候选样本池大小：{len(sample_list)}")
    if args.max_valid_samples > 0:
        print(f"目标有效样本数：{args.max_valid_samples}（攒够即停）")

    layer_batches = make_layer_batches(n_layers, args.layers_per_batch)
    print(f"36 层分为 {len(layer_batches)} 批处理：{layer_batches}")

    attributor = StartEndHeadAttributor(model, n_layers, NUM_HEADS, HEAD_DIM)

    sum_combined = np.zeros((n_layers, NUM_HEADS), dtype=np.float64)
    sum_grad = np.zeros((n_layers, NUM_HEADS), dtype=np.float64)
    sum_attn = np.zeros((n_layers, NUM_HEADS), dtype=np.float64)
    hit_count = np.zeros((n_layers, NUM_HEADS), dtype=np.int64)
    valid_count = 0
    failures = 0
    duration_cache: Dict[str, Optional[float]] = {}
    duration_skipped = 0
    duration_unknown = 0

    print(f"\n{'='*60}")
    print(f"开始 start/end token 梯度归因（候选池={len(sample_list)} 样本，"
          f"目标={args.max_valid_samples if args.max_valid_samples > 0 else '全部'} 有效样本，"
          f"目标=target logit，36层分{len(layer_batches)}批）")
    print(f"{'='*60}")

    t_start = time.time()

    for idx, s in enumerate(sample_list):
        # ── early stop：攒够指定数量的有效样本就停 ──
        if args.max_valid_samples > 0 and valid_count >= args.max_valid_samples:
            print(f"\n  已达到 --max-valid-samples={args.max_valid_samples}，提前停止"
                  f"（扫描了候选池中 {idx}/{len(sample_list)} 个样本）")
            break

        try:
            # 1. 采帧（训练集视频在子目录下，用 _resolve_video_path 定位）
            video_path = _resolve_video_path(video_dir, s)
            if video_path is None:
                print("    SKIP（视频文件缺失）")
                failures += 1
                continue

            # 时长过滤：ffprobe 探测真实时长，只保留 ≤ --max-duration 的视频
            if args.max_duration > 0:
                vp = str(video_path)
                if vp not in duration_cache:
                    duration_cache[vp] = get_duration_ffprobe(video_path)
                dur = duration_cache[vp]
                if dur is None:
                    print("    SKIP（ffprobe 无法获取时长）")
                    duration_unknown += 1
                    continue
                if dur > args.max_duration:
                    print(f"    SKIP（时长 {dur:.1f}s > {args.max_duration:.1f}s）")
                    duration_skipped += 1
                    continue
                s["probed_duration"] = dur

            frames = sample_frames(video_path, args.fps)
            sampled_frame_count = len(frames)
            # 训练集没有 duration 字段时，用采帧结果反推视频时长
            duration = float(s.get("duration") or 0.0)
            if duration <= 0:
                duration = sampled_frame_count / max(args.fps, 1e-6)

            frame_limit = resolve_max_frames(
                args.max_frames, args.total_tokens, args.min_tokens)
            if frame_limit > 0 and sampled_frame_count > frame_limit:
                frames = uniform_subsample_frames(frames, frame_limit)
                print(f"    [frames] {sampled_frame_count} -> {len(frames)} "
                      f"（完整时间轴均匀采样，上限={frame_limit}）")
            if args.total_tokens <= 0:
                frames = resize_frames(frames, max_side=args.max_side)
            if not frames:
                failures += 1
                continue

            # 2. 构建输入
            inputs, text_input, answer_text = build_inputs(
                processor, frames, s["query"], duration, device, args.fps,
                gt_start=s["gt_start"], gt_end=s["gt_end"],
                min_tokens=args.min_tokens, total_tokens=args.total_tokens,
            )
            input_ids_list = inputs["input_ids"][0].tolist()

            # 3. Token 位置解析 + start/end 数字 token 定位
            token_info = get_token_positions(input_ids_list, processor)

            # total_pixels 会受每帧最小分辨率和尺寸取整影响，因此第一次
            # processor 结果仍可能超过 total_tokens。按实测 token 比例继续
            # 减少帧数、覆盖完整时间轴均匀重采样，直到满足硬上限；不能
            # 因为超预算直接丢掉训练样本。
            while (args.total_tokens > 0
                   and token_info["n_video_tokens"] > args.total_tokens):
                old_frame_count = len(frames)
                old_video_tokens = token_info["n_video_tokens"]
                if old_frame_count <= 2:
                    raise RuntimeError(
                        f"最少 2 帧仍产生 {old_video_tokens} video tokens，"
                        f"无法满足 --total-tokens {args.total_tokens}"
                    )

                target_frame_count = int(
                    old_frame_count * args.total_tokens / old_video_tokens)
                # temporal_patch_size=2，保持偶数帧；同时保证每轮至少减少2帧。
                target_frame_count = min(
                    old_frame_count - 2, max(2, target_frame_count))
                target_frame_count -= target_frame_count % 2
                target_frame_count = max(2, target_frame_count)
                frames = uniform_subsample_frames(frames, target_frame_count)
                print(f"    [token-cap] video token {old_video_tokens} > "
                      f"{args.total_tokens}，均匀采样帧数 "
                      f"{old_frame_count} -> {len(frames)}")

                del inputs
                inputs, text_input, answer_text = build_inputs(
                    processor, frames, s["query"], duration, device, args.fps,
                    gt_start=s["gt_start"], gt_end=s["gt_end"],
                    min_tokens=args.min_tokens, total_tokens=args.total_tokens,
                )
                input_ids_list = inputs["input_ids"][0].tolist()
                token_info = get_token_positions(input_ids_list, processor)

            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            startend_info = locate_start_end_token_positions(
                input_ids_list, token_info, processor, s["gt_start"], s["gt_end"],
            )
            target_positions = startend_info["start_positions"] + startend_info["end_positions"]

            seq_len = inputs["input_ids"].shape[1]
            print(f"  [{idx+1}/{len(sample_list)}] "
                  f"(valid={valid_count}/{args.max_valid_samples if args.max_valid_samples > 0 else '∞'}) "
                  f"{s['video_id']} "
                  f"dur={s.get('probed_duration', duration):.1f}s "
                  f"seq={seq_len} video={token_info['n_video_tokens']} "
                  f"ans_tok={token_info['n_answer_tokens']} "
                  f"start_tok={startend_info['start_positions']} "
                  f"end_tok={startend_info['end_positions']} "
                  f"GT=[{s['gt_start']:.1f}, {s['gt_end']:.1f}] "
                  f"'{s['query'][:40]}'")

            if not target_positions:
                print("    SKIP（未能定位 start/end token）")
                del inputs, frames
                deep_gpu_cleanup(n_gpus)
                failures += 1
                continue

            n_video = token_info["n_video_tokens"]
            if n_video <= 0:
                print("    SKIP（无 video token）")
                del inputs, frames
                deep_gpu_cleanup(n_gpus)
                failures += 1
                continue
            gt_tok_s, gt_tok_e = compute_gt_video_token_range(
                s["gt_start"], s["gt_end"], duration, n_video,
            )

            # 4. 分批 forward + backward
            sample_grad = np.zeros((n_layers, NUM_HEADS), dtype=np.float32)
            sample_attn = np.zeros((n_layers, NUM_HEADS), dtype=np.float32)
            input_ids_t = inputs["input_ids"]

            decoder_layers = model.model.language_model.layers

            for batch_idx, layer_batch in enumerate(layer_batches):
                freeze_indices = list(range(0, layer_batch[0]))
                print(f"    [batch {batch_idx+1}/{len(layer_batches)}] "
                      f"layers={layer_batch}  frozen(no_grad)={len(freeze_indices)}层  "
                      f"仍需保留梯度到底={n_layers - layer_batch[0]}层")

                attributor.attach_hooks(layer_batch)

                try:
                    with FreezeLayersNoGrad(decoder_layers, freeze_indices):
                        if args.force_output_attentions:
                            outputs = model(**inputs, output_attentions=True)
                        else:
                            outputs = model(**inputs)
                        logits = outputs.logits  # (1, seq, vocab)

                        target, n_used = compute_target_logit_sum(
                            logits, input_ids_t, target_positions,
                        )
                        if target is None or n_used == 0 or torch.isnan(target):
                            print(f"    batch{batch_idx} SKIP（target 无效）")
                            attributor.remove_hooks()
                            del outputs, logits
                            model.zero_grad(set_to_none=True)
                            deep_gpu_cleanup(n_gpus)
                            continue

                        target.backward()
                except torch.cuda.OutOfMemoryError as oom:
                    print(f"    batch{batch_idx} OOM，跳过这一批"
                          f"（layers={layer_batch}），继续下一批: {oom}")
                    attributor.remove_hooks()
                    model.zero_grad(set_to_none=True)
                    deep_gpu_cleanup(n_gpus)
                    continue

                batch_results = attributor.compute_head_scores_batch(
                    layer_batch, target_positions, n_video, gt_tok_s, gt_tok_e,
                )
                for l, (hg, ha) in batch_results.items():
                    sample_grad[l] = hg
                    sample_attn[l] = ha

                attributor.remove_hooks()
                del outputs, logits, target
                model.zero_grad(set_to_none=True)
                deep_gpu_cleanup(n_gpus)

            print_gpu_memory(n_gpus, tag=f"after sample {idx+1}")

            # 5. 联合分数 + 累积
            sample_combined = normalize_and_combine(
                sample_grad, sample_attn, min_attn_ratio=args.min_attn_ratio,
            )
            sum_grad += sample_grad.astype(np.float64)
            sum_attn += sample_attn.astype(np.float64)
            sum_combined += sample_combined.astype(np.float64)
            valid_count += 1

            for l in range(n_layers):
                row = sample_combined[l]
                row_max = row.max()
                if row_max >= 0.999:
                    for h in np.where(row >= row_max - 1e-6)[0]:
                        hit_count[l, h] += 1

            if sample_combined.max() > 0:
                top3_idx = np.argsort(sample_combined.ravel())[::-1][:3]
                top3_str = ", ".join(
                    f"L{int(fi)//NUM_HEADS}H{int(fi)%NUM_HEADS}={sample_combined.ravel()[fi]:.4f}"
                    for fi in top3_idx
                )
                print(f"    top3: [{top3_str}]")

            del inputs, frames
            deep_gpu_cleanup(n_gpus)

        except Exception as e:
            print(f"  FAIL: {e}")
            attributor.remove_hooks()
            model.zero_grad(set_to_none=True)
            deep_gpu_cleanup(n_gpus)
            failures += 1
            continue

    elapsed = time.time() - t_start

    if valid_count == 0:
        print("ERROR: 无有效样本！")
        return 1

    mean_combined = (sum_combined / valid_count).astype(np.float32)
    mean_grad = (sum_grad / valid_count).astype(np.float32)
    mean_attn = (sum_attn / valid_count).astype(np.float32)

    if mean_grad.max() <= 0:
        print("  [WARN] mean_grad 矩阵全为 0：反向梯度可能没有真正流经 "
              "attention output（检查 backward hook / target logit 是否为 0）。")
    if mean_attn.max() <= 0:
        print("  [WARN] mean_attn 矩阵全为 0：attention 对齐分数没有被计算，"
              "检查 seq_len / n_video / gt_tok_s:gt_tok_e 的边界。")
    if mean_combined.max() <= 0:
        print("  [WARN] mean_combined 矩阵全为 0，Top-K 排名此时没有意义。")

    print(f"\n  [诊断] 跨 {valid_count} 个样本，各 (layer, head) 拿到"
          f"'本层冠军'的命中次数 Top-15：")
    hit_flat = hit_count.ravel()
    hit_top_idx = np.argsort(hit_flat)[::-1][:15]
    print(f"    {'Layer':>5}  {'Head':>4}  {'Hits':>6}  {'Hit_rate':>9}")
    for fi in hit_top_idx:
        l_idx, h_idx = divmod(int(fi), NUM_HEADS)
        hits = int(hit_flat[fi])
        if hits == 0:
            continue
        print(f"    {l_idx:>5}  {h_idx:>4}  {hits:>6}  {hits/valid_count:>8.1%}")

    grad_layer_max = mean_grad.max(axis=1)
    attn_layer_max = mean_attn.max(axis=1)
    print(f"\n  [诊断] 逐层原始最大值（未归一化）：")
    print(f"    {'Layer':>5}  {'Grad_max':>12}  {'Attn_max':>12}")
    for l in range(n_layers):
        print(f"    {l:>5}  {grad_layer_max[l]:>12.6f}  {attn_layer_max[l]:>12.6f}")

    flat = mean_combined.ravel()
    top_k_idx = np.argsort(flat)[::-1][:args.top_k]

    top_k_heads = []
    print(f"\n{'='*60}")
    print(f"Top-{args.top_k} start/end token 梯度归因 Head")
    print(f"{'='*60}")
    print(f"  {'Rank':>4}  {'Layer':>5}  {'Head':>4}  "
          f"{'Combined':>10}  {'Gradient':>10}  {'Attention':>10}")
    for rank, fi in enumerate(top_k_idx):
        l_idx, h_idx = divmod(int(fi), NUM_HEADS)
        top_k_heads.append({
            "rank": rank + 1,
            "layer": l_idx,
            "head": h_idx,
            "combined_score": round(float(mean_combined[l_idx, h_idx]), 6),
            "gradient_score": round(float(mean_grad[l_idx, h_idx]), 6),
            "attention_score": round(float(mean_attn[l_idx, h_idx]), 6),
        })
        print(f"  {rank+1:>4}  {l_idx:>5}  {h_idx:>4}  "
              f"{mean_combined[l_idx, h_idx]:>10.4f}  "
              f"{mean_grad[l_idx, h_idx]:>10.4f}  "
              f"{mean_attn[l_idx, h_idx]:>10.4f}")

    # ── 额外输出：纯粹按梯度归因分数（grad_score）单独排名的 Top-K
    #    （跟 attn 完全无关，是第三条独立候选筛选路径）──
    grad_only_heads: List[dict] = []
    if args.grad_only_top_k > 0:
        grad_only_heads = select_top_grad_only(
            mean_grad,
            top_k=args.grad_only_top_k,
        )
        print(f"\n{'='*60}")
        print(f"Top-{args.grad_only_top_k} 纯梯度归因分数 Head "
              f"(只看 grad_score，跟 GT 对齐无关)")
        print(f"{'='*60}")
        print(f"  {'Rank':>4}  {'Layer':>5}  {'Head':>4}  {'GradNorm':>9}  {'GradRaw':>10}")
        for entry in grad_only_heads:
            print(f"  {entry['rank']:>4}  {entry['layer']:>5}  {entry['head']:>4}  "
                  f"{entry['grad_score_norm']:>9.4f}  {entry['grad_score']:>10.4f}")
        if not grad_only_heads:
            print("  [WARN] 没有输出纯梯度排名列表。")

    # ── 额外输出：纯粹按 GT 对齐分数（attn_align）单独排名的 Top-K
    #    （跟 grad 完全无关，是第三条独立候选筛选路径）──
    attn_only_heads: List[dict] = []
    if args.attn_only_top_k > 0:
        attn_only_heads = select_top_attn_only(
            mean_attn,
            top_k=args.attn_only_top_k,
            min_attn_ratio=args.min_attn_ratio,
        )
        print(f"\n{'='*60}")
        print(f"Top-{args.attn_only_top_k} 纯 GT 对齐分数 Head "
              f"(只看 attn_align，跟梯度归因无关)")
        print(f"{'='*60}")
        print(f"  {'Rank':>4}  {'Layer':>5}  {'Head':>4}  {'AttnRatio':>10}")
        for entry in attn_only_heads:
            print(f"  {entry['rank']:>4}  {entry['layer']:>5}  {entry['head']:>4}  "
                  f"{entry['attn_align']:>10.4f}")
        if not attn_only_heads:
            print("  [WARN] 候选池为空（可能所有 head 的 attn_align 都低于 "
                  "--min-attn-ratio），没有输出纯 attn 排名列表。")

    print(f"\n生成可视化...")
    save_heatmaps(mean_combined, mean_grad, mean_attn,
                  args.top_k, valid_count, output_dir)
    save_per_layer_detail(mean_combined, mean_grad, mean_attn,
                           args.top_k, output_dir)

    result_json = {
        "_meta": {
            "method": "Start/End-token Gradient Attribution "
                      "(target logit, layer-batched retain_grad)",
            "formula": "target = sum_p logits[p-1, gt_token[p]] for p in start/end tokens; "
                       "HIS = E_p[|A_h[p]^T . dTarget/dA_h[p]|]; "
                       "attn_align = mean_p( sum_k(attn[h,p,gt_tok_s:gt_tok_e]) "
                       "/ (n_gt_tok/(p+1)) )  [ratio to causal-uniform baseline]; "
                       "Combined = min(norm(HIS), norm(attn_align))",
            "n_samples_total": len(sample_list),
            "n_valid": valid_count,
            "n_failures": failures,
            "max_samples": args.max_samples,
            "max_valid_samples": args.max_valid_samples,
            "max_duration": args.max_duration,
            "n_duration_skipped": duration_skipped,
            "n_duration_unknown": duration_unknown,
            "top_k": args.top_k,
            "min_attn_ratio": args.min_attn_ratio,
            "grad_only_top_k": args.grad_only_top_k,
            "attn_only_top_k": args.attn_only_top_k,
            "fps": args.fps,
            "max_frames": resolve_max_frames(
                args.max_frames, args.total_tokens, args.min_tokens),
            "min_tokens": args.min_tokens,
            "total_tokens": args.total_tokens,
            "max_side": args.max_side,
            "layers_per_batch": args.layers_per_batch,
            "n_layer_batches": len(layer_batches),
            "num_layers": n_layers,
            "num_heads": NUM_HEADS,
            "n_gpus": n_gpus,
            "gpu_mem_gib": args.gpu_mem_gib,
            "elapsed_seconds": round(elapsed, 1),
            "seconds_per_sample": round(elapsed / max(valid_count, 1), 2),
        },
        "top_k_heads": top_k_heads,
        "grad_only_top_heads": grad_only_heads,
        "attn_only_top_heads": attn_only_heads,
        "combined_score_matrix": mean_combined.tolist(),
        "gradient_score_matrix": mean_grad.tolist(),
        "attention_alignment_matrix": mean_attn.tolist(),
    }

    out_json = output_dir / "startend_gradient_head_attribution.json"
    out_json.write_text(
        json.dumps(result_json, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\n结果 JSON：{out_json}")

    attn_only_top = np.argsort(mean_attn.ravel())[::-1][:args.top_k]
    combined_set = set(int(fi) for fi in top_k_idx)
    attn_set = set(int(fi) for fi in attn_only_top)
    overlap = combined_set & attn_set
    print(f"\n  Combined vs Attention-only Top-{args.top_k} 重叠: {len(overlap)}/{args.top_k}")

    print(f"\n{'='*60}")
    print(f"完成  有效样本={valid_count}  失败={failures}  "
          f"时长过滤跳过={duration_skipped}  时长未知={duration_unknown}  "
          f"耗时={elapsed:.1f}s  ({elapsed/max(valid_count,1):.1f}s/sample)")
    print(f"Top-1 Head: L{top_k_heads[0]['layer']}H{top_k_heads[0]['head']}"
          f"  combined={top_k_heads[0]['combined_score']:.4f}")
    print(f"{'='*60}")
    return 0

def _set_default(argv: List[str], name: str, value: str) -> None:
    if name not in argv:
        argv.extend([name, value])


def _replace_arg(argv: List[str], name: str, value: str) -> None:
    if name in argv:
        idx = argv.index(name)
        if idx + 1 >= len(argv):
            raise ValueError(f"{name} 缺少参数值")
        argv[idx + 1] = value
    else:
        argv.extend([name, value])


def _build_rank_argv(argv: List[str], rank_dir: Path) -> List[str]:
    local_argv = list(argv)
    _set_default(local_argv, "--max-samples", "500")
    _set_default(local_argv, "--max-valid-samples", "0")
    _set_default(local_argv, "--max-duration", "0")
    _set_default(local_argv, "--fps", "2.0")
    _set_default(local_argv, "--min-tokens", "64")
    _set_default(local_argv, "--total-tokens", "14336")
    _set_default(local_argv, "--layers-per-batch", "2")
    _replace_arg(local_argv, "--output-dir", str(rank_dir))
    return local_argv


def _load_full_model_on_local_gpu(base_module, model_dir: Path,
                                  gpu_mem_gib: float = 21.0,
                                  cpu_mem_gib: float = 64.0):
    del gpu_mem_gib, cpu_mem_gib
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    n_layers = len(model.model.language_model.layers)
    print(f"  [model] rank-local 完整模型: cuda:0, "
          f"{n_layers} layers, {base_module.NUM_HEADS} heads")
    return model, processor, n_layers


def _make_distributed_loader(base_module, rank: int, world_size: int):
    original_load_samples = base_module.load_samples

    def distributed_load_samples(filtered_json, video_dir, max_samples):
        all_samples = original_load_samples(filtered_json, video_dir, max_samples)
        local_samples = all_samples[rank::world_size]
        print(f"  [rank {rank}] 全局候选={len(all_samples)}, "
              f"本 rank={len(local_samples)}")
        return local_samples

    return distributed_load_samples


def _top_combined(mean_combined, mean_grad, mean_attn, top_k: int):
    import numpy as np

    indices = np.argsort(mean_combined.ravel())[::-1][:top_k]
    result = []
    for rank, flat_idx in enumerate(indices, start=1):
        layer, head = divmod(int(flat_idx), mean_combined.shape[1])
        result.append({
            "rank": rank,
            "layer": layer,
            "head": head,
            "combined_score": round(float(mean_combined[layer, head]), 6),
            "gradient_score": round(float(mean_grad[layer, head]), 6),
            "attention_score": round(float(mean_attn[layer, head]), 6),
        })
    return result


def _merge_rank_results(base_module, output_dir: Path, rank_root: Path,
                        world_size: int, cli_args) -> Path:
    import numpy as np

    rank_results = []
    for rank in range(world_size):
        path = rank_root / f"rank_{rank}" / "startend_gradient_head_attribution.json"
        if not path.exists():
            raise FileNotFoundError(f"rank {rank} 未生成结果：{path}")
        rank_results.append(json.loads(path.read_text(encoding="utf-8")))

    total_valid = sum(int(r["_meta"]["n_valid"]) for r in rank_results)
    if total_valid <= 0:
        raise RuntimeError("所有 rank 的有效样本数之和为 0")

    def weighted_matrix(key: str):
        total = None
        for result in rank_results:
            weight = int(result["_meta"]["n_valid"])
            matrix = np.asarray(result[key], dtype=np.float64)
            total = matrix * weight if total is None else total + matrix * weight
        return (total / total_valid).astype(np.float32)

    mean_combined = weighted_matrix("combined_score_matrix")
    mean_grad = weighted_matrix("gradient_score_matrix")
    mean_attn = weighted_matrix("attention_alignment_matrix")

    top_k_heads = _top_combined(
        mean_combined, mean_grad, mean_attn, cli_args.top_k,
    )
    grad_only_heads = base_module.select_top_grad_only(
        mean_grad, top_k=cli_args.grad_only_top_k,
    ) if cli_args.grad_only_top_k > 0 else []
    attn_only_heads = base_module.select_top_attn_only(
        mean_attn,
        top_k=cli_args.attn_only_top_k,
        min_attn_ratio=cli_args.min_attn_ratio,
    ) if cli_args.attn_only_top_k > 0 else []

    meta = dict(rank_results[0]["_meta"])
    meta.update({
        "method": meta.get("method", "") + " + torchrun data parallel merge",
        "distributed": True,
        "world_size": world_size,
        "n_gpus": world_size,
        "n_samples_total": sum(
            int(r["_meta"]["n_samples_total"]) for r in rank_results
        ),
        "n_valid": total_valid,
        "n_failures": sum(int(r["_meta"]["n_failures"]) for r in rank_results),
        "n_duration_skipped": sum(
            int(r["_meta"].get("n_duration_skipped", 0)) for r in rank_results
        ),
        "n_duration_unknown": sum(
            int(r["_meta"].get("n_duration_unknown", 0)) for r in rank_results
        ),
        "elapsed_seconds": round(max(
            float(r["_meta"].get("elapsed_seconds", 0.0)) for r in rank_results
        ), 1),
        "per_rank_n_valid": [int(r["_meta"]["n_valid"]) for r in rank_results],
    })

    merged = {
        "_meta": meta,
        "top_k_heads": top_k_heads,
        "grad_only_top_heads": grad_only_heads,
        "attn_only_top_heads": attn_only_heads,
        "combined_score_matrix": mean_combined.tolist(),
        "gradient_score_matrix": mean_grad.tolist(),
        "attention_alignment_matrix": mean_attn.tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "startend_gradient_head_attribution.json"
    out_json.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    base_module.save_heatmaps(
        mean_combined, mean_grad, mean_attn,
        cli_args.top_k, total_valid, output_dir,
    )
    base_module.save_per_layer_detail(
        mean_combined, mean_grad, mean_attn, cli_args.top_k, output_dir,
    )

    print("\n" + "=" * 72)
    print(f"多卡合并完成：world_size={world_size}, valid={total_valid}")
    print(f"最终 JSON：{out_json}")
    if top_k_heads:
        top1 = top_k_heads[0]
        print(f"Top-1: L{top1['layer']}H{top1['head']} "
              f"combined={top1['combined_score']:.6f}")
    print("=" * 72)
    return out_json


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # 必须在首次 import torch/CUDA 前执行。每个 rank 只看见一张物理 GPU，
    # 因而原脚本中的 cuda:0 就是该 rank 自己的卡，不会发生模型分片。
    if world_size > 1:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            devices = [x.strip() for x in visible.split(",") if x.strip()]
            if local_rank >= len(devices):
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} 超出 CUDA_VISIBLE_DEVICES={visible}"
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    import torch
    import torch.distributed as dist
    this_module = sys.modules[__name__]

    if not torch.cuda.is_available():
        raise RuntimeError("多卡 head 探测要求 CUDA")
    torch.cuda.set_device(0)
    if world_size > 1:
        dist.init_process_group(backend="gloo", init_method="env://")

    parsed = this_module.build_parser().parse_args(argv)
    output_dir = Path(parsed.output_dir).expanduser().resolve()
    rank_root = output_dir / "_rank_outputs"
    rank_dir = rank_root / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    this_module.load_model_and_processor = lambda model_dir, gpu_mem_gib=21.0, cpu_mem_gib=64.0: (
        _load_full_model_on_local_gpu(this_module, model_dir, gpu_mem_gib, cpu_mem_gib)
    )
    this_module.load_samples = _make_distributed_loader(this_module, rank, world_size)

    print(f"[distributed] rank={rank}/{world_size}, local_rank={local_rank}, "
          f"visible_gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    rank_argv = _build_rank_argv(argv, rank_dir)
    rc = _run_local_attribution(rank_argv)
    if rc != 0:
        raise RuntimeError(f"rank {rank} head attribution 失败，return code={rc}")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        effective_args = this_module.build_parser().parse_args(rank_argv)
        _merge_rank_results(
            this_module, output_dir, rank_root, world_size, effective_args,
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def _gt_distributed_parser():
    parser = argparse.ArgumentParser(
        description="多卡数据并行纯 GT attention head 探测；每 rank 一张卡一份完整模型")
    parser.add_argument("--filtered-json", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--max-duration", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--total-tokens", type=int, default=14336)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-gt-ratio", type=float, default=1.0)
    parser.add_argument("--score-eps", type=float, default=1e-8)
    parser.add_argument("--timelens-model", action="store_true")
    return parser


def _gt_rank_model_loader(gt_module, model_dir, **_kwargs):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True)
    if hasattr(processor, "video_processor"):
        processor.video_processor.size["shortest_edge"] = 128
    if hasattr(processor, "image_processor"):
        processor.image_processor.size["shortest_edge"] = 128
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": "cuda:0"}, trust_remote_code=True, local_files_only=True)
    model.eval()
    model.config.use_cache = False
    text_config = getattr(model.config, "text_config", model.config)
    text_config.use_cache = False
    n_layers = len(model.model.language_model.layers)
    gt_module.validate_head_layout(model)
    print(f"  [model] rank-local SDPA 完整模型：cuda:0，{n_layers} layers")
    return model, processor, n_layers


def _merge_gt_rank_jsons(rank_root: Path, output_dir: Path, world_size: int,
                         top_k: int, min_ratio: float):
    rank_results = []
    for rank in range(world_size):
        path = rank_root / f"rank_{rank}" / "video_only_head_attribution.json"
        if not path.exists():
            raise FileNotFoundError(f"rank {rank} 未生成结果：{path}")
        rank_results.append(json.loads(path.read_text(encoding="utf-8")))

    matrices, counts = [], []
    total_valid = 0
    for result in rank_results:
        matrix = np.array([
            [np.nan if value is None else float(value) for value in row]
            for row in result["gt_alignment_score_matrix"]], dtype=np.float64)
        count = np.asarray(result["valid_sample_count_matrix"], dtype=np.int64)
        matrices.append(matrix)
        counts.append(count)
        total_valid += int(result["_meta"]["n_valid"])
    sum_counts = np.sum(counts, axis=0)
    weighted = np.zeros_like(matrices[0])
    for matrix, count in zip(matrices, counts):
        weighted += np.where(np.isfinite(matrix), matrix, 0.0) * count
    mean_score = np.divide(weighted, sum_counts,
                           out=np.full_like(weighted, np.nan), where=sum_counts > 0)
    complete = sum_counts == total_valid
    candidates = [(l, h) for l in range(mean_score.shape[0])
                  for h in range(mean_score.shape[1])
                  if complete[l, h] and mean_score[l, h] >= min_ratio]
    candidates.sort(key=lambda lh: (-mean_score[lh], lh[0], lh[1]))
    ranked = [{"rank": i + 1, "layer": l, "head": h,
               "video_gt_ratio": float(mean_score[l, h]),
               "gt_alignment_score": float(mean_score[l, h]),
               "valid_samples": total_valid}
              for i, (l, h) in enumerate(candidates[:top_k])]

    meta = dict(rank_results[0]["_meta"])
    meta.update({
        "distributed": True, "world_size": world_size, "n_gpus": world_size,
        "n_valid": total_valid,
        "n_failures": sum(int(r["_meta"].get("n_failures", 0)) for r in rank_results),
        "per_rank_n_valid": [int(r["_meta"]["n_valid"]) for r in rank_results],
        "gradient_attribution": False, "combined_selection": False,
        "extra_frame_cap": False,
    })
    merged = {
        "_meta": meta,
        "video_only_top_heads": ranked,
        "gt_alignment_score_matrix": [
            [float(v) if np.isfinite(v) else None for v in row] for row in mean_score],
        "valid_sample_count_matrix": sum_counts.tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "video_only_head_attribution.json"
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding="utf-8")
    print(f"\n多卡纯 GT 探测合并完成：valid={total_valid}，结果={output_path}")
    return output_path


def gt_distributed_main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _gt_distributed_parser().parse_args(argv)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            devices = [value.strip() for value in visible.split(",") if value.strip()]
            os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    import torch
    import torch.distributed as dist
    import new_d_head_attribution as gt_module

    if not torch.cuda.is_available():
        raise RuntimeError("多卡 GT head 探测要求 CUDA")
    torch.cuda.set_device(0)
    if world_size > 1:
        dist.init_process_group(backend="gloo", init_method="env://")

    output_dir = Path(args.output_dir).expanduser().resolve()
    rank_root = output_dir / "_rank_outputs"
    rank_dir = rank_root / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    original_load_samples = gt_module.load_samples

    def rank_samples(filtered_json, video_dir, max_samples):
        all_samples = original_load_samples(filtered_json, video_dir, max_samples)
        local = all_samples[rank::world_size]
        print(f"[distributed] rank={rank}/{world_size}：全局={len(all_samples)}，本 rank={len(local)}")
        return local

    gt_module.load_samples = rank_samples
    gt_module.load_model_and_processor = lambda model_dir, **kwargs: (
        _gt_rank_model_loader(gt_module, model_dir, **kwargs))
    # 多卡版按用户要求不设置额外帧数上限；仍按 2 FPS 采样并由 token budget 编码。
    gt_module.resolve_max_frames = lambda max_frames, total_tokens, min_tokens: 0
    rank_argv = [
        "--filtered-json", args.filtered_json,
        "--model-path", args.model_path,
        "--video-dir", args.video_dir,
        "--output-dir", str(rank_dir),
        "--max-samples", str(args.max_samples),
        "--max-valid-samples", str(args.max_valid_samples),
        "--max-duration", str(args.max_duration),
        "--fps", str(args.fps), "--max-frames", "0",
        "--min-tokens", str(args.min_tokens),
        "--total-tokens", str(args.total_tokens),
        "--top-k", str(args.top_k),
        "--min-gt-ratio", str(args.min_gt_ratio),
        "--score-eps", str(args.score_eps),
    ]
    if args.timelens_model:
        rank_argv.append("--timelens-model")
    rc = gt_module.gt_only_main(rank_argv)
    if rc != 0:
        raise RuntimeError(f"rank {rank} GT 探测失败：{rc}")
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        _merge_gt_rank_jsons(rank_root, output_dir, world_size,
                             args.top_k, args.min_gt_ratio)
        import shutil
        shutil.rmtree(rank_root)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(gt_distributed_main())
