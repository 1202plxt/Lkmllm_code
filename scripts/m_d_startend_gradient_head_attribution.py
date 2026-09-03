"""
new_d_head_attribution.py — Qwen3-VL-8B start/end 多 token、pre-o_proj head 归因。

基于 d_startend_gradient_head_attribution.py，新旧脚本互不依赖。
保留原版 CLI 采样/预算/显存默认值、时间戳子 token 定位方法、目标 logit
求和、按 query 取平均、按层最大值归一化后取 min，以及三条 head 排名输出。
仅默认输出目录独立为 outputs/new_d_head_attr，避免覆盖旧结果。

修正：
1. 校验每层 32 个 query heads（不是 8 个 KV heads）；head_dim 从配置读取，
   并检查每层 q/k/v/o 投影尺寸，不再以 28 个 head 静默截断。
2. 对全部 start/end 子 token 位置 P 求同一个目标：
       S = sum_{p in P} logits[0, p-1, input_ids[0,p]]
   激活归因与 GT attention 的 query 则用 Q = {p-1: p in P}。
   不只取第一个数字、不把目标位置重复减一、不改成逐 token 单独 backward。
       grad(l,h) = mean_{q in Q} abs(sum_d z[l,q,h,d] * dS/dz[l,q,h,d])
3. z 是 self_attn.o_proj 的输入，即 attention @ V 按 query heads 拼接后、
   尚未经过输出投影混合的张量；通过 forward_pre_hook 捕获真实计算图，
   不自行重算 RoPE / QK norm。默认 SDPA 在官方接口捕获 post-RoPE Q/K，
   只重算所需 query 行的 softmax（按官方 GQA 分组展开 K）。
4. GT 时间比例映射保持原版，但视频内部编号会映射回 video_mask 中的
   实际序列 key 位置（支持提示词前缀与多段视频 token）。
       ratio(q) = sum_{k in G, k<=q} attention[h,q,k]
                  / (number_of_visible_GT_keys / (q+1))
5. CPU 缓存仅保留所需 query 的激活/梯度和 GT mass，不复制完整 S×S attention。
   --attention-backend eager 对照模式仍有二次显存开销，不适合长视频。
   任一层批次失败的样本不进入全层平均，避免 OOM 层被当成零分。
6. 单进程多 GPU 模型分片：--device-map auto（默认）先在 meta 模型上规划，
   再按 GPU-only map 加载。每卡预留 --gpu-reserve-gib 8 给激活和反向，
   --gpu-mem-gib 16 仅为权重规划上限，不是进程总显存硬限制。
   禁止 CPU/disk 推理式卸载、禁止 torchrun/DDP；CPU 仍用于分数缓存。
7. --save-activations-on-cpu 使用 PyTorch save_on_cpu 暂存反向所需张量，
   权重仍常驻 GPU，反向按需取回原 GPU，不改变目标与梯度定义。
   这会增加主机内存和 PCIe 开销，不受 --cpu-mem-gib（旧参数）限制。
8. 默认 --attention-backend sdpa：全部层使用官方 SDPA 完整前向/反向；
   只有当前探测层额外计算 [1,32,len(Q),seq_len] 注意力行，不建完整 S×S 图。
   理论目标保持一致，但 fused 内核与 eager 的浮点误差可能改变接近分数的排名。
   不允许 SDPA 回退 math kernel；不支持高效内核时显式报错。
9. 独立输出 video_only_head_attribution.json：视频内部 GT 条件富集倍数
   (A_GT/A_video)/(N_visible_GT/N_visible_video)。逐 query 计算、样本内平均、
   样本间等权平均；独立排名不做层内归一化。combined 现仅使用视频内部
   ratio 与梯度分别按层最大值归一化后取 min；旧全上下文 ratio 仅保留独立排名。
   低视频 mass 的 query 记为无效，排名要求完整样本/query 覆盖。
10. 均匀抽帧时保留原始帧索引/native FPS，覆盖帧列表工具的伪时间元数据；
    校验最终输入中的逐时间块标记。GT 比例映射优先采用 ffprobe 时长，
    失败时用原始总帧数/native FPS，不用采样帧数/请求 FPS 推断时长。
    沿用 Qwen 的 frame_index/native_fps 约定；不是可变帧率视频的逐帧 PTS 实现。

官方结构参考（实现顺序为 reshape -> o_proj -> return）：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json

运行：把旧命令的脚本路径换成 scripts/new_d_head_attribution.py 即可；
已有参数名保留，默认仍是 50 个候选、30 秒过滤、3584 token、每批 6 层。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from contextlib import nullcontext
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
NUM_HEADS = 32  # Qwen3-VL-8B query heads；KV heads 不参与此处编号
HEAD_DIM = 128  # 文档常量，实际运行从配置读取并检查投影尺寸


# ═══════════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Qwen3-VL-8B 32-head 归因（全部时间戳 token logit 求和，p-1 query，pre-o_proj）"
    )
    p.add_argument("--filtered-json", type=str, default=None)
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--video-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str,
                   default=str(A_DATA_ROOT / "outputs" / "new_d_head_attr"))
    p.add_argument(
        "--timelens-model", action="store_true",
        help="探测官方 TimeLens-8B（Qwen3-VL 的 GRPO 权重）时开启：使用其官方的"
             "单 user-turn prompt，并保持处理器默认视觉尺寸；GT assistant answer 仍会"
             "附在输入末尾，只用于 teacher-forcing 的 start/end logit 归因。",
    )

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
                   help="仅控制原全上下文 GT ratio 的 attn-only 排名下限；不参与 combined。")
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
    p.add_argument("--video-only-top-k", type=int, default=30,
                   help="独立视频内部 GT ratio 排名数量；0 输出空排名，不关闭 combined。")
    p.add_argument("--min-video-ratio", type=float, default=1.0,
                   help="视频内部 GT 富集倍数下限，用于 combined 逐样本过滤及 video-only 排名。")
    p.add_argument("--min-video-mass", type=float, default=1e-8,
                   help="视频 attention mass 必须大于此值才计算条件 ratio；"
                        "无效项不填零，排名要求所有有效样本的全部 query 均有效。")
    p.add_argument("--layers-per-batch", type=int, default=6,
                   help="每批处理的层数。36 层默认按 6 层一批 → 6 批，"
                        "每批独立做一次 forward+backward，用于控制显存。"
                        "OOM 就调小（如 3 或 2），层数越少批数越多、"
                        "显存占用越低但耗时越长。")

    # 多卡显存控制
    p.add_argument("--attention-backend", choices=["sdpa", "eager"], default="sdpa",
                   help="默认内存高效 SDPA；探测层额外只计算目标 query 行。eager 用于小样本对照。")
    p.add_argument("--device-map", choices=["auto", "balanced", "balanced_low_0", "sequential"],
                   default="auto", help="单进程 GPU 模型分片策略；四卡建议 auto 或 balanced。")
    p.add_argument("--gpu-mem-gib", type=float, default=16.0,
                   help="每张 GPU 分配给模型权重的显存上限（GiB）。")
    p.add_argument("--gpu-reserve-gib", type=float, default=8.0,
                   help="从每卡当前空闲显存中预留给激活/反向的 GiB；不保证不会 OOM。")
    p.add_argument("--save-activations-on-cpu", action="store_true",
                   help="将 autograd 保存的反向张量暂存 CPU；降低 GPU 激活峰值，"
                        "但增加主机内存/传输耗时，模型权重仍在 GPU。")
    p.add_argument("--cpu-mem-gib", type=float, default=64.0,
                   help="保留旧命令兼容；新版归因禁用 CPU 权重卸载，此参数不参与分片。")

    p.add_argument("--force-output-attentions", action="store_true",
                   help="强制给 model() 调用传 output_attentions=True。"
                        "默认不传，本脚本用 forward hook 自己拿 attn_weights。"
                        "如果你环境里 self_attn 是否返回 attn_weights 依赖这个"
                        "flag（不传就直接 None），遇到 attn_align 全零再打开。")

    return p


# ═══════════════════════════════════════════════════════════════════════════════════
# 视频采帧：保留原视频坐标，不以重采样后的连续编号重建时间轴
# ═══════════════════════════════════════════════════════════════════════════════════

def make_video_metadata(native_fps, total_frames, indices, backend):
    """Qwen 的原视频索引/FPS 时间约定；不允许静默以请求采样 FPS 代替。"""
    if not np.isfinite(native_fps) or native_fps <= 0 or total_frames <= 0:
        raise ValueError("视频原始 FPS/总帧数无效，不能可靠构建时间轴")
    if not indices or any(i < 0 or i >= total_frames for i in indices):
        raise ValueError("采样原始帧索引为空或越界")
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("采样原始帧索引必须严格递增")
    return {"fps": float(native_fps), "total_num_frames": int(total_frames),
            "frames_indices": [int(i) for i in indices],
            "duration": float(total_frames / native_fps), "video_backend": backend}


def sample_frames(video_path: Path, fps: float) -> Tuple[list, dict]:
    """返回 PIL 帧及原始索引/FPS 元数据。抽帧策略保留原实现。"""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("采样 FPS 必须为有限正数")
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        native_fps = vr.get_avg_fps()
        if not np.isfinite(native_fps) or native_fps <= 0:
            raise ValueError("decord 未提供有效原始 FPS")
        step = max(1, int(native_fps / fps))
        indices = list(range(0, len(vr), step))
        metadata = make_video_metadata(native_fps, len(vr), indices, "decord")
        frames_t = vr.get_batch(indices)
        from PIL import Image
        return [Image.fromarray(frames_t[i].numpy()) for i in range(len(indices))], metadata
    except ImportError:
        pass
    try:
        import cv2
        from PIL import Image
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频：{video_path}")
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(native_fps) or native_fps <= 0:
                raise ValueError("OpenCV 未提供有效原始 FPS")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, int(native_fps / fps))
            indices = list(range(0, total_frames, step))
            metadata = make_video_metadata(native_fps, total_frames, indices, "opencv")
            frames = []
            for idx in indices:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
                    raise RuntimeError(f"OpenCV 无法定位原始帧 {idx}")
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"OpenCV 解码原始帧 {idx} 失败；不使用截短视频")
                decoded_idx = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                if decoded_idx != idx:
                    raise RuntimeError(f"OpenCV 帧定位不一致：请求 {idx}，实际 {decoded_idx}")
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            return frames, metadata
        finally:
            cap.release()
    except ImportError:
        pass
    sys.exit("ERROR 请安装 decord 或 opencv-python")


def uniform_subsample_frames(frames: list, max_frames: int, metadata: dict) -> Tuple[list, dict]:
    """图像和原视频索引同步均匀抽样，保留原 FPS、总帧数和 duration。"""
    if len(frames) != len(metadata["frames_indices"]):
        raise ValueError("帧数与原始索引数量不一致")
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames, {**metadata, "frames_indices": list(metadata["frames_indices"])}
    indices = np.rint(
        np.linspace(0, len(frames) - 1, num=max_frames)
    ).astype(np.int64)
    return ([frames[int(i)] for i in indices],
            {**metadata, "frames_indices": [metadata["frames_indices"][int(i)] for i in indices]})


def metadata_for_processor(metadata: dict, frame_count: int, processed_count: int) -> dict:
    """覆盖 qwen-vl-utils 的伪元数据；允许奇数帧末尾复制一次的官方补齐。"""
    indices = list(metadata["frames_indices"])
    if not indices or len(indices) != frame_count:
        raise ValueError("processor 输入帧数与原始索引不一致")
    if processed_count not in {frame_count, frame_count + frame_count % 2}:
        raise RuntimeError("视觉预处理改变了帧数，无法保证原始时间索引对应")
    indices.extend([indices[-1]] * (processed_count - frame_count))
    return {**metadata, "frames_indices": indices}


def verify_video_timestamps(processor, inputs, metadata: dict):
    """检查最终 token 中的时间标记，避免依赖版本静默丢失原始元数据。"""
    merge = int(processor.video_processor.temporal_patch_size)
    indices = list(metadata["frames_indices"])
    indices.extend([indices[-1]] * ((-len(indices)) % merge))
    expected = [f"{(indices[i] / metadata['fps'] + indices[i + merge - 1] / metadata['fps']) / 2:.1f}"
                for i in range(0, len(indices), merge)]
    decoded = processor.tokenizer.decode(inputs["input_ids"][0].tolist(), skip_special_tokens=False)
    # 只匹配紧邻视觉起始 token 的时间标签，避免 query 文字中出现同格式造成误判。
    actual = re.findall(r"<([0-9]+(?:\.[0-9]+)?) seconds>\s*<\|vision_start\|>", decoded)
    if actual != expected:
        raise RuntimeError(f"processor 时间标记不匹配原视频索引："
                           f"expected={expected[:3]}...{expected[-3:]} ({len(expected)}), "
                           f"actual={actual[:3]}...{actual[-3:]} ({len(actual)})")
    return expected


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


def validate_head_layout(model) -> Tuple[int, int, int]:
    """从 text_config 读取并逐层校验 query/KV head 布局；拒绝旧 28-head 模型。"""
    cfg = getattr(model.config, "text_config", model.config)
    num_heads = int(cfg.num_attention_heads)
    if num_heads != NUM_HEADS:
        raise ValueError(
            f"本脚本要求 Qwen3-VL-8B 的 {NUM_HEADS} query heads，"
            f"但模型配置是 {num_heads}；请检查 --model-path。"
        )
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        if int(cfg.hidden_size) % num_heads:
            raise ValueError("hidden_size 不能整除 num_attention_heads")
        head_dim = int(cfg.hidden_size) // num_heads
    head_dim = int(head_dim)
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
    if head_dim <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads:
        raise ValueError("无效的 head_dim / GQA 配置")
    for i, layer in enumerate(model.model.language_model.layers):
        expected = {
            "q_proj": (num_heads * head_dim, int(cfg.hidden_size)),
            "k_proj": (num_kv_heads * head_dim, int(cfg.hidden_size)),
            "v_proj": (num_kv_heads * head_dim, int(cfg.hidden_size)),
            "o_proj": (int(cfg.hidden_size), num_heads * head_dim),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(layer.self_attn, name).weight.shape)
            if actual != shape:
                raise ValueError(f"L{i}.{name} shape={actual}，配置要求 {shape}")
    return num_heads, head_dim, num_kv_heads


def get_prediction_positions(target_positions: list[int], seq_len: int) -> list[int]:
    """所有有效目标 token 各自左移一次；logit 函数仍接收原始目标位置 P。"""
    return sorted({int(p) - 1 for p in target_positions if 0 < int(p) < seq_len})


def activation_storage_context(enabled: bool):
    """只卸载 autograd 保存的张量，不启用推理式参数卸载、不切断梯度图。

    使用 pageable RAM（pin_memory=False），避免大规模长期锁页影响主机。
    注意 save_on_cpu 也可能保存反向所需的权重副本，主机 RAM 需求并非只有激活。
    官方接口：https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu
    """
    if not enabled:
        return nullcontext()
    import torch
    return torch.autograd.graph.save_on_cpu(pin_memory=False)


def attention_kernel_context(backend: str):
    """禁止 SDPA 静默回退到二次显存的 math 内核；不支持则显式报错。"""
    if backend == "eager":
        return nullcontext()
    from torch.nn.attention import sdpa_kernel, SDPBackend
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])


def video_ratios_from_masses(gt_mass, video_mass, gt_counts, video_counts,
                            min_video_mass=1e-8):
    """[H,Q] 条件富集倍数；float64，无分母 epsilon、无裁剪、无层内归一化。

    极低视频 mass、不可见 GT/video 或非有限输入返回 NaN，由新排名单独记覆盖数。
    不改变旧分数或旧样本有效性。GT/video counts 均指该 query 因果可见的 key。
    """
    if not np.isfinite(min_video_mass) or not 0 <= min_video_mass <= 1:
        raise ValueError("min_video_mass 必须在 [0,1] 内")
    g, v = np.broadcast_arrays(np.asarray(gt_mass, dtype=np.float64),
                               np.asarray(video_mass, dtype=np.float64))
    ng, nv = np.broadcast_arrays(np.asarray(gt_counts, dtype=np.float64),
                                 np.asarray(video_counts, dtype=np.float64))
    valid = (np.isfinite(g) & np.isfinite(v) & (g >= 0) & (v > min_video_mass)
             & (g <= v) & np.isfinite(ng) & np.isfinite(nv) & (ng > 0) & (nv >= ng))
    result = np.full(g.shape, np.nan, dtype=np.float64)
    # 两次条件归一化必须同时做，不能仅将 N_all 替换为 N_video。
    share = np.divide(g, v, out=np.zeros_like(g), where=valid)
    baseline = np.divide(ng, nv, out=np.zeros_like(ng), where=(nv > 0))
    np.divide(share, baseline, out=result, where=valid)
    return result


def video_attention_statistics(rows, query_positions, video_positions,
                               gt_key_positions, min_video_mass=1e-8):
    """rows=[H,Q,S]：复用已取得的 attention 行，只传回小型 CPU 统计。"""
    import torch
    with torch.no_grad():
        q = torch.tensor(query_positions, device=rows.device)
        v = torch.tensor(video_positions, device=rows.device)
        g = torch.tensor(gt_key_positions, device=rows.device)
        v_visible = v[None, :] <= q[:, None]
        g_visible = g[None, :] <= q[:, None]
        # 在 attention 原始精度之上以 float64 求和，减少条件比值的累计误差。
        vm = (rows.index_select(-1, v) * v_visible[None]).sum(-1, dtype=torch.float64).cpu().numpy()
        gm = (rows.index_select(-1, g) * g_visible[None]).sum(-1, dtype=torch.float64).cpu().numpy()
        gt_counts = g_visible.sum(-1).cpu().numpy().astype(np.float64)
        video_counts = v_visible.sum(-1).cpu().numpy().astype(np.float64)
        ratio = video_ratios_from_masses(
            gm, vm, gt_counts, video_counts, min_video_mass)
        baseline = np.divide(gt_counts, video_counts,
                             out=np.zeros_like(gt_counts), where=video_counts > 0)
        return {"ratio": ratio, "video_mass": vm, "gt_mass": gm,
                "gt_fraction_baseline": baseline}


def sample_gt_alignment_score(stats: dict, eps: float = 1e-8) -> np.ndarray:
    """截图公式：样本内先对 query 平均，再做 GT/video mass 与基线比值。"""
    gt_mass = np.asarray(stats["gt_mass"], dtype=np.float64)
    video_mass = np.asarray(stats["video_mass"], dtype=np.float64)
    baseline = np.asarray(stats["gt_fraction_baseline"], dtype=np.float64)
    if gt_mass.ndim != 2 or video_mass.shape != gt_mass.shape:
        raise ValueError("GT/video attention mass 必须为匹配的 [H,Q]")
    if baseline.ndim != 1 or baseline.shape[0] != gt_mass.shape[1]:
        raise ValueError("均匀基线必须为匹配的 [Q]")
    mean_video = video_mass.mean(axis=1)
    valid = (np.isfinite(gt_mass).all(axis=1) & np.isfinite(video_mass).all(axis=1)
             & np.isfinite(baseline).all() & (mean_video > eps)
             & (baseline.mean() > 0))
    score = np.full(gt_mass.shape[0], np.nan, dtype=np.float64)
    score[valid] = ((gt_mass.mean(axis=1)[valid] / (mean_video[valid] + eps))
                    / (baseline.mean() + eps))
    return score


def selected_attention_ratios(module, query, key, attention_mask,
                              query_positions, gt_key_positions, scaling,
                              video_positions=None, min_video_mass=1e-8):
    """直接使用官方 attention 接口处已完成 QK norm/RoPE 的 Q/K。

    仅生成 [B,H,Q_selected,S]。分母覆盖所有可见 key，而非只在 GT 内 softmax。
    仅支持本脚本 B=1、无 KV cache 的完整序列，不重建位置编码或改变模型前向。
    """
    import torch
    if query.shape[0] != 1 or key.shape[0] != 1 or query.shape[2] != key.shape[2]:
        raise RuntimeError("selected-row attribution requires B=1, full sequence, no KV cache")
    with torch.no_grad():
        q_idx = torch.tensor(query_positions, device=query.device)
        g_idx = torch.tensor(gt_key_positions, device=query.device)
        q = query.detach().index_select(2, q_idx)
        k = key.detach()
        if q.shape[1] % k.shape[1]:
            raise RuntimeError("Invalid query/KV head ratio")
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        scale = scaling if scaling is not None else query.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        visible = torch.arange(k.shape[2], device=query.device)[None, :] <= q_idx[:, None]
        if attention_mask is not None:
            if attention_mask.ndim != 4 or attention_mask.shape[-1] < k.shape[2]:
                raise RuntimeError("Expected None or 4D SDPA attention mask")
            mask = attention_mask[..., :k.shape[2]]
            if mask.shape[-2] != 1:
                if mask.shape[-2] != query.shape[2]:
                    raise RuntimeError("Attention mask query length mismatch")
                mask = mask.index_select(-2, q_idx.to(mask.device))
            mask = mask.to(scores.device)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask
        # In this pipeline the LM is causal, including when SDPA omits its mask.
        if not getattr(module, "is_causal", True):
            raise RuntimeError("Only causal text attention can be probed")
        scores = scores.masked_fill(~visible[None, None], float("-inf"))
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        gt_visible = visible.index_select(-1, g_idx)
        mass = (weights.index_select(-1, g_idx).float() * gt_visible[None, None]).sum(-1)
        baseline = gt_visible.sum(-1).float() / (q_idx.float() + 1)
        if (baseline <= 0).any():
            raise RuntimeError("No causally visible GT keys")
        original_ratio = (mass[0] / baseline[None]).cpu()
        if video_positions is None:
            return original_ratio
        return original_ratio, video_attention_statistics(
            weights[0], query_positions, video_positions, gt_key_positions, min_video_mass)


def gpu_weight_budget_bytes(free_bytes: int, total_bytes: int,
                            gpu_mem_gib: float, reserve_gib: float) -> int:
    """权重预算=min(用户上限, 当前空闲-激活预留)，不是进程显存限制。"""
    if not np.isfinite(gpu_mem_gib) or gpu_mem_gib <= 0:
        raise ValueError("--gpu-mem-gib 必须是有限正数")
    if not np.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("--gpu-reserve-gib 必须是有限非负数")
    gib = 1024 ** 3
    budget = min(int(gpu_mem_gib * gib),
                 min(int(free_bytes), int(total_bytes)) - int(reserve_gib * gib))
    if budget < gib:
        raise ValueError("显存预留后可用于权重的空间不足 1GiB；请释放显存或调整预留量")
    return budget


def validate_gpu_only_map(device_map: dict, n_gpus: int) -> set[int]:
    """拒绝推理式 CPU/disk 卸载；其权重迁移行为不能保证归因反向图。"""
    if not device_map:
        raise ValueError("自动 device map 为空")
    used = set()
    for name, device in device_map.items():
        if isinstance(device, int):
            index = device
        else:
            label = str(device)
            if not label.startswith("cuda:") or not label[5:].isdigit():
                raise ValueError(
                    f"自动分片把 {name or '<root>'} 放到了 {device}；"
                    "归因需要 GPU 常驻权重，禁止 CPU/disk/meta 卸载。"
                    "请检查可见 GPU、释放显存或适度增加 --gpu-mem-gib。")
            index = int(label[5:])
        if not 0 <= index < n_gpus:
            raise ValueError(f"device map 中 GPU {index} 不在可见 GPU 范围内")
        used.add(index)
    return used


def device_map_for_metadata(model) -> dict:
    """单卡实例可能没有 hf_device_map；优先实际 map，其次加载时保存的 map。"""
    for attr in ("hf_device_map", "_head_attr_device_map"):
        mapping = getattr(model, attr, None)
        if mapping:
            return {str(k): str(v) for k, v in mapping.items()}
    # 对外部加载的实例兜底，从实际参数读取设备，而不是臆造 cuda:0。
    return {name: str(param.device) for name, param in model.named_parameters()}


def plan_gpu_device_map(model_dir: Path, strategy: str, max_memory: dict):
    """遵循 HF meta-init -> balanced budgets -> infer map 流程，加载前拒绝卸载。"""
    import torch
    from accelerate import init_empty_weights, infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True)
    with init_empty_weights():
        empty = AutoModelForImageTextToText.from_config(
            config, trust_remote_code=True, dtype=torch.bfloat16,
            attn_implementation="eager")
    empty.tie_weights()
    validate_head_layout(empty)
    # 使用模型声明的残差 block 边界，避免把同一 decoder block 切到两张卡。
    no_split = sorted({name for module in empty.modules()
                       for name in (getattr(module, "_no_split_modules", None) or [])})
    no_split = sorted(set(no_split) | {"Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"})
    budgets = dict(max_memory)
    if strategy != "sequential":
        budgets = get_balanced_memory(
            empty, max_memory=budgets, no_split_module_classes=no_split,
            dtype=torch.bfloat16, low_zero=(strategy == "balanced_low_0"))
    planned = infer_auto_device_map(
        empty, max_memory=budgets, no_split_module_classes=no_split,
        dtype=torch.bfloat16)
    validate_gpu_only_map(planned, torch.cuda.device_count())
    del empty
    return planned


def load_model_and_processor(model_dir: Path, gpu_mem_gib: float = 16.0,
                              cpu_mem_gib: float = 64.0,
                              device_map: str = "auto", gpu_reserve_gib: float = 8.0,
                              attention_backend: str = "sdpa",
                              timelens_model: bool = False):
    """单进程 GPU-only 自动模型分片，保留 eager attention 的反向传播图。"""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("请用 CUDA_VISIBLE_DEVICES=0,1,2,3 python ... 单进程运行，不能用 torchrun/DDP")
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise RuntimeError("未检测到 CUDA GPU；本脚本的自动模型分片需要可见 GPU")
    if device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
        raise ValueError(f"不支持的 device-map 策略：{device_map}")

    processor_kwargs = dict(trust_remote_code=True, local_files_only=True)
    if timelens_model:
        # 与 e_head_eval.py 的官方 TimeLens-8B 路径保持一致，避免人为缩放
        # 覆盖该 checkpoint 自带的视觉预处理配置。
        processor_kwargs.update(padding_side="left", do_resize=False)
    processor = AutoProcessor.from_pretrained(str(model_dir), **processor_kwargs)
    if not timelens_model:
        if hasattr(processor, 'video_processor'):
            processor.video_processor.size['shortest_edge'] = 128
        if hasattr(processor, 'image_processor'):
            processor.image_processor.size['shortest_edge'] = 128

    max_memory = {"cpu": 0}
    for i in range(n_gpus):
        with torch.cuda.device(i):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        max_memory[i] = gpu_weight_budget_bytes(
            free_bytes, total_bytes, gpu_mem_gib, gpu_reserve_gib)
        print(f"  [model] GPU{i} {torch.cuda.get_device_name(i)}: "
              f"free={free_bytes / 1024**3:.2f}GiB, "
              f"weight_budget={max_memory[i] / 1024**3:.2f}GiB, "
              f"activation_reserve={gpu_reserve_gib:g}GiB")
    print(f"  [model] {device_map} GPU-only 分片；--cpu-mem-gib={cpu_mem_gib:g} "
          "仅为旧命令兼容，不启用 CPU 权重卸载")
    planned_map = plan_gpu_device_map(model_dir, device_map, max_memory)
    print("  [model] 自动分片规划：" + json.dumps(planned_map, ensure_ascii=False, default=str))

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        attn_implementation=attention_backend,
        device_map=planned_map,
        max_memory=max_memory,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    model.config.use_cache = False
    text_config = getattr(model.config, "text_config", model.config)
    text_config.use_cache = False
    actual_map = getattr(model, "hf_device_map", planned_map)
    model._head_attr_device_map = dict(actual_map)
    used_gpus = validate_gpu_only_map(actual_map, n_gpus)
    non_cuda_params = [name for name, param in model.named_parameters()
                       if param.device.type != "cuda"]
    if non_cuda_params:
        raise RuntimeError(f"分片加载后仍有非 CUDA 参数：{non_cuda_params[:5]}")
    print(f"  [model] 实际使用 GPU={sorted(used_gpus)}，可见 GPU 数={n_gpus}")
    if len(used_gpus) < n_gpus:
        print("  [WARN] 自动规划未使用所有可见 GPU；检查实际 map，必要时使用 --device-map balanced")
    n_layers = len(model.model.language_model.layers)

    from collections import Counter
    layer_devices = Counter()
    for i, layer in enumerate(model.model.language_model.layers):
        layer_devices[str(layer.self_attn.o_proj.weight.device)] += 1
    num_heads, head_dim, num_kv_heads = validate_head_layout(model)
    print(f"  [model] 加载完成：{n_layers} layers, {num_heads} query heads, "
          f"{num_kv_heads} KV heads, {head_dim} head_dim")
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
                 min_tokens: int = 64, total_tokens: int = 3584,
                 video_metadata: Optional[dict] = None,
                 timelens_model: bool = False) -> dict:
    """构建并校验含原视频时间轴的输入；原始 metadata 必须显式传入。"""
    from qwen_vl_utils import process_vision_info
    if video_metadata is None:
        raise ValueError("必须提供原始视频帧索引/FPS，不能按采样后帧号伪造时间轴")

    answer_text = _format_answer(query, duration, frames, sample_fps,
                                  gt_start=gt_start, gt_end=gt_end)
    if timelens_model:
        # TimeLens-8B 的 GRPO / 官方评测 prompt；不要加入基础模型使用的 system
        # prompt 或“帧前时间编号”说明，否则探测分布会和推理分布不一致。
        user_text = (
            f"Please find the visual event described by the sentence '{query}', determining its "
            "starting and ending times. The format should be: 'The event happens in "
            "<start time> - <end time> seconds'."
        )
    else:
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

    user_turn = {"role": "user", "content": [
        video_content,
        {"type": "text", "text": user_text},
    ]}
    assistant_turn = {"role": "assistant", "content": [
        {"type": "text", "text": answer_text}
    ]}
    messages = ([user_turn, assistant_turn] if timelens_model else [
        {"role": "system", "content": [
            {"type": "text", "text": "You are a video time analysis assistant."}
        ]}, user_turn, assistant_turn
    ])

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
        raise RuntimeError("视觉预处理没有返回视频")
    if len(video_tensors) != 1:
        raise RuntimeError("当前归因仅支持每条样本一个视频")
    # 工具对 PIL 列表自动生成连续编号；必须覆盖为我们保存的原视频坐标。
    source_metadata = metadata_for_processor(video_metadata, len(frames), int(video_tensors[0].shape[0]))
    video_metadatas = [source_metadata]

    scalar_video_kwargs = {
        k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
        for k, v in video_kwargs.items()
    }
    scalar_video_kwargs.pop("fps", None)
    scalar_video_kwargs["do_sample_frames"] = False

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_tensors,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        **scalar_video_kwargs,
    )
    timestamps = verify_video_timestamps(processor, inputs, source_metadata)
    print(f"    [time] 原始 fps={source_metadata['fps']:.6g}，"
          f"source_indices={source_metadata['frames_indices'][0]}..{source_metadata['frames_indices'][-1]}，"
          f"模型时间标记={timestamps[0]}..{timestamps[-1]}s，"
          f"共 {len(timestamps)} 个时间块（已校验）")

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
        # input_ids 与 logits 在模型分片时可能位于不同 GPU。
        # 使用 Python token ID，避免用另一张 GPU 的标量张量做索引。
        tgt_id = int(input_ids[0, p].item())
        val = logits[0, p - 1, tgt_id]
        total = val if total is None else total + val
        used += 1
    return total, used


# ═══════════════════════════════════════════════════════════════════════════════════
# 核心：梯度归因器（36 层分批 retain_grad）
# ═══════════════════════════════════════════════════════════════════════════════════

class StartEndHeadAttributor:
    """归因到官方 o_proj 输入 [B,S,H*D]；这里每个连续 D 维才是一个 head。

    对全部时间戳子 token 求联合 logit 目标 S，在全部 p-1 query 上取均值。
    eager 对照使用官方返回的 attention；sdpa 从官方 post-RoPE Q/K 计算目标行。
    全局 SDPA registry 只在当前批次临时包装，清理时恢复；本脚本单线程运行。
    """
    def __init__(self, model, n_layers: int, num_heads: int, head_dim: int,
                 attention_backend: str = "eager", min_video_mass: float = 1e-8):
        self.model = model
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.fwd_data: Dict[int, dict] = {}
        self.grad_data: Dict[int, dict] = {}
        self.hooks = []
        self.tensor_hooks = []
        self.query_positions: list[int] = []
        self.gt_key_positions: list[int] = []
        self.video_positions: list[int] = []
        self.min_video_mass = min_video_mass
        self.seq_len = 0
        self.n_video = 0
        self.gt_range = None
        self.attention_backend = attention_backend
        self._attention_registry = None
        self._original_sdpa = None

    def set_sample_context(self, target_positions, video_positions,
                           gt_tok_s, gt_tok_e, seq_len):
        """GT 范围是 video 内部索引；映射成 attention 的绝对 key 列索引。"""
        queries = get_prediction_positions(target_positions, seq_len)
        video_positions = np.asarray(video_positions, dtype=np.int64)
        if not queries:
            raise ValueError("没有可归因的时间戳预测位置 p-1")
        if (video_positions.ndim != 1 or len(video_positions) == 0
                or np.any(video_positions < 0)
                or np.any(video_positions >= seq_len)
                or np.any(np.diff(video_positions) <= 0)):
            raise ValueError("video_positions 必须是递增且有效的真实序列位置")
        if not 0 <= gt_tok_s < gt_tok_e <= len(video_positions):
            raise ValueError("GT 视频内部 token 范围无效")
        gt_keys = video_positions[gt_tok_s:gt_tok_e].tolist()
        if any(not any(k <= q for k in gt_keys) for q in queries):
            raise ValueError("时间戳 query 看不到任何 GT 视频 key")
        self.query_positions = queries
        self.gt_key_positions = gt_keys
        self.video_positions = video_positions.tolist()
        self.seq_len = int(seq_len)
        self.n_video = len(video_positions)
        self.gt_range = (gt_tok_s, gt_tok_e)

    def _make_pre_o_proj_hook(self, layer_idx: int):
        def hook(module, args):
            z = args[0]  # 官方 o_proj 输入，仍处于真实计算图中
            expected = (1, self.seq_len, self.num_heads * self.head_dim)
            if tuple(z.shape) != expected:
                raise RuntimeError(f"L{layer_idx} pre-o_proj {tuple(z.shape)} != {expected}")
            if not z.requires_grad:
                raise RuntimeError(f"L{layer_idx} pre-o_proj 无梯度图，检查 no_grad/freeze 配置")
            selected = z[0, self.query_positions, :].detach().float().cpu()
            self.fwd_data.setdefault(layer_idx, {})["head_output"] = selected.reshape(
                len(self.query_positions), self.num_heads, self.head_dim)

            queries = tuple(self.query_positions)
            def grad_hook(grad):
                selected_grad = grad[0, list(queries), :].detach().float().cpu()
                self.grad_data[layer_idx] = {
                    "head_grad": selected_grad.reshape(
                        len(queries), self.num_heads, self.head_dim)
                }
            # hook 原始 z；不能 hook 一个仅用于缓存、不参与 target 的新切片。
            self.tensor_hooks.append(z.register_hook(grad_hook))
        return hook

    def _make_attention_hook(self, layer_idx: int):
        def hook(module, args, output):
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(f"L{layer_idx} 未返回 (attn_output, attn_weights)")
            weights = output[1]
            if weights is None:
                raise RuntimeError(
                    f"L{layer_idx} attn_weights=None：需要 eager attention；"
                    "若当前 Transformers 版本要求，可加 --force-output-attentions"
                )
            expected = (1, self.num_heads, self.seq_len, self.seq_len)
            if tuple(weights.shape) != expected:
                raise RuntimeError(f"L{layer_idx} attention {tuple(weights.shape)} != {expected}")
            import torch
            with torch.no_grad():
                q_idx = torch.tensor(self.query_positions, device=weights.device)
                k_idx = torch.tensor(self.gt_key_positions, device=weights.device)
                rows = weights.detach()[0].index_select(1, q_idx)
                gt_weights = rows.index_select(2, k_idx).float()
                visible = k_idx[None, :] <= q_idx[:, None]
                gt_mass = (gt_weights * visible[None, :, :]).sum(dim=-1)
                baseline = visible.sum(dim=-1).float() / (q_idx.float() + 1)
                ratios = gt_mass / baseline[None, :]
                self.fwd_data.setdefault(layer_idx, {})["attn_ratio"] = ratios.cpu()
                self.fwd_data[layer_idx]["video_stats"] = video_attention_statistics(
                    rows, self.query_positions, self.video_positions,
                    self.gt_key_positions, self.min_video_mass)
        return hook

    def attach_hooks(self, layer_indices: list[int]):
        if not self.query_positions:
            raise RuntimeError("请先调用 set_sample_context")
        self.remove_hooks()
        try:
            if self.attention_backend == "sdpa":
                from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
                original = ALL_ATTENTION_FUNCTIONS["sdpa"]
                targets = {id(self.model.model.language_model.layers[i].self_attn): i
                           for i in layer_indices}

                def capture_sdpa(module, query, key, value, attention_mask,
                                 dropout=0.0, scaling=None, **kwargs):
                    layer_idx = targets.get(id(module))
                    if layer_idx is not None:
                        if module.training or dropout:
                            raise RuntimeError("Attribution requires eval mode with dropout=0")
                        for option in ("softcap", "sliding_window", "position_bias"):
                            if kwargs.get(option) is not None:
                                raise RuntimeError(f"Unsupported attention variant: {option}")
                        original_ratio, video_stats = selected_attention_ratios(
                            module, query, key, attention_mask,
                            self.query_positions, self.gt_key_positions, scaling,
                            self.video_positions, self.min_video_mass)
                        self.fwd_data.setdefault(layer_idx, {}).update(
                            attn_ratio=original_ratio, video_stats=video_stats)
                    # 原始 Q/K/V 原样传回官方 SDPA：完整模型梯度路径不变。
                    return original(module, query, key, value, attention_mask,
                                    dropout=dropout, scaling=scaling, **kwargs)

                self._attention_registry = ALL_ATTENTION_FUNCTIONS
                self._original_sdpa = original
                ALL_ATTENTION_FUNCTIONS.register("sdpa", capture_sdpa)
            for i in layer_indices:
                attn = self.model.model.language_model.layers[i].self_attn
                self.hooks.append(attn.o_proj.register_forward_pre_hook(
                    self._make_pre_o_proj_hook(i)))
                if self.attention_backend == "eager":
                    self.hooks.append(attn.register_forward_hook(
                        self._make_attention_hook(i)))
        except Exception:
            self.remove_hooks()
            raise
        print(f"    [hooks] {len(layer_indices)} 层：pre-o_proj 激活 + "
              f"GT attention；query_count={len(self.query_positions)}")

    def attach_gt_only_hooks(self, layer_indices: list[int]):
        """只捕获 SDPA 时间戳 query 行；不挂激活/梯度 hook，不需要 backward。"""
        if not self.query_positions:
            raise RuntimeError("请先调用 set_sample_context")
        if self.attention_backend != "sdpa":
            raise ValueError("纯 GT 探测固定使用 SDPA")
        self.remove_hooks()
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            original = ALL_ATTENTION_FUNCTIONS["sdpa"]
            targets = {id(self.model.model.language_model.layers[i].self_attn): i
                       for i in layer_indices}

            def capture_sdpa(module, query, key, value, attention_mask,
                             dropout=0.0, scaling=None, **kwargs):
                layer_idx = targets.get(id(module))
                if layer_idx is not None:
                    if module.training or dropout:
                        raise RuntimeError("GT probing requires eval mode with dropout=0")
                    original_ratio, video_stats = selected_attention_ratios(
                        module, query, key, attention_mask,
                        self.query_positions, self.gt_key_positions, scaling,
                        self.video_positions, self.min_video_mass)
                    self.fwd_data.setdefault(layer_idx, {}).update(
                        attn_ratio=original_ratio, video_stats=video_stats)
                return original(module, query, key, value, attention_mask,
                                dropout=dropout, scaling=scaling, **kwargs)

            self._attention_registry = ALL_ATTENTION_FUNCTIONS
            self._original_sdpa = original
            ALL_ATTENTION_FUNCTIONS.register("sdpa", capture_sdpa)
        except Exception:
            self.remove_hooks()
            raise
        print(f"    [hooks] 全 {len(layer_indices)} 层：仅 GT attention；"
              f"query_count={len(self.query_positions)}")

    def remove_hooks(self):
        if self._attention_registry is not None:
            self._attention_registry.register("sdpa", self._original_sdpa)
            self._attention_registry = None
            self._original_sdpa = None
        for handle in self.hooks + self.tensor_hooks:
            handle.remove()
        self.hooks.clear()
        self.tensor_hooks.clear()
        self.fwd_data.clear()
        self.grad_data.clear()

    def compute_head_scores_batch(self, layer_indices: list[int],
                                  target_positions: list[int],
                                  n_video: int, gt_tok_s: int,
                                  gt_tok_e: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """S 是全部目标 logit 总和；各 query 的 Taylor 点积取绝对值后均值。

        保留原版联合目标中的跨 token 贡献，不改成逐 token 单独 backward。
        """
        if get_prediction_positions(target_positions, self.seq_len) != self.query_positions:
            raise ValueError("归因 query 与 forward 捕获时的位置不一致")
        if n_video != self.n_video or (gt_tok_s, gt_tok_e) != self.gt_range:
            raise ValueError("GT/video 范围与 forward 捕获时不一致")
        results = {}
        for l in layer_indices:
            if (l not in self.fwd_data or l not in self.grad_data
                    or "head_output" not in self.fwd_data[l]
                    or "attn_ratio" not in self.fwd_data[l]):
                raise RuntimeError(f"L{l} 缺少 head 激活/梯度/attention，不能作为零分计入")
            fwd = self.fwd_data[l]["head_output"]  # [Q,H,D]
            grad = self.grad_data[l]["head_grad"]
            head_grad = (fwd * grad).sum(dim=-1).abs().mean(dim=0).numpy()
            head_attn = self.fwd_data[l]["attn_ratio"].mean(dim=1).numpy()
            if not (np.isfinite(head_grad).all() and np.isfinite(head_attn).all()):
                raise RuntimeError(f"L{l} 归因分数包含 NaN/Inf")
            results[l] = (head_grad, head_attn)
        return results


    def compute_video_scores_batch(self, layer_indices):
        """每个样本 query 等权；任一 query 无效则该 head 的样本 ratio 无效。

        返回 ratio 均值、video mass 均值、有效 query 数；不影响原分数计算。
        """
        results = {}
        for l in layer_indices:
            stats = self.fwd_data[l]["video_stats"]
            ratios = stats["ratio"]
            results[l] = (ratios.mean(axis=1), stats["video_mass"].mean(axis=1),
                          np.isfinite(ratios).sum(axis=1))
        return results


def build_video_only_result(sum_ratio, sum_mass, sample_counts, query_counts,
                            n_valid, n_queries, top_k=30, min_ratio=1.0,
                            min_video_mass=1e-8):
    """独立结果：缺失值为 JSON null；仅全覆盖 head 可入选，避免选择性均值。"""
    mean_ratio = np.divide(sum_ratio, sample_counts,
                           out=np.full_like(sum_ratio, np.nan, dtype=np.float64),
                           where=sample_counts > 0)
    mean_mass = sum_mass / n_valid
    eligible = ((sample_counts == n_valid) & (query_counts == n_queries)
                & np.isfinite(mean_ratio) & (mean_ratio >= min_ratio))
    candidates = [(int(l), int(h)) for l, h in np.argwhere(eligible)]
    candidates.sort(key=lambda lh: (-mean_ratio[lh], lh[0], lh[1]))
    ranked = [{"rank": rank + 1, "layer": l, "head": h,
               "video_gt_ratio": float(mean_ratio[l, h]),
               "mean_video_attention_mass": float(mean_mass[l, h]),
               "valid_samples": int(sample_counts[l, h]),
               "valid_queries": int(query_counts[l, h])}
              for rank, (l, h) in enumerate(candidates[:top_k])]
    return {
        "_meta": {
            "method": "Video-only GT attention enrichment (independent ranking)",
            "formula": "(A_GT/A_video)/(N_visible_GT/N_visible_video), each p-1 query",
            "aggregation": "equal query mean within sample; equal sample mean; no layer normalization",
            "ranking": "descending raw ratio; ties by layer/head; all original valid samples and queries required",
            "invalid_policy": "A_video <= min_video_mass or invalid counts/masses: undefined; no epsilon or zero filling; any invalid query invalidates this head/sample",
            "partial_means": "matrix may contain partial-sample means; consult valid_sample_count_matrix; partial coverage never ranked",
            "n_valid": n_valid, "n_queries": n_queries,
            "top_k": top_k, "min_video_ratio": min_ratio,
            "min_video_mass": min_video_mass,
            "num_layers": sum_ratio.shape[0], "num_heads": sum_ratio.shape[1],
            "combined_uses_video_ratio": True,
        },
        "video_only_top_heads": ranked,
        "video_gt_ratio_matrix": [[float(v) if np.isfinite(v) else None for v in row]
                                  for row in mean_ratio],
        "mean_video_attention_mass_matrix": mean_mass.tolist(),
        "valid_sample_count_matrix": sample_counts.tolist(),
        "valid_query_count_matrix": query_counts.tolist(),
    }


def normalize_and_combine(grad_score: np.ndarray, video_ratio: np.ndarray,
                          min_video_ratio: float = 1.0) -> np.ndarray:
    """每个样本：grad 与视频内部 ratio 分别除以本层最大值，再取 min。

    两路均变成 [0,1] 的层内相对强度，不直接混合原始量纲。
    视频 ratio 低于下限时置零；非有限 ratio 不参与本层最大值。
    无效项的零仅为累计占位，最终必须依据覆盖数排除，不能作为有效零分。
    不再接受或使用旧全上下文 attention ratio。
    """
    eps = 1e-9

    def _normalize_per_layer(matrix: np.ndarray) -> np.ndarray:
        normed = np.zeros_like(matrix)
        for l in range(matrix.shape[0]):
            row_max = matrix[l].max()
            if row_max > eps:
                normed[l] = matrix[l] / row_max
        return normed

    finite = np.isfinite(video_ratio)
    safe_video = np.where(finite, video_ratio, 0.0)
    combined = np.minimum(_normalize_per_layer(grad_score), _normalize_per_layer(safe_video))
    combined[(~finite) | (video_ratio < min_video_ratio)] = 0.0
    return combined


def select_top_grad_only(mean_grad: np.ndarray, top_k: int = 30) -> List[dict]:
    """
    纯粹按梯度归因分数（grad_score）单独排名，跟 attn_align 完全无关——
    是跟 combined_score（要求两边都强）、attn_only（纯看 attn）并列的
    第三条独立候选筛选路径中的一条：只看"这个 head 的输出对 target
    logit 的梯度响应强不强"，不管它生成 start/end 数字时 attention 有
    没有真的看向 GT 视频区间。

    做法：直接按跨层原始 mean_grad 全局排序，不做逐层归一化。
    这是单一梯度量的消融组：其排序必须对应“mask 这个 head 后，对目标
    logit 的一阶影响有多大”。逐层归一化只用于梯度与 video-GT ratio
    联合时的量纲对齐，不能改变纯梯度组的候选顺序。

    这类 head 可能是"确实参与了决定最终答案数字、但注意力模式本身不
    一定直接盯着 GT 视频段"的 head——比如做数值计算/格式化、或者从别的
    head 已经聚合好的信息里做进一步处理，跟 attn_only 选出来的那批
    "负责读取视频信息"的 head 角色可能完全不同。
    """
    n_layers, n_heads = mean_grad.shape
    safe_grad = np.where(np.isfinite(mean_grad), mean_grad, -np.inf)
    order = np.argsort(-safe_grad.ravel())[:top_k]
    global_max = float(np.max(safe_grad)) if np.isfinite(safe_grad).any() else 0.0

    results: List[dict] = []
    for rank, fi in enumerate(order):
        l, h = divmod(int(fi), n_heads)
        results.append({
            "rank": rank + 1,
            "layer": int(l),
            "head": int(h),
            "grad_score": round(float(mean_grad[l, h]), 6),
            # 仅用于显示相对最大梯度，不参与排序。
            "grad_score_norm": round(float(mean_grad[l, h] / global_max), 6)
            if global_max > 0 else 0.0,
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
        ("Combined Score (min-normalized, timestamp prediction queries)", combined),
        ("Pre-o-proj Taylor Score (timestamp prediction queries)", grad_score),
        ("GT Alignment Ratio (timestamp prediction queries)", attn_align),
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
        ("Pre-o-proj Taylor Score", grad_score),
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

def build_gt_only_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="仅用视频内部 GT attention 对齐分数筛选 Qwen3-VL heads（SDPA）")
    p.add_argument("--filtered-json", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--video-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--timelens-model", action="store_true")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--max-valid-samples", type=int, default=0)
    p.add_argument("--max-duration", type=float, default=0.0)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--max-frames", type=int, default=0,
                   help="0 按 2*total_tokens/min_tokens 推导；3584/64 对应 112 帧")
    p.add_argument("--min-tokens", type=int, default=64)
    p.add_argument("--total-tokens", type=int, default=3584)
    p.add_argument("--max-side", type=int, default=224)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--min-gt-ratio", type=float, default=1.0)
    p.add_argument("--score-eps", type=float, default=1e-8)
    p.add_argument("--device-map", choices=["auto", "balanced", "balanced_low_0", "sequential"],
                   default="auto")
    p.add_argument("--gpu-mem-gib", type=float, default=16.0)
    p.add_argument("--gpu-reserve-gib", type=float, default=4.0)
    p.add_argument("--cpu-mem-gib", type=float, default=64.0)
    return p


def gt_only_main(argv=None):
    """单次 SDPA forward 统计全部层/head，只写 video_only_head_attribution.json。"""
    args = build_gt_only_parser().parse_args(argv)
    if args.top_k < 0 or args.min_gt_ratio < 0 or args.score_eps < 0:
        raise ValueError("top-k、min-gt-ratio 和 score-eps 必须非负")
    import torch

    filtered_json = resolve_path(args.filtered_json, must_exist=True)
    model_dir = resolve_path(args.model_path, must_exist=True)
    video_dir = resolve_path(args.video_dir, must_exist=True)
    output_dir = ensure_directory(resolve_path(args.output_dir))
    output_path = output_dir / "video_only_head_attribution.json"

    n_gpus = torch.cuda.device_count()
    print(f"CUDA={torch.cuda.is_available()}，GPU={n_gpus}；纯 GT/SDPA，无 backward")
    model, processor, n_layers = load_model_and_processor(
        model_dir, gpu_mem_gib=args.gpu_mem_gib, cpu_mem_gib=args.cpu_mem_gib,
        device_map=args.device_map, gpu_reserve_gib=args.gpu_reserve_gib,
        attention_backend="sdpa", timelens_model=args.timelens_model)
    device = model.get_input_embeddings().weight.device
    num_heads, head_dim, num_kv_heads = validate_head_layout(model)
    samples = load_samples(filtered_json, video_dir, args.max_samples)
    attributor = StartEndHeadAttributor(
        model, n_layers, num_heads, head_dim,
        attention_backend="sdpa", min_video_mass=args.score_eps)

    shape = (n_layers, num_heads)
    sum_score = np.zeros(shape, dtype=np.float64)
    sum_gt_mass = np.zeros(shape, dtype=np.float64)
    sum_video_mass = np.zeros(shape, dtype=np.float64)
    valid_counts = np.zeros(shape, dtype=np.int64)
    valid_count = failures = duration_skipped = 0
    duration_cache: Dict[str, Optional[float]] = {}
    frame_limit = resolve_max_frames(args.max_frames, args.total_tokens, args.min_tokens)
    started = time.time()

    for idx, sample in enumerate(samples):
        if args.max_valid_samples > 0 and valid_count >= args.max_valid_samples:
            break
        frames = inputs = outputs = None
        try:
            video_path = _resolve_video_path(video_dir, sample)
            if video_path is None:
                raise FileNotFoundError("视频文件缺失")
            cache_key = str(video_path)
            if cache_key not in duration_cache:
                duration_cache[cache_key] = get_duration_ffprobe(video_path)
            probed_duration = duration_cache[cache_key]
            if (args.max_duration > 0 and probed_duration is not None
                    and probed_duration > args.max_duration):
                duration_skipped += 1
                continue

            frames, metadata = sample_frames(video_path, args.fps)
            sampled_count = len(frames)
            duration = float(probed_duration if probed_duration is not None
                             else metadata["duration"])
            metadata["duration"] = duration
            if frame_limit > 0 and len(frames) > frame_limit:
                frames, metadata = uniform_subsample_frames(frames, frame_limit, metadata)
                print(f"    [frames] {sampled_count}->{len(frames)}，2 FPS 后按预算上限均匀压缩")
            if args.total_tokens <= 0:
                frames = resize_frames(frames, args.max_side)

            inputs, _, _ = build_inputs(
                processor, frames, sample["query"], duration, device, args.fps,
                gt_start=sample["gt_start"], gt_end=sample["gt_end"],
                min_tokens=args.min_tokens, total_tokens=args.total_tokens,
                video_metadata=metadata, timelens_model=args.timelens_model)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            ids = inputs["input_ids"][0].tolist()
            token_info = get_token_positions(ids, processor)
            if token_info["n_video_tokens"] > args.total_tokens > 0:
                raise ValueError(f"video token={token_info['n_video_tokens']} 超过 {args.total_tokens}")
            positions = locate_start_end_token_positions(
                ids, token_info, processor, sample["gt_start"], sample["gt_end"])
            target_positions = positions["start_positions"] + positions["end_positions"]
            if not target_positions or token_info["n_video_tokens"] <= 0:
                raise ValueError("时间戳 query 或视频 token 为空")
            gt_s, gt_e = compute_gt_video_token_range(
                sample["gt_start"], sample["gt_end"], duration,
                token_info["n_video_tokens"])
            attributor.set_sample_context(
                target_positions, np.flatnonzero(token_info["video_mask"]),
                gt_s, gt_e, inputs["input_ids"].shape[1])

            attributor.attach_gt_only_hooks(list(range(n_layers)))
            with torch.inference_mode(), attention_kernel_context("sdpa"):
                outputs = model(**inputs)

            sample_score = np.full(shape, np.nan, dtype=np.float64)
            sample_gt = np.zeros(shape, dtype=np.float64)
            sample_video = np.zeros(shape, dtype=np.float64)
            for layer in range(n_layers):
                stats = attributor.fwd_data.get(layer, {}).get("video_stats")
                if stats is None:
                    raise RuntimeError(f"L{layer} 未捕获 GT attention")
                sample_score[layer] = sample_gt_alignment_score(stats, args.score_eps)
                sample_gt[layer] = np.asarray(stats["gt_mass"]).mean(axis=1)
                sample_video[layer] = np.asarray(stats["video_mass"]).mean(axis=1)
            finite = np.isfinite(sample_score)
            sum_score += np.where(finite, sample_score, 0.0)
            sum_gt_mass += sample_gt
            sum_video_mass += sample_video
            valid_counts += finite
            valid_count += 1
            best = np.nanargmax(sample_score)
            layer, head = divmod(int(best), num_heads)
            print(f"  [{idx+1}/{len(samples)}] valid={valid_count} "
                  f"{sample['video_id']} Q={len(attributor.query_positions)} "
                  f"top=L{layer}H{head} score={sample_score[layer, head]:.4f}")
        except Exception as exc:
            failures += 1
            print(f"  [{idx+1}/{len(samples)}] SKIP: {exc}")
        finally:
            attributor.remove_hooks()
            outputs = inputs = frames = None
            deep_gpu_cleanup(n_gpus)

    if valid_count == 0:
        raise RuntimeError("没有有效样本，未生成结果")
    mean_score = np.divide(sum_score, valid_counts,
                           out=np.full(shape, np.nan), where=valid_counts > 0)
    complete = valid_counts == valid_count
    candidates = [(l, h) for l in range(n_layers) for h in range(num_heads)
                  if complete[l, h] and mean_score[l, h] >= args.min_gt_ratio]
    candidates.sort(key=lambda lh: (-mean_score[lh], lh[0], lh[1]))
    ranked = [{"rank": rank + 1, "layer": l, "head": h,
               "video_gt_ratio": float(mean_score[l, h]),
               "gt_alignment_score": float(mean_score[l, h]),
               "mean_gt_attention_mass": float(sum_gt_mass[l, h] / valid_count),
               "mean_video_attention_mass": float(sum_video_mass[l, h] / valid_count),
               "valid_samples": valid_count}
              for rank, (l, h) in enumerate(candidates[:args.top_k])]
    result = {
        "_meta": {
            "method": "GT-only video-conditional attention alignment",
            "formula": "mean_i((mean_q A_GT)/(mean_q A_video + eps)/(mean_q N_GT/N_video + eps))",
            "gradient_attribution": False,
            "combined_selection": False,
            "attention_backend": "sdpa",
            "query_rows": "all start/end timestamp subtokens at p-1",
            "fps": args.fps, "max_frames": frame_limit,
            "min_tokens": args.min_tokens, "total_tokens": args.total_tokens,
            "score_eps": args.score_eps, "min_gt_ratio": args.min_gt_ratio,
            "n_valid": valid_count, "n_failures": failures,
            "n_duration_skipped": duration_skipped,
            "num_layers": n_layers, "num_heads": num_heads,
            "num_key_value_heads": num_kv_heads, "head_dim": head_dim,
            "video_timestamp_source": "original_frame_indices/native_fps",
            "video_timestamps_verified": True,
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "video_only_top_heads": ranked,
        "gt_alignment_score_matrix": [
            [float(v) if np.isfinite(v) else None for v in row] for row in mean_score],
        "valid_sample_count_matrix": valid_counts.tolist(),
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding="utf-8")
    print(f"\n完成：{output_path}；有效样本={valid_count}，Top heads={len(ranked)}")
    return 0

def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.video_only_top_k < 0 or not np.isfinite(args.min_video_ratio) or args.min_video_ratio < 0:
        raise ValueError("video-only-top-k 和 min-video-ratio 必须非负且有限")
    if not np.isfinite(args.min_video_mass) or not 0 <= args.min_video_mass <= 1:
        raise ValueError("min-video-mass 必须在 [0,1] 内")
    import torch
    if args.attention_backend == "sdpa" and args.force_output_attentions:
        raise ValueError("SDPA 模式只提取所需行，不支持 --force-output-attentions；对照请用 --attention-backend eager")

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

    print(f"\n加载模型（{args.attention_backend} 模式，分片到 {n_gpus} 张 GPU）：{model_dir}")
    model, processor, n_layers = load_model_and_processor(
        model_dir, gpu_mem_gib=args.gpu_mem_gib, cpu_mem_gib=args.cpu_mem_gib,
        device_map=args.device_map, gpu_reserve_gib=args.gpu_reserve_gib,
        attention_backend=args.attention_backend,
        timelens_model=args.timelens_model,
    )
    # model.device 只代表首个参数；分片模型以输入 embedding 的设备为入口。
    device = model.get_input_embeddings().weight.device
    print(f"  [model] input device={device}; lm_head device={model.lm_head.weight.device}")
    freeze_non_essential_params(model)
    num_heads, head_dim, num_kv_heads = validate_head_layout(model)
    if args.save_activations_on_cpu:
        print("  [memory] CPU saved-tensor offload 已开启：GPU 常驻权重，"
              "反向张量暂存主机 RAM；会变慢，主机内存不受 --cpu-mem-gib 限制。")

    sample_list = load_samples(filtered_json, video_dir, args.max_samples)
    print(f"\n候选样本池大小：{len(sample_list)}")
    if args.max_valid_samples > 0:
        print(f"目标有效样本数：{args.max_valid_samples}（攒够即停）")

    layer_batches = make_layer_batches(n_layers, args.layers_per_batch)
    print(f"36 层分为 {len(layer_batches)} 批处理：{layer_batches}")

    attributor = StartEndHeadAttributor(model, n_layers, num_heads, head_dim,
                                      attention_backend=args.attention_backend,
                                      min_video_mass=args.min_video_mass)

    sum_combined = np.zeros((n_layers, NUM_HEADS), dtype=np.float64)
    sum_grad = np.zeros((n_layers, NUM_HEADS), dtype=np.float64)
    sum_attn = np.zeros((n_layers, NUM_HEADS), dtype=np.float64)
    sum_video_ratio = np.zeros_like(sum_attn)
    sum_video_mass = np.zeros_like(sum_attn)
    video_sample_counts = np.zeros_like(sum_attn, dtype=np.int64)
    video_query_counts = np.zeros_like(sum_attn, dtype=np.int64)
    video_total_queries = 0
    hit_count = np.zeros((n_layers, NUM_HEADS), dtype=np.int64)
    valid_count = 0
    failures = 0
    duration_cache: Dict[str, Optional[float]] = {}
    duration_skipped = 0
    duration_unknown = 0
    incomplete_samples = 0
    timing_records = []

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

            # 即使关闭时长过滤，仍优先使用真实容器时长进行 GT 比例映射。
            vp = str(video_path)
            if vp not in duration_cache:
                duration_cache[vp] = get_duration_ffprobe(video_path)
            dur = duration_cache[vp]
            if args.max_duration > 0:
                if dur is None:
                    print("    SKIP（ffprobe 无法获取时长）")
                    duration_unknown += 1
                    continue
                if dur > args.max_duration:
                    print(f"    SKIP（时长 {dur:.1f}s > {args.max_duration:.1f}s）")
                    duration_skipped += 1
                    continue
                s["probed_duration"] = dur

            frames, source_metadata = sample_frames(video_path, args.fps)
            sampled_frame_count = len(frames)
            # 不使用缩减后的帧数或请求采样 FPS 推断原视频时长。
            duration = float(dur if dur is not None else source_metadata["duration"])
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError("原视频时长无效")
            duration_source = "ffprobe" if dur is not None else "decoder_total_frames/native_fps"
            source_metadata["duration"] = duration

            frame_limit = resolve_max_frames(
                args.max_frames, args.total_tokens, args.min_tokens)
            if frame_limit > 0 and sampled_frame_count > frame_limit:
                frames, source_metadata = uniform_subsample_frames(frames, frame_limit, source_metadata)
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
                video_metadata=source_metadata,
                timelens_model=args.timelens_model,
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            input_ids_list = inputs["input_ids"][0].tolist()

            # 3. Token 位置解析 + start/end 数字 token 定位
            token_info = get_token_positions(input_ids_list, processor)
            if (args.total_tokens > 0
                    and token_info["n_video_tokens"] > args.total_tokens):
                print(f"    SKIP（video token={token_info['n_video_tokens']} "
                      f"超过硬上限 {args.total_tokens}）")
                del inputs, frames
                deep_gpu_cleanup(n_gpus)
                failures += 1
                continue
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

            attributor.set_sample_context(
                target_positions, np.flatnonzero(token_info["video_mask"]),
                gt_tok_s, gt_tok_e, seq_len)
            print(f"    [positions] target_P={target_positions} -> "
                  f"prediction_Q={attributor.query_positions} (全部子 token)")

            # 4. 分批 forward + backward
            sample_grad = np.zeros((n_layers, NUM_HEADS), dtype=np.float32)
            sample_attn = np.zeros((n_layers, NUM_HEADS), dtype=np.float32)
            sample_video_ratio = np.full_like(sum_attn, np.nan)
            sample_video_mass = np.zeros_like(sum_attn)
            sample_video_queries = np.zeros_like(video_query_counts)
            input_ids_t = inputs["input_ids"]

            decoder_layers = model.model.language_model.layers

            completed_layers = set()
            for batch_idx, layer_batch in enumerate(layer_batches):
                freeze_indices = list(range(0, layer_batch[0]))
                print(f"    [batch {batch_idx+1}/{len(layer_batches)}] "
                      f"layers={layer_batch}  frozen(no_grad)={len(freeze_indices)}层  "
                      f"仍需保留梯度到底={n_layers - layer_batch[0]}层")
                outputs = logits = target = None
                try:
                    attributor.attach_hooks(layer_batch)
                    print("    [forward] 开始前向（完整序列，多 token 目标）", flush=True)
                    with attention_kernel_context(args.attention_backend), \
                            activation_storage_context(args.save_activations_on_cpu), \
                            FreezeLayersNoGrad(decoder_layers, freeze_indices):
                        if args.force_output_attentions:
                            outputs = model(**inputs, output_attentions=True)
                        else:
                            outputs = model(**inputs)
                        logits = outputs.logits
                        # 注意：仍传原始 P，由函数内部选 logits[p-1, token[p]]。
                        target, n_used = compute_target_logit_sum(
                            logits, input_ids_t, target_positions)
                        if target is None or n_used == 0 or not torch.isfinite(target):
                            print(f"    batch{batch_idx} SKIP（target 无效）")
                            break
                        print("    [backward] 前向完成，开始反向", flush=True)
                        target.backward()
                        print("    [backward] 完成", flush=True)

                    batch_results = attributor.compute_head_scores_batch(
                        layer_batch, target_positions, n_video, gt_tok_s, gt_tok_e)
                    video_results = attributor.compute_video_scores_batch(layer_batch)
                    for l, (hg, ha) in batch_results.items():
                        sample_grad[l] = hg
                        sample_attn[l] = ha
                        sample_video_ratio[l], sample_video_mass[l], sample_video_queries[l] = video_results[l]
                        completed_layers.add(l)
                except torch.cuda.OutOfMemoryError as oom:
                    print(f"    batch{batch_idx} OOM（layers={layer_batch}）: {oom}")
                    print("    该样本无法获得完整层分数，停止其余批次并跳过样本。")
                    if not args.save_activations_on_cpu:
                        print("    建议添加 --save-activations-on-cpu；仅降低 layers-per-batch "
                              "不能消除后续所有层的反向激活。")
                    break
                finally:
                    attributor.remove_hooks()
                    outputs = logits = target = None
                    model.zero_grad(set_to_none=True)
                    deep_gpu_cleanup(n_gpus)

            if len(completed_layers) != n_layers:
                print(f"    SKIP（仅 {len(completed_layers)}/{n_layers} 层成功；"
                      "不把缺失层当成零分纳入排名）")
                incomplete_samples += 1
                failures += 1
                del inputs, frames
                deep_gpu_cleanup(n_gpus)
                continue

            print_gpu_memory(n_gpus, tag=f"after sample {idx+1}")

            # 5. 联合分数 + 累积
            sample_combined = normalize_and_combine(
                sample_grad, sample_video_ratio, min_video_ratio=args.min_video_ratio,
            )
            sum_grad += sample_grad.astype(np.float64)
            sum_attn += sample_attn.astype(np.float64)
            sum_combined += sample_combined.astype(np.float64)
            video_valid = np.isfinite(sample_video_ratio)
            sum_video_ratio += np.where(video_valid, sample_video_ratio, 0.0)
            sum_video_mass += sample_video_mass
            video_sample_counts += video_valid
            video_query_counts += sample_video_queries
            video_total_queries += len(attributor.query_positions)
            valid_count += 1
            timing_records.append({"candidate_index": idx, "video_id": s["video_id"],
                                   "source_metadata": source_metadata,
                                   "duration_source": duration_source,
                                   "sampled_frame_count": sampled_frame_count,
                                   "selected_frame_count": len(frames),
                                   "processor_timestamps_verified": True})

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
    combined_eligible = ((video_sample_counts == valid_count)
                         & (video_query_counts == video_total_queries))
    # 与独立 video-only 一致：不以少量有效样本的平均值参与联合排名。
    # 保留数值矩阵/绘图接口，无效 head 用零占位并通过显式 mask 排除。
    mean_combined[~combined_eligible] = 0.0
    mean_video_ratio = np.divide(sum_video_ratio, video_sample_counts,
                                 out=np.full_like(sum_video_ratio, np.nan),
                                 where=video_sample_counts > 0)

    # 先落盘数值结果，后续元数据或绘图异常也不会丢失已完成的探测。
    checkpoint_path = output_dir / "head_score_checkpoint.npz"
    np.savez_compressed(checkpoint_path, sum_combined=sum_combined,
                        sum_grad=sum_grad, sum_attn=sum_attn,
                        sum_video_ratio=sum_video_ratio,
                        sum_video_mass=sum_video_mass,
                        video_sample_counts=video_sample_counts,
                        video_query_counts=video_query_counts,
                        video_total_queries=np.array(video_total_queries),
                        combined_eligible=combined_eligible,
                        combined_attention_metric=np.array("video_gt_ratio"),
                        combined_min_video_ratio=np.array(args.min_video_ratio),
                        hit_count=hit_count, n_valid=np.array(valid_count))
    print(f"  [save] 原始累计分数已保存：{checkpoint_path}", flush=True)
    timing_json = output_dir / "video_timing.json"
    timing_json.write_text(json.dumps(timing_records, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding="utf-8")

    # 独立 video-only 排名保持不变；联合排名改用同一视频内部 ratio。
    video_result = build_video_only_result(
        sum_video_ratio, sum_video_mass, video_sample_counts, video_query_counts,
        valid_count, video_total_queries, args.video_only_top_k,
        args.min_video_ratio, args.min_video_mass)
    video_result["_meta"].update(
        model_path=str(model_dir), filtered_json=str(filtered_json), video_dir=str(video_dir),
        fps=args.fps, max_frames=resolve_max_frames(args.max_frames, args.total_tokens, args.min_tokens),
        min_tokens=args.min_tokens, total_tokens=args.total_tokens, max_duration=args.max_duration,
        max_samples=args.max_samples, max_valid_samples=args.max_valid_samples,
        n_samples_total=len(sample_list), n_duration_skipped=duration_skipped,
        n_duration_unknown=duration_unknown, n_failures=failures,
        attention_backend=args.attention_backend,
        video_timestamp_source="original_frame_indices/native_fps",
        video_timestamps_verified=True, video_timing_json=timing_json.name,
        source_attribution_json="startend_gradient_head_attribution.json")
    video_json = output_dir / "video_only_head_attribution.json"
    video_json.write_text(json.dumps(video_result, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")
    print(f"  [video-only] 独立排名已保存：{video_json}；"
          f"入选 {len(video_result['video_only_top_heads'])} 个 head；"
          f"全覆盖 {int((video_sample_counts == valid_count).sum())}/{video_sample_counts.size}")

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
    combined_order = np.argsort(flat)[::-1]
    top_k_idx = combined_order[combined_eligible.ravel()[combined_order]][:args.top_k]

    top_k_heads = []
    print(f"\n{'='*60}")
    print(f"Top-{args.top_k} 联合归因 Head（Gradient + VideoRatio）")
    print(f"{'='*60}")
    print(f"  {'Rank':>4}  {'Layer':>5}  {'Head':>4}  "
          f"{'Combined':>10}  {'Gradient':>10}  {'VideoRatio':>10}  {'GlobalAttn':>10}")
    for rank, fi in enumerate(top_k_idx):
        l_idx, h_idx = divmod(int(fi), NUM_HEADS)
        top_k_heads.append({
            "rank": rank + 1,
            "layer": l_idx,
            "head": h_idx,
            "combined_score": round(float(mean_combined[l_idx, h_idx]), 6),
            "gradient_score": round(float(mean_grad[l_idx, h_idx]), 6),
            "attention_score": round(float(mean_attn[l_idx, h_idx]), 6),
            "video_gt_ratio": float(mean_video_ratio[l_idx, h_idx]),
        })
        print(f"  {rank+1:>4}  {l_idx:>5}  {h_idx:>4}  "
              f"{mean_combined[l_idx, h_idx]:>10.4f}  "
              f"{mean_grad[l_idx, h_idx]:>10.4f}  "
              f"{mean_video_ratio[l_idx, h_idx]:>10.4f}  "
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
        print(f"  {'Rank':>4}  {'Layer':>5}  {'Head':>4}  {'GlobalNorm':>10}  {'GradRaw':>10}")
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

    print(f"\n{'='*60}")
    print(f"Top-{args.video_only_top_k} 视频内部 GT ratio Head（独立排名，不使用梯度）")
    print(f"{'='*60}")
    print("  公式：(A_GT / A_video) / (N_visible_GT / N_visible_video)")
    print(f"  {'Rank':>4}  {'Layer':>5}  {'Head':>4}  {'VideoRatio':>12}  "
          f"{'VideoMass':>12}  {'ValidSamples':>14}  {'ValidQueries':>16}")
    for entry in video_result["video_only_top_heads"]:
        print(f"  {entry['rank']:>4}  {entry['layer']:>5}  {entry['head']:>4}  "
              f"{entry['video_gt_ratio']:>12.6f}  "
              f"{entry['mean_video_attention_mass']:>12.6g}  "
              f"{entry['valid_samples']:>7}/{valid_count:<6}  "
              f"{entry['valid_queries']:>7}/{video_total_queries:<8}")
    if not video_result["video_only_top_heads"]:
        print(f"  [video-only] 无排名条目：top-k={args.video_only_top_k}，"
              f"要求 ratio >= {args.min_video_ratio:g} 且所有样本/query 均有效。")
        print(f"  [video-only] 全样本覆盖 head="
              f"{int((video_sample_counts == valid_count).sum())}/{video_sample_counts.size}；"
              f"每个 query 的 A_video 必须 > {args.min_video_mass:g}。")
    print(f"  视频内部 GT ratio JSON：{video_json}")

    result_json = {
        "_meta": {
            "method": "Start/End-token Gradient Attribution "
                      "(multi-token target logit, p-1 queries, pre-o_proj)",
            "formula": "target = sum_p logits[p-1, gt_token[p]] for p in start/end tokens; "
                       "Q = sorted(unique(p-1)); "
                       "HIS = mean_q abs(sum_d Z_pre_o_proj[q,h,d] * dTarget/dZ[q,h,d]); "
                       "attn_align = mean_q(sum_{k in GT_seq_keys, k<=q} attn[h,q,k] "
                       "/ (count_visible_GT_keys/(q+1))); "
                       "video_ratio = mean_q((A_GT/A_video)/(N_visible_GT/N_visible_video)); "
                       "Combined = mean_sample(min(layer_max_norm(HIS), layer_max_norm(video_ratio)) "
                       "with per-sample video_ratio threshold)",
            "combined_attention_metric": "video_gt_ratio",
            "combined_uses_video_ratio": True,
            "combined_normalization": "per sample, each signal divided by its own layer maximum, then min; equal sample mean",
            "combined_invalid_policy": "require all valid samples/queries; ineligible matrix entries are zero placeholders, excluded from top_k_heads",
            "attention_score_semantics": "original full-context GT ratio, diagnostic only; combined uses video_gt_ratio",
            "min_video_ratio": args.min_video_ratio,
            "min_video_mass": args.min_video_mass,
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
            "grad_only_ranking": "raw_global_mean_grad",
            "attn_only_top_k": args.attn_only_top_k,
            "video_only_top_k": args.video_only_top_k,
            "fps": args.fps,
            "max_frames": resolve_max_frames(
                args.max_frames, args.total_tokens, args.min_tokens),
            "min_tokens": args.min_tokens,
            "total_tokens": args.total_tokens,
            "max_side": args.max_side,
            "layers_per_batch": args.layers_per_batch,
            "n_layer_batches": len(layer_batches),
            "num_layers": n_layers,
            "num_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "activation_site": "self_attn.o_proj input (pre-projection)",
            "query_position": "each timestamp subtoken p shifted to p-1",
            "video_timestamp_source": "original_frame_indices/native_fps",
            "video_timestamps_verified": True,
            "video_timing_json": timing_json.name,
            "duration_source": "ffprobe, fallback to decoder total_frames/native_fps",
            "gt_key_mapping": "video-local time-proportional range -> actual video_mask positions",
            "n_incomplete_samples": incomplete_samples,
            "n_gpus": n_gpus,
            "gpu_mem_gib": args.gpu_mem_gib,
            "gpu_reserve_gib": args.gpu_reserve_gib,
            "device_map_strategy": args.device_map,
            "hf_device_map": device_map_for_metadata(model),
            "cpu_weight_offload": False,
            "saved_tensors_on_cpu": args.save_activations_on_cpu,
            "attention_backend": args.attention_backend,
            "alignment_rows": "all timestamp prediction queries; all visible keys in softmax",
            "elapsed_seconds": round(elapsed, 1),
            "seconds_per_sample": round(elapsed / max(valid_count, 1), 2),
        },
        "top_k_heads": top_k_heads,
        "grad_only_top_heads": grad_only_heads,
        "attn_only_top_heads": attn_only_heads,
        # 同时嵌入独立 video-only 文件的排名，供下游单文件读取。
        "video_only_top_heads": video_result["video_only_top_heads"],
        "combined_score_matrix": mean_combined.tolist(),
        "gradient_score_matrix": mean_grad.tolist(),
        "attention_alignment_matrix": mean_attn.tolist(),
        "video_gt_ratio_matrix": video_result["video_gt_ratio_matrix"],
        "combined_eligible_matrix": combined_eligible.tolist(),
    }

    out_json = output_dir / "startend_gradient_head_attribution.json"
    out_json.write_text(
        json.dumps(result_json, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\n结果 JSON：{out_json}")

    print(f"\n生成可视化...")
    save_heatmaps(mean_combined, mean_grad, mean_attn,
                  args.top_k, valid_count, output_dir)
    save_per_layer_detail(mean_combined, mean_grad, mean_attn,
                           args.top_k, output_dir)

    attn_only_top = np.argsort(mean_attn.ravel())[::-1][:args.top_k]
    combined_set = set(int(fi) for fi in top_k_idx)
    attn_set = set(int(fi) for fi in attn_only_top)
    overlap = combined_set & attn_set
    print(f"\n  Combined vs Attention-only Top-{args.top_k} 重叠: {len(overlap)}/{args.top_k}")
    # 保留上面的原全上下文 attention 对照，并增加新联合信号对应的对照。
    # 使用相同 K、完整覆盖与 video-only 阈值，不受 video-only-top-k 输出数量影响。
    video_overlap_result = build_video_only_result(
        sum_video_ratio, sum_video_mass, video_sample_counts, video_query_counts,
        valid_count, video_total_queries, args.top_k, args.min_video_ratio, args.min_video_mass)
    video_set = {entry["layer"] * NUM_HEADS + entry["head"]
                 for entry in video_overlap_result["video_only_top_heads"]}
    video_overlap = combined_set & video_set
    print(f"  Combined vs Video-only Top-{args.top_k} 重叠: {len(video_overlap)} "
          f"（Combined={len(combined_set)}，Video-only={len(video_set)}）")

    print(f"\n{'='*60}")
    print(f"完成  有效样本={valid_count}  失败={failures}  "
          f"时长过滤跳过={duration_skipped}  时长未知={duration_unknown}  "
          f"耗时={elapsed:.1f}s  ({elapsed/max(valid_count,1):.1f}s/sample)")
    if top_k_heads:
        print(f"Top-1 Head: L{top_k_heads[0]['layer']}H{top_k_heads[0]['head']}"
              f"  combined={top_k_heads[0]['combined_score']:.4f}")
    else:
        print("无满足完整视频 ratio 覆盖条件的联合候选。")
    print(f"视频内部 GT ratio 独立结果：{video_json}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(gt_only_main())
