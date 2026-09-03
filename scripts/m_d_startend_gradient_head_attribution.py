"""Standalone TimeLens-8B GT-only head probing with torchrun data parallelism.

GT scoring, timestamp handling and SDPA query-row capture are copied directly
from new_d_head_attribution.py; there is no runtime dependency on that script.
Each rank loads one complete model on its own GPU and processes rank::world_size.
Default: TimeLens prompt/processor, 2 FPS, 14336 video tokens, no extra frame cap.
Only video_only_head_attribution.json is retained after the weighted merge.
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.project_paths import ensure_directory, resolve_path

NUM_LAYERS = 36
NUM_HEADS = 32
HEAD_DIM = 128


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone TimeLens-8B GT-only probing; one full model per GPU")
    parser.add_argument("--filtered-json", required=True)
    parser.add_argument("--model-path", default="../shared_models/TimeLens-8B")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--max-duration", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--total-tokens", type=int, default=14336)
    parser.add_argument("--max-side", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-gt-ratio", type=float, default=1.0)
    parser.add_argument("--score-eps", type=float, default=1e-8)
    parser.add_argument("--timelens-model", action="store_true", default=True,
                        help="TimeLens prompt/processor (already enabled by default)")
    return parser


def load_model_and_processor(model_dir):
    """Load one complete TimeLens model; preserve its processor resize settings."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True,
        padding_side="left", do_resize=False)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": "cuda:0"}, trust_remote_code=True, local_files_only=True)
    model.eval()
    model.config.use_cache = False
    text_config = getattr(model.config, "text_config", model.config)
    text_config.use_cache = False
    n_layers = len(model.model.language_model.layers)
    validate_head_layout(model)
    print(f"  [model] TimeLens-8B / SDPA: full rank-local model on cuda:0, {n_layers} layers")
    return model, processor, n_layers


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


def deep_gpu_cleanup(n_gpus: int):
    import torch
    gc.collect()
    if n_gpus <= 0:
        return
    for gi in range(n_gpus):
        with torch.cuda.device(gi):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class StartEndHeadAttributor:
    """Capture only timestamp query attention rows from official post-RoPE Q/K."""
    def __init__(self, model, n_layers: int, num_heads: int, head_dim: int,
                 attention_backend: str = "sdpa", min_video_mass: float = 1e-8):
        self.model = model
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.fwd_data: Dict[int, dict] = {}
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


def _run_rank_gt(args, rank, world_size):
    """单次 SDPA forward 统计全部层/head，只写 video_only_head_attribution.json。"""
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
    model, processor, n_layers = load_model_and_processor(model_dir)
    device = model.get_input_embeddings().weight.device
    num_heads, head_dim, num_kv_heads = validate_head_layout(model)
    all_samples = load_samples(filtered_json, video_dir, args.max_samples)
    samples = all_samples[rank::world_size]
    print(f"[distributed] rank={rank}/{world_size}: global={len(all_samples)}, local={len(samples)}")
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
    frame_limit = 0  # No extra frame cap after 2 FPS sampling.
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
            duration = float(probed_duration if probed_duration is not None
                             else metadata["duration"])
            metadata["duration"] = duration
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
            "model_path": str(model_dir), "timelens_model": True,
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
        "n_duration_skipped": sum(int(r["_meta"].get("n_duration_skipped", 0)) for r in rank_results),
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
    args = build_parser().parse_args(argv)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # Preserve the existing one-visible-GPU-per-rank mapping before CUDA init.
    if world_size > 1:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            devices = [v.strip() for v in visible.split(",") if v.strip()]
            os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU GT probing requires CUDA")
    torch.cuda.set_device(0)
    if world_size > 1:
        dist.init_process_group(backend="gloo", init_method="env://")
    try:
        output_dir = Path(args.output_dir).expanduser().resolve()
        rank_root = output_dir / "_rank_outputs"
        rank_dir = rank_root / f"rank_{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        rank_args = argparse.Namespace(**vars(args))
        rank_args.output_dir = str(rank_dir)
        rc = _run_rank_gt(rank_args, rank, world_size)
        if rc != 0:
            raise RuntimeError(f"rank {rank} GT probe failed: {rc}")
        if world_size > 1:
            dist.barrier()
        if rank == 0:
            _merge_gt_rank_jsons(rank_root, output_dir, world_size,
                                args.top_k, args.min_gt_ratio)
            import shutil
            shutil.rmtree(rank_root)
        if world_size > 1:
            dist.barrier()
    finally:
        if world_size > 1:
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(gt_distributed_main())
