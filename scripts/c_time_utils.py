"""
c_time_utils.py — C_time 回路共享工具函数

包含：视频采帧、输入构建、token 工具、时间戳解析、指标计算、
      模型加载、样本加载、GT digit 匹配。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]   # 项目根目录 = Lkmllm_code
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.project_paths import A_DATA_ROOT, ensure_directory, resolve_path, SHARED_MODELS_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# 视频采帧
# ─────────────────────────────────────────────────────────────────────────────

def sample_frames(video_path: Path, fps: float) -> list:
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
            sys.exit(f"ERROR 无法打开视频：{video_path}")
        native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        step = max(1, int(native_fps / fps))
        frames = []
        for idx in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), step):
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


# ─────────────────────────────────────────────────────────────────────────────
# 输入构建
# ─────────────────────────────────────────────────────────────────────────────

def build_inputs(processor, frames: list, query: str, device, sample_fps: float):
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames, "sample_fps": sample_fps},
            {"type": "text", "text": (
                f"When does '{query}' happen in the video? "
                "You are a video time analysis assistant. Always respond with exactly 'start: X, end: Y' where X and Y are numbers in seconds."
                "e.g. 'start: 3, end: 7'."
                "Do not include any other text. Only output the start and end times."
            )},
        ],
    }]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True
    )
    if video_inputs:
        video_tensors, video_metadatas = zip(*video_inputs)
        video_tensors   = list(video_tensors)
        video_metadatas = list(video_metadatas)
    else:
        video_tensors   = video_inputs
        video_metadatas = None

    scalar_video_kwargs = {
        k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
        for k, v in video_kwargs.items()
    }
    return processor(
        text=[text_input],
        images=image_inputs,
        videos=video_tensors,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        **scalar_video_kwargs,
    ).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Token 工具
# ─────────────────────────────────────────────────────────────────────────────

def get_gt_token_ids(tokenizer, gt_value: float) -> list[int]:
    gt_str = str(int(round(gt_value)))
    return tokenizer.encode(gt_str, add_special_tokens=False)


def get_gt_first_token_id(tokenizer, gt_value: float) -> Optional[int]:
    ids = get_gt_token_ids(tokenizer, gt_value)
    return ids[0] if ids else None


# ─────────────────────────────────────────────────────────────────────────────
# 时间戳解析 & 指标
# ─────────────────────────────────────────────────────────────────────────────

def parse_timespan(text: str) -> Optional[tuple[float, float]]:
    import re
    patterns = [
        r"start[:\s]+([\d.]+)[,\s]+end[:\s]+([\d.]+)",
        r"([\d.]+)\s*[,，]\s*([\d.]+)",
        r"([\d.]+)\s+to\s+([\d.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            s, e = float(m.group(1)), float(m.group(2))
            if s <= e:
                return s, e
    nums = re.findall(r"[\d.]+", text)
    if len(nums) >= 2:
        s, e = float(nums[0]), float(nums[1])
        if s <= e:
            return s, e
    return None


def compute_mae(pred, gt) -> float:
    return (abs(pred[0] - gt[0]) + abs(pred[1] - gt[1])) / 2.0


def compute_iou(pred, gt) -> float:
    inter = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / union if union > 0 else 0.0


def generate_prediction(model, processor, inputs: dict, max_new_tokens: int = 32):
    import torch
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    input_len = inputs["input_ids"].shape[1]
    new_ids = out_ids[0, input_len:].tolist()
    generated_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    return generated_text, parse_timespan(generated_text)


def match_gt_digits_to_steps(
    tokenizer, generated_token_ids: list[int],
    gt_start: float, gt_end: float,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """在生成序列里定位 GT digit 对应的 decode step"""
    start_tids = get_gt_token_ids(tokenizer, gt_start)
    end_tids   = get_gt_token_ids(tokenizer, gt_end)

    def find_subseq(seq, sub):
        for i in range(len(seq) - len(sub) + 1):
            if seq[i:i+len(sub)] == sub:
                return i
        return -1

    s_off = find_subseq(generated_token_ids, start_tids)
    e_off = find_subseq(generated_token_ids, end_tids)

    start_matches, end_matches = [], []

    if s_off >= 0:
        for i, tid in enumerate(start_tids):
            start_matches.append((tid, s_off + i))
    elif start_tids and start_tids[0] in generated_token_ids:
        start_matches.append((start_tids[0],
                               generated_token_ids.index(start_tids[0])))

    if e_off >= 0:
        for i, tid in enumerate(end_tids):
            end_matches.append((tid, e_off + i))
    elif end_tids:
        search_from = (s_off + len(start_tids)) if s_off >= 0 else 0
        sub = generated_token_ids[search_from:]
        if end_tids[0] in sub:
            end_matches.append((end_tids[0],
                                 search_from + sub.index(end_tids[0])))

    return start_matches, end_matches


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_processor(model_dir: Path):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto",
        trust_remote_code=True, local_files_only=True,
    )
    model.eval()
    return model, processor


def get_model_info(model) -> tuple[int, int]:
    """返回 (n_layers, intermediate_size)"""
    n_layers = len(model.model.language_model.layers)
    intermediate_size = (
        model.model.language_model.layers[0].mlp.down_proj.weight.shape[1]
    )
    return n_layers, intermediate_size


# ─────────────────────────────────────────────────────────────────────────────
# 样本加载
# ─────────────────────────────────────────────────────────────────────────────

def load_samples(
    anno_json_path: Path,
    video_dir: Path,
    video_ids: list[str] = None,
    max_samples: int = None,
) -> list[dict]:
    text = anno_json_path.read_text(encoding="utf-8").strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]

    sample_list = []

    if isinstance(raw, list):
        for info in raw:
            vp = info.get("video_path", "")  # e.g. "cosmo_cap/BVs52yd-RUQ.mp4"
            vid = Path(vp).stem              # e.g. "BVs52yd-RUQ"
            # 完整相对路径，用于拼接 video_dir
            video_rel = vp if vp.endswith(".mp4") else vp + ".mp4"

            for event in info.get("events", []):
                query = event.get("query", "")
                spans = event.get("span", [])
                if not spans:
                    continue
                s = spans[0] if isinstance(spans[0], (list, tuple)) else spans
                sample_list.append({
                    "video_id": vid,
                    "video_rel_path": video_rel,
                    "query": query,
                    "gt_start": float(s[0]),
                    "gt_end": float(s[1]),
                })
    elif isinstance(raw, dict):
        for vid, info in raw.items():
            for q, s in zip(info.get("queries", []), info["spans"]):
                sample_list.append({
                    "video_id": vid,
                    "video_rel_path": f"{vid}.mp4",
                    "query": q,
                    "gt_start": float(s[0]),
                    "gt_end": float(s[1]),
                })
    else:
        raise ValueError(f"Unsupported annotation format: {type(raw)}")

    if video_ids:
        ids = set(video_ids)
        sample_list = [s for s in sample_list if s["video_id"] in ids]
    elif max_samples:
        sample_list = sample_list[:max_samples]

    # 用完整相对路径查找视频
    sample_list = [s for s in sample_list
                   if (video_dir / s["video_rel_path"]).exists()]
    return sample_list


# ─────────────────────────────────────────────────────────────────────────────
# 路径解析（通用 CLI 参数）
# ─────────────────────────────────────────────────────────────────────────────

def resolve_common_paths(args) -> tuple[Path, Path, Path, Path, Path]:
    """返回 (anno_json_path, model_dir, video_dir, output_dir, png_dir)"""
    anno_json_path = resolve_path(
        args.anno_json or (
            A_DATA_ROOT / "datasets" / "Test" / "Charades_sta" / "charades-timelens.json"
        ),
        default=A_DATA_ROOT / "datasets" / "Test" / "Charades_sta" / "charades-timelens.json",
        must_exist=True,
    )
    model_dir = resolve_path(
        args.model_path or (SHARED_MODELS_ROOT / "Qwen3_VL_8B"),
        default=SHARED_MODELS_ROOT / "Qwen3_VL_8B",
        must_exist=True,
    )
    video_dir = resolve_path(
        args.video_dir or (
            A_DATA_ROOT / "datasets" / "Test" / "Charades_sta" / "charades"
        ),
        default=A_DATA_ROOT / "datasets" / "Test" / "Charades_sta" / "charades",
        must_exist=True,
    )
    output_dir = ensure_directory(resolve_path(
        args.output_dir, default=A_DATA_ROOT / "outputs" / "c_time"
    ))
    png_dir = ensure_directory(resolve_path(
        getattr(args, "png_dir", None) or args.output_dir,
        default=A_DATA_ROOT / "outputs" / "c_time",
    ))
    return anno_json_path, model_dir, video_dir, output_dir, png_dir