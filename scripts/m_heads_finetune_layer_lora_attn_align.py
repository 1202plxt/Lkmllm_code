"""
m_heads_finetune_layer_lora_attn_align.py
── Head-masked Q-LoRA + Attention Alignment 辅助 Loss（含 RoPE 修复）──

对归因选出的 top-K head 所在层做 LoRA，同时对 top-N head 手动计算
attention score（含 RoPE 旋转）做 alignment 监督。

关键修复：手动计算 Q/K 后调用模型自带的 rotary_emb + apply_rotary_pos_emb
做旋转，精确复现模型内部 attention 分布，而非使用位置无关的 Q_raw·K_raw。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Lkmllm_code.src.project_paths import A_DATA_ROOT, ensure_directory
from c_time_utils import load_samples, sample_frames, load_model_and_processor

DEFAULT_NUM_LAYERS = 36
DEFAULT_NUM_HEADS = 28
DEFAULT_HEAD_DIM = 128


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return distributed, local_rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def sync_gradients(params):
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


def seed_everything(seed: int):
    """设置当前训练路径使用的随机源；不强制确定性 kernel，以保留 FlashAttention。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ═══════════════ Prompt 模板 ═══════════════

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
TIMELENS_ASSISTANT_TEMPLATE = "The event happens in {start} - {end} seconds."

# ═══════════════ 工具函数 ═══════════════

def _find_assist_start(input_ids_1d: torch.Tensor, assist_ids: List[int]) -> int:
    n, m = int(input_ids_1d.shape[0]), len(assist_ids)
    if m == 0 or m > n:
        return -1
    assist_t = torch.tensor(assist_ids, device=input_ids_1d.device,
                            dtype=input_ids_1d.dtype)
    for start in range(n - m, -1, -1):
        if torch.equal(input_ids_1d[start:start + m], assist_t):
            return start
    return -1


def _find_subsequence(haystack: List[int], needle: List[int],
                      start_from: int = 0) -> int:
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return -1
    for i in range(start_from, n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1


def _build_numeric_mask(assist_ids, gt_start, gt_end, tokenizer):
    mask = [False] * len(assist_ids)
    start_pos = -1
    for fmt in (f"{gt_start:.1f}", f"{gt_start}", str(gt_start)):
        num_ids = tokenizer(fmt, add_special_tokens=False).input_ids
        pos = _find_subsequence(assist_ids, num_ids, 0)
        if pos >= 0:
            for i in range(pos, pos + len(num_ids)):
                mask[i] = True
            start_pos = pos + len(num_ids)
            break
    if start_pos < 0:
        return [False] * len(assist_ids)
    for fmt in (f"{gt_end:.1f}", f"{gt_end}", str(gt_end)):
        num_ids = tokenizer(fmt, add_special_tokens=False).input_ids
        pos = _find_subsequence(assist_ids, num_ids, start_pos)
        if pos >= 0:
            for i in range(pos, pos + len(num_ids)):
                mask[i] = True
            break
    return mask


# ═══════════════ 模型加载 ═══════════════

def load_model_with_attn(model_path: Path, attn_implementation: str = "flash_attention_2",
                         device: Optional[torch.device] = None):
    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn
            print(f"  [attn] flash_attn={flash_attn.__version__}")
        except ImportError:
            print("  [attn] flash_attn 不可用，降级为 sdpa")
            attn_implementation = "sdpa"
    if device is not None:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map={"": device},
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()
        print(f"  [attn] 使用 attn_implementation={attn_implementation} device={device}")
        return model, processor
    try:
        model, processor = load_model_and_processor(
            model_path, attn_implementation=attn_implementation)
        print(f"  [attn] 使用 attn_implementation={attn_implementation}")
        return model, processor
    except (TypeError, Exception) as e:
        print(f"  [attn] 参数加载失败: {e}，回退默认")
    model, processor = load_model_and_processor(model_path)
    return model, processor


# ═══════════════ CLI ═══════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TimeLens-aligned head/layer-masked LoRA + Attention Alignment (含 RoPE) 微调")
    p.add_argument("--attr-json", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--anno-json", required=True)
    p.add_argument("--video-dir", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--target-layers", type=str, default="12-19",
                   help="Optional layer override, e.g. '12-19' or '12,13,14'. "
                        "When set, use all heads in these layers.")
    p.add_argument("--random-heads", action="store_true")
    p.add_argument("--random-seed", type=int, default=42)

    p.add_argument("--align-top-n", type=int, default=10)
    p.add_argument("--align-weight", type=float, default=0.02)
    p.add_argument("--align-temperature", type=float, default=1.0)

    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.02)
    p.add_argument("--adapt-targets", nargs="+", default=["q_proj", "v_proj", "o_proj"],
                   choices=["q_proj", "v_proj", "o_proj"],
                   help="对选中层/head 的 q/v/o 投影做 masked LoRA；默认对齐当前 TimeLens 主线 q+v+o")

    p.add_argument("--max-samples-per-folder", type=int, default=800)
    p.add_argument("--folder-field", type=str, default=None)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=4000)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=0,
                   help="TimeLens 官方评测不额外设置 200 帧上限；0 表示不在脚本里截帧，只由 fps 和 total_tokens 控制")
    p.add_argument("--max-size", type=int, default=256)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--min-tokens", type=int, default=64)
    p.add_argument("--total-tokens", type=int, default=14336)
    p.add_argument("--attn-implementation", type=str, default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"])
    return p


# ═══════════════ Head 目标加载 ═══════════════

def load_head_targets(attr_json, top_k, align_top_n, n_layers, n_heads,
                      random_baseline=False, random_seed=42):
    if random_baseline:
        rng = random.Random(random_seed)
        all_h = [(l, h) for l in range(n_layers) for h in range(n_heads)]
        selected = rng.sample(all_h, min(top_k, len(all_h)))
        scores = {lh: 0.0 for lh in selected}
    else:
        data = json.loads(Path(attr_json).read_text(encoding="utf-8"))
        top_raw = data.get("top_k_heads", [])
        if not top_raw and "combined_score_matrix" in data:
            arr = np.array(data["combined_score_matrix"], dtype=np.float32)
            for fi in np.argsort(arr.ravel())[::-1][:top_k]:
                l, h = divmod(int(fi), arr.shape[1])
                top_raw.append({"layer": l, "head": h,
                                "combined_score": float(arr.ravel()[fi])})
        selected, scores = [], {}
        for e in top_raw[:top_k]:
            l, h = int(e.get("layer", 0)), int(e.get("head", 0))
            selected.append((l, h))
            scores[(l, h)] = float(e.get("combined_score", 0))
        print(f"  [attr] Top-{len(selected)} heads:")
        for i, (l, h) in enumerate(selected):
            print(f"    #{i+1}: L{l}H{h} score={scores.get((l,h),0):.6f}")

    target_heads: Dict[int, Set[int]] = {}
    for l, h in selected:
        target_heads.setdefault(l, set()).add(h)
    target_layers = set(target_heads.keys())

    sorted_by_score = sorted(selected, key=lambda lh: scores.get(lh, 0), reverse=True)
    align_heads = sorted_by_score[:align_top_n] if align_top_n > 0 else []

    print(f"  → {sum(len(v) for v in target_heads.values())} heads "
          f"in {len(target_layers)} layers: {sorted(target_layers)}")
    if align_heads:
        print(f"  → Align top-{len(align_heads)}: "
              f"{['L'+str(l)+'H'+str(h) for l,h in align_heads]}")
    return target_heads, target_layers, align_heads


# ═══════════════════════════════════════════════════════════════════════════
# Attention Alignment Hook & Loss（含 RoPE 修复）
# ═══════════════════════════════════════════════════════════════════════════

def parse_target_layers(spec: Optional[str], n_layers: int) -> Optional[List[int]]:
    if not spec:
        return None
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if end < start:
                start, end = end, start
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    out = sorted(l for l in layers if 0 <= l < n_layers)
    if not out:
        raise ValueError(f"--target-layers={spec!r} did not select any valid layer")
    return out


def make_all_head_targets(layers: List[int], n_heads: int):
    target_heads = {int(l): set(range(n_heads)) for l in layers}
    target_layers = set(target_heads.keys())
    print(f"  [layers] Override attribution heads with all heads in layers: {layers}")
    print(f"  -> {sum(len(v) for v in target_heads.values())} heads "
          f"in {len(target_layers)} layers")
    return target_heads, target_layers


def load_align_heads_from_attr(attr_json, align_top_n, allowed_layers=None):
    if align_top_n <= 0:
        return []
    data = json.loads(Path(attr_json).read_text(encoding="utf-8"))
    top_raw = data.get("top_k_heads", [])
    if not top_raw and "combined_score_matrix" in data:
        arr = np.array(data["combined_score_matrix"], dtype=np.float32)
        for fi in np.argsort(arr.ravel())[::-1]:
            l, h = divmod(int(fi), arr.shape[1])
            top_raw.append({"layer": l, "head": h,
                            "combined_score": float(arr.ravel()[fi])})
    allowed = set(allowed_layers) if allowed_layers is not None else None
    align_heads = []
    for e in top_raw:
        l, h = int(e.get("layer", 0)), int(e.get("head", 0))
        if allowed is not None and l not in allowed:
            continue
        align_heads.append((l, h))
        if len(align_heads) >= align_top_n:
            break
    print(f"  [align] Attr top heads for alignment: "
          f"{['L'+str(l)+'H'+str(h) for l, h in align_heads]}")
    return align_heads


def _find_rope_apply_fn(model):
    """
    找到 RoPE 旋转函数，同时返回一个"标准版本"作为降级备用。

    关键教训：不同模型的 mRoPE 实现细节不同——有的（如 Qwen2-VL）在顶层
    只算出"3 个维度堆叠"的 cos/sin（形状类似 (3, batch, seq, dim/2)），
    需要 apply_multimodal_rotary_pos_emb 按 mrope_section 切片合并；
    有的（如 Qwen3-VL）在 rotary_emb 内部就已经把 3 个维度合并成了标准
    形状 (batch, seq, head_dim)，直接用普通 apply_rotary_pos_emb 即可。

    返回 (primary_fn, is_mrope, plain_fn)：
      primary_fn: 优先尝试的函数（可能是 multimodal 或 plain）
      is_mrope:   primary_fn 是否是 multimodal 版本
      plain_fn:   标准 4 参数版本，用于运行时形状检测后降级
                  （不能直接用 primary_fn 降级调用，因为 multimodal 版本
                  的签名强制要求 mrope_section 这个第 5 个参数，少传会
                  直接 TypeError，必须换一个真正的标准函数）
    """
    import inspect
    layer0_attn = model.model.language_model.layers[0].self_attn
    own_mod = inspect.getmodule(type(layer0_attn))

    search_mods = [own_mod] if own_mod else []
    for modname in [
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    ]:
        try:
            search_mods.append(__import__(modname, fromlist=["*"]))
        except ImportError:
            continue

    # 先找一个标准版本备用（不管最终主选哪个，都需要它做降级保险）
    plain_fn = None
    for m in search_mods:
        if m and hasattr(m, "apply_rotary_pos_emb"):
            plain_fn = m.apply_rotary_pos_emb
            break

    # 主选：优先用模型自己模块内的函数，保证跟 rotary_emb 输出的 cos/sin 自洽
    if own_mod:
        if hasattr(own_mod, "apply_multimodal_rotary_pos_emb"):
            print(f"  [RoPE] 主选 mRoPE 函数: "
                  f"{own_mod.__name__}.apply_multimodal_rotary_pos_emb  "
                  f"(降级备用: {'有' if plain_fn else '无'})")
            return own_mod.apply_multimodal_rotary_pos_emb, True, plain_fn
        if hasattr(own_mod, "apply_rotary_pos_emb"):
            print(f"  [RoPE] 主选普通 RoPE 函数: "
                  f"{own_mod.__name__}.apply_rotary_pos_emb")
            return own_mod.apply_rotary_pos_emb, False, own_mod.apply_rotary_pos_emb

    # fallback：外部模块（架构可能不完全匹配，仅作最后手段）
    for m in search_mods:
        if m and hasattr(m, "apply_multimodal_rotary_pos_emb"):
            print(f"  [RoPE][fallback] {m.__name__}.apply_multimodal_rotary_pos_emb "
                  f"(警告: 非本模型模块，形状可能不匹配)")
            return m.apply_multimodal_rotary_pos_emb, True, plain_fn
    if plain_fn is not None:
        print(f"  [RoPE][fallback] 使用标准 apply_rotary_pos_emb")
        return plain_fn, False, plain_fn

    raise RuntimeError("找不到任何 apply_rotary_pos_emb / "
                       "apply_multimodal_rotary_pos_emb，请检查 transformers 版本")


def _get_mrope_section(model) -> Optional[List[int]]:
    """从 config.rope_scaling 中取 mrope_section（各维度通道数分配）。"""
    text_cfg = getattr(model.config, "text_config", model.config)
    rope_scaling = getattr(text_cfg, "rope_scaling", None)
    if rope_scaling is None:
        return None
    if isinstance(rope_scaling, dict):
        return rope_scaling.get("mrope_section")
    return getattr(rope_scaling, "mrope_section", None)


class AttnAlignHook:
    """
    Layer pre-hook 捕获 hidden_states + RoPE 相关张量，手动调 q_proj/k_proj
    得到 Q/K 后应用 RoPE 旋转，再算 attention score 做 alignment loss。

    现代 transformers 实现（Llama/Qwen2/Qwen3 系列）通常在模型顶层算一次
    rotary_emb(hidden_states, position_ids) 得到 (cos, sin)，然后把这个
    元组以 position_embeddings kwarg 逐层往下传，而不是每层重新用
    position_ids 算 —— 所以本 hook 优先捕获 position_embeddings，
    只有捕获不到时才退回 position_ids + 手动调 attn.rotary_emb()。
    """

    def __init__(self, align_heads: List[Tuple[int, int]],
                 num_heads: int, num_kv_heads: int, head_dim: int,
                 model):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_group_size = num_heads // num_kv_heads

        self.layer_heads: Dict[int, List[int]] = {}
        for l, h in align_heads:
            self.layer_heads.setdefault(l, []).append(h)

        self.captured_hidden: Dict[int, torch.Tensor] = {}
        self.captured_position_ids: Dict[int, torch.Tensor] = {}
        self.captured_position_embeddings: Dict[int, tuple] = {}
        self._layer_attns: Dict[int, nn.Module] = {}
        self._handles: List = []
        self._debug_printed = False

        self._apply_rotary, self._is_mrope, self._plain_apply_rotary = \
            _find_rope_apply_fn(model)
        self._mrope_section = _get_mrope_section(model) if self._is_mrope else None
        print(f"  [RoPE] is_mrope={self._is_mrope}  mrope_section={self._mrope_section}")
        self._debug_shape_printed = False
        self._qknorm_checked = False

    def _is_position_ids(self, t):
        if not isinstance(t, torch.Tensor):
            return False
        return t.dtype in (torch.long, torch.int, torch.int32, torch.int64) \
            and t.dim() >= 2

    def _make_layer_hook(self, layer_idx: int):
        def hook_fn(module, args, kwargs):
            if args:
                self.captured_hidden[layer_idx] = args[0]
            elif "hidden_states" in kwargs:
                self.captured_hidden[layer_idx] = kwargs["hidden_states"]

            # 一次性诊断：打出这一层 forward 实际收到哪些 kwargs
            if not self._debug_printed:
                print(f"  [debug-hook] layer{layer_idx} kwargs keys: "
                      f"{list(kwargs.keys())}")
                self._debug_printed = True

            # 优先捕获 position_embeddings=(cos, sin)（现代实现的标准传法）
            pe = kwargs.get("position_embeddings")
            if pe is not None and isinstance(pe, (tuple, list)) and len(pe) == 2:
                self.captured_position_embeddings[layer_idx] = tuple(pe)
                return

            # fallback: position_ids
            pos_ids = kwargs.get("position_ids")
            if pos_ids is not None:
                self.captured_position_ids[layer_idx] = pos_ids
                return
            for a in args[1:]:
                if self._is_position_ids(a):
                    self.captured_position_ids[layer_idx] = a
                    break
        return hook_fn

    def register(self, model):
        layers = model.model.language_model.layers
        for layer_idx in self.layer_heads:
            layer = layers[layer_idx]
            self._layer_attns[layer_idx] = layer.self_attn
            h = layer.register_forward_pre_hook(
                self._make_layer_hook(layer_idx), with_kwargs=True)
            self._handles.append(h)
        print(f"  [AttnAlignHook] {len(self._handles)} layer pre-hooks "
              f"on {len(self.layer_heads)} layers (RoPE enabled)")

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.captured_hidden.clear()
        self.captured_position_ids.clear()
        self.captured_position_embeddings.clear()
        self._layer_attns.clear()

    def clear(self):
        """每步后清空 hidden（释放显存），保留 position_ids/position_embeddings
        （同 batch 内不变，且 gc recompute 时 hook 未必会重新触发写入）。"""
        self.captured_hidden.clear()

    def clear_all(self):
        """换新 batch 前彻底清空。"""
        self.captured_hidden.clear()
        self.captured_position_ids.clear()
        self.captured_position_embeddings.clear()

    def compute_alignment_loss(
        self,
        answer_positions: List[int],
        gt_video_token_start: int,
        gt_video_token_end: int,
        temperature: float = 1.0,
    ) -> Optional[torch.Tensor]:
        """
        对每个 align head：
          1. hidden → q_proj/k_proj → Q_raw/K_raw
          2. rotary_emb(position_ids) → cos, sin
          3. apply_rotary_pos_emb(Q, K, cos, sin) → Q_rot, K_rot
          4. attention_score = softmax(Q_rot @ K_rot^T / √d)
          5. loss = -log(GT 区间 attention mass)
        """
        predict_positions = [int(p) - 1 for p in answer_positions if int(p) > 0]
        if not predict_positions or gt_video_token_end < gt_video_token_start:
            return None

        losses = []
        for layer_idx, head_list in self.layer_heads.items():
            hidden = self.captured_hidden.get(layer_idx)
            if hidden is None:
                continue
            attn = self._layer_attns.get(layer_idx)
            if attn is None:
                continue

            # Q/K raw（经过 LoRA patch）
            q_full = attn.q_proj(hidden)  # [1, seq, num_heads * head_dim]
            k_full = attn.k_proj(hidden)  # [1, seq, num_kv_heads * head_dim]
            seq_len = q_full.shape[1]

            # reshape: [1, seq, num_heads, head_dim]
            q = q_full.view(1, seq_len, self.num_heads, self.head_dim)
            k = k_full.view(1, seq_len, self.num_kv_heads, self.head_dim)

            # ── QK-Norm（Qwen3 系列架构特有）──
            # Qwen3 在 reshape 之后、RoPE 之前，对 Q/K 做逐 head 的 RMSNorm
            # （q_norm/k_norm 子模块，作用在最后一维 head_dim 上）。跳过这
            # 一步会导致 Q/K 数值量级跟模型真实前向完全不一致，是之前
            # align_loss 恒定不变的根因——不是 RoPE 的问题，是漏了这步。
            q_norm = getattr(attn, "q_norm", None)
            k_norm = getattr(attn, "k_norm", None)
            if not self._qknorm_checked:
                print(f"  [debug-qknorm] q_norm={q_norm}  k_norm={k_norm}")
                self._qknorm_checked = True
            if q_norm is not None:
                q = q_norm(q)
            if k_norm is not None:
                k = k_norm(k)

            # → [1, num_heads, seq, head_dim]（RoPE 按此维度顺序旋转）
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            # ── RoPE 旋转：优先用捕获到的 (cos, sin)，其次退回 position_ids ──
            pe = self.captured_position_embeddings.get(layer_idx)
            position_ids = self.captured_position_ids.get(layer_idx)
            if pe is not None:
                cos, sin = pe
            elif position_ids is not None:
                cos, sin = attn.rotary_emb(k, position_ids)
            else:
                cos, sin = None, None
                if not hasattr(self, '_warned_no_pos'):
                    print("  [WARN] position_embeddings/position_ids 均未捕获，"
                          "align loss 退化为无 RoPE")
                    self._warned_no_pos = True

            if cos is not None:
                if not self._debug_shape_printed:
                    print(f"  [debug-rope] cos.shape={tuple(cos.shape)} "
                          f"sin.shape={tuple(sin.shape)} q.shape={tuple(q.shape)} "
                          f"is_mrope={self._is_mrope} "
                          f"mrope_section={self._mrope_section}")
                    self._debug_shape_printed = True
                # 防御性检查：multimodal 函数要求 cos 有前置的 3 维堆叠
                # （形状形如 (3, batch, seq, dim)）。如果 cos 只有 3 维
                # （batch, seq, head_dim，说明 rotary_emb 内部已经把 3 个
                # 位置维度合并好了），说明这个模型不需要 multimodal 版本，
                # 自动降级为普通旋转，避免 IndexError。
                use_mrope_call = (self._is_mrope and self._mrope_section is not None
                                  and cos.dim() == 4 and cos.shape[0] == 3)
                if self._is_mrope and not use_mrope_call \
                        and not hasattr(self, '_mrope_shape_mismatch_warned'):
                    print(f"  [WARN] cos.shape={tuple(cos.shape)} 不是 mRoPE 堆叠"
                          f"格式 (3,batch,seq,dim)，自动降级为普通 RoPE 旋转")
                    self._mrope_shape_mismatch_warned = True
                if use_mrope_call:
                    q, k = self._apply_rotary(q, k, cos, sin, self._mrope_section)
                elif self._plain_apply_rotary is not None:
                    q, k = self._plain_apply_rotary(q, k, cos, sin)
                else:
                    raise RuntimeError(
                        "cos 形状不是 mRoPE 堆叠格式，且找不到标准 "
                        "apply_rotary_pos_emb 做降级，无法完成 RoPE 旋转")

            for head_idx in head_list:
                kv_head_idx = head_idx // self.kv_group_size

                # q_h: [n_ans, head_dim]  k_h: [seq, head_dim]
                ans_pos_t = torch.tensor(predict_positions, device=q.device)
                q_h = q[0, head_idx, ans_pos_t, :]    # [n_ans, head_dim]
                k_h = k[0, kv_head_idx, :, :]          # [seq, head_dim]

                # attention score + causal mask
                scale = (self.head_dim ** 0.5) * temperature
                attn_logits = torch.matmul(q_h, k_h.T) / scale  # [n_ans, seq]

                causal_mask = torch.zeros_like(attn_logits, dtype=torch.bool)
                for i, pos in enumerate(predict_positions):
                    causal_mask[i, pos + 1:] = True
                attn_logits = attn_logits.masked_fill(causal_mask, float("-inf"))

                attn_weights = F.softmax(attn_logits, dim=-1)

                gt_s = max(gt_video_token_start, 0)
                gt_e = min(gt_video_token_end + 1, seq_len)
                if gt_e <= gt_s:
                    continue

                gt_mass = attn_weights[:, gt_s:gt_e].sum(dim=-1).clamp(min=1e-8)
                losses.append(-torch.log(gt_mass).mean())

        if not losses:
            return None
        return torch.stack(losses).mean()


# ═══════════════ GT Video Token 区间定位 ═══════════════

def locate_gt_video_token_range(input_ids, processor, gt_start, gt_end,
                                fps, num_frames, video_duration):
    ids = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
    tokenizer = processor.tokenizer

    VISION_PAD_IDS = set()
    for attr in ("video_token_id", "image_token_id", "vision_token_id"):
        tid = getattr(processor, attr, None) or getattr(tokenizer, attr, None)
        if tid is not None and isinstance(tid, int):
            VISION_PAD_IDS.add(tid)
    try:
        for name, tid in tokenizer.get_added_vocab().items():
            if any(kw in name.lower() for kw in ["image_pad", "video_pad", "vision_pad"]):
                VISION_PAD_IDS.add(tid)
    except Exception:
        pass
    if not VISION_PAD_IDS:
        VISION_PAD_IDS = {151654, 151655, 151656}

    vision_positions = [i for i, tid in enumerate(ids) if tid in VISION_PAD_IDS]
    if not vision_positions or video_duration <= 0:
        return None
    if gt_start >= video_duration:
        return None

    total = len(vision_positions)
    ratio_s = min(1.0, max(0.0, gt_start / video_duration))
    ratio_e = min(1.0, gt_end / video_duration)
    tok_s = min(int(ratio_s * total), total - 1)
    tok_e = min(int(ratio_e * total), total - 1)
    tok_s = min(tok_s, tok_e)
    return (vision_positions[tok_s], vision_positions[tok_e])


# ═══════════════ LoRA ═══════════════

class LinearLoRA(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int,
                 alpha: float, dropout: float = 0.0,
                 input_mask: Optional[torch.Tensor] = None,
                 output_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        dtype = original_linear.weight.dtype
        dev = original_linear.weight.device
        self.lora_A = nn.Parameter(
            torch.zeros(rank, self.in_features, dtype=dtype, device=dev))
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, dtype=dtype, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if input_mask is not None:
            self.register_buffer("input_mask", input_mask.to(device=dev, dtype=dtype))
        else:
            self.input_mask = None
        if output_mask is not None:
            self.register_buffer("output_mask", output_mask.to(device=dev, dtype=dtype))
        else:
            self.output_mask = None
        object.__setattr__(self, "_orig_forward", original_linear.forward)
        object.__setattr__(self, "_orig_weight", original_linear.weight)

    def forward(self, x):
        lora_input = x if self.input_mask is None else x * self.input_mask
        delta = (self.dropout(lora_input) @ self.lora_A.T) @ self.lora_B.T
        if self.output_mask is not None:
            delta = delta * self.output_mask
        return self._orig_forward(x) + delta * self.scaling

    @torch.no_grad()
    def merge_into_weights(self):
        delta = self.lora_B @ self.lora_A
        if self.input_mask is not None:
            delta = delta * self.input_mask.unsqueeze(0)
        if self.output_mask is not None:
            delta = delta * self.output_mask.unsqueeze(1)
        self._orig_weight.data += (delta * self.scaling).to(self._orig_weight.dtype)


class LayerLoRAOverlay(nn.Module):
    def __init__(self, self_attn, layer_idx, selected_heads, adapt_targets,
                 rank, alpha, dropout, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.layer_idx = layer_idx
        self.lora_modules = nn.ModuleDict()
        selected_heads = sorted(set(selected_heads))
        kv_group_size = max(num_heads // num_kv_heads, 1)
        selected_kv_heads = sorted({h // kv_group_size for h in selected_heads})
        for name in adapt_targets:
            proj = getattr(self_attn, name, None)
            if proj is None:
                continue
            input_mask = output_mask = None
            if name == "q_proj":
                output_mask = torch.zeros(proj.out_features)
                for h in selected_heads:
                    output_mask[h * head_dim:(h + 1) * head_dim] = 1
            elif name in ("k_proj", "v_proj"):
                output_mask = torch.zeros(proj.out_features)
                for h in selected_kv_heads:
                    output_mask[h * head_dim:(h + 1) * head_dim] = 1
            elif name == "o_proj":
                input_mask = torch.zeros(proj.in_features)
                for h in selected_heads:
                    input_mask[h * head_dim:(h + 1) * head_dim] = 1
            else:
                raise ValueError(f"Head-masked LoRA 不支持投影 {name!r}")
            self.lora_modules[name] = LinearLoRA(
                proj, rank, alpha, dropout,
                input_mask=input_mask, output_mask=output_mask)
        object.__setattr__(self, "_self_attn", self_attn)
        self._orig_forwards = {}

    def patch(self):
        for name, lora in self.lora_modules.items():
            proj = getattr(self._self_attn, name)
            self._orig_forwards[name] = proj.forward
            proj.forward = lora.forward

    def unpatch(self):
        for name in self.lora_modules:
            if name in self._orig_forwards:
                getattr(self._self_attn, name).forward = self._orig_forwards[name]
        self._orig_forwards.clear()

    @torch.no_grad()
    def merge_into_weights(self):
        for lora in self.lora_modules.values():
            lora.merge_into_weights()


def apply_head_lora(model, target_heads, adapt_targets, rank, alpha, dropout,
                    num_heads, num_kv_heads, head_dim):
    for param in model.parameters():
        param.requires_grad = False
    overlays = nn.ModuleList()
    total_params = 0
    for idx in range(len(model.model.language_model.layers)):
        if idx not in target_heads:
            continue
        attn = model.model.language_model.layers[idx].self_attn
        ov = LayerLoRAOverlay(
            attn, idx, target_heads[idx], adapt_targets, rank, alpha, dropout,
            num_heads, num_kv_heads, head_dim)
        ov.patch()
        overlays.append(ov)
        lp = sum(p.numel() for p in ov.parameters())
        total_params += lp
        print(f"  L{idx:02d} heads={sorted(target_heads[idx])}: "
              f"masked LoRA [{','.join(ov.lora_modules.keys())}] params={lp:,}")
    print(f"  Head-masked LoRA total: {total_params:,}")
    return overlays


# ═══════════════ Checkpoint ═══════════════

def save_checkpoint(overlays, optimizer, scheduler, epoch, step, path):
    torch.save({"architecture": "head_masked_lora_v1",
                "overlay_state_dict": overlays.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch, "step": step}, path)

def load_checkpoint(path, overlays, optimizer, scheduler):
    if not os.path.exists(path):
        return 0, 0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("architecture") != "head_masked_lora_v1":
        print("  [WARN] 检测到旧版 Layer-LoRA checkpoint，不能用于新的 "
              "Head-masked LoRA；本次将从头训练并在首次保存时覆盖它")
        return 0, 0
    overlays.load_state_dict(ckpt["overlay_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["step"]


def get_video_duration(video_path: Path) -> float:
    try:
        import decord
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        fps = vr.get_avg_fps()
        if fps and fps > 0:
            return float(len(vr) / fps)
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap.release()
            if fps > 0 and count > 0:
                return float(count / fps)
    except Exception:
        pass
    return 0.0


# ═══════════════ Dataset ═══════════════

class VideoDataset(Dataset):
    def __init__(self, samples, video_dir, fps, processor,
                 max_frames=0, max_size=256, min_tokens=64, total_tokens=14336):
        self.samples = samples
        self.video_dir = Path(video_dir)
        self.fps = fps
        self.processor = processor
        self.max_frames = max_frames
        self.max_size = max_size
        self.min_tokens = min_tokens
        self.total_tokens = total_tokens

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        vp = self.video_dir / s.get("video_rel_path", f"{s['video_id']}.mp4")
        frames, duration = [], 0.0
        if "duration" in s:
            duration = float(s["duration"])
        elif "video_duration" in s:
            duration = float(s["video_duration"])
        if vp.exists():
            try:
                if duration <= 0:
                    duration = get_video_duration(vp)
                frames = sample_frames(vp, self.fps)
            except Exception as e:
                if idx < 10:
                    print(f"  [WARN] 视频读取失败 idx={idx} vp={vp}: {e}")
            if self.max_frames and self.max_frames > 0 and len(frames) > self.max_frames:
                ii = [int(i * len(frames) / self.max_frames)
                      for i in range(self.max_frames)]
                frames = [frames[i] for i in ii]
            if self.total_tokens <= 0:
                out = []
                for f in frames:
                    w, h = f.size
                    if max(w, h) > self.max_size:
                        sc = self.max_size / max(w, h)
                        f = f.resize((int(w * sc), int(h * sc)))
                    out.append(f)
                frames = out
        else:
            if idx < 10:
                print(f"  [WARN] 视频文件不存在 idx={idx} vp={vp}")
        if duration <= 0 and self.fps > 0:
            duration = len(frames) / self.fps
        gt_end = float(s["gt_end"])
        if gt_end > duration:
            duration = gt_end
        effective_fps = (len(frames) / duration) if duration > 0 else self.fps
        return {"frames": frames, "query": s["query"],
                "gt_start": float(s["gt_start"]), "gt_end": gt_end,
                "duration": duration, "num_frames": len(frames), "fps": effective_fps,
                "requested_fps": self.fps,
                "min_tokens": self.min_tokens, "total_tokens": self.total_tokens}


def collate_fn_with_processor(batch, processor, device):
    from qwen_vl_utils import process_vision_info
    item = batch[0]
    if not item["frames"]:
        return None
    # Qwen3-VL uses a 16x16 vision patch and a 2x2 spatial merge, so one
    # merged visual token corresponds to 32x32 input pixels.
    PATCH_PIXELS = 32 * 32
    video_content = {"type": "video", "video": item["frames"],
                     "sample_fps": item["fps"]}
    if item.get("total_tokens", 0) > 0:
        video_content["min_pixels"] = item["min_tokens"] * PATCH_PIXELS
        video_content["total_pixels"] = item["total_tokens"] * PATCH_PIXELS

    messages = [
        {"role": "system", "content": [{"type": "text", "text": TIMELENS_SYSTEM}]},
        {"role": "user", "content": [
            video_content,
            {"type": "text", "text": TIMELENS_USER_TEMPLATE.format(query=item["query"])},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": TIMELENS_ASSISTANT_TEMPLATE.format(
                start=item["gt_start"], end=item["gt_end"])},
        ]},
    ]
    full_text = processor.apply_chat_template(messages, tokenize=False)
    imgs, vids, vkw = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if vids:
        vtens, vmetas = zip(*vids)
        vtens, vmetas = list(vtens), list(vmetas)
    else:
        vtens, vmetas = vids, None
    svkw = {k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
            for k, v in vkw.items()}
    inputs = processor(text=[full_text], images=imgs, videos=vtens,
                       video_metadata=vmetas, padding=True,
                       return_tensors="pt", **svkw)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    labels = inputs["input_ids"].clone()
    labels[:] = -100
    assist_text = TIMELENS_ASSISTANT_TEMPLATE.format(
        start=item["gt_start"], end=item["gt_end"])
    assist_ids = processor.tokenizer(assist_text, add_special_tokens=False).input_ids
    start = _find_assist_start(inputs["input_ids"][0], assist_ids)

    answer_positions = []
    if start >= 0:
        num_mask = _build_numeric_mask(assist_ids, item["gt_start"],
                                       item["gt_end"], processor.tokenizer)
        for i, keep in enumerate(num_mask):
            if keep:
                labels[0, start + i] = inputs["input_ids"][0, start + i]
                answer_positions.append(start + i)
    inputs["labels"] = labels
    inputs["_answer_positions"] = answer_positions
    inputs["_gt_start"] = item["gt_start"]
    inputs["_gt_end"] = item["gt_end"]
    inputs["_duration"] = item["duration"]
    inputs["_num_frames"] = item["num_frames"]
    inputs["_effective_fps"] = item["fps"]
    return inputs


# ═══════════════ 分层采样 ═══════════════

def infer_source_folder(s, field=None):
    if field and field in s: return str(s[field])
    for k in ("source", "dataset", "domain", "folder"):
        if k in s and s[k]: return str(s[k])
    for k in ("video_rel_path", "video_path", "video"):
        if k in s and s[k]:
            parts = str(s[k]).replace("\\", "/").split("/")
            if len(parts) > 1: return parts[0]
    vid = str(s.get("video_id", ""))
    return vid.split("_", 1)[0] if "_" in vid else "_unknown_"


def stratified_sample(samples, max_per_folder, seed=42, folder_field=None):
    if not max_per_folder or max_per_folder <= 0:
        return samples
    rng = random.Random(seed)
    groups: Dict[str, list] = {}
    for s in samples:
        groups.setdefault(infer_source_folder(s, folder_field), []).append(s)
    selected = []
    for folder in sorted(groups):
        items = groups[folder]
        rng.shuffle(items)
        chosen = items[:max_per_folder]
        selected.extend(chosen)
        print(f"    [{folder}] {len(items)} → {len(chosen)}")
    rng.shuffle(selected)
    print(f"  Stratified: {len(selected)} samples")
    return selected


def _fmt_layers(layer_set) -> str:
    if not layer_set:
        return "[]"
    return "[" + ",".join(str(x) for x in sorted(layer_set)) + "]"


def _fmt_heads(heads) -> str:
    if not heads:
        return "[]"
    return "[" + ", ".join(f"L{int(l)}H{int(h)}" for l, h in heads) + "]"


def print_launch_config(args, distributed: bool, local_rank: int, world_size: int):
    """训练启动前先打印 CLI/default 解析后的关键参数。"""
    if not is_main_process():
        return
    print("\n" + "=" * 88)
    print("TimeLens LoRA fine-tuning launch config")
    print("=" * 88)
    print("[paths]")
    print(f"  model_path       = {args.model_path}")
    print(f"  attr_json        = {args.attr_json}")
    print(f"  anno_json        = {args.anno_json}")
    print(f"  video_dir        = {args.video_dir}")
    print(f"  output_dir       = {args.output_dir}")
    print("[distributed]")
    print(f"  distributed      = {distributed}")
    print(f"  local_rank       = {local_rank}")
    print(f"  world_size       = {world_size}")
    print(f"  effective_global_batch = {world_size} x 1 x "
          f"{args.gradient_accumulation_steps} = "
          f"{world_size * args.gradient_accumulation_steps}")
    print("[TimeLens video/input]")
    print(f"  fps              = {args.fps}")
    print(f"  max_frames       = {args.max_frames} "
          f"({'no extra frame cap' if args.max_frames == 0 else 'uniform frame cap'})")
    print(f"  max_size         = {args.max_size} "
          f"({'unused when total_tokens>0' if args.total_tokens > 0 else 'manual resize path'})")
    print(f"  min_tokens       = {args.min_tokens}")
    print(f"  total_tokens     = {args.total_tokens}")
    print(f"  attn_impl        = {args.attn_implementation}")
    print("[sampling/data]")
    print(f"  max_samples      = {args.max_samples}")
    print(f"  max_samples_per_folder = {args.max_samples_per_folder}")
    print(f"  folder_field     = {args.folder_field}")
    print(f"  sample_seed      = {args.sample_seed}")
    print("[LoRA/alignment]")
    print(f"  target_layers    = {args.target_layers}")
    print(f"  top_k            = {args.top_k}")
    print(f"  random_heads     = {args.random_heads}")
    print(f"  adapt_targets    = {args.adapt_targets}")
    print(f"  lora_rank        = {args.lora_rank}")
    print(f"  lora_alpha       = {args.lora_alpha}")
    print(f"  lora_dropout     = {args.lora_dropout}")
    print(f"  align_top_n      = {args.align_top_n}")
    print(f"  align_weight     = {args.align_weight}")
    print(f"  align_temperature= {args.align_temperature}")
    print("[optimization]")
    print(f"  lr               = {args.lr}")
    print(f"  epochs           = {args.epochs}")
    print(f"  grad_accum       = {args.gradient_accumulation_steps}")
    print(f"  grad_clip        = {args.grad_clip}")
    print(f"  save_every       = {args.save_every}")
    print("=" * 88 + "\n")


def print_resolved_training_config(args, samples, target_heads, target_layers,
                                   align_heads, n_layers: int, n_heads: int,
                                   distributed: bool, world_size: int):
    """选完实际层/head 后，在 epoch 开始前打印最终训练配置。"""
    if not is_main_process():
        return
    n_target_heads = sum(len(v) for v in target_heads.values())
    print("\n" + "=" * 88)
    print("Resolved training config")
    print("=" * 88)
    print(f"  loaded_samples   = {len(samples)}")
    print(f"  model_layers     = {n_layers}")
    print(f"  attention_heads  = {n_heads}")
    print(f"  target_layers    = {_fmt_layers(target_layers)}")
    print(f"  target_head_count= {n_target_heads}")
    print(f"  adapt_targets    = {args.adapt_targets}")
    print(f"  align_enabled    = {bool(align_heads and args.align_weight > 0)}")
    print(f"  align_heads      = {_fmt_heads(align_heads)}")
    print(f"  align_weight     = {args.align_weight}")
    print(f"  video_encoding   = fps={args.fps}, max_frames={args.max_frames}, "
          f"min_tokens={args.min_tokens}, total_tokens={args.total_tokens}")
    print(f"  optimizer        = AdamW(lr={args.lr}), epochs={args.epochs}, "
          f"grad_accum={args.gradient_accumulation_steps}, grad_clip={args.grad_clip}")
    print(f"  distributed      = {distributed}, world_size={world_size}, "
          f"effective_global_batch={world_size * args.gradient_accumulation_steps}")
    print("=" * 88 + "\n")


# ═══════════════ 训练循环 ═══════════════

def train(model, processor, dataloader, device, target_heads, target_layers,
          align_heads, args, sampler=None):
    out_dir = ensure_directory(Path(args.output_dir))
    ckpt_path = str(out_dir / "lora_layer_checkpoint.pt")
    use_align = len(align_heads) > 0 and args.align_weight > 0

    print("=" * 60)
    print(f"Head-masked Q-LoRA + {'Attn Align (RoPE)' if use_align else 'CE only'}")
    print(f"  Layers={sorted(target_layers)}  rank={args.lora_rank}")
    if use_align:
        print(f"  Align heads={['L'+str(l)+'H'+str(h) for l,h in align_heads]}  "
              f"λ={args.align_weight}")
    print("=" * 60)

    text_cfg = getattr(model.config, "text_config", model.config)
    num_heads = getattr(text_cfg, "num_attention_heads", DEFAULT_NUM_HEADS)
    num_kv_heads = getattr(text_cfg, "num_key_value_heads", num_heads)
    head_dim = getattr(text_cfg, "head_dim", None)
    if head_dim is None:
        head_dim = text_cfg.hidden_size // num_heads

    overlays = apply_head_lora(
        model, target_heads, args.adapt_targets, args.lora_rank,
        args.lora_alpha, args.lora_dropout,
        num_heads, num_kv_heads, head_dim)
    overlays.to(device)

    attn_hook = None
    if use_align:

        attn_hook = AttnAlignHook(
            align_heads, num_heads, num_kv_heads, head_dim, model)
        attn_hook.register(model)
        print(f"  GQA: heads={num_heads}, kv_heads={num_kv_heads}, "
              f"head_dim={head_dim}")

    lora_params = list(overlays.parameters())
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(total_steps // 10, 1), max(total_steps, 1))
    start_epoch, start_step = load_checkpoint(ckpt_path, overlays, optimizer, scheduler)
    model.train()

    for epoch in range(start_epoch, args.epochs):
        tot_ce = tot_align = tot_loss = 0.0
        n_valid = n_oom = n_skip = n_empty = n_align = 0
        optimizer.zero_grad()

        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc=f"Epoch {epoch+1}/{args.epochs}",
                    disable=not is_main_process())
        for step, batch in pbar:
            if epoch == start_epoch and step < start_step:
                continue
            if batch is None:
                n_skip += 1
                if n_skip <= 5:
                    pbar.write(f"  [skip] step={step+1}: batch is None "
                               f"(collate_fn 返回 None，通常是视频文件"
                               f"读取失败或 frames 为空)")
                continue

            ans_pos = batch.pop("_answer_positions", [])
            gt_s = batch.pop("_gt_start", 0.0)
            gt_e = batch.pop("_gt_end", 0.0)
            dur = batch.pop("_duration", 0.0)
            nf = batch.pop("_num_frames", 0)
            eff_fps = batch.pop("_effective_fps", args.fps)

            if (batch["labels"] != -100).sum().item() == 0:
                n_empty += 1
                if attn_hook: attn_hook.clear_all()
                continue

            try:
                if attn_hook: attn_hook.clear_all()  # 新 batch，彻底清空
                outputs = model(**batch)
            except torch.cuda.OutOfMemoryError:
                n_oom += 1
                optimizer.zero_grad(); torch.cuda.empty_cache()
                if attn_hook: attn_hook.clear_all()
                continue
            except Exception:
                n_skip += 1; optimizer.zero_grad()
                if attn_hook: attn_hook.clear_all()
                continue

            ce_loss = outputs.loss
            align_loss = None

            if use_align and ans_pos and dur > 0:
                # 前 3 步打印诊断
                if n_valid < 3:
                    caps = list(attn_hook.captured_hidden.keys())
                    pos_caps = list(attn_hook.captured_position_ids.keys())
                    pe_caps = list(attn_hook.captured_position_embeddings.keys())
                    pbar.write(
                        f"  [diag] step={step+1}: hidden_layers={caps}, "
                        f"pos_id_layers={pos_caps}, pe_layers={pe_caps}, "
                        f"ans={len(ans_pos)}, gt=[{gt_s:.1f},{gt_e:.1f}] "
                        f"dur={dur:.1f} nf={nf} eff_fps={eff_fps:.4f}")

                gt_range = locate_gt_video_token_range(
                    batch["input_ids"], processor, gt_s, gt_e, args.fps, nf, dur)

                if n_valid < 3:
                    pbar.write(f"  [diag] gt_range={gt_range}")

                if gt_range is not None:
                    align_loss = attn_hook.compute_alignment_loss(
                        ans_pos, gt_range[0], gt_range[1], args.align_temperature)
                    if n_valid < 3:
                        pbar.write(f"  [diag] align_loss={align_loss}")

            combined = ce_loss + args.align_weight * align_loss \
                if align_loss is not None else ce_loss
            if align_loss is not None:
                tot_align += align_loss.item()
                n_align += 1

            (combined / args.gradient_accumulation_steps).backward()
            if attn_hook: attn_hook.clear()  # 只清 hidden 释放显存，保留 pos_ids
            tot_ce += ce_loss.item()
            tot_loss += combined.item()
            n_valid += 1

            if (step + 1) % args.gradient_accumulation_steps == 0 or \
                    (step + 1) == len(dataloader):
                sync_gradients(lora_params)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            if n_valid > 0:
                pf = dict(ce=f"{tot_ce/n_valid:.4f}",
                          total=f"{tot_loss/n_valid:.4f}", oom=n_oom)
                if use_align and n_align > 0:
                    pf["align"] = f"{tot_align/n_align:.4f}"
                    pf["a_hit"] = f"{n_align}/{n_valid}"
                pbar.set_postfix(pf)

            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                if is_main_process():
                    save_checkpoint(overlays, optimizer, scheduler,
                                    epoch, step + 1, ckpt_path)
        pbar.close()
        if is_main_process():
            print(f"  Epoch {epoch+1}: valid={n_valid} oom={n_oom} skip={n_skip} "
                  f"empty={n_empty}")
            print(f"    ce={tot_ce/max(n_valid,1):.4f}  "
                  f"align={tot_align/max(n_align,1):.4f}({n_align}/{n_valid})  "
                  f"total={tot_loss/max(n_valid,1):.4f}")
            save_checkpoint(overlays, optimizer, scheduler, epoch + 1, 0, ckpt_path)

    if attn_hook: attn_hook.remove()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # 保存 adapter
    if is_main_process():
        torch.save({
            "overlay_state_dict": overlays.state_dict(),
            "target_heads": {l: sorted(s) for l, s in target_heads.items()},
            "target_layers": sorted(target_layers),
            "align_heads": align_heads,
            "adapt_targets": args.adapt_targets,
            "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
            "align_weight": args.align_weight,
        }, out_dir / "lora_layer_adapter.pt")

        # merge 回原始权重
        print("Merging LoRA...")
        for ov in overlays:
            ov.merge_into_weights(); ov.unpatch()
        model.save_pretrained(out_dir)
        processor.save_pretrained(out_dir)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return overlays


# ═══════════════ Main ═══════════════

def main(argv=None):
    args = build_parser().parse_args(argv)
    distributed, local_rank, world_size = setup_distributed()
    ensure_directory(Path(args.output_dir))
    seed_everything(args.sample_seed + local_rank)

    print_launch_config(args, distributed, local_rank, world_size)

    if is_main_process():
        print(f"Distributed: {distributed} world_size={world_size}")
        print("Loading model...")
    load_device = torch.device(f"cuda:{local_rank}") if distributed else None
    model, processor = load_model_with_attn(
        Path(args.model_path), args.attn_implementation, device=load_device)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if is_main_process():
            print("  gradient checkpointing enabled")
    device = next(model.parameters()).device

    if is_main_process():
        print("\nLoading data...")
    if args.max_samples_per_folder > 0:
        try:
            samples = load_samples(Path(args.anno_json), Path(args.video_dir),
                                   max_samples=None)
        except TypeError:
            samples = load_samples(Path(args.anno_json), Path(args.video_dir),
                                   max_samples=10**9)
        samples = stratified_sample(samples, args.max_samples_per_folder,
                                    args.sample_seed, args.folder_field)
    else:
        samples = load_samples(Path(args.anno_json), Path(args.video_dir),
                               max_samples=args.max_samples)
    if not samples:
        print("[ERROR] No samples"); return 1
    if is_main_process():
        print(f"Samples: {len(samples)}")

    dataset = VideoDataset(samples, args.video_dir, args.fps, processor,
                           args.max_frames, args.max_size,
                           args.min_tokens, args.total_tokens)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=local_rank,
        shuffle=True, seed=args.sample_seed) if distributed else None
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.sample_seed)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=(sampler is None),
        sampler=sampler,
        generator=loader_generator if sampler is None else None,
        collate_fn=lambda x: collate_fn_with_processor(x, processor, device))

    text_cfg = getattr(model.config, "text_config", model.config)
    n_layers = len(model.model.language_model.layers)
    n_heads = getattr(text_cfg, "num_attention_heads", DEFAULT_NUM_HEADS)

    manual_layers = parse_target_layers(args.target_layers, n_layers)
    if manual_layers is not None:
        target_heads, target_layers = make_all_head_targets(
            manual_layers, n_heads)
        align_heads = load_align_heads_from_attr(
            args.attr_json, args.align_top_n, allowed_layers=target_layers)
    else:
        target_heads, target_layers, align_heads = load_head_targets(
            args.attr_json, args.top_k, args.align_top_n,
            n_layers, n_heads, args.random_heads, args.random_seed)
    if not target_layers:
        print("[ERROR] No target layers"); return 1

    print_resolved_training_config(
        args, samples, target_heads, target_layers, align_heads,
        n_layers, n_heads, distributed, world_size)

    if is_main_process():
        Path(args.output_dir, "config.json").write_text(json.dumps({
        "architecture": "head_masked_lora_v1",
        "distributed": distributed,
        "world_size": world_size,
        "command": [str(x) for x in sys.argv],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "model_path": str(Path(args.model_path).resolve()),
        "attr_json": str(Path(args.attr_json).resolve()),
        "anno_json": str(Path(args.anno_json).resolve()),
        "video_dir": str(Path(args.video_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "target_heads": {str(l): sorted(hs) for l, hs in target_heads.items()},
        "target_layers": sorted(target_layers),
        "target_layers_arg": args.target_layers,
        "align_heads": [list(lh) for lh in align_heads],
        "top_k": args.top_k,
        "random_heads": args.random_heads,
        "random_seed": args.random_seed,
        "align_top_n": args.align_top_n,
        "align_weight": args.align_weight,
        "align_temperature": args.align_temperature,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "adapt_targets": args.adapt_targets,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "max_size": args.max_size,
        "min_tokens": args.min_tokens,
        "total_tokens": args.total_tokens,
        "attn_implementation": args.attn_implementation,
        "lr": args.lr,
        "epochs": args.epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "grad_clip": args.grad_clip,
        "save_every": args.save_every,
        "max_samples": args.max_samples,
        "max_samples_per_folder": args.max_samples_per_folder,
        "folder_field": args.folder_field,
        "sample_seed": args.sample_seed,
        "num_loaded_samples": len(samples),
        }, indent=2))

    train(model, processor, dataloader, device, target_heads, target_layers,
          align_heads, args, sampler=sampler)
    if is_main_process():
        print("\nDone!")
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
