"""
heads_finetune_layer_lora_attn_align_multigpu.py
── Layer LoRA + Attention Alignment 辅助 Loss（含 RoPE 修复）── 多卡并行版

基于 PyTorch DistributedDataParallel (DDP) 的多卡训练适配。
启动方式：
  torchrun --nproc_per_node=${NUM_GPUS} \
  scripts/heads_finetune_layer_lora_attn_align.py \
  --attr-json   ../Lkmllm_data/outputs/startend_gradient_head_attr/startend_gradient_head_attribution.json \
  --model-path  ../shared_models/Qwen3-VL-8B-Instruct \
  --anno-json   ../Lkmllm_data/datasets/Train/timelens-100k/timelens-100k.jsonl \
  --video-dir   ../Lkmllm_data/datasets/Train/timelens-100k \
  --output-dir  ../Lkmllm_data/checkpoints/lora_layer \
  --top-k 20 --align-top-n 20 --align-weight 0.1 \
  --lr 1e-5 --epochs 10
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]   # 项目根目录 = Lkmllm_code
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.project_paths import A_DATA_ROOT, ensure_directory
from c_time_utils import load_samples, sample_frames, load_model_and_processor

DEFAULT_NUM_LAYERS = 36
DEFAULT_NUM_HEADS = 28
DEFAULT_HEAD_DIM = 128

# ═══════════════ 分布式工具 ═══════════════

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def print_rank0(*args, **kwargs):
    """仅 rank 0 打印"""
    if is_main_process():
        print(*args, **kwargs)

def setup_distributed():
    """初始化分布式环境（由 torchrun 自动设置环境变量）"""
    if "RANK" not in os.environ:
        print("[WARN] 未检测到分布式环境变量，以单卡模式运行")
        return 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[DDP] 初始化完成: world_size={world_size}")
    dist.barrier()
    return rank, world_size, device

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def reduce_mean(tensor, world_size):
    """跨进程平均一个标量 tensor"""
    if world_size <= 1 or not dist.is_initialized():
        return tensor
    t = tensor.clone().detach()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / world_size


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
                         device=None):
    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn
            print_rank0(f"  [attn] flash_attn={flash_attn.__version__}")
        except ImportError:
            print_rank0("  [attn] flash_attn 不可用，降级为 sdpa")
            attn_implementation = "sdpa"

    # 在 DDP 下，强制将整套模型直接加载到当前 rank 指定的单卡设备上，
    # 避免 Accelerate 自动跨卡切分 (device_map="auto") 导致的显存叠加与 Hook 冲突
    extra_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": attn_implementation,
    }
    if device is not None:
        extra_kwargs["device_map"] = {"": device}

    try:
        model, processor = load_model_and_processor(model_path, **extra_kwargs)
        print_rank0(f"  [attn] 成功加载模型，使用 attn_implementation={attn_implementation}")
        return model, processor
    except (TypeError, Exception) as e:
        print_rank0(f"  [attn] 扩展参数加载失败 ({e})，尝试回退基础加载模式")
        model, processor = load_model_and_processor(model_path)
        
        # 仅当模型未挂载 accelerate hook 且未量化时，才显式移动到 device
        if device is not None and not hasattr(model, "_hf_hook") \
           and not getattr(model, "is_loaded_in_8bit", False) \
           and not getattr(model, "is_quantized", False):
            model = model.to(device)
        return model, processor


# ═══════════════ CLI ═══════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Layer LoRA + Attention Alignment (含 RoPE) 微调 - 多卡并行版")
    p.add_argument("--attr-json", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--anno-json", required=True)
    p.add_argument("--video-dir", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--random-heads", action="store_true")
    p.add_argument("--random-seed", type=int, default=42)

    p.add_argument("--align-top-n", type=int, default=5)
    p.add_argument("--align-weight", type=float, default=0.1)
    p.add_argument("--align-temperature", type=float, default=1.0)

    p.add_argument("--lora-rank", type=int, default=4)
    p.add_argument("--lora-alpha", type=float, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--adapt-targets", nargs="+", default=["q_proj", "v_proj"])

    p.add_argument("--max-samples-per-folder", type=int, default=6000)
    p.add_argument("--folder-field", type=str, default=None)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=50000)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--max-size", type=int, default=256)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--min-tokens", type=int, default=16)
    p.add_argument("--total-tokens", type=int, default=2048)
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
        print_rank0(f"  [attr] Top-{len(selected)} heads:")
        for i, (l, h) in enumerate(selected):
            print_rank0(f"    #{i+1}: L{l}H{h} score={scores.get((l,h),0):.6f}")

    target_heads: Dict[int, Set[int]] = {}
    for l, h in selected:
        target_heads.setdefault(l, set()).add(h)
    target_layers = set(target_heads.keys())

    sorted_by_score = sorted(selected, key=lambda lh: scores.get(lh, 0), reverse=True)
    align_heads = sorted_by_score[:align_top_n] if align_top_n > 0 else []

    print_rank0(f"  → {sum(len(v) for v in target_heads.values())} heads "
                f"in {len(target_layers)} layers: {sorted(target_layers)}")
    if align_heads:
        print_rank0(f"  → Align top-{len(align_heads)}: "
                    f"{['L'+str(l)+'H'+str(h) for l,h in align_heads]}")
    return target_heads, target_layers, align_heads


# ═══════════════════════════════════════════════════════════════════════════
# Attention Alignment Hook & Loss（含 RoPE 修复）
# ═══════════════════════════════════════════════════════════════════════════

def _find_rope_apply_fn(model):
    """
    找到 RoPE 旋转函数。model 可能是 DDP 包裹后的，需要用 unwrapped model。
    """
    import inspect
    raw_model = model.module if hasattr(model, "module") else model
    layer0_attn = raw_model.model.language_model.layers[0].self_attn
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

    plain_fn = None
    for m in search_mods:
        if m and hasattr(m, "apply_rotary_pos_emb"):
            plain_fn = m.apply_rotary_pos_emb
            break

    if own_mod:
        if hasattr(own_mod, "apply_multimodal_rotary_pos_emb"):
            print_rank0(f"  [RoPE] 主选 mRoPE 函数: "
                        f"{own_mod.__name__}.apply_multimodal_rotary_pos_emb  "
                        f"(降级备用: {'有' if plain_fn else '无'})")
            return own_mod.apply_multimodal_rotary_pos_emb, True, plain_fn
        if hasattr(own_mod, "apply_rotary_pos_emb"):
            print_rank0(f"  [RoPE] 主选普通 RoPE 函数: "
                        f"{own_mod.__name__}.apply_rotary_pos_emb")
            return own_mod.apply_rotary_pos_emb, False, own_mod.apply_rotary_pos_emb

    for m in search_mods:
        if m and hasattr(m, "apply_multimodal_rotary_pos_emb"):
            print_rank0(f"  [RoPE][fallback] {m.__name__}.apply_multimodal_rotary_pos_emb")
            return m.apply_multimodal_rotary_pos_emb, True, plain_fn
    if plain_fn is not None:
        print_rank0(f"  [RoPE][fallback] 使用标准 apply_rotary_pos_emb")
        return plain_fn, False, plain_fn

    raise RuntimeError("找不到任何 apply_rotary_pos_emb / "
                       "apply_multimodal_rotary_pos_emb，请检查 transformers 版本")


def _get_mrope_section(model) -> Optional[List[int]]:
    raw_model = model.module if hasattr(model, "module") else model
    text_cfg = getattr(raw_model.config, "text_config", raw_model.config)
    rope_scaling = getattr(text_cfg, "rope_scaling", None)
    if rope_scaling is None:
        return None
    if isinstance(rope_scaling, dict):
        return rope_scaling.get("mrope_section")
    return getattr(rope_scaling, "mrope_section", None)


class AttnAlignHook:
    """
    Layer pre-hook 捕获 hidden_states + RoPE 相关张量。
    多卡版：hook 注册在底层模型（非 DDP wrapper）上。
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
        print_rank0(f"  [RoPE] is_mrope={self._is_mrope}  mrope_section={self._mrope_section}")
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

            if not self._debug_printed:
                print_rank0(f"  [debug-hook] layer{layer_idx} kwargs keys: "
                            f"{list(kwargs.keys())}")
                self._debug_printed = True

            pe = kwargs.get("position_embeddings")
            if pe is not None and isinstance(pe, (tuple, list)) and len(pe) == 2:
                self.captured_position_embeddings[layer_idx] = tuple(pe)
                return

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
        """注册 hook 到底层模型（unwrap DDP）"""
        raw_model = model.module if hasattr(model, "module") else model
        layers = raw_model.model.language_model.layers
        for layer_idx in self.layer_heads:
            layer = layers[layer_idx]
            self._layer_attns[layer_idx] = layer.self_attn
            h = layer.register_forward_pre_hook(
                self._make_layer_hook(layer_idx), with_kwargs=True)
            self._handles.append(h)
        print_rank0(f"  [AttnAlignHook] {len(self._handles)} layer pre-hooks "
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
        self.captured_hidden.clear()

    def clear_all(self):
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
        if not answer_positions or gt_video_token_end < gt_video_token_start:
            return None

        losses = []
        for layer_idx, head_list in self.layer_heads.items():
            hidden = self.captured_hidden.get(layer_idx)
            if hidden is None:
                continue
            attn = self._layer_attns.get(layer_idx)
            if attn is None:
                continue

            q_full = attn.q_proj(hidden)
            k_full = attn.k_proj(hidden)
            seq_len = q_full.shape[1]

            q = q_full.view(1, seq_len, self.num_heads, self.head_dim)
            k = k_full.view(1, seq_len, self.num_kv_heads, self.head_dim)

            q_norm = getattr(attn, "q_norm", None)
            k_norm = getattr(attn, "k_norm", None)
            if not self._qknorm_checked:
                print_rank0(f"  [debug-qknorm] q_norm={q_norm}  k_norm={k_norm}")
                self._qknorm_checked = True
            if q_norm is not None:
                q = q_norm(q)
            if k_norm is not None:
                k = k_norm(k)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            pe = self.captured_position_embeddings.get(layer_idx)
            position_ids = self.captured_position_ids.get(layer_idx)
            if pe is not None:
                cos, sin = pe
            elif position_ids is not None:
                cos, sin = attn.rotary_emb(k, position_ids)
            else:
                cos, sin = None, None
                if not hasattr(self, '_warned_no_pos'):
                    print_rank0("  [WARN] position_embeddings/position_ids 均未捕获，"
                                "align loss 退化为无 RoPE")
                    self._warned_no_pos = True

            if cos is not None:
                if not self._debug_shape_printed:
                    print_rank0(f"  [debug-rope] cos.shape={tuple(cos.shape)} "
                                f"sin.shape={tuple(sin.shape)} q.shape={tuple(q.shape)} "
                                f"is_mrope={self._is_mrope} "
                                f"mrope_section={self._mrope_section}")
                    self._debug_shape_printed = True
                use_mrope_call = (self._is_mrope and self._mrope_section is not None
                                  and cos.dim() == 4 and cos.shape[0] == 3)
                if self._is_mrope and not use_mrope_call \
                        and not hasattr(self, '_mrope_shape_mismatch_warned'):
                    print_rank0(f"  [WARN] cos.shape={tuple(cos.shape)} 不是 mRoPE 堆叠"
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

                ans_pos_t = torch.tensor(answer_positions, device=q.device)
                q_h = q[0, head_idx, ans_pos_t, :]
                k_h = k[0, kv_head_idx, :, :]

                scale = (self.head_dim ** 0.5) * temperature
                attn_logits = torch.matmul(q_h, k_h.T) / scale

                causal_mask = torch.zeros_like(attn_logits, dtype=torch.bool)
                for i, pos in enumerate(answer_positions):
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

    total = len(vision_positions)
    ratio_s = max(0.0, gt_start / video_duration)
    ratio_e = min(1.0, gt_end / video_duration)
    tok_s = min(int(ratio_s * total), total - 1)
    tok_e = min(int(ratio_e * total), total - 1)
    tok_s = min(tok_s, tok_e)
    return (vision_positions[tok_s], vision_positions[tok_e])


# ═══════════════ LoRA ═══════════════

class LinearLoRA(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int,
                 alpha: float, dropout: float = 0.0):
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
        object.__setattr__(self, "_orig_forward", original_linear.forward)
        object.__setattr__(self, "_orig_weight", original_linear.weight)

    def forward(self, x):
        return self._orig_forward(x) + \
               (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling

    @torch.no_grad()
    def merge_into_weights(self):
        self._orig_weight.data += \
            ((self.lora_B @ self.lora_A) * self.scaling).to(self._orig_weight.dtype)


class LayerLoRAOverlay(nn.Module):
    def __init__(self, self_attn, layer_idx, adapt_targets, rank, alpha, dropout):
        super().__init__()
        self.layer_idx = layer_idx
        self.lora_modules = nn.ModuleDict()
        for name in adapt_targets:
            proj = getattr(self_attn, name, None)
            if proj is None:
                continue
            self.lora_modules[name] = LinearLoRA(proj, rank, alpha, dropout)
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


def apply_layer_lora(model, target_layers, adapt_targets, rank, alpha, dropout):
    """在底层模型上应用 LoRA（unwrap DDP）"""
    raw_model = model.module if hasattr(model, "module") else model
    for param in raw_model.parameters():
        param.requires_grad = False
    overlays = nn.ModuleList()
    total_params = 0
    for idx in range(len(raw_model.model.language_model.layers)):
        if idx not in target_layers:
            continue
        attn = raw_model.model.language_model.layers[idx].self_attn
        ov = LayerLoRAOverlay(attn, idx, adapt_targets, rank, alpha, dropout)
        ov.patch()
        overlays.append(ov)
        lp = sum(p.numel() for p in ov.parameters())
        total_params += lp
        print_rank0(f"  L{idx:02d}: LoRA [{','.join(ov.lora_modules.keys())}] params={lp:,}")
    print_rank0(f"  LoRA total: {total_params:,}")
    return overlays


# ═══════════════ Checkpoint ═══════════════

def save_checkpoint(overlays, optimizer, scheduler, epoch, step, path):
    """仅 rank 0 保存"""
    if not is_main_process():
        return
    torch.save({"overlay_state_dict": overlays.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch, "step": step}, path)

def load_checkpoint(path, overlays, optimizer, scheduler):
    if not os.path.exists(path):
        return 0, 0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    overlays.load_state_dict(ckpt["overlay_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["step"]


# ═══════════════ Dataset ═══════════════

class VideoDataset(Dataset):
    def __init__(self, samples, video_dir, fps, processor,
                 max_frames=80, max_size=256, min_tokens=16, total_tokens=2048):
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
        if vp.exists():
            try:
                frames = sample_frames(vp, self.fps)
                duration = len(frames) / self.fps if self.fps > 0 else 0
            except Exception as e:
                if idx < 10 and is_main_process():
                    print(f"  [WARN] 视频读取失败 idx={idx} vp={vp}: {e}")
            if len(frames) > self.max_frames:
                ii = [int(i * len(frames) / self.max_frames)
                      for i in range(self.max_frames)]
                frames = [frames[i] for i in ii]
                duration = len(frames) / self.fps if self.fps > 0 else 0
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
            if idx < 10 and is_main_process():
                print(f"  [WARN] 视频文件不存在 idx={idx} vp={vp}")
        if "duration" in s:
            duration = float(s["duration"])
        elif "video_duration" in s:
            duration = float(s["video_duration"])
        return {"frames": frames, "query": s["query"],
                "gt_start": float(s["gt_start"]), "gt_end": float(s["gt_end"]),
                "duration": duration, "num_frames": len(frames),
                "min_tokens": self.min_tokens, "total_tokens": self.total_tokens}


def collate_fn_with_processor(batch, processor, device):
    from qwen_vl_utils import process_vision_info
    item = batch[0]
    if not item["frames"]:
        return None
    PATCH_PIXELS = 28 * 28
    video_content = {"type": "video", "video": item["frames"], "sample_fps": 1.0}
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
        messages, return_video_kwargs=True, return_video_metadata=True)
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
        print_rank0(f"    [{folder}] {len(items)} → {len(chosen)}")
    rng.shuffle(selected)
    print_rank0(f"  Stratified: {len(selected)} samples")
    return selected


# ═══════════════ 训练循环（多卡版） ═══════════════

def train(model, processor, dataloader, device, target_heads, target_layers,
          align_heads, args, sampler=None):
    """
    多卡训练主循环。
    关键区别：
      - LoRA overlay 用 DDP 包裹，自动同步梯度
      - Loss 做 all_reduce 用于日志
      - 仅 rank 0 保存 checkpoint
      - 每个 epoch 设置 sampler.set_epoch 以保证 shuffle 正确
    """
    out_dir = ensure_directory(Path(args.output_dir))
    ckpt_path = str(out_dir / "lora_layer_checkpoint.pt")
    use_align = len(align_heads) > 0 and args.align_weight > 0
    world_size = get_world_size()
    rank = get_rank()

    print_rank0("=" * 60)
    print_rank0(f"Layer LoRA + {'Attn Align (RoPE)' if use_align else 'CE only'} "
                f"[{world_size} GPUs]")
    print_rank0(f"  Layers={sorted(target_layers)}  rank={args.lora_rank}")
    if use_align:
        print_rank0(f"  Align heads={['L'+str(l)+'H'+str(h) for l,h in align_heads]}  "
                    f"λ={args.align_weight}")
    print_rank0("=" * 60)

    # LoRA 应用在底层模型上
    overlays = apply_layer_lora(model, target_layers, args.adapt_targets,
                                args.lora_rank, args.lora_alpha, args.lora_dropout)
    overlays.to(device)

    # 用 DDP 包裹 overlays 以同步梯度
    if dist.is_initialized() and world_size > 1:
        overlays_ddp = DDP(overlays, device_ids=[device.index] if device.type == "cuda" else None,
                           find_unused_parameters=True)
    else:
        overlays_ddp = overlays

    attn_hook = None
    if use_align:
        raw_model = model.module if hasattr(model, "module") else model
        text_cfg = getattr(raw_model.config, "text_config", raw_model.config)
        num_heads = getattr(text_cfg, "num_attention_heads", DEFAULT_NUM_HEADS)
        num_kv_heads = getattr(text_cfg, "num_key_value_heads", num_heads)
        head_dim = getattr(text_cfg, "head_dim", DEFAULT_HEAD_DIM)
        if head_dim == DEFAULT_HEAD_DIM and hasattr(text_cfg, "hidden_size"):
            head_dim = text_cfg.hidden_size // num_heads

        attn_hook = AttnAlignHook(
            align_heads, num_heads, num_kv_heads, head_dim, model)
        attn_hook.register(model)
        print_rank0(f"  GQA: heads={num_heads}, kv_heads={num_kv_heads}, "
                    f"head_dim={head_dim}")

    # 优化器作用在 overlays（非 DDP 包裹）的参数上
    lora_params = list(overlays.parameters())
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)

    # 计算总步数时考虑 world_size：每个 epoch 每张卡走 len(dataloader) 步
    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(total_steps // 10, 1), max(total_steps, 1))
    start_epoch, start_step = load_checkpoint(ckpt_path, overlays, optimizer, scheduler)

    # 广播 start_epoch/start_step 以确保所有卡一致
    if dist.is_initialized():
        resume_tensor = torch.tensor([start_epoch, start_step],
                                     dtype=torch.long, device=device)
        dist.broadcast(resume_tensor, src=0)
        start_epoch, start_step = int(resume_tensor[0]), int(resume_tensor[1])

    model.train()

    for epoch in range(start_epoch, args.epochs):
        # 设置 sampler epoch 以确保每 epoch shuffle 不同
        if sampler is not None:
            sampler.set_epoch(epoch)

        tot_ce = tot_align = tot_loss = 0.0
        n_valid = n_oom = n_skip = n_empty = n_align = 0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc=f"Epoch {epoch+1}/{args.epochs} [GPU{rank}]",
                    disable=not is_main_process())
        for step, batch in pbar:
            if epoch == start_epoch and step < start_step:
                continue
            if batch is None:
                n_skip += 1
                if n_skip <= 5 and is_main_process():
                    pbar.write(f"  [skip] step={step+1}: batch is None")
                continue

            ans_pos = batch.pop("_answer_positions", [])
            gt_s = batch.pop("_gt_start", 0.0)
            gt_e = batch.pop("_gt_end", 0.0)
            dur = batch.pop("_duration", 0.0)
            nf = batch.pop("_num_frames", 0)

            if (batch["labels"] != -100).sum().item() == 0:
                n_empty += 1
                if attn_hook: attn_hook.clear_all()
                continue

            try:
                if attn_hook: attn_hook.clear_all()
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
                if n_valid < 3 and is_main_process():
                    caps = list(attn_hook.captured_hidden.keys())
                    pos_caps = list(attn_hook.captured_position_ids.keys())
                    pe_caps = list(attn_hook.captured_position_embeddings.keys())
                    pbar.write(
                        f"  [diag] step={step+1}: hidden_layers={caps}, "
                        f"pos_id_layers={pos_caps}, pe_layers={pe_caps}, "
                        f"ans={len(ans_pos)}, gt=[{gt_s:.1f},{gt_e:.1f}]")

                gt_range = locate_gt_video_token_range(
                    batch["input_ids"], processor, gt_s, gt_e, args.fps, nf, dur)

                if n_valid < 3 and is_main_process():
                    pbar.write(f"  [diag] gt_range={gt_range}")

                if gt_range is not None:
                    align_loss = attn_hook.compute_alignment_loss(
                        ans_pos, gt_range[0], gt_range[1], args.align_temperature)
                    if n_valid < 3 and is_main_process():
                        pbar.write(f"  [diag] align_loss={align_loss}")

            combined = ce_loss + args.align_weight * align_loss \
                if align_loss is not None else ce_loss
            if align_loss is not None:
                tot_align += align_loss.item()
                n_align += 1

            (combined / args.gradient_accumulation_steps).backward()

            # DDP overlays 的 backward 会自动触发梯度同步（在 accumulation 的最后一步）
            if attn_hook: attn_hook.clear()
            tot_ce += ce_loss.item()
            tot_loss += combined.item()
            n_valid += 1

            if (step + 1) % args.gradient_accumulation_steps == 0 or \
                    (step + 1) == len(dataloader):
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            if n_valid > 0 and is_main_process():
                pf = dict(ce=f"{tot_ce/n_valid:.4f}",
                          total=f"{tot_loss/n_valid:.4f}", oom=n_oom)
                if use_align and n_align > 0:
                    pf["align"] = f"{tot_align/n_align:.4f}"
                    pf["a_hit"] = f"{n_align}/{n_valid}"
                pbar.set_postfix(pf)

            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                # 同步所有卡后再保存
                if dist.is_initialized():
                    dist.barrier()
                save_checkpoint(overlays, optimizer, scheduler,
                                epoch, step + 1, ckpt_path)

        pbar.close()

        # epoch 结束同步
        if dist.is_initialized():
            dist.barrier()

        print_rank0(f"  Epoch {epoch+1}: valid={n_valid} oom={n_oom} skip={n_skip} "
                    f"empty={n_empty}")
        print_rank0(f"    ce={tot_ce/max(n_valid,1):.4f}  "
                    f"align={tot_align/max(n_align,1):.4f}({n_align}/{n_valid})  "
                    f"total={tot_loss/max(n_valid,1):.4f}")
        save_checkpoint(overlays, optimizer, scheduler, epoch + 1, 0, ckpt_path)

    if attn_hook: attn_hook.remove()

    # 仅 rank 0 保存最终模型
    if dist.is_initialized():
        dist.barrier()

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
        raw_model = model.module if hasattr(model, "module") else model
        for ov in overlays:
            ov.merge_into_weights(); ov.unpatch()
        raw_model.save_pretrained(out_dir)
        processor.save_pretrained(out_dir)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    if dist.is_initialized():
        dist.barrier()

    return overlays


# ═══════════════ Main ═══════════════

def main(argv=None):
    args = build_parser().parse_args(argv)

    # 初始化分布式环境
    rank, world_size, device = setup_distributed()
    torch.cuda.empty_cache()

    ensure_directory(Path(args.output_dir))

    print_rank0("Loading model...")
    model, processor = load_model_with_attn(
        Path(args.model_path), args.attn_implementation, device=device)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        print_rank0("  gradient checkpointing enabled")

    print_rank0("\nLoading data...")
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
        print_rank0("[ERROR] No samples")
        cleanup_distributed()
        return 1
    print_rank0(f"Samples: {len(samples)}")

    dataset = VideoDataset(samples, args.video_dir, args.fps, processor,
                           args.max_frames, args.max_size,
                           args.min_tokens, args.total_tokens)

    # 多卡：使用 DistributedSampler
    sampler = None
    if dist.is_initialized() and world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=True, seed=args.sample_seed)
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False,  # sampler 控制 shuffle
            sampler=sampler,
            collate_fn=lambda x: collate_fn_with_processor(x, processor, device))
    else:
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=True,
            collate_fn=lambda x: collate_fn_with_processor(x, processor, device))

    raw_model = model.module if hasattr(model, "module") else model
    text_cfg = getattr(raw_model.config, "text_config", raw_model.config)
    n_layers = len(raw_model.model.language_model.layers)
    n_heads = getattr(text_cfg, "num_attention_heads", DEFAULT_NUM_HEADS)

    target_heads, target_layers, align_heads = load_head_targets(
        args.attr_json, args.top_k, args.align_top_n,
        n_layers, n_heads, args.random_heads, args.random_seed)
    if not target_layers:
        print_rank0("[ERROR] No target layers")
        cleanup_distributed()
        return 1

    if is_main_process():
        Path(args.output_dir, "config.json").write_text(json.dumps({
            "target_layers": sorted(target_layers),
            "align_heads": [list(lh) for lh in align_heads],
            "align_weight": args.align_weight,
            "lora_rank": args.lora_rank,
            "world_size": world_size,
        }, indent=2))

    train(model, processor, dataloader, device, target_heads, target_layers,
          align_heads, args, sampler=sampler)

    print_rank0("\nDone!")
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
