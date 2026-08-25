"""
m_d_startend_gradient_head_attribution.py
— start/end 时间 token 的多卡 Head 梯度归因（torchrun 数据并行版）

每个 rank 在一张 GPU 上加载一份完整模型，并处理全局候选样本的 rank::world_size
切片。各 rank 独立保存中间 JSON，最后由 rank 0 按有效样本数加权合并三个
score matrix，重新计算 Top-K，并生成与单卡脚本同名的最终 JSON/热力图。

默认探测 500 条样本，并与当前 TimeLens 微调/评测输入保持一致：
  fps=2.0, max_frames=0（不截帧）, min_tokens=64, total_tokens=14336。
注意：归因必须取得 attention weights，因此 attention backend 仍为 eager，不能使用
微调时的 flash_attention_2；这是归因算法要求，不是输入参数不一致。

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
    --layers-per-batch 2

最终结果：
  <output-dir>/startend_gradient_head_attribution.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List


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
    import d_startend_gradient_head_attribution as base

    if not torch.cuda.is_available():
        raise RuntimeError("多卡 head 探测要求 CUDA")
    torch.cuda.set_device(0)
    if world_size > 1:
        dist.init_process_group(backend="gloo", init_method="env://")

    parsed = base.build_parser().parse_args(argv)
    output_dir = Path(parsed.output_dir).expanduser().resolve()
    rank_root = output_dir / "_rank_outputs"
    rank_dir = rank_root / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    base.load_model_and_processor = lambda model_dir, gpu_mem_gib=21.0, cpu_mem_gib=64.0: (
        _load_full_model_on_local_gpu(base, model_dir, gpu_mem_gib, cpu_mem_gib)
    )
    base.load_samples = _make_distributed_loader(base, rank, world_size)

    print(f"[distributed] rank={rank}/{world_size}, local_rank={local_rank}, "
          f"visible_gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    rank_argv = _build_rank_argv(argv, rank_dir)
    rc = base.main(rank_argv)
    if rc != 0:
        raise RuntimeError(f"rank {rank} head attribution 失败，return code={rc}")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        effective_args = base.build_parser().parse_args(rank_argv)
        _merge_rank_results(
            base, output_dir, rank_root, world_size, effective_args,
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
