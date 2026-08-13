#!/usr/bin/env python3
"""
organize_timelens.py — 把下载下来的 TimeLens-100K 数据集整理成项目约定结构。

约定目标结构：
    Lkmllm_data/datasets/Train/timelens-100k/
        ├── cosmo_cap/            # 视频文件（.mp4）
        ├── didemo/
        ├── hirest/
        ├── internvid_vtime/
        ├── queryd/
        └── timelens-100k.jsonl   # 标注文件

用法（在项目根目录 LK_OPD 下运行）：
    # 默认整理项目根目录下的 timelens-100k（移动，不保留原下载目录）
    python Lkmllm_code/tools/organize_timelens.py

    # 指定其它下载位置
    python Lkmllm_code/tools/organize_timelens.py --src /path/to/timelens-100k

    # 复制（保留原下载目录）
    python Lkmllm_code/tools/organize_timelens.py --copy

    # 只预览要做什么，不实际执行
    python Lkmllm_code/tools/organize_timelens.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# tools/ 在 Lkmllm_code/ 下，向上两级即项目根（LK_OPD）
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "timelens-100k"
DEFAULT_DST = REPO_ROOT / "Lkmllm_data" / "datasets" / "Train" / "timelens-100k"

# TimeLens-100K 训练集的 5 个来源子文件夹
VIDEO_SUBDIRS = ["cosmo_cap", "didemo", "hirest", "internvid_vtime", "queryd"]
ANNO_FILENAME = "timelens-100k.jsonl"


def find_file(src: Path, name: str):
    """在 src 下找指定文件（优先根目录，其次递归）。"""
    direct = src / name
    if direct.is_file():
        return direct
    for c in src.rglob(name):
        if c.is_file():
            return c
    return None


def find_dir(src: Path, name: str):
    """在 src 下找指定目录（优先根目录，其次递归）。"""
    direct = src / name
    if direct.is_dir():
        return direct
    for c in src.rglob(name):
        if c.is_dir():
            return c
    return None


def transfer(src: Path, dst: Path, *, copy: bool, dry_run: bool) -> None:
    """把单个条目从 src 搬到 dst（copy=False 为移动）。"""
    if src.resolve() == dst.resolve():
        print(f"  [skip] 源与目标相同：{dst}")
        return
    if dst.exists():
        print(f"  [覆盖] 目标已存在，先删除：{dst}")
        if not dry_run:
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
    if dry_run:
        print(f"  [计划] {'复制' if copy else '移动'} {src} → {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))
    print(f"  [完成] {src.name} → {dst}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="整理 TimeLens-100K 到项目约定结构")
    p.add_argument("--src", default=str(DEFAULT_SRC),
                   help=f"下载下来的数据集目录（默认 {DEFAULT_SRC}）")
    p.add_argument("--dst", default=str(DEFAULT_DST),
                   help=f"目标目录（默认 {DEFAULT_DST}）")
    p.add_argument("--copy", action="store_true", help="复制而非移动（保留原目录）")
    p.add_argument("--dry-run", action="store_true", help="只打印计划，不实际执行")
    args = p.parse_args(argv)

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    if not src.is_dir():
        print(f"[ERROR] 源目录不存在：{src}")
        return 1

    if src == dst:
        print(f"[OK] 源目录已经是目标位置，无需整理：{dst}")
        return 0

    # 定位标注文件（优先标准名，其次任意 .jsonl）
    anno = find_file(src, ANNO_FILENAME)
    if anno is None:
        anno = next((c for c in src.rglob("*.jsonl") if c.is_file()), None)

    # 定位 5 个视频子文件夹
    video_dirs = {name: find_dir(src, name) for name in VIDEO_SUBDIRS}
    found_video = {k: v for k, v in video_dirs.items() if v is not None}
    missing = [k for k in VIDEO_SUBDIRS if k not in found_video]

    print("=" * 70)
    print("TimeLens-100K 整理计划")
    print("=" * 70)
    print(f"  源目录   : {src}")
    print(f"  目标目录 : {dst}")
    print(f"  模式     : {'复制' if args.copy else '移动'}")
    print(f"  标注文件 : {anno if anno else '未找到 (!)'}")
    for k in VIDEO_SUBDIRS:
        print(f"  视频目录 : {k} → {video_dirs.get(k) or '未找到 (!)'}")

    if anno is None:
        print("[ERROR] 找不到标注文件 timelens-100k.jsonl（或任何 .jsonl），"
              "请确认下载完整。")
        return 1
    if missing:
        print(f"[WARN] 未找到这些子文件夹: {missing}，将继续整理已找到的部分。")

    # 逐个搬运：标注文件 + 已找到的视频子文件夹
    transfer(anno, dst / ANNO_FILENAME, copy=args.copy, dry_run=args.dry_run)
    for name in VIDEO_SUBDIRS:
        v = video_dirs.get(name)
        if v is not None:
            transfer(v, dst / name, copy=args.copy, dry_run=args.dry_run)

    print("=" * 70)
    print("整理完成。")
    print(f"  标注: {dst / ANNO_FILENAME}")
    for name in found_video:
        print(f"  视频: {dst / name}")
    if args.dry_run:
        print("  （以上为 dry-run 预览，未实际改动任何文件）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
