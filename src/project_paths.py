from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def find_project_root(start: Optional[Path] = None) -> Path:
    """Locate the Lkmllm_code directory (project root) by walking up from this file."""
    base = (start or Path(__file__)).resolve()
    if base.is_file():
        base = base.parent
    for candidate in [base, *base.parents]:
        if candidate.name == "Lkmllm_code":
            return candidate
    raise RuntimeError(
        "Could not locate project root. Expected directory: Lkmllm_code."
    )


# 项目根目录 = Lkmllm_code（代码根）；数据/模型是它的兄弟目录。
PROJECT_ROOT = find_project_root()              # Lkmllm_code
WORKSPACE_ROOT = PROJECT_ROOT.parent            # 上层工作区（含 Lkmllm_data / shared_models）
A_CODE_ROOT = PROJECT_ROOT
A_DATA_ROOT = WORKSPACE_ROOT / "Lkmllm_data"
SHARED_MODELS_ROOT = WORKSPACE_ROOT / "shared_models"
DEFAULT_MODEL_PATH = SHARED_MODELS_ROOT / "Qwen3-VL-8B-Instruct"


def resolve_path(value: Optional[os.PathLike[str] | str], *, default: Optional[Path] = None, must_exist: bool = False) -> Path:
    """Resolve a CLI or env path relative to the current working directory or project root."""
    if value is None:
        resolved = default or PROJECT_ROOT
    else:
        raw = Path(str(value)).expanduser()
        if raw.is_absolute():
            resolved = raw
        else:
            cwd_candidate = (Path.cwd() / raw).resolve()
            if not must_exist or cwd_candidate.exists():
                resolved = cwd_candidate
            else:
                resolved = (WORKSPACE_ROOT / raw).resolve()

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Required path does not exist: {resolved}")
    return resolved


def env_path(name: str, default: Optional[Path] = None) -> Path:
    return resolve_path(os.getenv(name), default=default)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
