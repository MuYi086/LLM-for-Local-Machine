#!/usr/bin/env python3
"""为 8390 配置 WanGP 的本地 LongCat 权重和显存运行档。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from wangp_runtime import (
    DEFAULT_MEMORY_PROFILE,
    ensure_links,
    update_wangp_config,
    validate_memory_profile,
)


def expand_path(value: str) -> Path:
    """展开用户目录和环境变量。"""
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(value))))


def main() -> int:
    """确保 WanGP 环境、配置文件和权重别名就绪。"""
    wangp_dir = expand_path(os.getenv("LONGCAT_AVATAR_WANGP_DIR", "~/LLM-depency/WanGP"))
    wangp_python = expand_path(
        os.getenv("LONGCAT_AVATAR_WANGP_PYTHON", str(wangp_dir / ".venv/bin/python"))
    )
    model_dir = expand_path(
        os.getenv(
            "LONGCAT_AVATAR_MODEL_DIR",
            "~/hf-mirror/meituan/LongCat-Video-Avatar-1.5",
        )
    )
    base_model_dir = expand_path(
        os.getenv("LONGCAT_AVATAR_BASE_MODEL_DIR", "~/hf-mirror/meituan/LongCat-Video")
    )
    memory_profile = validate_memory_profile(
        os.getenv("LONGCAT_AVATAR_MEMORY_PROFILE", str(DEFAULT_MEMORY_PROFILE))
    )
    if not wangp_dir.is_dir() or not (wangp_dir / "shared/api.py").is_file():
        raise FileNotFoundError(f"WanGP 源码目录不可用：{wangp_dir}")
    if not wangp_python.is_file():
        raise FileNotFoundError(f"WanGP Python 环境不可用：{wangp_python}")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    config_path = wangp_dir / "wgp_config.json"
    if not config_path.exists():
        sys.path.insert(0, str(wangp_dir))
        from shared.api import init

        init(
            root=wangp_dir,
            config_path=config_path,
            output_dir=wangp_dir / "outputs",
            cli_args=[
                "--offline",
                "--profile",
                str(memory_profile),
                "--attention",
                "sdpa",
                "--gpu",
                "cuda:0",
            ],
            console_output=False,
        )

    ensure_links(wangp_dir, model_dir, base_model_dir)
    update_wangp_config(
        wangp_dir,
        memory_profile=memory_profile,
        attention_mode="sdpa",
    )
    print(f"WanGP 已配置：{wangp_dir}")
    print(f"显存运行档：MMGP Profile {memory_profile}，attention=sdpa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
