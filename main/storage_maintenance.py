#!/usr/bin/env python3
"""显式维护 LLM Local Backend 生成文件，默认只预览不删除。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm_local_runtime import prune_generated_outputs, storage_disk_status


def expand_path(value: str) -> Path:
    """展开存储相关环境变量，统一为绝对目录。"""
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(value))))


def parse_args() -> argparse.Namespace:
    """解析明确的执行确认开关。"""
    parser = argparse.ArgumentParser(description="预览或执行 LLM Local Backend 生成音频保留策略")
    parser.add_argument("--apply", action="store_true", help="实际删除匹配的历史生成 WAV")
    return parser.parse_args()


def main() -> int:
    """按环境中的保留策略输出计划；未显式确认绝不删除文件。"""
    args = parse_args()
    storage_dir = expand_path(os.getenv("STORAGE_DIR", "storage"))
    timbre_dir = expand_path(os.getenv("TIMBRE_STORAGE_DIR", str(storage_dir / "timbre")))
    clone_dir = expand_path(os.getenv("CLONE_STORAGE_DIR", str(storage_dir / "clone")))
    bgm_dir = expand_path(os.getenv("BGM_STORAGE_DIR", str(storage_dir / "bgm")))
    status = storage_disk_status(storage_dir)
    print(status)
    if not status["retention_enabled"]:
        print(
            "保留策略未启用：设置 STORAGE_RETENTION_HOURS 或 STORAGE_RETENTION_MAX_BYTES 后重试。"
        )
        return 0

    planned = prune_generated_outputs(
        {
            timbre_dir: ("qwen_voicedesign_",),
            clone_dir: ("qwen3_tts_", "voxcpm2_", "indexTTS25_"),
            bgm_dir: ("ace_step_1_5_", "yue2_3b_"),
        },
        retention_hours=float(status["retention_hours"]),
        retention_max_bytes=int(status["retention_max_bytes"]),
        apply=args.apply,
        timbre_dir=timbre_dir,
    )
    action = "已删除" if args.apply else "将删除（预览）"
    for path in planned:
        print(f"{action}: {path}")
    print(f"匹配 {len(planned)} 个生成文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
