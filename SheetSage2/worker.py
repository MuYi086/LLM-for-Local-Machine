#!/usr/bin/env python3
"""运行一次 SheetSage2 音乐转谱并打包模型原生结果。"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Any


def load_model(payload: dict[str, Any]):
    """按官方 Transformers 接口加载本地模型与固定版本 MERT-v2 父模型。"""
    import torch
    from transformers import AutoModel

    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = str(payload["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，SheetSage2 服务默认要求 GPU。")
    model = AutoModel.from_pretrained(
        payload["model_dir"],
        base_model_path=payload["base_model_dir"],
        local_files_only=bool(payload["local_files_only"]),
        trust_remote_code=True,
    )
    model.eval().to(device)
    return torch, model


def transcribe(payload: dict[str, Any], archive_path: Path) -> None:
    """调用模型原生 transcribe，并将 ABC、MIDI 与标注文件写入 ZIP。"""
    torch, model = load_model(payload)
    try:
        with tempfile.TemporaryDirectory(
            prefix="sheetsage2_result_",
            dir=payload["worker_tmp_dir"],
        ) as temporary_dir:
            output_dir = Path(temporary_dir)
            audio_path = payload["audio_path"]
            options = {
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "audio_path",
                    "base_model_dir",
                    "device",
                    "local_files_only",
                    "model_dir",
                    "worker_tmp_dir",
                }
                and value is not None
            }
            model.transcribe(audio_path, output_dir=output_dir, **options)
            files = sorted(
                path for path in output_dir.rglob("*") if path.is_file() and not path.is_symlink()
            )
            if not files:
                raise RuntimeError("SheetSage2 未生成转写文件。")

            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(
                archive_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                for path in files:
                    archive.write(path, path.relative_to(output_dir).as_posix())
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError as exc:
                print(f"[SheetSage2 worker] CUDA synchronize 跳过: {exc}", file=sys.stderr)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def parse_args() -> argparse.Namespace:
    """解析 API 进程传入的一次性 JSON 请求和 ZIP 输出路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-zip", required=True)
    return parser.parse_args()


def main() -> int:
    """读取一次请求并完成转写；错误通过退出码交给 API 进程处理。"""
    args = parse_args()
    try:
        with Path(args.input_json).open(encoding="utf-8") as request_file:
            payload = json.load(request_file)
        if not isinstance(payload, dict):
            raise ValueError("worker JSON 顶层必须是对象。")
        transcribe(payload, Path(args.output_zip).resolve())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
