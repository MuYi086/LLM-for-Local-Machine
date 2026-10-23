#!/usr/bin/env python3
"""Qwen3-ASR-1.7B 一次性音频转写 worker。"""

from __future__ import annotations

# 模型和 Torch 只在一次性 worker 中导入；进程退出后由 CUDA 回收权重显存。
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def parse_args():
    """解析 worker JSON 输入和 JSON 结果输出路径。"""
    import argparse

    parser = argparse.ArgumentParser(description="One-shot Qwen3-ASR transcription worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_request(path: str) -> dict[str, Any]:
    """读取并确认 worker 输入 JSON 顶层为对象。"""
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("worker input JSON 必须是对象。")
    return payload


def require_path(path: str, label: str, *, directory: bool = False) -> Path:
    """验证文件或目录存在，并返回规范化绝对路径。"""
    resolved = Path(path).expanduser().resolve()
    expected = resolved.is_dir() if directory else resolved.is_file()
    if not expected:
        kind = "目录" if directory else "文件"
        raise FileNotFoundError(f"{label}{kind}不存在: {resolved}")
    return resolved


def parse_bool(value: Any, default: bool = False) -> bool:
    """兼容 JSON 布尔值与环境变量式字符串。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def write_result(path: Path, payload: dict[str, Any]) -> None:
    """写入 worker 结果 JSON，供父 HTTP 进程在 worker 退出后读取。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())


def transcribe_audio(request: dict[str, Any]) -> dict[str, Any]:
    """加载本地 Qwen3-ASR 权重并识别单个上传音频。"""
    local_files_only = parse_bool(request.get("local_files_only"), True)
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model_dir = require_path(str(request.get("model_dir") or ""), "Qwen3-ASR 模型", directory=True)
    audio_path = require_path(str(request.get("audio_path") or ""), "待识别音频")
    device = str(request.get("device") or "cuda:0")
    dtype_name = str(request.get("dtype") or "bfloat16")
    dtype_names = {"bfloat16", "float16", "float32"}
    if dtype_name not in dtype_names:
        raise ValueError("dtype 仅支持 bfloat16、float16 或 float32。")

    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-ASR 运行时导入失败，请先为 Qwen3_ASR_1.7B 执行 uv sync。"
            f"缺少或无法导入: {exc.name or exc}"
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，Qwen3-ASR 默认要求 GPU。")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    model = None
    started = time.perf_counter()
    try:
        model = Qwen3ASRModel.from_pretrained(
            str(model_dir),
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=int(request.get("max_inference_batch_size") or 1),
            max_new_tokens=int(request.get("max_new_tokens") or 256),
            local_files_only=local_files_only,
        )
        results = model.transcribe(
            audio=str(audio_path),
            language=request.get("language") or None,
            context=str(request.get("context") or ""),
        )
        if not results:
            raise RuntimeError("Qwen3-ASR 未返回识别结果。")
        text = getattr(results[0], "text", None)
        language = getattr(results[0], "language", None)
        if not isinstance(text, str) or (language is not None and not isinstance(language, str)):
            raise RuntimeError("Qwen3-ASR 返回了无法解析的结果。")
        return {
            "text": text,
            "language": language,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError as exc:
                print(f"[Qwen3-ASR worker] CUDA synchronize 跳过: {exc}")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main() -> int:
    """执行一次音频识别，并将文本和语言写入结果 JSON。"""
    args = parse_args()
    try:
        result = transcribe_audio(load_request(args.input_json))
        write_result(Path(args.output_json).expanduser().resolve(), result)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
