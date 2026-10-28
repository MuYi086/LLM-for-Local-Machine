#!/usr/bin/env python3
"""IndexTTS-2.5 一次性模型推理 worker。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析 worker JSON 输入和 WAV 输出路径。"""
    parser = argparse.ArgumentParser(description="One-shot IndexTTS-2.5 synthesis worker")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-wav", required=True)
    return parser.parse_args()


def load_request(path: str | os.PathLike[str]) -> dict[str, Any]:
    """读取 worker 输入，并验证 JSON 顶层为对象。"""
    with Path(path).open(encoding="utf-8") as request_file:
        payload = json.load(request_file)
    if not isinstance(payload, dict):
        raise ValueError("worker input JSON 必须是对象。")
    return payload


def require_path(path: str, label: str, *, directory: bool = False) -> Path:
    """确认本地文件或目录存在并返回绝对路径。"""
    resolved = Path(path).expanduser().resolve()
    exists = resolved.is_dir() if directory else resolved.is_file()
    if not exists:
        kind = "目录" if directory else "文件"
        raise FileNotFoundError(f"{label}{kind}不存在: {resolved}")
    return resolved


def configure_offline_environment(cache_dir: Path) -> None:
    """将模型库缓存指向临时目录，并阻止推理期间隐式联网。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["MODELSCOPE_CACHE"] = str(cache_dir / "modelscope")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def generate_audio(request: dict[str, Any], output_path: Path) -> None:
    """按 IndexTTS 官方 Python API 合成并写出 22.05 kHz WAV。"""
    model_dir = require_path(
        str(request.get("model_dir") or ""), "IndexTTS-2.5 模型", directory=True
    )
    code_path = require_path(str(request.get("code_path") or ""), "IndexTTS 源码", directory=True)
    reference_audio = require_path(str(request.get("reference_audio_path") or ""), "参考音频")
    require_path(str(model_dir / "config.yaml"), "IndexTTS-2.5 配置")
    require_path(str(code_path / "indextts" / "infer_v2_5.py"), "IndexTTS-2.5 推理源码")

    text = str(request.get("text") or "").strip()
    if not text:
        raise ValueError("text 不能为空。")
    lang = str(request.get("lang") or "").strip().upper()
    if not lang:
        raise ValueError("lang 不能为空。")

    cache_dir = output_path.parent / "model-cache"
    configure_offline_environment(cache_dir)
    code_string = str(code_path)
    if code_string not in sys.path:
        sys.path.insert(0, code_string)

    try:
        from indextts.infer_v2_5 import IndexTTS2
    except ImportError as exc:
        raise RuntimeError(f"IndexTTS-2.5 推理依赖导入失败: {exc}") from exc

    device = str(request.get("device") or "cuda:0")
    model = IndexTTS2(
        cfg_path=str(model_dir / "config.yaml"),
        model_dir=str(model_dir),
        use_bf16=bool(request.get("use_bf16", True)),
        device=device,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_qwen_emo=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.infer(
        spk_audio_prompt=str(reference_audio),
        text=text,
        lang=lang,
        output_path=str(output_path),
        duration_factor=float(request.get("duration_factor", 1.0)),
        emo_vector=request.get("emo_vector"),
        emo_alpha=float(request.get("emo_alpha", 1.0)),
        verbose=bool(request.get("verbose", False)),
    )


def main() -> int:
    """加载 JSON 请求、运行一次推理，并以退出码报告错误。"""
    args = parse_args()
    try:
        request = load_request(args.input_json)
        generate_audio(request, Path(args.output_wav))
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
