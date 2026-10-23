#!/usr/bin/env python3
"""将已校验的音乐理解请求交给一次性 MOSS GGUF 进程。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_command(payload: dict[str, object], output: Path, prompt_file: Path) -> list[str]:
    """构造固定 CLI 参数，不经过 shell，也不开放任意命令或模型路径给客户端。"""
    return [
        str(payload["caption_bin"]),
        "--models",
        str(payload["model_dir"]),
        "--src-audio",
        str(payload["audio_path"]),
        "--mode",
        "prose",
        "--prompt-file",
        str(prompt_file),
        "--max-tokens",
        str(payload["max_new_tokens"]),
        "--max-seconds",
        str(payload["max_audio_seconds"]),
        "--temperature",
        str(payload["temperature"]),
        "-o",
        str(output),
    ]


def main() -> None:
    """用 exec 替换 worker，保证 C++ 推理与父进程管理的是同一个进程组。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    request_path = Path(args.input_json).resolve()
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    prompt_file = request_path.with_suffix(".prompt.txt")
    prompt_file.write_text(str(payload["prompt"]), encoding="utf-8")
    os.environ["GGML_BACKEND"] = str(payload["backend"])
    command = build_command(payload, Path(args.output_file).resolve(), prompt_file)
    # 不再创建第二层 subprocess；成功/失败/超时都由 API 统一回收整组并清理请求与 prompt。
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
