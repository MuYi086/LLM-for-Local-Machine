#!/usr/bin/env python3
"""启动一次性 llama-server，并调用 Index-Echo GGUF 文本主干翻译。"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析 worker JSON 输入和输出路径。"""
    parser = argparse.ArgumentParser(description="One-shot Index-Echo GGUF translation worker")
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


def require_file(path: str, label: str) -> Path:
    """验证文件存在并返回规范化绝对路径。"""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label}文件不存在: {resolved}")
    return resolved


def find_free_port() -> int:
    """为本次 llama-server 分配本机临时端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def read_log_excerpt(log_file) -> str:
    """读取 llama-server 日志末尾，避免把完整日志写入 API 错误。"""
    log_file.flush()
    log_file.seek(0, os.SEEK_END)
    size = log_file.tell()
    log_file.seek(max(0, size - 12_000))
    return log_file.read().decode("utf-8", errors="replace").strip()


def stop_server(process: subprocess.Popen | None) -> None:
    """正常退出时关闭共享 worker 进程组内的 llama-server 子进程。"""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def wait_for_server(
    process: subprocess.Popen,
    port: int,
    startup_timeout: float,
    log_file,
) -> None:
    """等待 llama-server 完成模型加载，或返回启动错误。"""
    deadline = time.monotonic() + startup_timeout
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server 提前退出，退出码={process.returncode}: {read_log_excerpt(log_file)}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    raise TimeoutError(
        f"llama-server 加载模型超时（>{startup_timeout:.0f}s）: {read_log_excerpt(log_file)}"
    )


def translate(request: dict[str, Any]) -> str:
    """运行本地 OpenAI 兼容服务并提取纯译文。"""
    model_path = require_file(str(request.get("model_path") or ""), "Index-Echo GGUF 模型")
    server_bin = require_file(str(request.get("llama_server_bin") or ""), "llama-server")
    text = str(request.get("text") or "").strip()
    source_language_name = str(request.get("source_language_name") or "").strip()
    target_language = str(request.get("target_language") or "").strip()
    target_name = str(request.get("target_language_name") or "").strip()
    if not text or not source_language_name or not target_name or not target_language:
        raise ValueError("源文本和目标语言不能为空。")

    port = find_free_port()
    startup_timeout = float(request.get("startup_timeout_seconds", 300))
    if startup_timeout <= 0:
        raise ValueError("llama-server 启动超时必须大于零。")

    command = [
        str(server_bin),
        "--model",
        str(model_path),
        "--alias",
        "index-echo-s2st-9b",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--n-gpu-layers",
        str(int(request.get("n_gpu_layers", 99))),
        "--ctx-size",
        str(int(request.get("context_size", 4096))),
    ]

    prompt = (
        f"Translate the following spoken text from {source_language_name} into {target_name}. "
        "Preserve its meaning, intent, names, and punctuation. "
        "Return only one line with the translation. Do not repeat the source or add an explanation.\n\n"
        f"{text}"
    )
    payload = {
        "model": "index-echo-s2st-9b",
        "messages": [
            {"role": "system", "content": "You are a precise speech translation assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": int(request.get("max_new_tokens", 1024)),
        "stream": False,
    }

    process: subprocess.Popen | None = None
    with tempfile.TemporaryFile() as log_file:
        try:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=False,
                env=os.environ.copy(),
            )
            wait_for_server(process, port, startup_timeout, log_file)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            http_request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    http_request,
                    timeout=float(request.get("inference_timeout_seconds", 600)),
                ) as response:
                    result = json.loads(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read(4_000).decode("utf-8", errors="replace")
                raise RuntimeError(f"llama-server 返回 HTTP {exc.code}: {detail}") from exc

            try:
                translated = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("llama-server 返回了无法识别的聊天完成结果。") from exc
            if not isinstance(translated, str) or not translated.strip():
                raise RuntimeError("Index-Echo GGUF 未生成译文。")
            cleaned = translated.strip()
            if "</think>" in cleaned:
                cleaned = cleaned.rsplit("</think>", 1)[-1].strip()
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError("Index-Echo GGUF 未生成译文。")
            return lines[-1]
        finally:
            stop_server(process)


def main() -> int:
    """执行单次文本翻译并写入 JSON 结果。"""
    args = parse_args()
    output_path = Path(args.output_json).expanduser().resolve()
    try:
        translated_text = translate(load_request(args.input_json))
        output_path.write_text(
            json.dumps({"translated_text": translated_text}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
