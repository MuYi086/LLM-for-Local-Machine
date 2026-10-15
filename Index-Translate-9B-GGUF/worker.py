#!/usr/bin/env python3
"""启动一次性 llama-server，并调用 Index-Translate GGUF 翻译原生输入文本。"""

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
    parser = argparse.ArgumentParser(description="One-shot Index-Translate GGUF translation worker")
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


def build_translation_prompt(
    text: str,
    target_language_name: str,
    glossary: dict[str, str],
    hard_constraints: list[str],
    soft_constraints: list[str],
    genre: str,
    source_language_name: str = "",
    instruction: str = "",
) -> str:
    """复现官方 translate.py 的普通翻译、instTrans 和 JSON 保持模板。

    来源：https://github.com/bilibili/Index-Translate/blob/main/inference/llm/translate.py
    """
    if not text.strip() or not target_language_name.strip():
        raise ValueError("源文本和目标语言不能为空。")
    constraints = []
    for values, prefixes, default_prefix in (
        (hard_constraints, ("〖硬性要求〗",), "〖硬性要求〗"),
        (soft_constraints, ("〖注意〗", "〖软性要求〗"), "〖注意〗"),
    ):
        if default_prefix == "〖注意〗" and glossary:
            terms = "、".join(f"{term}→{translation}" for term, translation in glossary.items())
            constraints.append(f"〖硬性要求〗专名/术语对照: {terms}")
        for value in values:
            for line in value.splitlines():
                line = line.strip().lstrip("0123456789. ")
                if line:
                    constraints.append(
                        line if line.startswith(prefixes) else f"{default_prefix}{line}"
                    )
    for line in instruction.splitlines():
        line = line.strip().lstrip("0123456789. ")
        if line:
            prefixes = ("〖硬性要求〗", "〖注意〗", "〖软性要求〗")
            constraints.append(line if line.startswith(prefixes) else f"〖注意〗{line}")
    source = source_language_name.strip()
    if not constraints:
        return f"请将以下{source}文本翻译为{target_language_name}，直接输出翻译结果，不要进行任何解释。\n\n{text.strip()}"
    suffix = "只输出译文，不要有任何额外说明。"
    if any("json" in value.lower() for value in constraints):
        try:
            if isinstance(json.loads(text), dict):
                suffix = "请以相同的 JSON 格式输出翻译结果，key 保持不变，value 为对应译文。只输出 JSON，不要有任何额外说明。"
        except ValueError:
            pass
    numbered = "\n".join(f"{index}. {value}" for index, value in enumerate(constraints, 1))
    return (
        f"请将以下{source}{genre}翻译成{target_language_name}，并且严格遵循所有约束要求。\n\n"
        f"〖源文〗\n{text.strip()}\n\n〖约束要求〗\n{numbered}\n\n{suffix}"
    )


def build_messages(request: dict[str, Any]) -> list[dict]:
    """保持官方文本或图片消息结构，不执行外部 OCR。"""
    text = str(request.get("text") or "").strip()
    source = str(request.get("source_language_name") or "")
    target = str(request.get("target_language_name") or "").strip()
    if not target:
        raise ValueError("目标语言不能为空。")
    image = request.get("image")
    if image:
        prompt = f"请将图片中的{source}文本翻译为{target}，直接输出翻译结果，不要进行任何解释。"
        extras = []
        if text:
            extras.append(f"〖参考文本/关注内容〗\n{text}")
        if request.get("instruction"):
            extras.append(f"〖指令要求〗{request['instruction']}")
        extras.extend(f"〖硬性要求〗{value}" for value in request.get("hard_constraints", []))
        extras.extend(f"〖注意〗{value}" for value in request.get("soft_constraints", []))
        if request.get("glossary"):
            terms = "、".join(f"{key}→{value}" for key, value in request["glossary"].items())
            extras.append(f"〖专名/术语对照〗{terms}")
        if extras:
            prompt += "\n\n" + "\n\n".join(extras) + "\n\n只输出翻译结果，不要进行任何解释。"
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image}},
        ]
    elif request.get("raw_prompt"):
        if not text:
            raise ValueError("原始提示词不能为空。")
        content = text
    else:
        content = build_translation_prompt(
            text,
            target,
            request.get("glossary") or {},
            request.get("hard_constraints") or [],
            request.get("soft_constraints") or [],
            str(request.get("genre") or "文本"),
            source,
            str(request.get("instruction") or ""),
        )
    return [{"role": "user", "content": content}]


def translate(request: dict[str, Any]) -> str:
    """运行本地 OpenAI 兼容服务并提取纯译文。"""
    model_path = require_file(str(request.get("model_path") or ""), "Index-Translate GGUF 模型")
    server_bin = require_file(str(request.get("llama_server_bin") or ""), "llama-server")

    port = find_free_port()
    startup_timeout = float(request.get("startup_timeout_seconds", 300))
    if startup_timeout <= 0:
        raise ValueError("llama-server 启动超时必须大于零。")

    command = [
        str(server_bin),
        "--model",
        str(model_path),
        "--alias",
        "index-translate-9b",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--n-gpu-layers",
        str(int(request.get("n_gpu_layers", 99))),
        "--ctx-size",
        str(int(request.get("context_size", 32768))),
        "--parallel",
        "1",
        "--no-context-shift",
        "--jinja",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
    ]

    if request.get("image"):
        projector = require_file(str(request.get("mmproj_path") or ""), "Index-Translate mmproj")
        command.extend(["--mmproj", str(projector)])
    payload = {
        "model": "index-translate-9b",
        "messages": build_messages(request),
        "temperature": float(request.get("temperature", 0)),
        "max_tokens": int(request.get("max_new_tokens", 1024)),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
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
                raise RuntimeError("Index-Translate GGUF 未生成译文。")
            cleaned = translated.strip()
            if "</think>" in cleaned:
                cleaned = cleaned.split("</think>", 1)[1].strip()
            if cleaned.startswith("<think>"):
                cleaned = cleaned.removeprefix("<think>").strip()
            if not cleaned:
                raise RuntimeError("Index-Translate GGUF 未生成译文。")
            return cleaned
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
