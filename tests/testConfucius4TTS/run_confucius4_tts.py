#!/usr/bin/env python3
"""读取 JSON 配置，调用 8361 端口的 Confucius4-TTS 并保存返回 WAV。"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_CONFIG_PATH = Path(__file__).with_name("request.json")
UPLOAD_ENDPOINT = "/v1/upload_audio"
GENERATE_ENDPOINT = "/v1/confucius4TTS/generate"
DEFAULT_SERVER_URL = "http://127.0.0.1:8361"
DEFAULT_TIMEOUT_SECONDS = 900.0
UPLOAD_CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    """解析可选的 JSON 配置路径。"""
    parser = argparse.ArgumentParser(description="调用本地 Confucius4-TTS 服务并保存生成 WAV。")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON 配置路径，默认: {DEFAULT_CONFIG_PATH.name}",
    )
    return parser.parse_args()


def require_text(value: Any, label: str) -> str:
    """提取非空文本配置。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串。")
    return value.strip()


def load_config(config_path: Path) -> dict[str, Any]:
    """读取并校验客户端配置，同时解析本地参考音频路径。"""
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"找不到配置文件: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"配置文件 JSON 格式错误: {config_path}:{exc.lineno}:{exc.colno}") from exc

    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是 JSON 对象。")

    reference_value = config.get("reference_audio")
    reference_audio = Path(require_text(reference_value, "reference_audio")).expanduser()
    if not reference_audio.is_absolute():
        reference_audio = config_path.parent / reference_audio
    reference_audio = reference_audio.resolve()
    if not reference_audio.is_file():
        raise ValueError(f"找不到参考 WAV: {reference_audio}")

    generation = config.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("generation 必须是 JSON 对象。")
    generation_payload = dict(generation)
    generation_payload["text"] = require_text(generation_payload.get("text"), "generation.text")
    generation_payload.setdefault("lang", "zh")

    full_path = str(config.get("full_path") or reference_audio.name).strip()
    if not full_path:
        raise ValueError("full_path 不能为空。")
    generation_payload["audio_path"] = full_path

    prompt_text = config.get("prompt_text")
    if prompt_text is not None and not isinstance(prompt_text, str):
        raise ValueError("prompt_text 必须是字符串或 null。")

    output_filename = str(config.get("output_filename") or "confucius4_output.wav").strip()
    output_path = Path(output_filename)
    if output_path.name != output_filename or output_path.suffix.lower() != ".wav":
        raise ValueError("output_filename 必须是 output 子目录下的 WAV 文件名，不能包含路径。")

    try:
        timeout_seconds = float(config.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_seconds 必须是正数。") from exc
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须是正数。")

    return {
        "server_url": str(config.get("server_url") or DEFAULT_SERVER_URL).rstrip("/"),
        "reference_audio": reference_audio,
        "full_path": full_path,
        "prompt_text": prompt_text.strip() if isinstance(prompt_text, str) else None,
        "generation": generation_payload,
        "output_path": config_path.parent / "output" / output_path.name,
        "timeout_seconds": timeout_seconds,
    }


def _connection_target(server_url: str, endpoint: str, timeout: float):
    """创建 HTTP(S) 连接，并拼接服务地址和接口路径。"""
    parsed = urlsplit(server_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"server_url 必须是带主机名的 http/https 地址: {server_url}")
    if parsed.query or parsed.fragment:
        raise ValueError("server_url 不应包含 query 或 fragment。")

    base_path = parsed.path.rstrip("/")
    target = f"{base_path}{endpoint}" or endpoint
    connection_type = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    return connection, target


def _response_error(status: int, reason: str, body: bytes) -> RuntimeError:
    """将服务端错误响应转换成包含 detail 的可读异常。"""
    message = body.decode("utf-8", errors="replace").strip()
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(payload, dict) and payload.get("detail"):
            message = str(payload["detail"])
    return RuntimeError(f"HTTP {status} {reason}: {message[:1_000] or '服务端未返回错误详情'}")


def post_json(
    server_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
) -> bytes:
    """向服务端发送 JSON 请求并返回响应字节。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    connection, target = _connection_target(server_url, endpoint, timeout)
    try:
        connection.request(
            "POST",
            target,
            body=body,
            headers={
                "Accept": "audio/wav, application/json",
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        response_body = response.read()
        if not 200 <= response.status < 300:
            raise _response_error(response.status, response.reason, response_body)
        return response_body
    except (OSError, TimeoutError) as exc:
        raise RuntimeError(f"请求 {server_url}{endpoint} 失败: {exc}") from exc
    finally:
        connection.close()


def upload_audio(
    server_url: str,
    audio_path: Path,
    full_path: str,
    prompt_text: str | None,
    timeout: float,
) -> None:
    """以 multipart/form-data 流式上传参考音频，避免一次性复制大 WAV。"""
    boundary = f"----LLMLocalBackendConfucius4TTS{uuid.uuid4().hex}"
    text_parts = [
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="full_path"\r\n\r\n'
            f"{full_path}\r\n"
        ).encode()
    ]
    if prompt_text:
        text_parts.append(
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="prompt_text"\r\n\r\n'
                f"{prompt_text}\r\n"
            ).encode()
        )
    file_header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="{audio_path.name}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    closing = f"\r\n--{boundary}--\r\n".encode("ascii")
    prefix = b"".join([*text_parts, file_header])
    content_length = len(prefix) + audio_path.stat().st_size + len(closing)

    connection, target = _connection_target(server_url, UPLOAD_ENDPOINT, timeout)
    try:
        connection.putrequest("POST", target)
        connection.putheader("Accept", "application/json")
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(content_length))
        connection.endheaders()
        connection.send(prefix)
        with audio_path.open("rb") as audio_file:
            while chunk := audio_file.read(UPLOAD_CHUNK_SIZE):
                connection.send(chunk)
        connection.send(closing)
        response = connection.getresponse()
        response_body = response.read()
        if not 200 <= response.status < 300:
            raise _response_error(response.status, response.reason, response_body)
    except (OSError, TimeoutError) as exc:
        raise RuntimeError(f"上传 {audio_path} 失败: {exc}") from exc
    finally:
        connection.close()


def save_wav(output_path: Path, audio_bytes: bytes) -> None:
    """校验服务响应并原子写入 output 子目录。"""
    if not audio_bytes.startswith(b"RIFF") or b"WAVE" not in audio_bytes[:16]:
        raise RuntimeError("服务返回的内容不是有效 WAV 文件。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as output_file:
            output_file.write(audio_bytes)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    """执行上传、生成和输出三个步骤。"""
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    try:
        config = load_config(config_path)
        print(f"[1/3] 上传参考音频: {config['reference_audio']}")
        upload_audio(
            config["server_url"],
            config["reference_audio"],
            config["full_path"],
            config["prompt_text"],
            config["timeout_seconds"],
        )
        print(f"[2/3] 调用模型: {config['server_url']}{GENERATE_ENDPOINT}")
        audio_bytes = post_json(
            config["server_url"],
            GENERATE_ENDPOINT,
            config["generation"],
            config["timeout_seconds"],
        )
        save_wav(config["output_path"], audio_bytes)
        print(f"[3/3] 输出完成: {config['output_path']}")
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
