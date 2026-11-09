"""调用本机 VoiceDesign 服务生成并检查一个中文音频示例。"""

from __future__ import annotations

import argparse
import email.message
import io
import json
import os
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROUTE = "/v1/qwen/Qwen3-TTS-1.7B-VoiceDesign"


def build_parser() -> argparse.ArgumentParser:
    """构造本机生成示例的命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8301")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--voice-description",
        default="成年女性，声音清晰自然，温和亲切，语速中等。",
    )
    parser.add_argument("--text", default="你好，欢迎使用 Qwen3-TTS 音色设计。")
    return parser


def _stored_response_path(headers: email.message.Message, audio: bytes) -> Path:
    """确认服务已将响应原子写入共享的音色目录。"""
    filename = headers.get_filename()
    if not filename or Path(filename).name != filename or not filename.endswith(".wav"):
        raise ValueError("服务响应缺少有效的 WAV 文件名。")
    storage_dir = (
        Path(os.getenv("TIMBRE_STORAGE_DIR", ROOT / "storage/timbre")).expanduser().resolve()
    )
    stored_path = (storage_dir / filename).resolve()
    if not stored_path.is_relative_to(storage_dir) or not stored_path.is_file():
        raise ValueError(f"服务未将生成音频保存在预期目录：{stored_path}")
    if stored_path.read_bytes() != audio:
        raise ValueError("HTTP 响应与 storage/timbre 中的已保存 WAV 不一致。")
    return stored_path


def validate_wav(audio: bytes) -> tuple[int, int, int]:
    """检查响应为带采样帧的 PCM WAV，返回采样率、声道数和帧数。"""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wave_file:
            params = wave_file.getparams()
            frames = wave_file.readframes(params.nframes)
    except (EOFError, OSError, wave.Error) as exc:
        raise ValueError("服务响应不是有效的 RIFF/WAVE 文件。") from exc
    if (
        params.comptype != "NONE"
        or params.nchannels not in {1, 2}
        or params.framerate <= 0
        or params.nframes <= 0
        or len(frames) != params.nframes * params.nchannels * params.sampwidth
    ):
        raise ValueError("服务响应没有有效 PCM 采样帧。")
    return params.framerate, params.nchannels, params.nframes


def generate_example(base_url: str, timeout: float, description: str, text: str) -> Path:
    """通过 HTTP 请求音色设计并验证输出文件。"""
    request_data = json.dumps(
        {"voice_description": description, "text": text}, ensure_ascii=False
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + ROUTE,
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            audio = response.read()
            headers = email.message.Message()
            for name, value in response.headers.items():
                headers[name] = value
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"VoiceDesign 请求失败：HTTP {exc.code}: {body}") from exc
    sample_rate, channels, frames = validate_wav(audio)
    stored_path = _stored_response_path(headers, audio)
    duration_seconds = frames / sample_rate
    print(f"生成文件：{stored_path}")
    print(
        f"WAV 校验通过：{sample_rate} Hz，{channels} 声道，"
        f"{frames} 帧，{duration_seconds:.2f} 秒，{len(audio)} 字节。"
    )
    return stored_path


def main() -> int:
    """运行本机 VoiceDesign 音频示例。"""
    arguments = build_parser().parse_args()
    try:
        generate_example(
            arguments.base_url,
            arguments.timeout,
            arguments.voice_description,
            arguments.text,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
