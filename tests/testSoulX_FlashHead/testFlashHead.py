#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.13"
# dependencies = ["httpx==0.28.1"]
# ///
"""上传同目录的 1.png 和 demo.wav，调用 8391 服务并保存 dist/demo.mp4。

在脚本目录运行，uv 会按脚本声明自动准备依赖：
    uv run ./testFlashHead.py

也可以在仓库根目录运行：
    uv run tests/testSoulX_FlashHead/testFlashHead.py

这是手动执行的真实模型演示，不会在 unittest discover 中自动执行。
"""

from __future__ import annotations

import argparse
import math
import mimetypes
import os
import sys
import tempfile
import uuid
from pathlib import Path

import httpx

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SERVER_URL = "http://127.0.0.1:8391"
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    """解析输入文件、服务地址和官方模型选项。"""
    parser = argparse.ArgumentParser(description="用人像图片和音频生成 SoulX 数字人视频。")
    parser.add_argument("--image", type=Path, default=BASE_DIR / "1.png", help="人像图片路径")
    parser.add_argument("--audio", type=Path, default=BASE_DIR / "demo.wav", help="驱动音频路径")
    parser.add_argument(
        "--output", type=Path, default=BASE_DIR / "dist" / "demo.mp4", help="输出 MP4 路径"
    )
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="SoulX 服务地址")
    parser.add_argument("--model-type", choices=("lite", "pro"), default="lite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-face-crop", action="store_true", help="启用官方人脸裁剪")
    parser.add_argument(
        "--timeout", type=float, default=3600.0, help="请求超时秒数，包含 GPU 排队与推理时间"
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout 必须是有限的正数。")
    if not 0 <= args.seed <= 4_294_967_295:
        parser.error("--seed 必须在 0 至 4294967295 之间。")
    for label, path in (("图片", args.image), ("音频", args.audio)):
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"{label}文件不存在或为空: {path}")
    if args.output.suffix.lower() != ".mp4":
        parser.error("--output 必须使用 .mp4 扩展名。")
    if args.output.resolve() in {args.image.resolve(), args.audio.resolve()}:
        parser.error("输出路径不能覆盖输入素材。")
    return args


def check_response(response: httpx.Response) -> None:
    """显示服务端错误详情，避免将 JSON 错误响应保存成视频。"""
    if response.is_success:
        return
    response.read()
    try:
        detail = response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        detail = response.text
    raise RuntimeError(f"HTTP {response.status_code}: {str(detail)[:2000]}")


def upload_file(client: httpx.Client, path: Path, field: str, full_path: str) -> None:
    """流式上传素材，并使用同一个 full_path 构造后续生成请求。"""
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    endpoint = "/v1/upload_image" if field == "image" else "/v1/upload_audio"
    with path.open("rb") as source:
        response = client.post(
            endpoint,
            data={"full_path": full_path},
            files={field: (path.name, source, content_type)},
        )
    check_response(response)
    result = response.json()
    if result.get("code") != 200:
        raise RuntimeError(f"上传失败: {result}")
    print(f"已上传 {path.name}（{path.stat().st_size:,} 字节）", flush=True)


def generate_video(
    client: httpx.Client, payload: dict[str, str | int | bool], output_path: Path
) -> None:
    """流式下载 MP4 并原子提交；请求失败时保留已有成品。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.stem}_", suffix=".part"
    )
    temporary = Path(name)
    try:
        with (
            os.fdopen(descriptor, "wb") as destination,
            client.stream("POST", "/v1/soulX/flashHead", json=payload) as response,
        ):
            check_response(response)
            media_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
            if media_type != "video/mp4":
                raise RuntimeError(f"生成接口未返回 video/mp4，实际为 {media_type!r}。")
            for chunk in response.iter_bytes(CHUNK_SIZE):
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        with temporary.open("rb") as source:
            header = source.read(12)
        if len(header) != 12 or header[4:8] != b"ftyp":
            raise RuntimeError("返回内容不是有效的 MP4 文件。")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    """检查依赖、上传人像和音频，再请求一次数字人视频生成。"""
    args = parse_args()
    namespace = f"testSoulX_FlashHead/{uuid.uuid4().hex}"
    image_reference = f"{namespace}/{args.image.name}"
    audio_reference = f"{namespace}/{args.audio.name}"
    try:
        with httpx.Client(
            base_url=args.server_url.rstrip("/") + "/",
            timeout=httpx.Timeout(args.timeout, connect=10.0),
            trust_env=False,
        ) as client:
            health = client.get("/v1/health")
            check_response(health)
            available = health.json().get("available", {})
            required = (
                "code",
                "wav2vec",
                f"model_{args.model_type}",
                "ffmpeg",
                "ffprobe",
                "torch",
                "cuda",
            )
            missing = [name for name in required if not available.get(name)]
            if missing:
                raise RuntimeError("SoulX 服务依赖未就绪: " + ", ".join(missing))
            upload_file(client, args.image, "image", image_reference)
            upload_file(client, args.audio, "audio", audio_reference)
            payload: dict[str, str | int | bool] = {
                "image_path": image_reference,
                "audio_path": audio_reference,
                "model_type": args.model_type,
                "seed": args.seed,
                "use_face_crop": args.use_face_crop,
            }
            print(f"正在调用 {args.server_url} 生成 {args.model_type} 数字人视频……", flush=True)
            generate_video(client, payload, args.output)
        print(f"视频已保存: {args.output.resolve()}（{args.output.stat().st_size:,} 字节）")
        return 0
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        print(f"生成失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
