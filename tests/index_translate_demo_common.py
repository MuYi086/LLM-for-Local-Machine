"""调用 8382–8385 原生 Index 接口，保存模型原始 WAV/TXT。"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from index_demo_environment import check_service_environment, print_checks

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SOURCE_TEXT = REPOSITORY_DIR / "tests/testIndexTranslate/subtitles.md"
SOURCE_AUDIO = REPOSITORY_DIR / "tests/testIndexTranslate/speech.wav"


def save_output(path: Path, content: bytes) -> None:
    """原子保存响应内容，保留 UTF-8、换行和 WAV 字节。"""
    if not content:
        raise SystemExit("模型返回了空输出。")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"原始输出：{path}；{len(content)} bytes")


def request_output(request: urllib.request.Request, *, timeout: float, expected_type: str) -> bytes:
    """调用单个模型接口，拒绝把错误 JSON 或其他模型输出保存为成功产物。"""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            if content_type != expected_type:
                raise SystemExit(f"响应类型为 {content_type}，预期 {expected_type}。")
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(4_000).decode("utf-8", errors="replace")
        raise SystemExit(f"模型请求失败 HTTP {exc.code}：{detail}") from exc
    except OSError as exc:
        raise SystemExit(f"模型请求失败：{exc}") from exc


def run_text_model_demo(
    *,
    demo_name: str,
    env_prefix: str,
    port: int,
    route: str,
    output_name: str,
    demo_dir: Path,
) -> None:
    """发送模型原生文本参数并将响应直接保存为 TXT，不串接其他服务。"""
    parser = argparse.ArgumentParser(description=f"运行 {demo_name} 原生文本翻译 Demo")
    parser.add_argument("--input", type=Path, default=SOURCE_TEXT, help="UTF-8 原文")
    parser.add_argument("--output", type=Path, default=demo_dir / "dist" / output_name)
    parser.add_argument(
        "--base-url", default=os.getenv(f"{env_prefix}_BASE_URL", f"http://127.0.0.1:{port}")
    )
    parser.add_argument("--check-only", action="store_true", help="只检查，条件不足时跳过")
    parser.add_argument("--timeout", type=float, default=1800)
    if route == "homura":
        parser.add_argument("--syllables", type=int, default=36)
        parser.add_argument("--target-language", default="en")
        parser.add_argument("--glossary", type=json.loads, default={})
    elif route == "naiLong":
        parser.add_argument(
            "--direction", choices=("zh-en", "en-zh", "zh-ja", "ja-zh"), default="zh-en"
        )
    else:
        parser.add_argument("--source-language", default="zh")
        parser.add_argument("--target-language", default="en")
        parser.add_argument("--glossary", type=json.loads, default={})
        parser.add_argument("--hard", action="append", default=[])
        parser.add_argument("--soft", action="append", default=[])
        parser.add_argument("--instruction", default="")
        parser.add_argument("--genre", default="文本")
        parser.add_argument("--raw-prompt", action="store_true")
        parser.add_argument("--image", type=Path, help="可选原生图片输入，需要服务已配置 mmproj")
    args = parser.parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout 必须大于零。")
    image = getattr(args, "image", None)
    checks = check_service_environment(
        input_path=image or args.input,
        base_url=args.base_url,
        expected_backend=f"{demo_name} GGUF native",
        output_type="text/plain",
        text_input=image is None,
        image_input=image is not None,
    )
    if not print_checks(checks):
        raise SystemExit("前置条件未满足，跳过推理；未写入模型输出。")
    if args.check_only:
        return
    payload = {"text": args.input.read_text(encoding="utf-8")}
    if route == "homura":
        payload.update(
            target_language=args.target_language,
            target_syllables=args.syllables,
            glossary=args.glossary,
        )
    elif route == "naiLong":
        payload["direction"] = args.direction
    else:
        payload.update(
            source_language=args.source_language,
            target_language=args.target_language,
            glossary=args.glossary,
            hard_constraints=args.hard,
            soft_constraints=args.soft,
            genre=args.genre,
            instruction=args.instruction,
            raw_prompt=args.raw_prompt,
        )
        if image:
            media_type = mimetypes.guess_type(image.name)[0] or "image/png"
            payload["image"] = f"data:{media_type};base64," + base64.b64encode(
                image.read_bytes()
            ).decode("ascii")
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/indexTranslate/{route}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content = request_output(request, timeout=args.timeout, expected_type="text/plain")
    save_output(args.output, content)


def run_echo_demo(demo_dir: Path) -> None:
    """流式上传源语音至 Echo 原生接口，保存 WAV 或 WAV/info ZIP。"""
    parser = argparse.ArgumentParser(description="运行 Index-Echo-S2ST 原生语音翻译 Demo")
    parser.add_argument("--audio", type=Path, default=SOURCE_AUDIO)
    parser.add_argument(
        "--base-url", default=os.getenv("INDEX_ECHO_S2ST_BASE_URL", "http://127.0.0.1:8382")
    )
    parser.add_argument("--target-language", choices=("en", "es", "ja", "zh"), default="en")
    parser.add_argument("--return-info", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout 必须大于零。")
    checks = check_service_environment(
        input_path=args.audio,
        base_url=args.base_url,
        expected_backend="Index-Echo-S2ST DubbingBridgeModel",
        output_type="audio/wav",
        text_input=False,
    )
    if not print_checks(checks):
        raise SystemExit("前置条件未满足，跳过 Echo 原生推理；未写入 WAV。")
    if args.check_only:
        return
    boundary = f"index-echo-{uuid.uuid4().hex}"
    prefix = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="target_language"\r\n\r\n{args.target_language}\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="return_info"\r\n\r\n{str(args.return_info).lower()}\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="speech.wav"\r\nContent-Type: audio/wav\r\n\r\n'
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()

    def chunks():
        yield prefix
        with args.audio.open("rb") as source:
            yield from iter(lambda: source.read(1024 * 1024), b"")
        yield suffix

    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/indexTranslate/echoS2ST",
        data=chunks(),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(prefix) + args.audio.stat().st_size + len(suffix)),
        },
        method="POST",
    )
    expected_type = "application/zip" if args.return_info else "audio/wav"
    content = request_output(request, timeout=args.timeout, expected_type=expected_type)
    extension = "zip" if args.return_info else "wav"
    output = (
        args.output
        or demo_dir / "dist" / f"index_echo_s2st_zh_to_{args.target_language}.{extension}"
    )
    save_output(output, content)
