"""通过 8501 HTTP API 生成一张 MOSS-Image-SFT 示例图。"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

API_ROUTE = "/v1/openmoss/MOSS-Image-SFT-Q8_0"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _default_output() -> Path:
    """为每次生成创建独立运行产物路径。"""
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    unique_id = uuid.uuid4().hex[:8]
    return (
        Path(__file__).resolve().parents[2]
        / "storage/llm_tests/MOSS-Image-SFT-Q8_0"
        / f"{timestamp}-{unique_id}-MOSS-Image-SFT-Q8_0-example.png"
    )


def _save_exclusive(path: Path, content: bytes) -> None:
    """将图片原子发布到新路径，避免覆盖既有生成结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def generate_example(base_url: str, output: Path, timeout: float) -> Path:
    """调用模型 API 并保存 Base64 响应中的 PNG。

    Args:
        base_url: 图片服务的 HTTP 根地址。
        output: 不得覆盖已有文件的输出路径。
        timeout: 单次 HTTP 请求的最长等待秒数。

    Returns:
        已原子发布的 PNG 文件路径。
    """
    payload = {
        "model": "MOSS-Image-SFT-Q8_0",
        "prompt": "A cinematic product photograph of a red bicycle leaning against a quiet canal, soft evening light, realistic details",
        "size": "1024x1024",
        "steps": 20,
        "cfg_scale": 3.0,
        "seed": 13579,
        "n": 1,
        "response_format": "b64_json",
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{API_ROUTE}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    image_data = result["data"][0]["b64_json"]
    image = base64.b64decode(image_data, validate=True)
    if not image.startswith(PNG_SIGNATURE):
        raise ValueError("API 返回内容不是 PNG 图片。")
    _save_exclusive(output, image)
    return output


def main() -> int:
    """解析服务地址和输出路径后生成模型样例。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("MOSS_IMAGE_BASE_URL", "http://127.0.0.1:8501"),
    )
    parser.add_argument("--output", type=Path, default=_default_output())
    parser.add_argument("--timeout", type=float, default=3600)
    arguments = parser.parse_args()
    if not 0 < arguments.timeout < float("inf"):
        parser.error("--timeout 必须为有限正数。")
    output = generate_example(arguments.base_url, arguments.output, arguments.timeout)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
