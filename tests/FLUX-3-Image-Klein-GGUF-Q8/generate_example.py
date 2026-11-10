"""通过 8502 HTTP API 生成并编辑一张 FLUX.3 Klein 示例图。"""

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

API_ROUTE = "/v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _default_output(kind: str, unique_id: str) -> Path:
    """为生成结果或编辑结果创建独立运行产物路径。"""
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return (
        Path(__file__).resolve().parents[2]
        / "storage/llm_tests/FLUX-3-Image-Klein-GGUF-Q8"
        / f"{timestamp}-{unique_id}-FLUX-3-Image-Klein-GGUF-Q8-{kind}.png"
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


def _decode_png(response: object) -> bytes:
    """读取 OpenAI Images 风格响应并确认返回内容为 PNG。"""
    result = json.load(response)  # type: ignore[arg-type]
    image = base64.b64decode(result["data"][0]["b64_json"], validate=True)
    if not image.startswith(PNG_SIGNATURE):
        raise ValueError("API 返回内容不是 PNG 图片。")
    return image


def _multipart_body(fields: dict[str, str], image: bytes, boundary: str) -> bytes:
    """构造包含表单字段和 PNG 文件的 multipart 请求体。"""
    delimiter = f"--{boundary}".encode("ascii")
    chunks = []
    for name, value in fields.items():
        chunks.extend(
            (
                delimiter,
                b'Content-Disposition: form-data; name="' + name.encode("ascii") + b'"',
                b"",
                value.encode("utf-8"),
            )
        )
    chunks.extend(
        (
            delimiter,
            b'Content-Disposition: form-data; name="image"; filename="generated.png"',
            b"Content-Type: image/png",
            b"",
            image,
            f"--{boundary}--".encode("ascii"),
            b"",
        )
    )
    return b"\r\n".join(chunks)


def generate_and_edit_example(
    base_url: str,
    generated_output: Path,
    edited_output: Path,
    timeout: float,
) -> tuple[Path, Path]:
    """先生成示例图，再将其作为参考图发送到编辑 API。

    Args:
        base_url: 图片服务的 HTTP 根地址。
        generated_output: 文生图结果的独占输出路径。
        edited_output: 编辑结果的独占输出路径。
        timeout: 单次 HTTP 请求的最长等待秒数。

    Returns:
        已原子发布的文生图和编辑结果路径。
    """
    payload = {
        "model": "FLUX-3-Image-Klein-GGUF-Q8",
        "prompt": "A cinematic photograph of a red bicycle beside a quiet canal at sunset",
        "size": "1024x1024",
        "steps": 16,
        "cfg_scale": 1.0,
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
        generated_image = _decode_png(response)
    _save_exclusive(generated_output, generated_image)

    boundary = uuid.uuid4().hex
    edit_fields = {
        "prompt": "Change the red bicycle to bright blue and preserve the scene composition",
        "size": "1024x1024",
        "steps": "16",
        "cfg_scale": "1.0",
        "seed": "13580",
    }
    edit_request = urllib.request.Request(
        f"{base_url.rstrip('/')}{API_ROUTE}/edit",
        data=_multipart_body(edit_fields, generated_image, boundary),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(edit_request, timeout=timeout) as response:
        edited_image = _decode_png(response)
    _save_exclusive(edited_output, edited_image)
    return generated_output, edited_output


def main() -> int:
    """解析服务地址与输出路径后依次生成和编辑模型样例。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("FLUX3_IMAGE_BASE_URL", "http://127.0.0.1:8502"),
    )
    unique_id = uuid.uuid4().hex[:8]
    parser.add_argument(
        "--generated-output", type=Path, default=_default_output("generated", unique_id)
    )
    parser.add_argument("--edited-output", type=Path, default=_default_output("edited", unique_id))
    parser.add_argument("--timeout", type=float, default=3600)
    arguments = parser.parse_args()
    if not 0 < arguments.timeout < float("inf"):
        parser.error("--timeout 必须为有限正数。")
    generated, edited = generate_and_edit_example(
        arguments.base_url,
        arguments.generated_output,
        arguments.edited_output,
        arguments.timeout,
    )
    print(generated)
    print(edited)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
