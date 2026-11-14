#!/usr/bin/env python3
"""调用本机 Ornith MTP 9B Q8 API 并保存生成文本。"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
API_ROUTE = "/v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8"
DEFAULT_PROMPT = "请用简洁的中文介绍一下你自己，并说明你擅长处理哪些任务。"
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png"}


def extract_assistant_text(payload: Any) -> str:
    """从 OpenAI Chat Completions 响应中提取非空 assistant 文本。"""
    try:
        text = payload["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("API 响应缺少 choices[0].message.content。") from exc
    if not isinstance(text, str) or not text.strip():
        raise ValueError("模型返回了空的 assistant 文本。")
    return text.strip()


def build_user_content(prompt: str, image_path: str | Path | None) -> str | list[dict[str, Any]]:
    """构造文本或包含本地 PNG/JPEG 图片的 user content。"""
    if image_path is None:
        return prompt

    path = Path(image_path)
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError("图片必须是 PNG 或 JPEG 文件。")
    try:
        image_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"无法读取图片文件：{path}") from exc
    if not image_bytes:
        raise ValueError("图片文件不能为空。")
    image_data = base64.b64encode(image_bytes).decode("ascii")
    return [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{image_data}"},
        },
    ]


def request_completion(
    *,
    base_url: str,
    prompt: str,
    max_tokens: int = 256,
    timeout_seconds: float = 1800.0,
    image_path: str | Path | None = None,
) -> str:
    """请求 MTP 9B 固定路由并返回 assistant 文本。"""
    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": build_user_content(prompt, image_path)}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{API_ROUTE}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ornith MTP 9B API 返回 HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"无法连接 Ornith MTP 9B API：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ornith MTP 9B API 返回了无效 JSON。") from exc
    return extract_assistant_text(result)


def write_sample(
    text: str,
    output_dir: str | Path,
    *,
    timestamp: datetime | None = None,
    pid: int | None = None,
) -> Path:
    """将生成文本以独占方式写入运行数据目录。"""
    if not text.strip():
        raise ValueError("不能保存空的生成文本。")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    created_at = timestamp or datetime.now(UTC)
    process_id = pid if pid is not None else os.getpid()
    filename = f"ornith-mtp-9b-q8-example-{created_at:%Y%m%dT%H%M%S%fZ}-{process_id}.txt"
    output_path = directory / filename
    with output_path.open("x", encoding="utf-8") as output_file:
        output_file.write(text.strip())
        output_file.write("\n")
    return output_path


def main() -> int:
    """生成一条真实模型样例并输出保存路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("ORNITH_MTP_9B_TEST_BASE_URL", "http://127.0.0.1:8504"),
        help="Ornith MTP 9B API 根地址。",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="发送给模型的提示。")
    parser.add_argument("--image", type=Path, help="可选的本地 PNG/JPEG 图片路径。")
    parser.add_argument("--max-tokens", type=int, default=256, help="生成 token 上限。")
    parser.add_argument("--timeout", type=float, default=1800.0, help="请求超时秒数。")
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "ORNITH_MTP_9B_TEST_OUTPUT_DIR",
            str(
                Path(os.getenv("STORAGE_DIR", ROOT / "storage"))
                / "llm_tests/ornith_1_5_9b_mtp_gguf_q8"
            ),
        ),
        help="生成文本输出目录。",
    )
    args = parser.parse_args()
    if not args.prompt.strip() or args.max_tokens <= 0 or args.timeout <= 0:
        parser.error("prompt 必须非空，max-tokens 和 timeout 必须为正数。")
    try:
        text = request_completion(
            base_url=args.base_url,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout,
            image_path=args.image,
        )
        output_path = write_sample(text, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Ornith MTP 9B 样例生成失败：{exc}", file=sys.stderr)
        return 1
    print(f"Ornith MTP 9B 样例已保存：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
