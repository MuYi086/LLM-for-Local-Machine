#!/usr/bin/env python3
"""调用本机 Ornith-1.5-9B-GGUF API 并保存生成文本。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
API_ROUTE = "/v1/ornith-ai/Ornith-1.5-9B-GGUF"
DEFAULT_PROMPT = "请用简洁的中文介绍一下你自己，并说明你擅长处理哪些任务。"


def extract_assistant_text(payload: Any) -> str:
    """从 OpenAI Chat Completions 响应中提取非空 assistant 文本。"""
    try:
        text = payload["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("API 响应缺少 choices[0].message.content。") from exc
    if not isinstance(text, str) or not text.strip():
        raise ValueError("模型返回了空的 assistant 文本。")
    return text.strip()


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
    filename = f"ornith-9b-example-{created_at:%Y%m%dT%H%M%S%fZ}-{process_id}.txt"
    output_path = directory / filename
    with output_path.open("x", encoding="utf-8") as output_file:
        output_file.write(text.strip())
        output_file.write("\n")
    return output_path


def request_completion(
    *,
    base_url: str,
    prompt: str,
    max_tokens: int = 256,
    timeout_seconds: float = 1800.0,
) -> str:
    """请求 9B 固定路由并返回 assistant 文本。"""
    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
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
        raise RuntimeError(f"Ornith 9B API 返回 HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"无法连接 Ornith 9B API：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ornith 9B API 返回了无效 JSON。") from exc
    return extract_assistant_text(result)


def main() -> int:
    """生成一条真实模型样例并输出保存路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("ORNITH_9B_TEST_BASE_URL", "http://127.0.0.1:8503"),
        help="Ornith 9B API 根地址。",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="发送给模型的中文提示。")
    parser.add_argument("--max-tokens", type=int, default=256, help="生成 token 上限。")
    parser.add_argument("--timeout", type=float, default=1800.0, help="请求超时秒数。")
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "ORNITH_9B_TEST_OUTPUT_DIR",
            str(Path(os.getenv("STORAGE_DIR", ROOT / "storage")) / "llm_tests/ornith_1_5_9b_gguf"),
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
        )
        output_path = write_sample(text, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Ornith 9B 样例生成失败：{exc}", file=sys.stderr)
        return 1
    print(f"Ornith 9B 样例已保存：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
