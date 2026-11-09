#!/usr/bin/env python3
"""调用本机 Qwen3.6 API 并将生成文本保存到运行数据目录。"""

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
API_ROUTE = "/v1/qwen/Qwen3.6-35B-A3B-MTP"
DEFAULT_PROMPT = "请用简洁的中文解释混合专家模型如何分配 token。"


def extract_assistant_text(payload: Any) -> str:
    """从 Chat Completions 响应中提取非空 assistant 文本。"""
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
) -> Path:
    """将生成文本以独占方式写入运行数据目录。"""
    if not text.strip():
        raise ValueError("不能保存空的生成文本。")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    created_at = timestamp or datetime.now(UTC)
    filename = f"qwen3-6-example-{created_at:%Y%m%dT%H%M%S}-{os.getpid()}.txt"
    output_path = directory / filename
    with output_path.open("x", encoding="utf-8") as output_file:
        output_file.write(text.strip())
        output_file.write("\n")
    return output_path


def request_completion(
    *,
    base_url: str,
    prompt: str,
    timeout_seconds: float = 1800.0,
) -> str:
    """请求固定 Qwen 路由，并关闭本次样例请求的 thinking 输出。"""
    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
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
        raise RuntimeError(f"Qwen3.6 API 返回 HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"无法连接 Qwen3.6 API：{exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Qwen3.6 API 返回了无效 JSON。") from exc
    return extract_assistant_text(result)


def main() -> int:
    """生成一条真实模型样例并输出保存路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("QWEN36_TEST_BASE_URL", "http://127.0.0.1:8402"),
        help="Qwen3.6 API 根地址。",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="发送给模型的中文提示。")
    parser.add_argument("--timeout", type=float, default=1800.0, help="请求超时秒数。")
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "QWEN36_TEST_OUTPUT_DIR",
            str(Path(os.getenv("STORAGE_DIR", ROOT / "storage")) / "llm_tests/qwen3_6_35b_a3b"),
        ),
        help="生成文本输出目录。",
    )
    args = parser.parse_args()
    if not args.prompt.strip() or args.timeout <= 0:
        parser.error("prompt 必须非空且 timeout 必须为正数。")
    try:
        text = request_completion(
            base_url=args.base_url,
            prompt=args.prompt,
            timeout_seconds=args.timeout,
        )
        output_path = write_sample(text, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Qwen3.6 样例生成失败：{exc}", file=sys.stderr)
        return 1
    print(f"Qwen3.6 样例已保存：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
