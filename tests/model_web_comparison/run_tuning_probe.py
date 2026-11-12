#!/usr/bin/env python3
"""以原网页提示词执行短输出探测，为完整复测筛选后端参数。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from run_comparison import (
    MODELS,
    PROMPT_PATH,
    ROOT,
    find_llama_server_pid,
    parse_stream,
    verify_service,
)


def main() -> None:
    """串行发送限长请求并保存原始 SSE、后端命令和测速结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--models", nargs="+", choices=("ornith",), default=["ornith"]
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "storage/llm_tests/web_comparison_probes",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("label 只能包含英文、数字、下划线和连字符。")
    if args.max_tokens <= 0 or args.repeats <= 0:
        parser.error("max-tokens 和 repeats 必须为正数。")
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / f"{args.label}.json"
    if summary_path.exists():
        parser.error(f"已有同名探测记录：{summary_path}")
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    results = []
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    cache_dir = Path(os.getenv("RUNTIME_CACHE_DIR", ROOT / "storage/.cache/runtime"))
    lock_path = Path(os.getenv("GPU_LOCK_FILE", cache_dir / "gpu-runtime.lock"))
    for short_name, model in zip(("ornith",), MODELS, strict=True):
        if short_name not in args.models:
            continue
        for repeat in range(1, args.repeats + 1):
            if Path(f"{lock_path}.quarantine").exists():
                raise RuntimeError("GPU 已隔离，停止探测并保留标记。")
            verify_service(model)
            pid = find_llama_server_pid(model["backend_port"])
            backend_args = [
                item.decode()
                for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                if item
            ]
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request = urllib.request.Request(
                f"{model['api_base']}{model['route']}",
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            stem = f"{args.label}_{short_name}_{repeat}_{datetime.now(UTC):%Y%m%dT%H%M%S%f}"
            print(f"探测 {args.label}: {short_name} 第 {repeat} 次", flush=True)
            with (directory / f"{stem}.sse").open("xb") as raw_output:
                started = time.monotonic()
                with opener.open(request, timeout=1800) as response:

                    def lines():
                        """读取到 HTTP EOF，确保服务端完成卸载和锁释放。"""
                        for line in response:
                            raw_output.write(line)
                            yield line

                    text, tokens, first, last, reasons = parse_stream(lines())
                elapsed = time.monotonic() - started
            if not text or not tokens or first is None or last is None or last <= first:
                raise RuntimeError("探测响应缺少文本、token 用量或有效时间戳，原始 SSE 已保留。")
            verify_service(model)
            if Path(f"{lock_path}.quarantine").exists():
                raise RuntimeError("请求完成后仍有 GPU 隔离标记，停止探测。")
            result = {
                "label": args.label,
                "model": short_name,
                "repeat": repeat,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "request": payload,
                "backend_arguments": backend_args,
                "completion_tokens": tokens,
                "finish_reasons": reasons,
                "elapsed_seconds": round(elapsed, 3),
                "time_to_first_token_seconds": round(first - started, 3),
                "generation_seconds": round(last - first, 3),
                "tokens_per_second": round(tokens / (last - first), 3),
                "raw_sse": f"{stem}.sse",
            }
            (directory / f"{stem}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            )
            results.append(result)
            print(f"  {tokens} token，{result['tokens_per_second']:.2f} token/s", flush=True)
    with summary_path.open("x", encoding="utf-8") as output:
        json.dump(results, output, ensure_ascii=False, indent=2)
        output.write("\n")
    for name in args.models:
        rates = [item["tokens_per_second"] for item in results if item["model"] == name]
        print(f"{name} 中位数：{statistics.median(rates):.2f} token/s", flush=True)


if __name__ == "__main__":
    main()
