#!/usr/bin/env python3
"""串行比较 96k 后端参数；真实推理必须显式运行并使用共享 GPU 锁。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from benchmark_live import ResourceSampler, summarize_resources

from llm_local_runtime import gpu_runtime_lock

ROOT = Path(__file__).resolve().parents[2]

PROFILES = {
    "baseline": [],
    "baseline-b1024": ["--batch-size", "1024", "--ubatch-size", "1024"],
    "mtp2": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
    "mtp4": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"],
    "mtp4-b1024": [
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "4",
        "--batch-size",
        "1024",
        "--ubatch-size",
        "1024",
    ],
    "mtp4-q8kv": [
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "4",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
    ],
}
PROMPT = (
    "请实现一个完整 Python LRUCache，包含 get、put、容量检查、类型注解、"
    "中文说明和使用示例。只输出 Python 源码，尽可能完整。"
)


def request_probe(
    base_url: str,
    prompt: str,
    pid: int,
    *,
    messages: list[dict] | None = None,
    max_tokens: int = 512,
    extra: dict | None = None,
) -> dict:
    """记录真实 SSE 首 token、原生 timings 及系统资源，不按字符估算 token。"""
    payload = {
        "model": "Ornith-1.5-9B-GGUF",
        "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload.update(extra or {})
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    sampler = ResourceSampler(pid, 0.25)
    sampler.start()
    start = time.monotonic()
    first = None
    text = ""
    timings = {}
    usage = {}
    tool_calls: dict[int, dict] = {}
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            for raw in response:
                if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                    continue
                event = json.loads(raw[6:])
                timings.update(event.get("timings") or {})
                usage.update(event.get("usage") or {})
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    chunk = delta.get("content") or delta.get("reasoning_content") or ""
                    for call in delta.get("tool_calls") or []:
                        target = tool_calls.setdefault(
                            call["index"],
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        target["id"] += call.get("id", "")
                        for key in ("name", "arguments"):
                            target["function"][key] += call.get("function", {}).get(key, "")
                        if call.get("function") and first is None:
                            first = time.monotonic() - start
                    if chunk:
                        if first is None:
                            first = time.monotonic() - start
                        text += chunk
        elapsed = time.monotonic() - start
    finally:
        sampler.stop()
    if (not text and not tool_calls) or not usage.get("completion_tokens"):
        raise RuntimeError("模型未返回正文或实际 token 用量。")
    return {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "ttft_seconds": first,
        "elapsed_seconds": elapsed,
        "timings": timings,
        "usage": usage,
        "text": text,
        "tool_calls": list(tool_calls.values()),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "resources": summarize_resources(sampler.samples, interval_seconds=0.25),
    }


def main() -> None:
    """在独立 loopback 端口启动实验后端，并通过既有生命周期助手清理。"""
    sys.path.insert(0, str(ROOT / "model-text"))
    from llama_cpp_runtime.runtime import create_quarantine_marker, wait_for_sleeping

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", required=True, type=Path)
    parser.add_argument("--model-file", required=True, type=Path)
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=Path(
            os.getenv("GPU_LOCK_FILE", str(ROOT / "storage/.cache/runtime/gpu-runtime.lock"))
        ),
    )
    parser.add_argument(
        "--existing-backend", default=os.getenv("ORNITH_9B_LLAMA_BASE_URL", "http://127.0.0.1:8505")
    )
    parser.add_argument(
        "--profiles", nargs="+", choices=PROFILES, default=["baseline", "mtp2", "mtp4", "mtp4-q8kv"]
    )
    parser.add_argument("--port", type=int, default=28504)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--context-copies", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 0 or args.context_copies < 1:
        parser.error("repeats 必须非负，context-copies 必须为正数。")
    output = (
        ROOT
        / "storage/llm_tests/ornith_1_5_9b_gguf"
        / datetime.now(UTC).strftime("tuning-%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "model-text"))
    helper = [sys.executable, "-m", "llama_cpp_runtime.runtime"]
    base = f"http://127.0.0.1:{args.port}"
    records = []
    source = (ROOT / "model-text/llama_cpp_runtime/runtime.py").read_text()
    prompts = [PROMPT] * args.repeats + [
        "以下是参考源码，阅读后完成末尾任务。\n"
        + source * args.context_copies
        + "\n任务："
        + PROMPT
    ]
    (output / "long-prompt.txt").write_text(prompts[-1])
    for profile in args.profiles:
        pid_file = output / f"{profile}.pid"
        backend_args = [
            "--model",
            str(args.model_file.resolve()),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--alias",
            "Ornith-1.5-9B-GGUF",
            "--device",
            "CUDA0",
            "--gpu-layers",
            "99",
            "--flash-attn",
            "on",
            "--jinja",
            "--ctx-size",
            "98304",
            "--batch-size",
            "512",
            "--threads",
            "6",
            "--threads-batch",
            "8",
            "--load-mode",
            "none",
            "--parallel",
            "1",
            "--sleep-idle-seconds",
            "1",
            *PROFILES[profile],
        ]
        common = [
            "--lock-file",
            str(args.lock_file),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--executable",
            str(args.server_bin.resolve()),
            "--lock-timeout",
            "900",
            "--model",
            "Ornith-9B-tuning",
        ]
        with (output / f"{profile}.log").open("xb") as log:
            subprocess.run(
                helper
                + [
                    "startup",
                    *common,
                    "--pid-file",
                    str(pid_file),
                    "--base-url",
                    base,
                    "--timeout",
                    "180",
                    "--",
                    *backend_args,
                ],
                env=env,
                stdout=log,
                stderr=log,
                check=True,
            )
            pid = int(pid_file.read_text())
            try:
                # 同一提示重复测量，每次卸载；第二种提示增加约万 token 的输入。
                for index, prompt in enumerate(prompts):
                    with gpu_runtime_lock(args.lock_file, f"tuning/{profile}", timeout_seconds=900):
                        try:
                            wait_for_sleeping(
                                args.existing_backend, timeout_seconds=30, poll_interval=0.25
                            )
                            wait_for_sleeping(base, timeout_seconds=30, poll_interval=0.25)
                            record = request_probe(base, prompt, pid)
                        finally:
                            try:
                                wait_for_sleeping(base, timeout_seconds=60, poll_interval=0.25)
                                time.sleep(2)
                            except Exception:
                                create_quarantine_marker(
                                    args.lock_file,
                                    host="127.0.0.1",
                                    port=args.port,
                                    pid=pid,
                                    pgid=pid,
                                    executable=str(args.server_bin.resolve()),
                                )
                                raise
                    record.update(profile=profile, case=index, backend_args=backend_args)
                    records.append(record)
                    (output / "results.json").write_text(
                        json.dumps(records, ensure_ascii=False, indent=2)
                    )
                    print(
                        json.dumps(
                            {
                                "profile": profile,
                                "case": index,
                                "ttft": record["ttft_seconds"],
                                "timings": record["timings"],
                            }
                        ),
                        flush=True,
                    )
            finally:
                subprocess.run(
                    helper + ["shutdown", *common, "--pid", str(pid), "--pgid", str(pid)],
                    env=env,
                    stdout=log,
                    stderr=log,
                    check=True,
                )
    print(f"report_dir={output}", flush=True)


if __name__ == "__main__":
    main()
