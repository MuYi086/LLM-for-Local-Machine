#!/usr/bin/env python3
"""显式验证多轮对话跨 sleep 的缓存复用、失效及显存释放。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from benchmark_live import ResourceSampler, summarize_resources, take_resource_sample
from tune_backend import ROOT, request_probe

from llm_local_runtime import gpu_runtime_lock


def main() -> None:
    """对独立 backend 运行冷请求、追加、分叉、换题与重新冷算对照。"""
    sys.path.insert(0, str(ROOT / "model-text"))
    from llama_cpp_runtime.runtime import create_quarantine_marker, wait_for_sleeping

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--mtp", choices=("none", "draft-mtp"), default="none")
    parser.add_argument("--cache", choices=("0", "1"), default="1")
    parser.add_argument("--cache-mib", type=int, default=4096)
    parser.add_argument("--context-copies", type=int, default=1)
    parser.add_argument("--expect-cache", action="store_true")
    parser.add_argument("--tool-turn", action="store_true")
    parser.add_argument(
        "--source-file", type=Path, default=ROOT / "model-text/llama_cpp_runtime/runtime.py"
    )
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=28504)
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=Path(
            os.getenv("GPU_LOCK_FILE", str(ROOT / "storage/.cache/runtime/gpu-runtime.lock"))
        ),
    )
    parser.add_argument(
        "--existing-backend", default=os.getenv("ORNITH_9B_LLAMA_BASE_URL", "http://127.0.0.1:8504")
    )
    args = parser.parse_args()
    if args.cache_mib <= 0 or args.context_copies < 1:
        parser.error("cache-mib 和 context-copies 必须为正数。")
    directory = (
        ROOT
        / "storage/llm_tests/ornith_1_5_9b_gguf"
        / datetime.now(UTC).strftime("sleep-cache-%Y%m%dT%H%M%S%fZ")
    )
    directory.mkdir(parents=True, exist_ok=False)
    base = f"http://127.0.0.1:{args.port}"
    env = dict(
        os.environ, PYTHONPATH=str(ROOT / "model-text"), LLAMA_SERVER_SLEEP_PROMPT_CACHE=args.cache
    )
    helper = [sys.executable, "-m", "llama_cpp_runtime.runtime"]
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
        "Ornith-sleep-cache-test",
    ]
    backend = [
        "--model",
        str(args.model_file),
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
        "1024",
        "--ubatch-size",
        "1024",
        "--threads",
        "6",
        "--threads-batch",
        "8",
        "--spec-type",
        args.mtp,
        "--spec-draft-n-max",
        "4",
        "--load-mode",
        "none",
        "--parallel",
        "1",
        "--sleep-idle-seconds",
        "1",
        "--cache-ram",
        str(args.cache_mib),
    ]
    (directory / "config.json").write_text(
        json.dumps(
            {
                "cache": args.cache,
                "backend_args": backend,
                "binary": str(args.server_bin.resolve()),
            },
            indent=2,
        )
    )
    pid_file = directory / "backend.pid"
    results = []
    with (directory / "backend.log").open("xb") as log:
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
                *backend,
            ],
            env=env,
            stdout=log,
            stderr=log,
            check=True,
        )
        pid = int(pid_file.read_text())
        try:

            def run(label: str, messages: list[dict], extra: dict | None = None) -> dict:
                """在同一 GPU 租约内请求并等待卸载，保留包括保存开销的指标。"""
                with gpu_runtime_lock(args.lock_file, f"sleep-cache/{label}", timeout_seconds=900):
                    wait_for_sleeping(args.existing_backend, timeout_seconds=60, poll_interval=0.25)
                    start = time.monotonic()
                    lease_sampler = ResourceSampler(pid, 0.25)
                    lease_sampler.start()
                    try:
                        result = request_probe(
                            base,
                            json.dumps(messages, ensure_ascii=False),
                            pid,
                            messages=messages,
                            max_tokens=64,
                            extra=extra,
                        )
                    finally:
                        try:
                            wait_for_sleeping(base, timeout_seconds=90, poll_interval=0.1)
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
                        finally:
                            lease_sampler.stop()
                    result.update(
                        label=label,
                        lease_seconds=time.monotonic() - start,
                        sleeping_resources=asdict(take_resource_sample(pid)),
                        lease_resources=summarize_resources(
                            lease_sampler.samples, interval_seconds=0.25
                        ),
                    )
                    with urlopen(base + "/props", timeout=5) as response:
                        props = json.loads(response.read())
                    result["props"] = {
                        k: props.get(k)
                        for k in (
                            "is_sleeping",
                            "sleep_prompt_cache_supported",
                            "sleep_prompt_cache_enabled",
                        )
                    }
                results.append(result)
                (directory / "results.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=2)
                )
                print(
                    json.dumps(
                        {
                            "label": label,
                            "timings": result["timings"],
                            "ttft": result["ttft_seconds"],
                            "lease_seconds": result["lease_seconds"],
                            "text": result["text"],
                        }
                    ),
                    flush=True,
                )
                return result

            source = args.source_file.read_text() * args.context_copies
            instruction = (
                "只调用 report_tag 工具，tag 使用本轮目标标签。"
                if args.tool_turn
                else "只回复目标标签，不解释。"
            )
            prompt = (
                "本轮目标标签是 ORNITH-4827。以下源码仅为参考。\n" + source + "\n" + instruction
            )
            (directory / "prompt.txt").write_text(prompt)
            messages = [{"role": "user", "content": prompt}]
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "report_tag",
                        "description": "记录目标标签",
                        "parameters": {
                            "type": "object",
                            "properties": {"tag": {"type": "string"}},
                            "required": ["tag"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]
            extra = {"tools": tools, "tool_choice": "none"} if args.tool_turn else None
            cold = run(
                "cold",
                messages,
                {"tools": tools, "tool_choice": "required"} if args.tool_turn else None,
            )
            if args.tool_turn:
                calls = cold["tool_calls"]
                if (
                    len(calls) != 1
                    or json.loads(calls[0]["function"]["arguments"]).get("tag") != "ORNITH-4827"
                ):
                    raise RuntimeError("初次工具调用标签无效。")
                prefix = messages + [
                    {"role": "assistant", "content": cold["text"] or None, "tool_calls": calls},
                    {
                        "role": "tool",
                        "tool_call_id": calls[0]["id"],
                        "content": "目标标签已记录：ORNITH-4827",
                    },
                ]
            else:
                prefix = messages + [{"role": "assistant", "content": cold["text"]}]
            continued = prefix + [{"role": "user", "content": "再次只回复本轮目标标签。"}]
            warm = run("append", continued, extra)
            branch = prefix + [
                {"role": "user", "content": "目标标签中的四位数字是什么？只输出数字。"}
            ]
            branched = run("branch", branch, extra)
            mutated_prompt = prompt.replace("ORNITH-4827", "ORNITH-9136", 1)
            mutated = run(
                "mutated_prefix",
                [
                    {
                        "role": "user",
                        "content": mutated_prompt.replace(
                            "只调用 report_tag 工具，tag 使用本轮目标标签。",
                            "只回复目标标签，不解释。",
                        ),
                    }
                ],
                extra,
            )
            unrelated = run(
                "unrelated", [{"role": "user", "content": "9加8等于几？只输出阿拉伯数字。"}]
            )
            # 独立冷算对照通过另一次 cache=0 的 backend 进程提供。
            expected = ([] if args.tool_turn else [(cold, "ORNITH-4827")]) + [
                (warm, "ORNITH-4827"),
                (branched, "4827"),
                (mutated, "ORNITH-9136"),
                (unrelated, "17"),
            ]
            for record, answer in expected:
                if record["text"].strip().strip('`"\n ') != answer:
                    raise RuntimeError(f"{record['label']} 输出不符合目标标签：{record['text']!r}")
            if (
                unrelated["timings"].get("cache_n", 0) != 0
                or mutated["timings"].get("cache_n", 0) > 512
            ):
                raise RuntimeError("无关提示或前部改写错误地复用了历史状态。")
            if (
                args.expect_cache
                and warm["timings"].get("cache_n", 0) < 0.8 * cold["usage"]["prompt_tokens"]
            ):
                raise RuntimeError("跨 sleep 的追加请求未命中至少 80% 的历史输入。")
            if (args.cache == "0" or args.cache_mib == 1) and warm["timings"].get(
                "cache_n", 0
            ) != 0:
                raise RuntimeError("禁用/超限缓存请求发生了意外命中。")
        finally:
            subprocess.run(
                helper + ["shutdown", *common, "--pid", str(pid), "--pgid", str(pid)],
                env=env,
                stdout=log,
                stderr=log,
                check=True,
            )
    print(f"report_dir={directory}", flush=True)


if __name__ == "__main__":
    main()
