#!/usr/bin/env python3
"""显式通过真实适配器与 backend 核对分词，并拒绝挤占摘要空间的请求。"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from benchmark_live import ROOT, find_backend_pid
from fastapi.testclient import TestClient


def main() -> None:
    """复用已运行的 loopback backend，所有唤醒和生成均由 API 的 GPU 租约保护。"""
    sys.path.insert(0, str(ROOT / "model-text"))
    from llama_cpp_runtime.app import create_app

    base = os.getenv("ORNITH_9B_LLAMA_BASE_URL", "http://127.0.0.1:8504")
    parsed = urlsplit(base)
    pid = find_backend_pid(parsed.port or 8504)
    directory = (
        ROOT
        / "storage/llm_tests/ornith_1_5_9b_gguf"
        / datetime.now(UTC).strftime("context-budget-%Y%m%dT%H%M%S%fZ")
    )
    directory.mkdir(parents=True, exist_ok=False)
    original = {
        key: os.environ.get(key) for key in ("MODEL_TELEMETRY_ENABLED", "MODEL_TELEMETRY_DIR")
    }
    os.environ.update(MODEL_TELEMETRY_ENABLED="true", MODEL_TELEMETRY_DIR=str(directory))
    try:
        app = create_app(
            model_alias="Ornith-1.5-9B-GGUF",
            backend_env_prefix="ORNITH_9B",
            backend_base_url=base,
            backend_host=parsed.hostname,
            backend_port=parsed.port or 8504,
            backend_pid=pid,
            backend_pgid=pid,
            backend_executable=str(Path(f"/proc/{pid}/exe").resolve()),
            lock_path=Path(
                os.getenv("GPU_LOCK_FILE", str(ROOT / "storage/.cache/runtime/gpu-runtime.lock"))
            ),
        )
        with TestClient(app) as client:
            small = client.post(
                "/v1/chat/completions",
                json={
                    "llm_context_guard": True,
                    "max_completion_tokens": 64,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": "只回复 OK。"}],
                },
            )
            if small.status_code != 200:
                raise RuntimeError(f"小请求失败：{small.status_code} {small.text[:300]}")
            oversized = client.post(
                "/v1/chat/completions",
                json={
                    "llm_context_guard": True,
                    "stream": True,
                    "max_completion_tokens": 4096,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": "x " * 97000}],
                },
            )
            if (
                oversized.status_code != 400
                or oversized.json().get("error", {}).get("code") != "context_length_exceeded"
            ):
                raise RuntimeError(f"大请求未正确拒绝：{oversized.status_code}")
        rows = [
            json.loads(line)
            for path in directory.glob("requests-*.jsonl")
            for line in path.read_text().splitlines()
        ]
        success = next(row for row in rows if row["status"] == 200)
        rejected = next(row for row in rows if row["status"] == 400)
        budget = success["metadata"]["context_budget"]
        if budget["input_tokens"] != small.json()["usage"]["prompt_tokens"]:
            raise RuntimeError("实际模板分词与生成返回的 prompt_tokens 不一致。")
        if "backend_send" in rejected["stages_ms"]:
            raise RuntimeError("超限请求仍然提交了生成。")
        report = {
            "small_input_count": budget["input_tokens"],
            "backend_prompt_tokens": small.json()["usage"]["prompt_tokens"],
            "rejected_budget": rejected["metadata"]["context_budget"],
            "rejected_error": oversized.json(),
            "report_directory": str(directory),
        }
        (directory / "verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        print(json.dumps(report, ensure_ascii=False))
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
