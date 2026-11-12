"""以标准库旁路记录请求阶段与流式性能，不保存模型输入输出正文。"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
WRITE_LOCK = threading.Lock()
MAX_EVENT_BYTES = 65536
TIMING_KEYS = {
    "prompt_n",
    "prompt_ms",
    "prompt_per_token_ms",
    "prompt_per_second",
    "predicted_n",
    "predicted_ms",
    "predicted_per_token_ms",
    "predicted_per_second",
    "cache_n",
    "draft_n",
    "draft_n_accepted",
}


def enabled() -> bool:
    """读取默认开启的性能记录开关。"""
    return os.getenv("MODEL_TELEMETRY_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def telemetry_directory() -> Path:
    """返回当前启动批次的性能目录，单独启动服务时使用 standalone 目录。"""
    cache = Path(
        os.getenv(
            "RUNTIME_CACHE_DIR", str(Path(os.getenv("STORAGE_DIR", "storage")) / ".cache/runtime")
        )
    )
    return Path(os.getenv("MODEL_TELEMETRY_DIR", str(cache / "telemetry/standalone")))


def append_record(path: Path, record: dict[str, Any]) -> bool:
    """追加单条 JSONL；磁盘失败只告警，不影响推理和租约清理。"""
    try:
        with WRITE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return True
    except (OSError, ValueError):
        LOGGER.warning("性能记录写入失败；请检查性能目录和磁盘空间。")
        return False


def numeric_fields(value: Any, keys: set[str]) -> dict[str, int | float]:
    """仅保留允许的有限数值字段，避免上游扩展字段泄露正文。"""
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, number in value.items():
        if key not in keys or type(number) not in (int, float):
            continue
        try:
            if math.isfinite(number) and number >= 0:
                result[key] = number
        except OverflowError:
            continue
    return result


class Trace:
    """保存单次调用的时间戳及允许的数值统计，内存占用有界。"""

    def __init__(self, service: str, request_id: str | None = None) -> None:
        self.active = enabled()
        self.service = service
        self.request_id = (
            request_id
            if request_id and re.fullmatch(r"[a-f0-9]{32}", request_id)
            else uuid.uuid4().hex
        )
        self.trace_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.started_at = datetime.now(UTC).isoformat()
        self.stages_ms: dict[str, float] = {}
        self.first_content_ms: float | None = None
        self.first_tool_ms: float | None = None
        self.first_reasoning_ms: float | None = None
        self.last_content_ms: float | None = None
        self.usage: dict[str, int | float] = {}
        self.timings: dict[str, int | float] = {}
        self.finish_reasons: set[str] = set()
        self.buffer = b""
        self.dropping = False
        self.parse_incomplete = False
        self.done_seen = False
        self.status: int | None = None
        self.metadata: dict[str, Any] = {}

    def elapsed_ms(self) -> float:
        """使用单调时钟计算本次请求耗时。"""
        return round((time.monotonic() - self.started) * 1000, 3)

    def mark(self, stage: str) -> None:
        """标记由服务代码定义的请求阶段。"""
        self.stages_ms[stage] = self.elapsed_ms()

    def event(self, payload: Any) -> None:
        """提取完成用量、原生 timing 和首个正文/工具/推理事件。"""
        if not isinstance(payload, dict):
            return
        self.usage.update(
            numeric_fields(
                payload.get("usage"), {"prompt_tokens", "completion_tokens", "total_tokens"}
            )
        )
        self.timings.update(numeric_fields(payload.get("timings"), TIMING_KEYS))
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            reason = choice.get("finish_reason")
            if reason in ("stop", "length", "tool_calls", "content_filter", "function_call"):
                self.finish_reasons.add(reason)
            delta = choice.get("delta", choice.get("message", {}))
            if not isinstance(delta, dict):
                continue
            now = self.elapsed_ms()
            if isinstance(delta.get("content"), str) and delta["content"]:
                if self.first_content_ms is None:
                    self.first_content_ms = now
                self.last_content_ms = now
            if delta.get("tool_calls") and self.first_tool_ms is None:
                self.first_tool_ms = now
            if (
                delta.get("reasoning_content") or delta.get("reasoning")
            ) and self.first_reasoning_ms is None:
                self.first_reasoning_ms = now

    def feed(self, chunk: bytes) -> None:
        """跨传输块解析 SSE，超大事件丢弃统计但保持原始响应透传。"""
        # 限制每次切片大小，避免单个巨大 HTTP chunk 造成第二份整块缓冲。
        for offset in range(0, len(chunk), 8192):
            self.buffer = (self.buffer + chunk[offset : offset + 8192]).replace(b"\r\n", b"\n")
            while b"\n\n" in self.buffer:
                event, self.buffer = self.buffer.split(b"\n\n", 1)
                if len(event) > MAX_EVENT_BYTES:
                    self.parse_incomplete = True
                    continue
                if self.dropping:
                    self.dropping = False
                    continue
                data = b"\n".join(
                    line[5:].lstrip() for line in event.split(b"\n") if line.startswith(b"data:")
                )
                if data == b"[DONE]":
                    self.done_seen = True
                    continue
                if not data:
                    continue
                try:
                    self.event(json.loads(data))
                except (ValueError, UnicodeError, RecursionError):
                    self.parse_incomplete = True
            if len(self.buffer) > MAX_EVENT_BYTES:
                self.buffer = self.buffer[-2:]
                self.dropping = True
                self.parse_incomplete = True

    def begin(self) -> None:
        """先写请求起点，使异常退出或收尾未完成的调用也能被发现。"""
        if self.active:
            append_record(
                telemetry_directory() / f"starts-{self.service}-{os.getpid()}.jsonl",
                {
                    "trace_id": self.trace_id,
                    "request_id": self.request_id,
                    "service": self.service,
                    "pid": os.getpid(),
                    "started_at": self.started_at,
                },
            )

    def finish(self, outcome: str = "complete") -> None:
        """将最终记录写入本进程专属文件；不使用客户端内容构造路径。"""
        if not self.active:
            return
        append_record(
            telemetry_directory() / f"requests-{self.service}-{os.getpid()}.jsonl",
            {
                "schema_version": 1,
                "service": self.service,
                "pid": os.getpid(),
                "trace_id": self.trace_id,
                "request_id": self.request_id,
                "started_at": self.started_at,
                "ended_at": datetime.now(UTC).isoformat(),
                "duration_ms": self.elapsed_ms(),
                "status": self.status,
                "outcome": outcome,
                "stages_ms": self.stages_ms,
                "first_content_ms": self.first_content_ms,
                "last_content_ms": self.last_content_ms,
                "first_tool_ms": self.first_tool_ms,
                "first_reasoning_ms": self.first_reasoning_ms,
                "usage": self.usage,
                "timings": self.timings,
                "finish_reasons": sorted(self.finish_reasons),
                "done_seen": self.done_seen,
                "parse_incomplete": self.parse_incomplete,
                "metadata": self.metadata,
            },
        )


class TelemetryMiddleware:
    """纯 ASGI 观测器，不读取请求体、不改变响应字节或 GPU 生命周期。"""

    def __init__(self, app: Any, *, service: str, paths: set[str]) -> None:
        self.app = app
        self.service = service
        self.paths = paths

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("path") not in self.paths or not enabled():
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        request_id = headers.get(b"x-model-request-id", b"").decode("ascii", errors="ignore")
        trace = Trace(self.service, request_id)
        scope["model_telemetry"] = trace
        trace.metadata["measurement_position"] = self.service
        stream = False
        json_response = False
        body = bytearray()
        outcome = "complete"

        async def observed_send(message: dict) -> None:
            nonlocal stream, json_response
            if message["type"] == "http.response.start":
                trace.status = message["status"]
                response_headers = list(message.get("headers", []))
                content_type = dict(response_headers).get(b"content-type", b"")
                stream = b"text/event-stream" in content_type
                json_response = b"application/json" in content_type
                trace.metadata["response_mode"] = "sse" if stream else "nonstream"
                response_headers.append((b"x-model-request-id", trace.request_id.encode()))
                message = {**message, "headers": response_headers}
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                if stream:
                    trace.feed(chunk)
                elif json_response:
                    if len(body) + len(chunk) <= MAX_EVENT_BYTES:
                        body.extend(chunk)
                    else:
                        trace.parse_incomplete = True
                    if not message.get("more_body") and not trace.parse_incomplete:
                        try:
                            trace.event(json.loads(body))
                        except (ValueError, UnicodeError, RecursionError):
                            trace.parse_incomplete = True
                if not message.get("more_body"):
                    trace.mark("response_body_end")
            await send(message)

        await asyncio.to_thread(trace.begin)
        try:
            await self.app(scope, receive, observed_send)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except BaseException:
            outcome = "error"
            raise
        finally:
            if stream and not trace.done_seen:
                trace.parse_incomplete = True
                if outcome == "complete":
                    outcome = "incomplete_stream"
            await asyncio.to_thread(trace.finish, outcome)
