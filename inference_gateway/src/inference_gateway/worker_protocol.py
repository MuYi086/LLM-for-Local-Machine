"""LocalAI completion transport 中的版本化一次性 worker envelope。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

PROTOCOL_NAME = "unitale.worker"
PROTOCOL_VERSION = 1
MAX_ENVELOPE_BYTES = 2 * 1024 * 1024
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class WorkerProtocolError(ValueError):
    """版本化 worker envelope 无效。"""


@dataclass(frozen=True)
class WorkerEnvelope:
    """发送给固定外部 backend 的 worker 请求。"""

    model: str
    operation: str
    request_id: str
    payload: dict[str, Any]


def encode_worker_request(
    *, model: str, operation: str, request_id: str, payload: dict[str, Any]
) -> str:
    """序列化并限制 completion transport 的 JSON envelope 大小。"""
    _validate_identifier(model, "model")
    _validate_identifier(operation, "operation")
    _validate_identifier(request_id, "request_id")
    if not isinstance(payload, dict):
        raise WorkerProtocolError("worker payload 必须是对象。")
    encoded = json.dumps(
        {
            "protocol": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "model": model,
            "operation": operation,
            "request_id": request_id,
            "payload": payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > MAX_ENVELOPE_BYTES:
        raise WorkerProtocolError("worker envelope 超过 2 MiB 上限。")
    return encoded


def decode_worker_request(prompt: str, *, model_identity: str = "") -> WorkerEnvelope:
    """解析并验证 gRPC Predict prompt 与 LocalAI ModelIdentity。"""
    if not isinstance(prompt, str) or not prompt:
        raise WorkerProtocolError("worker envelope 不能为空。")
    if len(prompt.encode("utf-8")) > MAX_ENVELOPE_BYTES:
        raise WorkerProtocolError("worker envelope 超过 2 MiB 上限。")
    try:
        value = json.loads(prompt)
    except json.JSONDecodeError as exc:
        raise WorkerProtocolError("worker envelope 必须是 UTF-8 JSON 对象。") from exc
    if not isinstance(value, dict):
        raise WorkerProtocolError("worker envelope 顶层必须是对象。")
    if value.get("protocol") != PROTOCOL_NAME or value.get("version") != PROTOCOL_VERSION:
        raise WorkerProtocolError("worker envelope protocol/version 不受支持。")
    model = _validate_identifier(value.get("model"), "model")
    if model_identity and model_identity != model:
        raise WorkerProtocolError("worker envelope model 与 LocalAI ModelIdentity 不一致。")
    operation = _validate_identifier(value.get("operation"), "operation")
    request_id = _validate_identifier(value.get("request_id"), "request_id")
    payload = value.get("payload")
    if not isinstance(payload, dict):
        raise WorkerProtocolError("worker envelope payload 必须是对象。")
    return WorkerEnvelope(model, operation, request_id, payload)


def encode_worker_response(
    *, model: str, operation: str, request_id: str, result: dict[str, Any]
) -> bytes:
    """编码可由 OpenAI-compatible completion text 承载的结构化结果。"""
    _validate_identifier(model, "model")
    _validate_identifier(operation, "operation")
    _validate_identifier(request_id, "request_id")
    return json.dumps(
        {
            "protocol": PROTOCOL_NAME,
            "version": PROTOCOL_VERSION,
            "model": model,
            "operation": operation,
            "request_id": request_id,
            "result": result,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_worker_response_text(
    payload: bytes, *, model: str, operation: str, request_id: str
) -> dict[str, Any]:
    """解析 bytes 编码的 gRPC 响应并复用同一身份校验。"""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WorkerProtocolError("gRPC worker response 不是 UTF-8。") from exc
    return decode_worker_response(text, model=model, operation=operation, request_id=request_id)


def decode_worker_response(
    value: object, *, model: str, operation: str, request_id: str
) -> dict[str, Any]:
    """验证 LocalAI completion 返回的 worker response envelope。"""
    if not isinstance(value, str) or len(value.encode("utf-8")) > 180 * 1024 * 1024:
        raise WorkerProtocolError("LocalAI worker response 文本无效或超过大小上限。")
    try:
        response = json.loads(value)
    except json.JSONDecodeError as exc:
        raise WorkerProtocolError("LocalAI worker response 不是有效 JSON。") from exc
    if (
        not isinstance(response, dict)
        or response.get("protocol") != PROTOCOL_NAME
        or response.get("version") != PROTOCOL_VERSION
        or response.get("model") != model
        or response.get("operation") != operation
        or response.get("request_id") != request_id
        or not isinstance(response.get("result"), dict)
    ):
        raise WorkerProtocolError("LocalAI worker response 与请求身份或协议版本不匹配。")
    return response["result"]


def _validate_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise WorkerProtocolError(f"worker {label} 无效。")
    return value
