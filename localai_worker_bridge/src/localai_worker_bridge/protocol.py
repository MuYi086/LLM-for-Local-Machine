"""worker bridge 与内部 LocalAI client 共用的协议函数。"""

from inference_gateway.worker_protocol import (
    MAX_ENVELOPE_BYTES,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    encode_worker_response,
)
from inference_gateway.worker_protocol import (
    WorkerEnvelope as BridgeRequest,
)
from inference_gateway.worker_protocol import (
    WorkerProtocolError as BridgeProtocolError,
)
from inference_gateway.worker_protocol import (
    decode_worker_request as decode_request,
)


class BridgeResponse:
    """worker response envelope，保持 adapter 内部简洁的构造形式。"""

    def __init__(self, model, operation, request_id, result):
        self.model = model
        self.operation = operation
        self.request_id = request_id
        self.result = result


def encode_response(response):
    """序列化 adapter 收到的 worker result。"""
    return encode_worker_response(
        model=response.model,
        operation=response.operation,
        request_id=response.request_id,
        result=response.result,
    )


__all__ = [
    "MAX_ENVELOPE_BYTES",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "BridgeProtocolError",
    "BridgeRequest",
    "BridgeResponse",
    "decode_request",
    "encode_response",
]
