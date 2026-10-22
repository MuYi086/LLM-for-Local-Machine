"""LocalAI 外部 gRPC worker bridge。"""

from .protocol import BridgeProtocolError, BridgeRequest, BridgeResponse

__all__ = ["BridgeProtocolError", "BridgeRequest", "BridgeResponse"]
