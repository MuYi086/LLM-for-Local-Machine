"""LLM Local Backend 内部推理 provider 与 LocalAI 接入。"""

from .config import GatewayConfig, GatewayConfigurationError, load_gateway_config
from .errors import (
    BackendQuarantinedError,
    LocalAIRequestError,
    LocalAIUnavailableError,
    UnsupportedCapabilityError,
)
from .localai_client import LocalAIClient

__all__ = [
    "BackendQuarantinedError",
    "GatewayConfig",
    "GatewayConfigurationError",
    "LocalAIClient",
    "LocalAIRequestError",
    "LocalAIUnavailableError",
    "UnsupportedCapabilityError",
    "load_gateway_config",
]
