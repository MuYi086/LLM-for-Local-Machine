"""Unitale 内部推理 provider 与 LocalAI 接入。"""

from .config import GatewayConfig, GatewayConfigurationError, load_gateway_config
from .errors import (
    BackendQuarantinedError,
    LocalAIRequestError,
    LocalAIUnavailableError,
    UnsupportedCapabilityError,
)
from .execution import (
    execute_localai_worker,
    execute_localai_worker_file,
    execute_localai_worker_file_reference,
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
    "execute_localai_worker",
    "execute_localai_worker_file",
    "execute_localai_worker_file_reference",
]
