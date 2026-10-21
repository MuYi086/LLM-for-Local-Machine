"""推理网关的稳定内部错误类型。"""


class GatewayError(RuntimeError):
    """推理网关错误基类。"""


class GatewayConfigurationError(GatewayError):
    """启动配置不合法或请求 provider 未启用。"""


class UnsupportedCapabilityError(GatewayError):
    """选定 provider 无法保留当前请求能力。"""


class LocalAIUnavailableError(GatewayError):
    """LocalAI 连接失败或未就绪。"""


class LocalAIRequestError(GatewayError):
    """LocalAI 返回错误或不符合约定的响应。"""


class BackendQuarantinedError(GatewayError):
    """后端停止无法确认，GPU 准入已隔离。"""
