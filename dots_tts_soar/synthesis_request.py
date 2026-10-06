"""独立 dots.tts-soar 服务共用的请求契约。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class CloneSynthesisRequest(BaseModel):
    """``/v1/dotsTTS/clone`` 参考音频克隆请求的基础模型。

    WebUI 按模型分别构造字段，因此未知字段必须拒绝，避免错误被静默忽略。
    此接口只克隆上传的声音，不进行音色设计，所以会拒绝风格提示。
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def reject_style_prompt(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "style_prompt" in value:
            raise ValueError("style_prompt 不适用于 /v1/dotsTTS/clone；该接口仅用于参考音频克隆。")
        return value
