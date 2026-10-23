"""语音克隆合成接口共用的请求契约。"""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class CloneSynthesisRequest(BaseModel):
    """``/v1/qwen/clone`` 语音克隆请求的基础模型。

    WebUI 按模型分别构造字段，因此未知字段必须拒绝，避免错误被静默忽略。
    ``style_prompt`` 是特例：此接口只克隆上传的参考音频，不进行音色设计；
    如果误将风格提示当成待朗读文本，会改变合成结果。参考音频转写属于
    具体模型能力，因此支持 ``prompt_text`` 的模型应在自己的请求模型中声明。
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def reject_style_prompt(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "style_prompt" in value:
            raise ValueError("style_prompt 不适用于 /v1/qwen/clone；该接口仅用于参考音频克隆。")
        return value
