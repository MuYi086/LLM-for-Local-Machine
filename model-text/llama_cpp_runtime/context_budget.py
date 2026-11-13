"""在既有 GPU 租约内核验摘要的实际模板 token 与输出空间。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

SAFETY_TOKENS = 256


@dataclass(frozen=True)
class ContextBudget:
    """只保存 token 数量，不保存模板正文。"""

    context_tokens: int
    input_tokens: int
    output_tokens: int
    safety_tokens: int = SAFETY_TOKENS


class ContextBudgetExceeded(ValueError):
    """实际输入挤占了本次请求所需的输出空间。"""

    def __init__(self, budget: ContextBudget) -> None:
        self.budget = budget
        super().__init__(
            f"This model's maximum context length is {budget.context_tokens} tokens. "
            f"Your messages contain {budget.input_tokens} tokens and request "
            f"{budget.output_tokens} output tokens plus {budget.safety_tokens} safety tokens. "
            "Reduce the messages before retrying; the requested summary output will not be silently shortened."
        )


class ContextBudgetUnavailable(RuntimeError):
    """无法取得可信的后端预算，拒绝不受保护的摘要生成。"""


async def check_context_budget(client: httpx.AsyncClient, payload: dict[str, Any]) -> ContextBudget:
    """使用与聊天相同的后端模板和 tokenizer 计数，调用方必须已取得 GPU 锁。

    Args:
        client: 只连接 loopback 后端的现有客户端。
        payload: 已剥离本地扩展字段的原始聊天参数。

    Returns:
        实际输入、请求输出与上下文容量，供无正文遥测使用。

    Raises:
        ValueError: 未提供有限正整数输出预算。
        ContextBudgetExceeded: 无法同时容纳实际输入与所需输出。
        ContextBudgetUnavailable: 后端模板、分词或容量不可核验。
    """
    outputs = [payload[key] for key in ("max_tokens", "max_completion_tokens") if key in payload]
    if not outputs or any(type(value) is not int or value <= 0 for value in outputs):
        raise ValueError("摘要预算保护要求提供正整数 max_tokens 或 max_completion_tokens。")
    try:
        props = await client.get("/props", timeout=5)
        props.raise_for_status()
        context = props.json()["default_generation_settings"]["n_ctx"]
        if type(context) is not int or context <= 0:
            raise ValueError("后端未报告实际上下文容量。")
        template = await client.post("/apply-template", json=payload)
        template.raise_for_status()
        prompt = template.json()["prompt"]
        if not isinstance(prompt, str):
            raise ValueError("摘要预算保护只接受可完整分词的文本模板。")
        tokenized = await client.post(
            "/tokenize",
            json={
                "content": prompt,
                "add_special": True,
                "parse_special": True,
            },
        )
        tokenized.raise_for_status()
        tokens = tokenized.json()["tokens"]
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ValueError("后端未返回完整 token ID 数组。")
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise ContextBudgetUnavailable("无法核验实际摘要 token 预算，未提交生成请求。") from exc
    budget = ContextBudget(context, len(tokens), max(outputs))
    if budget.input_tokens + budget.output_tokens + budget.safety_tokens > budget.context_tokens:
        raise ContextBudgetExceeded(budget)
    return budget
