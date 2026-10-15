"""读取 GGUF 元数据中的特殊 token，不加载模型或读取张量。"""

from __future__ import annotations

import os
import struct
from pathlib import Path


def load_stop_tokens(model_path: Path, token_ids: tuple[int, ...] = (248044, 248046)) -> list[str]:
    """将官方 NativeLong 停止 token ID 映射为 llama.cpp 的 stop 字符串。

    llama.cpp 聊天 API 不接受 vLLM 的 stop_token_ids。读取实际 GGUF 的词表，
    避免猜测不同 Tokenizer 的特殊 token 文本；缺失映射时明确失败。
    """
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with model_path.open("rb") as file:

        def number(format_string: str, size: int) -> int:
            data = file.read(size)
            if len(data) != size:
                raise ValueError("GGUF 元数据不完整。")
            return struct.unpack(format_string, data)[0]

        def u64() -> int:
            return number("<Q", 8)

        def u32() -> int:
            return number("<I", 4)

        def string() -> str:
            size = u64()
            if size > 16 * 1024 * 1024:
                raise ValueError("GGUF 字符串元数据过大。")
            data = file.read(size)
            if len(data) != size:
                raise ValueError("GGUF 字符串元数据不完整。")
            return data.decode("utf-8")

        def skip(kind: int) -> None:
            if kind in sizes:
                file.seek(sizes[kind], os.SEEK_CUR)
            elif kind == 8:
                size = u64()
                file.seek(size, os.SEEK_CUR)
            elif kind == 9:
                element_kind, count = u32(), u64()
                if element_kind in sizes:
                    file.seek(sizes[element_kind] * count, os.SEEK_CUR)
                else:
                    for _ in range(count):
                        skip(element_kind)
            else:
                raise ValueError(f"不支持的 GGUF 元数据类型：{kind}")

        if file.read(4) != b"GGUF" or u32() not in (2, 3):
            raise ValueError("NativeLong 权重不是受支持的 GGUF。")
        u64()
        for _ in range(u64()):
            key, kind = string(), u32()
            if key == "tokenizer.ggml.tokens":
                if kind != 9 or u32() != 8:
                    raise ValueError("GGUF 词表类型错误。")
                tokens = {}
                for index in range(u64()):
                    token = string()
                    if index in token_ids:
                        tokens[index] = token
                if any(index not in tokens for index in token_ids):
                    raise ValueError("GGUF 缺少官方 NativeLong 停止 token。")
                return [tokens[index] for index in token_ids]
            skip(kind)
    raise ValueError("GGUF 缺少词表元数据。")
