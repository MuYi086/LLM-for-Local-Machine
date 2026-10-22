"""只在无模型 RPC smoke 中调用的 echo worker。"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


def main() -> int:
    """回显 input JSON，不导入模型依赖。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-file")
    args = parser.parse_args()
    request = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    if args.output_file:
        # 生成有效的静音 WAV，覆盖超过 Go gRPC/base64 及旧 Python artifact 上限的传输。
        remaining = int(request["size_bytes"]) // 6 * 6
        with wave.open(args.output_file, "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(3)
            output.setframerate(48_000)
            block = bytes(6 * 1024 * 1024)
            while remaining:
                chunk = block[: min(len(block), remaining)]
                output.writeframesraw(chunk)
                remaining -= len(chunk)
    else:
        Path(args.output_json).write_text(
            json.dumps({"echo": request["text"]}, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
