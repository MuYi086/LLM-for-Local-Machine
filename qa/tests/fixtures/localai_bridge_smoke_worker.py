"""只在无模型 RPC smoke 中调用的 echo worker。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """回显 input JSON，不导入模型依赖。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    Path(args.output_json).write_text(
        json.dumps({"echo": request["text"]}, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
