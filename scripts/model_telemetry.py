"""提供无需额外安装依赖的性能采样与汇总入口。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_local_runtime/src"))

from llm_local_runtime.telemetry_cli import main  # noqa: E402

if __name__ == "__main__":
    main()
