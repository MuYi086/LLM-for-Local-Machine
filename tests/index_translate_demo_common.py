"""调用 Index-Translate 家族官方推理脚本并保留原始文本输出。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SOURCE_TEXT = REPOSITORY_DIR / "tests/testIndexTranslate/subtitles.md"
UPSTREAM_REPOSITORY = Path(
    os.getenv("INDEX_TRANSLATE_SOURCE_DIR", str(REPOSITORY_DIR.parent / "Index-Translate"))
)
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000/v1"


def run_text_model_demo(
    *,
    demo_name: str,
    env_prefix: str,
    script_relative_path: str,
    model_name: str,
    output_name: str,
    demo_dir: Path,
    model_arguments: list[str] | None = None,
    stdin_input: bool = False,
    default_target_syllables: int | None = None,
) -> None:
    """通过家族官方 CLI/API 执行文本模型，并原样保存其标准输出。"""
    parser = argparse.ArgumentParser(description=f"运行 {demo_name} 官方文本翻译 Demo")
    parser.add_argument("--input", type=Path, default=SOURCE_TEXT, help="UTF-8 源文本文件")
    parser.add_argument(
        "--output",
        type=Path,
        default=demo_dir / "dist" / output_name,
        help="保存官方 API 原始文本输出的路径",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=UPSTREAM_REPOSITORY,
        help="官方 Index-Translate 源码目录",
    )
    parser.add_argument(
        "--python",
        default=os.getenv("INDEX_TRANSLATE_PYTHON", sys.executable),
        help="安装了官方推理依赖的 Python 解释器",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv(
            f"{env_prefix}_API_BASE_URL",
            os.getenv("OPENAI_BASE_URL", DEFAULT_API_BASE_URL),
        ),
        help="该模型 OpenAI 兼容服务的 /v1 地址",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv(f"{env_prefix}_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")),
        help="该模型 API 密钥；本地服务通常使用 EMPTY",
    )
    parser.add_argument("--model", default=model_name, help="官方模型 ID")
    if default_target_syllables is not None:
        parser.add_argument(
            "--syllables",
            type=int,
            default=default_target_syllables,
            help="Homura 的目标音节数",
        )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("INDEX_TRANSLATE_DEMO_TIMEOUT", "1800")),
        help="单次官方 API 请求超时秒数",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"找不到源文本：{args.input}")
    if args.timeout <= 0:
        raise SystemExit("--timeout 必须大于零。")
    if default_target_syllables is not None and args.syllables <= 0:
        raise SystemExit("--syllables 必须大于零。")

    script_path = args.repo_dir / script_relative_path
    if not script_path.is_file():
        raise SystemExit(
            f"找不到官方推理脚本：{script_path}；请通过 --repo-dir 或 "
            "INDEX_TRANSLATE_SOURCE_DIR 指向 bilibili/Index-Translate 源码目录。"
        )

    source_text = args.input.read_text(encoding="utf-8")
    command = [
        args.python,
        str(script_path),
        "-" if stdin_input else source_text,
        *([] if model_arguments is None else model_arguments),
    ]
    if default_target_syllables is not None:
        command.extend(["--syllables", str(args.syllables)])
    command.extend(
        [
            "--model",
            args.model,
        ]
    )
    environment = os.environ.copy()
    environment["OPENAI_BASE_URL"] = args.api_base_url
    environment["OPENAI_API_KEY"] = args.api_key
    environment["INDEX_MODEL"] = args.model
    try:
        result = subprocess.run(
            command,
            input=source_text.encode("utf-8") if stdin_input else None,
            capture_output=True,
            check=False,
            env=environment,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"{demo_name} 官方 API 请求超时（>{args.timeout:.0f}s）。") from exc
    except OSError as exc:
        raise SystemExit(f"无法启动官方推理脚本：{exc}") from exc

    if result.returncode != 0:
        detail = (
            result.stderr.decode("utf-8", errors="replace").strip()
            or result.stdout.decode("utf-8", errors="replace").strip()
            or "官方推理脚本执行失败。"
        )
        raise SystemExit(f"{demo_name} 请求失败（退出码 {result.returncode}）：{detail}")
    if not result.stdout.strip():
        raise SystemExit(f"{demo_name} 官方 API 返回了空文本。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_name(f".{args.output.name}.tmp")
    try:
        temporary_path.write_bytes(result.stdout)
        temporary_path.replace(args.output)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"模型：{demo_name} ({args.model})")
    print(f"源文本：{args.input}")
    print(f"模型 API：{args.api_base_url}")
    print(f"原始输出：{args.output}")
    print(f"文件大小：{args.output.stat().st_size} bytes")
