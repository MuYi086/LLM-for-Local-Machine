#!/usr/bin/env python3
"""本机 sd.cpp 运维入口；状态检查不执行二进制或加载权重。"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "inference_gateway/src"), str(ROOT / "llm_local_runtime/src")]

from inference_gateway.sdcpp import GenerationRequest, SdcppConfig, load_models  # noqa: E402
from inference_gateway.sdcpp.config import local_path  # noqa: E402
from inference_gateway.sdcpp.execution import generate  # noqa: E402
from inference_gateway.sdcpp.models import TASKS  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.getenv("SDCPP_MODEL_MANIFEST", str(ROOT / "sdcpp/manifests/models.json"))),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="只报告清单、版本和配置状态")
    validate = commands.add_parser("validate", help="校验清单、组件文件和二进制摘要，不加载模型")
    validate.add_argument("--model")
    run = commands.add_parser("generate", help="在全局 GPU 锁中执行一个已配置任务")
    run.add_argument("--model", required=True)
    run.add_argument("--task", choices=sorted(TASKS), required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--negative-prompt", default="")
    run.add_argument("--init-image")
    for name in ("width", "height", "steps", "frames", "fps"):
        run.add_argument("--" + name, type=int)
    run.add_argument("--cfg-scale", type=float)
    run.add_argument("--seed", type=int, default=-1)
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行本机运维命令，stdout 保持单个 JSON，运行日志写 stderr。"""
    arguments = _parser().parse_args(argv)
    try:
        models = load_models(arguments.manifest)
        if arguments.command == "status":
            versions = json.loads((ROOT / "sdcpp/manifests/versions.json").read_text())
            configuration_error = None
            try:
                config = SdcppConfig.from_env()
            except ValueError as exc:
                configuration_error = str(exc)
            result = {
                "provider": "stable-diffusion.cpp",
                "source_commit": versions["commit"],
                "model_validation": versions["model_validation"],
                "models": [{"id": model.id, "tasks": model.tasks} for model in models.values()],
                "configured": configuration_error is None,
                "configuration_error": configuration_error,
                "binary_exists": config.binary.is_file() if configuration_error is None else False,
                "binary_digest_checked": False,
                "http_service": False,
            }
        else:
            selected = getattr(arguments, "model", None)
            if selected is not None and selected not in models:
                raise ValueError(f"模型未配置：{selected}；请先选择模型并填写本地清单。")
            if not models:
                raise ValueError("模型清单为空；请先选择模型并填写本地清单。")
            config = SdcppConfig.from_env()
            if arguments.command == "validate":
                config.validate_binary()
                for model in [models[selected]] if selected else models.values():
                    model.validate_files()
                result = {
                    "validated_models": [selected] if selected else list(models),
                    "binary_digest_checked": True,
                    "model_inference_verified": False,
                }
            else:
                request = GenerationRequest(
                    task=arguments.task,
                    prompt=arguments.prompt,
                    negative_prompt=arguments.negative_prompt,
                    width=arguments.width,
                    height=arguments.height,
                    steps=arguments.steps,
                    cfg_scale=arguments.cfg_scale,
                    seed=arguments.seed,
                    frames=arguments.frames,
                    fps=arguments.fps,
                    init_image=local_path(arguments.init_image) if arguments.init_image else None,
                )
                with contextlib.redirect_stdout(sys.stderr):
                    generated = generate(config, models[selected], request)
                result = {
                    "request_id": generated.request_id,
                    "output": str(generated.output),
                    "metrics": generated.metrics,
                }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            json.dumps(
                {"error": "生成被中断；若存在 quarantine，请运行恢复脚本。"}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
