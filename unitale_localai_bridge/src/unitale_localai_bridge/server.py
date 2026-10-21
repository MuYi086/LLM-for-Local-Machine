"""LocalAI v4.11.0 外部 gRPC backend 服务。"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import signal
from concurrent import futures
from pathlib import Path
from typing import Any

import grpc

from . import localai_backend_pb2 as pb2
from . import localai_backend_pb2_grpc as pb2_grpc
from .protocol import BridgeProtocolError, BridgeResponse, decode_request, encode_response
from .runner import WorkerExecutionError, WorkerRunner, WorkerSpec

ROOT = Path(__file__).resolve().parents[3]
_LOGGER = logging.getLogger("unitale_localai_bridge")


class LocalAIWorkerBackend(pb2_grpc.BackendServicer):
    """将固定 Predict envelope 路由到注册的一次性 worker。"""

    def __init__(self, runner: WorkerRunner, allowed_models: set[str]) -> None:
        self.runner = runner
        self.allowed_models = allowed_models

    def Health(self, request: Any, context: grpc.ServicerContext) -> Any:
        del request, context
        return pb2.Reply(message=b"OK")

    def Status(self, request: Any, context: grpc.ServicerContext) -> Any:
        del request, context
        return pb2.StatusResponse(state=pb2.StatusResponse.READY)

    def LoadModel(self, request: Any, context: grpc.ServicerContext) -> Any:
        del context
        # Worker 在每个 Predict 请求中启动；LoadModel 只确认 adapter 已连接。
        _LOGGER.info("LocalAI 请求装载惰性 worker 模型：%s", request.Model)
        return pb2.Result(message="one-shot worker adapter 已就绪", success=True)

    def Free(self, request: Any, context: grpc.ServicerContext) -> Any:
        del request, context
        # 每个 RPC 的 worker 都由该次 Predict 确认退出，无需保留模型上下文。
        return pb2.Result(message="one-shot worker 已退出", success=True)

    def Predict(self, request: Any, context: grpc.ServicerContext) -> Any:
        prompt = str(request.Prompt)
        identity = str(request.ModelIdentity)
        try:
            decoded = decode_request(prompt, model_identity=identity)
            if decoded.model not in self.allowed_models:
                raise BridgeProtocolError("该 worker model 未在本机 manifest 登记。")
            result = self.runner.run(
                decoded.model,
                decoded.operation,
                decoded.payload,
                request_id=decoded.request_id,
                cancelled=context.is_active,
            )
            response = BridgeResponse(
                model=decoded.model,
                operation=decoded.operation,
                request_id=decoded.request_id,
                result=result,
            )
            return pb2.Reply(message=encode_response(response))
        except BridgeProtocolError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except WorkerExecutionError as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    def PredictStream(self, request: Any, context: grpc.ServicerContext) -> Any:
        del request
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "worker bridge 不支持流式 RPC。")


def load_worker_specs(path: Path, repository_root: Path = ROOT) -> dict[str, WorkerSpec]:
    """读取 JSON manifest，并将所有路径限制在仓库内部。"""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"worker manifest 不可读或 JSON 无效：{path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("worker manifest schema_version 必须为 1。")
    entries = manifest.get("workers")
    if not isinstance(entries, dict):
        raise ValueError("worker manifest workers 必须是对象。")

    root = repository_root.resolve()
    specs: dict[str, WorkerSpec] = {}
    for model, raw in entries.items():
        if not isinstance(model, str) or not model or not isinstance(raw, dict):
            raise ValueError("worker manifest entry 格式无效。")
        project_value = raw.get("project")
        script_value = raw.get("script")
        output_kind = raw.get("output_kind")
        timeout_seconds = raw.get("timeout_seconds")
        output_argument = raw.get("output_argument")
        output_suffix = raw.get("output_suffix")
        output_suffix_field = raw.get("output_suffix_field")
        output_suffixes = raw.get("output_suffixes", {})
        raw_environment = raw.get("environment", {})
        if (
            not isinstance(project_value, str)
            or Path(project_value).is_absolute()
            or ".." in Path(project_value).parts
            or not isinstance(script_value, str)
            or Path(script_value).is_absolute()
            or ".." in Path(script_value).parts
            or not isinstance(output_kind, str)
            or output_kind not in {"json", "file", "file_ref"}
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 3600
            or (
                output_argument is not None
                and (
                    not isinstance(output_argument, str)
                    or output_argument
                    not in {
                        "--output-json",
                        "--output-file",
                        "--output-zip",
                        "--output-wav",
                        "--output-video",
                    }
                )
            )
            or (
                output_suffix is not None
                and (
                    not isinstance(output_suffix, str)
                    or output_suffix not in {".zip", ".json", ".npz", ".npy", ".wav", ".mp4"}
                )
            )
            or (
                output_suffix_field is not None
                and (
                    not isinstance(output_suffix_field, str)
                    or output_suffix_field not in {"operation", "output"}
                )
            )
            or not isinstance(output_suffixes, dict)
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, str)
                or value not in {".zip", ".json", ".npz", ".npy", ".wav", ".mp4"}
                for key, value in output_suffixes.items()
            )
            or (output_suffix is not None and (output_suffix_field is not None or output_suffixes))
            or (output_suffix_field is None and output_suffixes)
            or (output_suffix_field is not None and not output_suffixes)
            or (
                output_kind in {"file", "file_ref"}
                and (
                    output_argument is None
                    or (output_suffix is None and output_suffix_field is None)
                )
            )
            or not isinstance(raw_environment, dict)
            or any(
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name)
                or not isinstance(value, str)
                or "\0" in value
                for name, value in raw_environment.items()
            )
        ):
            raise ValueError(f"worker manifest entry 无效：{model}")
        project = (root / project_value).resolve()
        script = (project / script_value).resolve()
        if not project.is_relative_to(root) or not script.is_relative_to(project):
            raise ValueError(f"worker manifest 路径越界：{model}")
        if not project.is_dir() or not script.is_file():
            raise ValueError(f"worker manifest 指向的项目或脚本不存在：{model}")
        specs[model] = WorkerSpec(
            project=project,
            script=script,
            output_kind=output_kind,
            timeout_seconds=float(timeout_seconds),
            output_argument=output_argument,
            output_suffix=output_suffix,
            output_suffix_field=output_suffix_field,
            output_suffixes=dict(output_suffixes),
            environment=dict(raw_environment),
        )
    return specs


def create_server(address: str, manifest_path: Path) -> grpc.Server:
    """创建并配置 gRPC server，限定回环监听和单请求推理。"""
    host, separator, port_text = address.rpartition(":")
    if not separator or host not in {"127.0.0.1", "localhost", "[::1]"}:
        raise ValueError("UNITALE_LOCALAI_BRIDGE_ADDRESS 只允许本机回环地址。")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("LocalAI bridge gRPC 端口无效。") from exc
    if not 1 <= port <= 65535:
        raise ValueError("LocalAI bridge gRPC 端口超出有效范围。")

    specs = load_worker_specs(manifest_path)
    options = [
        ("grpc.max_receive_message_length", 4 * 1024 * 1024),
        ("grpc.max_send_message_length", 180 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=options)
    pb2_grpc.add_BackendServicer_to_server(
        LocalAIWorkerBackend(WorkerRunner(ROOT, specs), set(specs)), server
    )
    bound_port = server.add_insecure_port(address)
    if bound_port != port:
        raise OSError(f"无法绑定 LocalAI bridge 回环端口：{address}")
    return server


def main() -> int:
    """启动外部 backend 并在 TERM/INT 时有界退出。"""
    parser = argparse.ArgumentParser(description="Unitale LocalAI one-shot worker bridge")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            os.getenv(
                "UNITALE_LOCALAI_BRIDGE_WORKERS", ROOT / "localai/manifests/worker-bridge.json"
            )
        ),
    )
    parser.add_argument(
        "--address", default=os.getenv("UNITALE_LOCALAI_BRIDGE_ADDRESS", "127.0.0.1:50052")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = create_server(args.address, args.manifest)
    server.start()
    _LOGGER.info("Unitale LocalAI worker bridge listening on %s", args.address)

    def stop(signum: int, frame: Any) -> None:
        del frame
        _LOGGER.info("收到信号 %s，停止 bridge", signum)
        server.stop(grace=5).wait(timeout=7)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server.wait_for_termination()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
