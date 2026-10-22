"""LocalAI 外部 worker bridge 的无模型协议与子进程回归。"""

from __future__ import annotations

import json
import tempfile
import unittest
from concurrent import futures
from pathlib import Path
from unittest.mock import patch

import grpc
from localai_worker_bridge.protocol import decode_request, encode_response
from localai_worker_bridge.runner import WorkerExecutionError, WorkerRunner, WorkerSpec
from localai_worker_bridge.server import LocalAIWorkerBackend, load_worker_specs

from localai_worker_bridge import (
    BridgeProtocolError,
    BridgeResponse,
)
from localai_worker_bridge import localai_backend_pb2 as pb2
from localai_worker_bridge import localai_backend_pb2_grpc as pb2_grpc


def envelope(*, model: str = "llm-local.test.echo", request_id: str = "req-1") -> str:
    """序列化固定测试请求。"""
    return json.dumps(
        {
            "protocol": "llm-local.worker",
            "version": 1,
            "model": model,
            "operation": "run",
            "request_id": request_id,
            "payload": {"worker_input": {"text": "你好"}},
        },
        ensure_ascii=False,
    )


class _FakeRunner:
    def run(self, model, operation, payload, *, request_id, cancelled=None):
        if cancelled is not None and not cancelled():
            raise AssertionError("active test request was marked cancelled")
        return {
            "kind": "json",
            "value": payload["worker_input"],
            "elapsed_seconds": 0.01,
        }


class LocalAIWorkerBridgeTests(unittest.TestCase):
    def test_repository_manifest_fixes_worker_paths_and_output_extensions(self) -> None:
        manifest = Path(__file__).resolve().parents[1] / "localai/manifests/worker-bridge.json"
        specs = load_worker_specs(manifest)

        self.assertEqual(
            set(specs),
            {
                "llm-local.sheetsage2.worker.v1",
                "llm-local.mert-v2-fullsong.worker.v1",
                "llm-local.yue2-vae.worker.v1",
                "llm-local.yue2-3b.worker.v1",
                "llm-local.tiger-dnr.worker.v1",
            },
        )
        self.assertEqual(
            specs["llm-local.mert-v2-fullsong.worker.v1"].output_suffixes,
            {"embedding": ".json", "frames": ".npz"},
        )
        self.assertEqual(
            specs["llm-local.yue2-vae.worker.v1"].output_suffixes,
            {"decode": ".wav", "encode": ".npy"},
        )

    def test_decode_enforces_protocol_identity_and_shape(self) -> None:
        request = decode_request(envelope(), model_identity="llm-local.test.echo")
        self.assertEqual(request.model, "llm-local.test.echo")
        self.assertEqual(request.payload["worker_input"]["text"], "你好")

        with self.assertRaises(BridgeProtocolError):
            decode_request(envelope(), model_identity="llm-local.other")
        with self.assertRaises(BridgeProtocolError):
            decode_request("[]")

    def test_response_round_trips_as_versioned_json(self) -> None:
        encoded = encode_response(
            BridgeResponse(
                model="llm-local.test.echo",
                operation="run",
                request_id="req-1",
                result={"kind": "json", "value": {"text": "你好"}},
            )
        )
        response = json.loads(encoded)
        self.assertEqual(response["protocol"], "llm-local.worker")
        self.assertEqual(response["version"], 1)
        self.assertEqual(response["result"]["value"]["text"], "你好")

    def test_backend_serves_fixed_localai_rpc_paths(self) -> None:
        server = grpc.server(futures.ThreadPoolExecutor(1))
        service = LocalAIWorkerBackend(_FakeRunner(), {"llm-local.test.echo"})
        pb2_grpc.add_BackendServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        self.assertGreater(port, 0)
        server.start()
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        stub = pb2_grpc.BackendStub(channel)
        try:
            self.assertEqual(stub.Health(pb2.HealthMessage(), timeout=2).message, b"OK")
            self.assertEqual(
                stub.Status(pb2.HealthMessage(), timeout=2).state, pb2.StatusResponse.READY
            )
            self.assertTrue(stub.LoadModel(pb2.ModelOptions(Model="unused"), timeout=2).success)
            result = stub.Predict(
                pb2.PredictOptions(
                    Prompt=envelope(),
                    ModelIdentity="llm-local.test.echo",
                ),
                timeout=2,
            )
            self.assertEqual(json.loads(result.message)["result"]["value"]["text"], "你好")
            with self.assertRaises(grpc.RpcError) as raised:
                stub.Predict(
                    pb2.PredictOptions(
                        Prompt=envelope(model="llm-local.unknown"),
                        ModelIdentity="llm-local.unknown",
                    ),
                    timeout=2,
                )
            self.assertEqual(raised.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        finally:
            channel.close()
            server.stop(grace=0).wait(timeout=2)

    def test_runner_uses_fixed_argv_and_returns_worker_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            script = project / "worker.py"
            script.write_text(
                "import argparse, json\n"
                "from pathlib import Path\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--input-json', required=True)\n"
                "parser.add_argument('--output-json', required=True)\n"
                "args = parser.parse_args()\n"
                "payload = json.loads(Path(args.input_json).read_text())\n"
                "Path(args.output_json).write_text(json.dumps({'echo': payload['text']}))\n",
                encoding="utf-8",
            )
            uv = project / "uv"
            uv.write_text(
                "#!/bin/sh\n"
                '[ "$1" = run ] || exit 91\n'
                "shift\n"
                '[ "$1" = --project ] || exit 92\n'
                "shift 2\n"
                '[ "$1" = --no-sync ] || exit 93\n'
                "shift\n"
                'exec "$@"\n',
                encoding="utf-8",
            )
            uv.chmod(0o755)
            runner = WorkerRunner(
                project,
                {
                    "llm-local.test.echo": WorkerSpec(
                        project=project,
                        script=script,
                        output_kind="json",
                        output_argument="--output-json",
                        timeout_seconds=5,
                    )
                },
                uv_executable=str(uv),
            )

            result = runner.run(
                "llm-local.test.echo",
                "run",
                {"worker_input": {"text": "hello"}},
                request_id="req-2",
            )

        self.assertEqual(result["kind"], "json")
        self.assertEqual(result["value"]["echo"], "hello")

    def test_runner_selects_only_declared_file_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            script = project / "worker.py"
            script.write_text(
                "import argparse\n"
                "from pathlib import Path\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--input-json', required=True)\n"
                "parser.add_argument('--output-file', required=True)\n"
                "args = parser.parse_args()\n"
                "Path(args.output_file).write_text(Path(args.output_file).suffix)\n",
                encoding="utf-8",
            )
            uv = project / "uv"
            uv.write_text(
                '#!/bin/sh\nshift 4\nexec "$@"\n',
                encoding="utf-8",
            )
            uv.chmod(0o755)
            runner = WorkerRunner(
                project,
                {
                    "llm-local.test.frames": WorkerSpec(
                        project=project,
                        script=script,
                        output_kind="file",
                        timeout_seconds=5,
                        output_argument="--output-file",
                        output_suffix_field="output",
                        output_suffixes={"frames": ".npz"},
                    )
                },
                uv_executable=str(uv),
            )

            result = runner.run(
                "llm-local.test.frames",
                "run",
                {"worker_input": {"output": "frames"}},
                request_id="req-suffix",
            )

        self.assertEqual(result["kind"], "file")
        self.assertEqual(result["size_bytes"], len(b".npz"))
        with self.assertRaisesRegex(WorkerExecutionError, "扩展名"):
            runner.run(
                "llm-local.test.frames",
                "run",
                {"worker_input": {"output": "unexpected"}},
                request_id="req-invalid-suffix",
            )

    def test_runner_publishes_large_artifact_as_scoped_file_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            artifact_root = root / "artifacts"
            script = project / "worker.py"
            script.write_text(
                "import argparse\n"
                "from pathlib import Path\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('--input-json', required=True)\n"
                "parser.add_argument('--output-zip', required=True)\n"
                "args = parser.parse_args()\n"
                "Path(args.output_zip).write_bytes(b'zip artifact')\n",
                encoding="utf-8",
            )
            uv = project / "uv"
            uv.write_text('#!/bin/sh\nshift 4\nexec "$@"\n', encoding="utf-8")
            uv.chmod(0o755)
            runner = WorkerRunner(
                root,
                {
                    "llm-local.test.file-ref": WorkerSpec(
                        project=project,
                        script=script,
                        output_kind="file_ref",
                        output_argument="--output-zip",
                        output_suffix=".zip",
                        timeout_seconds=5,
                    )
                },
                uv_executable=str(uv),
            )
            request_id = "a" * 32
            with patch.dict("os.environ", {"LLM_LOCALAI_ARTIFACT_DIR": str(artifact_root)}):
                result = runner.run(
                    "llm-local.test.file-ref",
                    "run",
                    {"worker_input": {}},
                    request_id=request_id,
                )

            self.assertEqual(result["kind"], "file_ref")
            self.assertEqual(result["artifact_id"], request_id)
            self.assertNotIn("data_base64", result)
            self.assertEqual((artifact_root / f"{request_id}.zip").read_bytes(), b"zip artifact")


if __name__ == "__main__":
    unittest.main()
