"""验证 Dagu 任务对现有模型接口的调用与重复提交保护。"""

import base64
import hashlib
import importlib.util
import io
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
import zlib
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/dagu/job.py"
spec = importlib.util.spec_from_file_location("dagu_job", SCRIPT)
job_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job_module)
worker_spec = importlib.util.spec_from_file_location(
    "dagu_http_worker", SCRIPT.with_name("http_worker.py")
)
worker_module = importlib.util.module_from_spec(worker_spec)
worker_spec.loader.exec_module(worker_module)


def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


PNG = (
    b"\x89PNG\r\n\x1a\n"
    + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    + png_chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
    + png_chunk(b"IEND", b"")
)
audio_buffer = io.BytesIO()
with wave.open(audio_buffer, "wb") as audio:
    audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
    audio.writeframes(b"\x00\x00" * 240)
WAV = audio_buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    requests = []
    fail = False
    corrupt = False
    redirect = False
    delay = 0

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.requests.append((self.path, body))
        if self.delay:
            time.sleep(self.delay)
        if self.redirect:
            self.send_response(307)
            self.send_header("Location", "/unexpected")
            self.end_headers()
            return
        self.send_response(500 if self.fail else 200)
        self.end_headers()
        if self.fail:
            return
        if "MOSS" in self.path:
            self.wfile.write(
                json.dumps(
                    {
                        "data": [
                            {
                                "b64_json": base64.b64encode(
                                    PNG[:8] if self.corrupt else PNG
                                ).decode()
                            }
                        ]
                    }
                ).encode()
            )
        else:
            try:
                self.wfile.write(WAV[:12] if self.corrupt else WAV)
            except BrokenPipeError:
                pass

    def log_message(self, *_args):
        pass


class DaguJobTests(unittest.TestCase):
    def setUp(self):
        Handler.requests = []
        Handler.fail = False
        Handler.corrupt = False
        Handler.redirect = False
        Handler.delay = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "inputs").mkdir()
        self.env = os.environ | {
            "DAGU_JOB_ROOT": str(self.root),
            "DAGU_MOSS_BASE_URL": f"http://127.0.0.1:{self.server.server_port}",
            "DAGU_VOXCPM_BASE_URL": f"http://127.0.0.1:{self.server.server_port}",
        }

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def run_job(self, kind, job_id="test-1"):
        return subprocess.run(
            [sys.executable, str(SCRIPT), kind, job_id],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_moss_saves_png_and_prevents_duplicate_submission(self):
        (self.root / "inputs/test-1.json").write_text(json.dumps({"prompt": "hello"}))
        self.assertEqual(self.run_job("moss").returncode, 0)
        self.assertEqual((self.root / "jobs/test-1/result.png").read_bytes(), PNG)
        self.assertIn("MOSS", Handler.requests[0][0])
        self.assertEqual(json.loads(Handler.requests[0][1])["prompt"], "hello")
        self.assertNotEqual(self.run_job("moss").returncode, 0)
        self.assertEqual(len(Handler.requests), 1)

    def test_voxcpm_sends_form_and_saves_wav(self):
        (self.root / "inputs/test-1.json").write_text(json.dumps({"text": "hello"}))
        self.assertEqual(self.run_job("voxcpm").returncode, 0)
        self.assertEqual((self.root / "jobs/test-1/result.wav").read_bytes(), WAV)
        self.assertIn(b"hello", Handler.requests[0][1])
        self.assertIn("VoxCPM2", Handler.requests[0][0])

    def test_failed_request_cannot_be_repeated_automatically(self):
        (self.root / "inputs/test-1.json").write_text(json.dumps({"text": "hello"}))
        Handler.fail = True
        self.assertNotEqual(self.run_job("voxcpm").returncode, 0)
        self.assertNotEqual(self.run_job("voxcpm").returncode, 0)
        self.assertFalse((self.root / "jobs/test-1/result.wav").exists())
        self.assertEqual(len(Handler.requests), 1)

    def test_rejects_corrupt_media_without_completed_record(self):
        Handler.corrupt = True
        for kind, params in (("moss", {"prompt": "test"}), ("voxcpm", {"text": "test"})):
            with self.subTest(kind=kind):
                (self.root / f"inputs/{kind}.json").write_text(json.dumps(params))
                self.assertNotEqual(self.run_job(kind, kind).returncode, 0)
                self.assertFalse((self.root / f"jobs/{kind}/completed.json").exists())

    def test_invalid_params_do_not_reserve_id(self):
        for kind, params in (
            ("moss", {"prompt": "x" * 65537}),
            ("moss", {"prompt": "a\\0b".replace("\\0", "\0")}),
            ("voxcpm", {"text": "x" * 12001}),
            ("voxcpm", {"text": "hello", "instruction": "x" * 1001}),
        ):
            (self.root / "inputs/test-1.json").write_text(json.dumps(params))
            self.assertNotEqual(self.run_job(kind).returncode, 0)
            self.assertFalse((self.root / "jobs/test-1").exists())
        self.assertEqual(Handler.requests, [])

    def test_snapshot_and_hash_identify_actual_request(self):
        params = {"text": "hello"}
        source = self.root / "inputs/test-1.json"
        source.write_text(json.dumps(params))
        self.assertEqual(self.run_job("voxcpm").returncode, 0)
        source.write_text('{"text":"changed"}')
        job = self.root / "jobs/test-1"
        snapshot = (job / "request.json").read_bytes()
        self.assertEqual(json.loads(snapshot)["params"], params)
        submitted = json.loads((job / "submitted.json").read_text())
        self.assertEqual(submitted["request_sha256"], hashlib.sha256(snapshot).hexdigest())
        completed = json.loads((job / "completed.json").read_text())
        self.assertEqual(completed["output_sha256"], hashlib.sha256(WAV).hexdigest())

    def test_concurrent_duplicate_only_submits_once(self):
        (self.root / "inputs/test-1.json").write_text('{"text":"hello"}')
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: self.run_job("voxcpm"), range(2)))
        self.assertEqual(sorted(result.returncode for result in results), [0, 1])
        self.assertEqual(len(Handler.requests), 1)

    def test_png_missing_end_chunk_is_rejected(self):
        image = self.root / "truncated.png"
        image.write_bytes(PNG[:-12])
        with self.assertRaises(ValueError):
            job_module._validate_media(image, "moss")

    def test_completion_write_failure_keeps_output_unconfirmed(self):
        (self.root / "inputs/test-1.json").write_text('{"text":"hello"}')
        publish = job_module._publish

        def fail_completion(path, content):
            if path.name == "completed.json":
                raise OSError("disk fault")
            publish(path, content)

        with (
            patch.dict(os.environ, self.env),
            patch.object(job_module, "_publish", side_effect=fail_completion),
        ):
            with self.assertRaises(OSError):
                job_module.execute("voxcpm", "test-1", self.root)
        job = self.root / "jobs/test-1"
        self.assertTrue((job / "result.wav").exists())
        self.assertFalse((job / "completed.json").exists())
        self.assertEqual(json.loads((job / "failure.json").read_text())["stage"], "publishing")
        self.assertNotEqual(self.run_job("voxcpm").returncode, 0)
        self.assertEqual(len(Handler.requests), 1)

    def test_total_http_deadline_keeps_task_unknown(self):
        Handler.delay = 1
        (self.root / "inputs/test-1.json").write_text('{"text":"hello"}')
        with patch.dict(os.environ, self.env), patch.object(job_module, "REQUEST_TIMEOUT", 0.2):
            with self.assertRaises(subprocess.TimeoutExpired):
                job_module.execute("voxcpm", "test-1", self.root)
        failure = json.loads((self.root / "jobs/test-1/failure.json").read_text())
        self.assertEqual(failure["status"], "unknown")
        self.assertFalse((self.root / "jobs/test-1/completed.json").exists())

    def test_interruption_preserves_reservation_and_blocks_resubmit(self):
        Handler.delay = 0.5
        (self.root / "inputs/test-1.json").write_text('{"text":"hello"}')
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "voxcpm", "test-1"],
            env=self.env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 5
            while not Handler.requests and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(Handler.requests), 1)
        finally:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        self.assertTrue((self.root / "jobs/test-1/submitted.json").exists())
        self.assertFalse((self.root / "jobs/test-1/completed.json").exists())
        self.assertNotEqual(self.run_job("voxcpm").returncode, 0)
        self.assertEqual(len(Handler.requests), 1)

    def test_redirect_is_not_followed(self):
        Handler.redirect = True
        (self.root / "inputs/test-1.json").write_text('{"text":"hello"}')
        self.assertNotEqual(self.run_job("voxcpm").returncode, 0)
        self.assertEqual(len(Handler.requests), 1)

    def test_response_limit_rejects_oversize(self):
        with patch.object(worker_module, "LIMIT", 10):
            with self.assertRaises(ValueError):
                worker_module.receive(
                    self.env["DAGU_VOXCPM_BASE_URL"] + "/tts",
                    "text/plain",
                    b"hello",
                    self.root / "response",
                )

    def test_failed_atomic_link_never_publishes_partial_record(self):
        destination = self.root / "record.json"
        with patch.object(job_module.os, "link", side_effect=OSError("disk fault")):
            with self.assertRaises(OSError):
                job_module._publish(destination, b'{"complete":true}')
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".partial-*")), [])

    def test_record_failure_prevents_network_submission(self):
        (self.root / "inputs/test-1.json").write_text('{"text":"hello"}')
        with (
            patch.dict(os.environ, self.env),
            patch.object(job_module.os, "fsync", side_effect=OSError("disk fault")),
        ):
            with self.assertRaises(OSError):
                job_module.execute("voxcpm", "test-1", self.root)
        self.assertEqual(Handler.requests, [])

    def test_smoke_does_not_inherit_production_paths(self):
        smoke_spec = importlib.util.spec_from_file_location(
            "dagu_smoke", Path(__file__).with_name("dagu_smoke.py")
        )
        smoke = importlib.util.module_from_spec(smoke_spec)
        smoke_spec.loader.exec_module(smoke)
        with patch.dict(
            os.environ,
            {
                "DAGU_DATA_DIR": "/production",
                "DAGU_QUEUE_DIR": "/production/queue",
                "DAGU_AUTH_MODE": "none",
                "DAGU_BIN": "/custom/dagu",
            },
        ):
            env = smoke.smoke_environment(self.root, self.server.server_port)
        self.assertNotIn("DAGU_DATA_DIR", env)
        self.assertNotIn("DAGU_QUEUE_DIR", env)
        self.assertNotIn("DAGU_AUTH_MODE", env)
        self.assertEqual(env["DAGU_BIN"], "/custom/dagu")
        self.assertEqual(env["DAGU_HOME"], str(self.root / "server"))

    def test_rejects_path_traversal_before_request(self):
        self.assertNotEqual(self.run_job("moss", "../escape").returncode, 0)
        self.assertEqual(Handler.requests, [])


if __name__ == "__main__":
    unittest.main()
