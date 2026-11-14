"""执行一次独占登记的 Dagu 模型任务，不直接管理 GPU。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
import wave
import zlib
from pathlib import Path

ROUTES = {
    "moss": "/v1/openmoss/MOSS-Image-SFT-Q8_0",
    "voxcpm": "/v1/OpenBMB/VoxCPM2-GGUF",
}
LIMIT = 64 * 1024 * 1024
REQUEST_TIMEOUT = 7200


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(path: Path, content: bytes) -> None:
    """同步文件并以独占链接原子发布，随后同步目录项。"""
    descriptor, name = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _validate_media(path: Path, kind: str) -> None:
    if kind == "voxcpm":
        with wave.open(str(path), "rb") as audio:
            expected = audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
            if expected <= 0 or expected > LIMIT:
                raise ValueError("WAV 必须包含有效 PCM 采样。")
            actual = 0
            while chunk := audio.readframes(16384):
                actual += len(chunk)
            if actual != expected:
                raise ValueError("WAV 采样已截断。")
    else:
        content = path.read_bytes()
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("模型未返回 PNG。")
        offset = 8
        chunks = []
        while offset < len(content):
            length = int.from_bytes(content[offset : offset + 4], "big")
            end = offset + length + 12
            if end > len(content):
                raise ValueError("PNG 数据块已截断。")
            kind_bytes = content[offset + 4 : offset + 8]
            crc = int.from_bytes(content[end - 4 : end], "big")
            if zlib.crc32(content[offset + 4 : end - 4]) != crc:
                raise ValueError("PNG 数据块校验失败。")
            chunks.append(kind_bytes)
            if kind_bytes == b"IEND" and (length != 0 or end != len(content)):
                raise ValueError("PNG 结束块不合法。")
            offset = end
        if not chunks or chunks[0] != b"IHDR" or chunks[-1] != b"IEND" or b"IDAT" not in chunks:
            raise ValueError("PNG 缺少必要数据块。")
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-threads",
                "1",
                "-i",
                str(path),
                "-f",
                "null",
                "-",
            ],
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def execute(kind: str, job_id: str, root: Path) -> Path:
    """校验输入并仅提交一次请求，将有效产物原子发布到任务目录。"""
    if kind not in ROUTES:
        raise ValueError("未知模型任务。")
    if kind == "moss" and shutil.which("ffmpeg") is None:
        raise ValueError("PNG 校验需要 FFmpeg。")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", job_id):
        raise ValueError("任务 ID 只能包含字母、数字、下划线和短横线。")
    input_path = root / "inputs" / f"{job_id}.json"
    with input_path.open("rb") as stream:
        input_bytes = stream.read(1024 * 1024 + 1)
    if len(input_bytes) > 1024 * 1024:
        raise ValueError("任务输入超过 1 MiB。")
    params = json.loads(input_bytes)
    if not isinstance(params, dict):
        raise ValueError("任务输入必须是 JSON 对象。")
    if kind == "moss":
        if (
            set(params) != {"prompt"}
            or not isinstance(params["prompt"], str)
            or not params["prompt"].strip()
            or len(params["prompt"]) > 65536
            or "\0" in params["prompt"]
        ):
            raise ValueError("MOSS 输入需要非空 prompt。")
        body = json.dumps(
            {
                "model": "MOSS-Image-SFT-Q8_0",
                "prompt": params["prompt"],
                "response_format": "b64_json",
            }
        ).encode()
        content_type = "application/json"
        base = os.environ.get("DAGU_MOSS_BASE_URL", "http://127.0.0.1:8501")
        filename = "result.png"
    else:
        if (
            set(params) - {"text", "language", "instruction"}
            or not isinstance(params.get("text"), str)
            or not params["text"].strip()
            or any(not isinstance(value, str) for value in params.values())
            or len(params["text"]) > 12000
            or len(params.get("instruction", "")) > 1000
            or len(params.get("language", "")) > 32
        ):
            raise ValueError("VoxCPM 输入需要非空 text，可选 language 和 instruction。")
        boundary = f"dagu-{uuid.uuid4().hex}"
        parts = []
        for key, value in params.items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
            )
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        content_type = f"multipart/form-data; boundary={boundary}"
        base = os.environ.get("DAGU_VOXCPM_BASE_URL", "http://127.0.0.1:8313")
        filename = "result.wav"
    parsed = urllib.parse.urlsplit(base)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.username
    ):
        raise ValueError("模型服务地址必须是 loopback HTTP 根地址。")
    jobs = root / "jobs"
    jobs.mkdir(parents=True, exist_ok=True, mode=0o700)
    _sync_directory(jobs.parent)
    job_dir = jobs / job_id
    job_dir.mkdir(mode=0o700)
    _sync_directory(jobs)
    stage = "registering"
    response_path = job_dir / ".response"
    try:
        snapshot = _json_bytes({"kind": kind, "params": params})
        _publish(job_dir / "request.json", snapshot)
        _publish(
            job_dir / "submitted.json",
            _json_bytes(
                {
                    "job_id": job_id,
                    "kind": kind,
                    "status": "submitted_or_unknown",
                    "created_at": time.time(),
                    "request_sha256": hashlib.sha256(snapshot).hexdigest(),
                    "dag_run_id": os.environ.get("DAG_RUN_ID"),
                    "endpoint": base.rstrip("/") + ROUTES[kind],
                }
            ),
        )
        stage = "request"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("http_worker.py")),
                base.rstrip("/") + ROUTES[kind],
                content_type,
                str(response_path),
            ],
            input=body,
            check=True,
            timeout=REQUEST_TIMEOUT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        stage = "validation"
        if kind == "moss":
            content = base64.b64decode(
                json.loads(response_path.read_bytes())["data"][0]["b64_json"], validate=True
            )
            response_path.write_bytes(content)
        _validate_media(response_path, kind)
        stage = "publishing"
        content = response_path.read_bytes()
        destination = job_dir / filename
        _publish(destination, content)
        _publish(
            job_dir / "completed.json",
            _json_bytes(
                {
                    "job_id": job_id,
                    "kind": kind,
                    "output": filename,
                    "output_sha256": hashlib.sha256(content).hexdigest(),
                    "completed_at": time.time(),
                }
            ),
        )
        return destination
    except Exception as exc:
        try:
            _publish(
                job_dir / "failure.json",
                _json_bytes(
                    {
                        "stage": stage,
                        "error_type": type(exc).__name__,
                        "failed_at": time.time(),
                        "status": "unknown" if stage == "request" else "failed",
                    }
                ),
            )
        except OSError:
            pass  # 磁盘故障时保留目录作为禁止重复提交的登记，不掩盖原始错误。
        raise
    finally:
        response_path.unlink(missing_ok=True)


def main() -> int:
    """解析任务参数并输出产物位置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=ROUTES)
    parser.add_argument("job_id")
    args = parser.parse_args()
    root = Path(os.environ.get("DAGU_JOB_ROOT", str(Path.home() / ".local/share/llm-dagu")))
    try:
        print(execute(args.kind, args.job_id, root))
    except Exception as exc:
        print(f"任务失败或状态未知：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
