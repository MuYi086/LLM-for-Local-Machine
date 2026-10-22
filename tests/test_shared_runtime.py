"""共享上传和 GPU 队列的无模型回归测试。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_local_runtime import (
    AudioReferenceStore,
    AudioUploadError,
    GpuLockTimeoutError,
    StagedUpload,
    UploadPolicy,
    gpu_runtime_lock,
    stage_audio_upload,
)


class FakeUpload:
    """模拟分块读取的 UploadFile，避免测试依赖 FastAPI 临时文件。"""

    def __init__(self, content: bytes, *, filename: str = "reference.wav") -> None:
        self.filename = filename
        self.content_type = "audio/wav"
        self._content = content
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._content):
            return b""
        read_size = len(self._content) if size < 0 else size
        chunk = self._content[self._offset : self._offset + read_size]
        self._offset += len(chunk)
        return chunk


class SharedRuntimeTests(unittest.TestCase):
    """验证跨服务复用的存储边界与队列失败语义。"""

    def test_upload_limit_rejects_and_removes_staging_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staging_dir = Path(temporary)
            upload = FakeUpload(b"RIFF" + b"x" * 32)
            with self.assertRaises(AudioUploadError) as raised:
                asyncio.run(
                    stage_audio_upload(
                        upload,
                        staging_dir,
                        policy=UploadPolicy(max_bytes=16, chunk_size=8),
                    )
                )
            self.assertEqual(raised.exception.status_code, 413)
            self.assertEqual(list(staging_dir.iterdir()), [])

    def test_timbre_reference_is_relative_and_does_not_copy_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = AudioReferenceStore(root / "clone", root / "timbre")
            content = b"RIFF-designed"
            timbre_path = root / "timbre" / "designed.wav"
            timbre_path.write_bytes(content)
            store.register_timbre_file(timbre_path)
            staged_path = root / "upload.wav.part"
            staged_path.write_bytes(content)
            staged = StagedUpload(
                path=staged_path,
                sha256=store.register_timbre_file(timbre_path),
                size_bytes=len(content),
                suffix=".wav",
            )

            result = store.commit_staged_upload(staged, "preview.wav", "准确参考文案")

            self.assertEqual(result["storage"], "timbre_reference")
            self.assertFalse(store.clone_path("preview.wav").exists())
            self.assertEqual(store.prompt_audio_path("preview.wav"), timbre_path)
            reference = store.reference_path("preview.wav").read_text(encoding="utf-8")
            self.assertNotIn(str(timbre_path), reference)
            self.assertIn("relative_path", reference)
            self.assertEqual(store.load_prompt_text("preview.wav"), "准确参考文案")

    def test_reference_commit_failure_restores_existing_clone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = AudioReferenceStore(root / "clone", root / "timbre")
            old_path = store.clone_path("reference.wav")
            old_path.write_bytes(b"RIFF-old")
            timbre_path = root / "timbre" / "designed.wav"
            timbre_path.write_bytes(b"RIFF-new")
            digest = store.register_timbre_file(timbre_path)
            staged_path = root / "upload.wav.part"
            staged_path.write_bytes(b"RIFF-new")
            staged = StagedUpload(staged_path, digest, staged_path.stat().st_size, ".wav")

            with patch(
                "llm_local_runtime.storage._atomic_write_json", side_effect=OSError("disk full")
            ):
                with self.assertRaises(OSError):
                    store.commit_staged_upload(staged, "reference.wav")

            self.assertEqual(old_path.read_bytes(), b"RIFF-old")
            self.assertFalse(store.reference_path("reference.wav").exists())

    def test_gpu_queue_returns_timeout_instead_of_waiting_forever(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "gpu.lock"
            with gpu_runtime_lock(lock_path, "holder", timeout_seconds=1):
                with self.assertRaises(GpuLockTimeoutError):
                    with gpu_runtime_lock(lock_path, "waiter", timeout_seconds=0.05):
                        self.fail("超时队列不应获得锁")


if __name__ == "__main__":
    unittest.main()
