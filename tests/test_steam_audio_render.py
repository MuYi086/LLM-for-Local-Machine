"""对象级上传、Steam CLI 调用和纯母带路径的无模型测试。"""

from __future__ import annotations

import array
import json
import math
import shutil
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIR / "main"))

from spatial_schema import SpatialRenderManifest  # noqa: E402
from steam_audio_render import (  # noqa: E402
    SteamAudioRenderError,
    SteamAudioRenderProcessor,
)

from llm_local_runtime import StagedUpload  # noqa: E402


def render_manifest() -> SpatialRenderManifest:
    """构造同时覆盖 point 与 preserve_stereo 的任务。"""
    return SpatialRenderManifest.model_validate(
        {
            "version": "1.0",
            "sample_rate": 48000,
            "timeline_duration_ms": 5000,
            "scene": {"acoustic_quality": "balanced"},
            "sources": [
                {
                    "id": "dialogue_1",
                    "kind": "dialogue",
                    "asset_id": "dialogue_asset",
                    "asset_filename": "dialogue.wav",
                    "start_ms": 250,
                    "trim_start_ms": 125,
                    "duration_ms": 1500,
                    "playback_rate": 1.25,
                    "gain_db": -2,
                    "spatial": {
                        "mode": "point",
                        "location": "front_left",
                        "distance": "conversational",
                        "movement": "static",
                        "occlusion": "none",
                        "spatial_blend": "medium",
                    },
                },
                {
                    "id": "bgm_1",
                    "kind": "bgm",
                    "asset_id": "bgm_asset",
                    "asset_filename": "bgm.wav",
                    "start_ms": 0,
                    "duration_ms": 5000,
                    "gain_db": -12,
                    "loop": True,
                    "spatial": {
                        "mode": "preserve_stereo",
                        "location": "front_center",
                        "distance": "conversational",
                        "movement": "static",
                        "occlusion": "none",
                        "spatial_blend": "subtle",
                    },
                },
            ],
        }
    )


def write_regression_wav(path: Path) -> None:
    """生成可触发 FFmpeg 6.1.1 atempo/SoXR EOF 问题的短音频。"""
    sample_rate = 24000
    frame_count = 84497
    samples = array.array(
        "h",
        (
            round(8000 * math.sin(2 * math.pi * 220 * frame / sample_rate))
            for frame in range(frame_count)
        ),
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


class SteamAudioRenderProcessorTests(unittest.TestCase):
    def test_render_normalizes_objects_and_mastering_never_adds_legacy_spatial_filters(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="steam-render-tests-") as temporary_dir:
            root = Path(temporary_dir)
            renderer = root / "steam-audio-render"
            renderer.write_text("stub", encoding="utf-8")
            renderer.chmod(0o755)
            staged_assets: dict[str, StagedUpload] = {}
            for filename in ("dialogue.wav", "bgm.wav"):
                path = root / f"{filename}.part"
                path.write_bytes(b"RIFF-upload")
                staged_assets[filename] = StagedUpload(
                    path=path,
                    sha256="a" * 64,
                    size_bytes=path.stat().st_size,
                    suffix=".wav",
                )

            processor = SteamAudioRenderProcessor(
                root / "cache",
                renderer_bin=str(renderer),
                ffmpeg_bin="ffmpeg-test",
                timeout_seconds=30,
                mastering_timeout_seconds=30,
            )
            ffmpeg_commands: list[list[str]] = []
            renderer_manifest: dict[str, object] = {}

            def fake_ffmpeg(command: list[str]) -> str:
                ffmpeg_commands.append(command)
                if "null" in command:
                    return (
                        '{"input_i":"-21","input_tp":"-5","input_lra":"6",'
                        '"input_thresh":"-31","target_offset":"0"}'
                    )
                Path(command[-1]).write_bytes(b"normalized-or-mastered")
                return ""

            def fake_renderer(command: list[str], **_kwargs) -> str:
                manifest_path = Path(command[command.index("--manifest") + 1])
                renderer_manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))
                Path(command[command.index("--output") + 1]).write_bytes(b"steam-pre-master")
                return '{"ok":true,"sample_rate":48000,"channels":2,"duration_ms":5000,"sources":2}'

            with (
                patch.object(processor.mastering, "_run_ffmpeg", side_effect=fake_ffmpeg),
                patch.object(processor, "_run_renderer", side_effect=fake_renderer),
            ):
                progress_updates: list[tuple[str, int, str]] = []
                result = processor.render(
                    render_manifest(),
                    staged_assets,
                    profile="immersive",
                    output_format="wav",
                    progress_callback=lambda stage, progress, message: progress_updates.append(
                        (stage, progress, message)
                    ),
                )

            self.assertEqual(len(ffmpeg_commands), 4)
            self.assertIn("-ac", ffmpeg_commands[0])
            self.assertEqual(ffmpeg_commands[0][ffmpeg_commands[0].index("-ac") + 1], "1")
            self.assertIn("atempo=1.25", ffmpeg_commands[0][ffmpeg_commands[0].index("-af") + 1])
            self.assertIn("-stream_loop", ffmpeg_commands[1])
            self.assertEqual(ffmpeg_commands[1][ffmpeg_commands[1].index("-ac") + 1], "2")
            self.assertNotIn(
                "atempo=",
                ffmpeg_commands[1][ffmpeg_commands[1].index("-af") + 1],
            )
            all_filters = " ".join(" ".join(command) for command in ffmpeg_commands)
            self.assertNotIn("haas=", all_filters)
            self.assertNotIn("aecho=", all_filters)
            self.assertIn("loudnorm=", all_filters)

            self.assertEqual(renderer_manifest["scene"]["acoustic_quality"], "immersive")  # type: ignore[index]
            normalized_sources = renderer_manifest["sources"]  # type: ignore[assignment]
            self.assertEqual(normalized_sources[0]["asset_filename"], "source_0000.wav")
            self.assertEqual(normalized_sources[0]["trim_start_ms"], 0)
            self.assertEqual(normalized_sources[0]["playback_rate"], 1.0)
            self.assertFalse(normalized_sources[1]["loop"])
            self.assertTrue(result.path.is_file())
            self.assertTrue(all(not staged.path.exists() for staged in staged_assets.values()))
            self.assertIn(("rendering", 60, "Steam Audio renderer 已启动"), progress_updates)
            self.assertIn(("mastering", 90, "正在执行响度归一化与最终编码"), progress_updates)
            normalizing_messages = [
                message for stage, _progress, message in progress_updates if stage == "normalizing"
            ]
            self.assertIn("完成首个对象后估算剩余时间", normalizing_messages[0])
            self.assertIn("预计剩余", normalizing_messages[1])
            self.assertIn("音频对象标准化完成 2/2", normalizing_messages[-1])
            result.cleanup()
            self.assertFalse(result.path.parent.exists())

    @unittest.skipUnless(shutil.which("ffmpeg"), "本机未安装 FFmpeg")
    def test_one_x_normalization_does_not_stall_at_ffmpeg_eof(self) -> None:
        with tempfile.TemporaryDirectory(prefix="steam-render-regression-") as temporary_dir:
            root = Path(temporary_dir)
            source_path = root / "source.wav"
            output_path = root / "normalized.wav"
            write_regression_wav(source_path)
            staged = StagedUpload(
                source_path,
                "c" * 64,
                source_path.stat().st_size,
                ".wav",
            )
            processor = SteamAudioRenderProcessor(
                root / "cache",
                renderer_bin=sys.executable,
                mastering_timeout_seconds=1,
            )

            started = time.monotonic()
            processor._normalize_source(
                staged=staged,
                output_path=output_path,
                trim_start_ms=0,
                duration_ms=3520,
                playback_rate=1.0,
                preserve_stereo=False,
                loop=False,
            )

            self.assertLess(time.monotonic() - started, 1)
            self.assertGreater(output_path.stat().st_size, 44)

    def test_renderer_failure_cleans_job_and_staged_assets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="steam-render-tests-") as temporary_dir:
            root = Path(temporary_dir)
            renderer = root / "renderer"
            renderer.write_text("stub", encoding="utf-8")
            renderer.chmod(0o755)
            upload = root / "dialogue.wav.part"
            upload.write_bytes(b"RIFF-upload")
            staged = StagedUpload(upload, "b" * 64, upload.stat().st_size, ".wav")
            manifest = render_manifest().model_copy(deep=True)
            manifest.sources = manifest.sources[:1]
            manifest.timeline_duration_ms = 2000
            processor = SteamAudioRenderProcessor(root / "cache", renderer_bin=str(renderer))

            def fake_ffmpeg(command: list[str]) -> str:
                Path(command[-1]).write_bytes(b"normalized")
                return ""

            with (
                patch.object(processor.mastering, "_run_ffmpeg", side_effect=fake_ffmpeg),
                patch.object(
                    processor,
                    "_run_renderer",
                    side_effect=SteamAudioRenderError("renderer failed"),
                ),
                self.assertRaisesRegex(SteamAudioRenderError, "renderer failed"),
            ):
                processor.render(
                    manifest,
                    {"dialogue.wav": staged},
                    profile="balanced",
                    output_format="wav",
                )

            self.assertFalse(upload.exists())
            jobs = list((root / "cache").glob("steam_audio_*"))
            self.assertEqual(jobs, [])

    def test_renderer_report_rejects_logs_or_wrong_format(self) -> None:
        with self.assertRaisesRegex(SteamAudioRenderError, "一行 JSON"):
            SteamAudioRenderProcessor._parse_renderer_report(
                'debug\n{"ok":true,"sample_rate":48000,"channels":2,"sources":1}', 1
            )
        with self.assertRaisesRegex(SteamAudioRenderError, "48 kHz"):
            SteamAudioRenderProcessor._parse_renderer_report(
                '{"ok":true,"sample_rate":44100,"channels":2,"sources":1}', 1
            )

    def test_renderer_timeout_terminates_its_process_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="steam-render-tests-") as temporary_dir:
            processor = SteamAudioRenderProcessor(
                temporary_dir,
                renderer_bin=sys.executable,
                timeout_seconds=0.05,
            )
            started = time.monotonic()
            with self.assertRaisesRegex(SteamAudioRenderError, "超过"):
                processor._run_renderer([sys.executable, "-c", "import time; time.sleep(60)"])
            self.assertLess(time.monotonic() - started, 3)

    def test_renderer_nonzero_exit_includes_bounded_stderr_excerpt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="steam-render-tests-") as temporary_dir:
            processor = SteamAudioRenderProcessor(
                temporary_dir,
                renderer_bin=sys.executable,
                timeout_seconds=5,
            )
            command = [
                sys.executable,
                "-c",
                "import sys; [print(f'line-{i}', file=sys.stderr) for i in range(100)]; sys.exit(7)",
            ]
            with self.assertRaises(SteamAudioRenderError) as raised:
                processor._run_renderer(command)
            message = str(raised.exception)
            self.assertIn("退出码 7", message)
            self.assertNotIn("line-0\n", message)
            self.assertIn("line-99", message)


if __name__ == "__main__":
    unittest.main()
