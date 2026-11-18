"""确保每个 audio.cpp 模型有独立的真实 API 冒烟入口。"""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from unittest.mock import patch

from tests.audio_cpp import generate_examples

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "localai/manifests/audio-cpp-models.json"


class AudioCppModelTestDirectoryTests(unittest.TestCase):
    def test_single_model_runner_pins_the_requested_service(self) -> None:
        with patch.object(generate_examples, "generate_examples", return_value=[]) as generate:
            result = generate_examples.run_single_model(
                "midashenglm_gen",
                argv=["--timeout", "12"],
            )

        self.assertEqual(result, 0)
        self.assertEqual(generate.call_args.kwargs["model_keys"], ["midashenglm_gen"])
        self.assertEqual(generate.call_args.kwargs["input_audio"], None)
        self.assertEqual(generate.call_args.kwargs["transcript"], None)
        self.assertEqual(generate.call_args.kwargs["language"], None)

    def test_each_audio_cpp_service_has_a_model_specific_generation_runner(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        candidates = [
            candidate
            for candidate in manifest["models"]
            if candidate["service"].get("runtime") == "audio_cpp_runtime"
        ]
        self.assertEqual(len(candidates), 14)

        for candidate in candidates:
            key = candidate["key"]
            directory_name = Path(candidate["service"]["directory"]).name
            test_directory = ROOT / "tests" / directory_name
            with self.subTest(model=key):
                readme = test_directory / "README.md"
                runner = test_directory / "generate_example.py"
                self.assertTrue(readme.is_file(), f"缺少模型测试说明：{readme}")
                self.assertTrue(runner.is_file(), f"缺少模型真实生成入口：{runner}")

                module_spec = importlib.util.spec_from_file_location(
                    f"model_api_smoke_{key}", runner
                )
                self.assertIsNotNone(module_spec)
                self.assertIsNotNone(module_spec.loader)
                module = importlib.util.module_from_spec(module_spec)
                with patch.object(generate_examples, "run_single_model", return_value=0) as run:
                    module_spec.loader.exec_module(module)
                    self.assertEqual(module.main(), 0)
                run.assert_called_once_with(key)

    def test_source_separation_demo_saves_and_checks_the_complete_zip(self) -> None:
        candidate = next(
            item for item in generate_examples._model_candidates() if item["key"] == "bs_roformer"
        )
        wav = io.BytesIO()
        with wave.open(wav, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(44_100)
            output.writeframes(b"\x00\x00" * 4)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("vocals.wav", wav.getvalue())
            output.writestr("instrumental.wav", wav.getvalue())

        with (
            patch.object(
                generate_examples,
                "_post",
                return_value=(archive.getvalue(), "application/zip"),
            ) as post,
            patch.object(
                generate_examples, "_save_zip", wraps=generate_examples._save_zip
            ) as save_zip,
            tempfile.TemporaryDirectory() as temporary,
        ):
            source = Path(temporary) / "music.wav"
            source.write_bytes(wav.getvalue())
            output_dir = Path(temporary) / "output"
            path = generate_examples._generate_one(
                candidate,
                host="127.0.0.1",
                timeout=30,
                reference_audio=None,
                reference_text=None,
                source_audio=None,
                input_audio=source,
                transcript=None,
                language="Chinese",
                output_dir=output_dir,
            )

            with zipfile.ZipFile(path) as saved_archive:
                self.assertEqual(
                    set(saved_archive.namelist()),
                    {"vocals.wav", "instrumental.wav"},
                )
            self.assertEqual(save_zip.call_count, 1)
            self.assertEqual(post.call_args.args[1:3], (8344, candidate["service"]["route"]))

    def test_forced_aligner_demo_sends_transcript_and_saves_timestamps(self) -> None:
        candidate = next(
            item
            for item in generate_examples._model_candidates()
            if item["key"] == "qwen3_forced_aligner"
        )
        result = {"text": "Hello world.", "words": [{"text": "Hello", "start": 0, "end": 0.4}]}
        wav = io.BytesIO()
        with wave.open(wav, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(b"\x00\x00" * 4)
        with (
            patch.object(
                generate_examples,
                "_post",
                return_value=(json.dumps(result).encode(), "application/json"),
            ) as post,
            tempfile.TemporaryDirectory() as temporary,
        ):
            source = Path(temporary) / "speech.wav"
            source.write_bytes(wav.getvalue())
            output_dir = Path(temporary) / "output"
            path = generate_examples._generate_one(
                candidate,
                host="127.0.0.1",
                timeout=30,
                reference_audio=None,
                reference_text=None,
                source_audio=None,
                input_audio=source,
                transcript="Hello world.",
                language="English",
                output_dir=output_dir,
            )

            self.assertEqual(json.loads(path.read_text()), result)
            self.assertEqual(post.call_args.args[1:3], (8345, candidate["service"]["route"]))
            self.assertEqual(
                post.call_args.kwargs["form_fields"],
                {"text": "Hello world.", "language": "English"},
            )


if __name__ == "__main__":
    unittest.main()
