"""验证 sd.cpp 的本地配置、模型清单和参数边界，不加载权重。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference_gateway.sdcpp import GenerationRequest, SdcppConfig, load_models


def model_document() -> dict:
    """返回仅用于无模型测试的清单。"""
    return {
        "schema_version": 1,
        "models": [
            {
                "id": "fixture",
                "tasks": ["text-to-image", "image-to-video"],
                "components": {"diffusion_model": "weights.gguf", "vae": "vae.safetensors"},
                "defaults": {"width": 64, "height": 64, "steps": 2, "frames": 5},
                "limits": {
                    "max_width": 128,
                    "max_height": 128,
                    "max_pixels": 8192,
                    "max_steps": 4,
                    "max_frames": 9,
                    "width_multiple": 8,
                    "height_multiple": 8,
                    "frame_multiple": 4,
                    "frame_offset": 1,
                },
                "runtime": {"backend": "cuda0", "params_backend": "cpu", "vae_tiling": True},
            }
        ],
    }


class SdcppConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "models.json"
        self.manifest.write_text(json.dumps(model_document()))

    def test_default_manifest_declares_moss_and_flux_components_and_runtime(self) -> None:
        root = Path(__file__).resolve().parents[1]
        mirror = self.root
        with patch.dict(os.environ, {"HF_MIRROR_DIR": str(mirror)}):
            models = load_models(root / "sdcpp/manifests/models.json")

        self.assertEqual(list(models), ["MOSS-Image-SFT-Q8_0", "FLUX-3-Image-Klein-GGUF-Q8"])
        moss = models["MOSS-Image-SFT-Q8_0"]
        self.assertEqual(moss.tasks, ("text-to-image",))
        self.assertEqual(set(moss.components), {"diffusion_model", "llm", "vae", "tokenizer"})
        self.assertEqual(moss.components["diffusion_model"].name, "MOSS-Image-SFT-Q8_0.gguf")
        self.assertEqual(moss.components["llm"].name, "Qwen3.5-9B-Q6_K.gguf")
        self.assertEqual(moss.components["vae"].name, "flux2-vae.safetensors")
        self.assertEqual(moss.components["tokenizer"].name, "tokenizer.json")
        self.assertEqual(moss.defaults["width"], 1024)
        self.assertEqual(moss.defaults["height"], 1024)
        self.assertEqual(moss.defaults["steps"], 20)
        self.assertEqual(moss.defaults["cfg_scale"], 3.0)
        self.assertEqual(moss.limits["width_multiple"], 16)
        self.assertEqual(moss.limits["height_multiple"], 16)
        self.assertTrue(moss.runtime["offload_to_cpu"])

        moss_arguments = moss.arguments(
            GenerationRequest(task="text-to-image", prompt="a cat"), self.root / "result.png"
        )
        self.assertIn("--offload-to-cpu", moss_arguments)
        for option, path in (
            ("--diffusion-model", moss.components["diffusion_model"]),
            ("--llm", moss.components["llm"]),
            ("--vae", moss.components["vae"]),
            ("--tokenizer", moss.components["tokenizer"]),
        ):
            self.assertEqual(moss_arguments[moss_arguments.index(option) + 1], str(path))

        flux = models["FLUX-3-Image-Klein-GGUF-Q8"]
        self.assertEqual(flux.tasks, ("text-to-image", "image-to-image"))
        self.assertEqual(set(flux.components), {"diffusion_model", "llm", "llm_vision", "vae"})
        self.assertEqual(flux.components["diffusion_model"].name, "flux3-image-klein-Q8_0.gguf")
        self.assertEqual(flux.components["llm"].name, "Qwen3VL-4B-Instruct-Q8_0.gguf")
        self.assertEqual(flux.components["llm_vision"].name, "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf")
        self.assertEqual(flux.components["vae"].name, "ae.safetensors")
        self.assertEqual(flux.defaults["width"], 1024)
        self.assertEqual(flux.defaults["height"], 1024)
        self.assertEqual(flux.defaults["steps"], 16)
        self.assertEqual(flux.defaults["cfg_scale"], 1.0)
        self.assertEqual(flux.defaults["sampler"], "euler")
        self.assertEqual(flux.runtime["image_to_image_flag"], "ref-image")

    def test_tokenizer_component_and_cpu_offload_map_to_cli_arguments(self) -> None:
        document = model_document()
        document["models"][0]["components"]["tokenizer"] = "tokenizer.json"
        document["models"][0]["components"]["llm_vision"] = "mmproj.gguf"
        document["models"][0]["runtime"] = {"offload_to_cpu": True}
        self.manifest.write_text(json.dumps(document))

        model = load_models(self.manifest)["fixture"]
        arguments = model.arguments(
            GenerationRequest(task="text-to-image", prompt="cat"), self.root / "result.png"
        )

        self.assertEqual(
            arguments[arguments.index("--tokenizer") + 1], str(self.root / "tokenizer.json")
        )
        self.assertEqual(
            arguments[arguments.index("--llm_vision") + 1], str(self.root / "mmproj.gguf")
        )
        self.assertIn("--offload-to-cpu", arguments)

    def test_image_to_image_uses_manifest_reference_flag(self) -> None:
        document = model_document()
        document["models"][0]["tasks"].append("image-to-image")
        document["models"][0]["runtime"]["image_to_image_flag"] = "ref-image"
        self.manifest.write_text(json.dumps(document))
        image = self.root / "reference.png"
        image.write_bytes(b"fixture")

        model = load_models(self.manifest)["fixture"]
        arguments = model.arguments(
            GenerationRequest(task="image-to-image", prompt="edit this image", init_image=image),
            self.root / "result.png",
        )

        self.assertEqual(arguments[arguments.index("--ref-image") + 1], str(image.resolve()))
        self.assertNotIn("--init-img", arguments)

    def test_structured_arguments_keep_prompt_as_one_literal_argument(self) -> None:
        model = load_models(self.manifest)["fixture"]
        prompt = "--output /tmp/escape; $(touch bad) 中文"
        command = model.arguments(
            GenerationRequest(task="text-to-image", prompt=prompt), self.root / "result.png"
        )
        self.assertEqual(command[command.index("--prompt") + 1], prompt)
        self.assertEqual(command[command.index("--mode") + 1], "img_gen")
        self.assertNotIn("--video-frames", command)
        self.assertEqual(
            command[command.index("--diffusion-model") + 1], str(self.root / "weights.gguf")
        )

    def test_request_limits_reject_invalid_numbers_and_unsupported_tasks(self) -> None:
        model = load_models(self.manifest)["fixture"]
        for values in [
            {"width": True},
            {"width": 65},
            {"width": 0},
            {"width": 256},
            {"width": 128, "height": 128},
            {"steps": 5},
            {"cfg_scale": float("nan")},
            {"seed": -2},
            {"task": "image-to-image"},
            {"frames": 5},
            {"fps": 24},
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                model.arguments(
                    GenerationRequest(**{"task": "text-to-image", "prompt": "cat", **values}),
                    self.root / "result.png",
                )

    def test_video_requires_input_and_valid_frame_count(self) -> None:
        model = load_models(self.manifest)["fixture"]
        image = self.root / "input.png"
        image.write_bytes(b"fixture")
        for values in [
            {},
            {"init_image": image, "frames": 1},
            {"init_image": image, "frames": 6},
            {"init_image": image, "frames": 13},
        ]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                model.arguments(
                    GenerationRequest(task="image-to-video", prompt="cat", **values),
                    self.root / "result.avi",
                )
        command = model.arguments(
            GenerationRequest(task="image-to-video", prompt="cat", init_image=image),
            self.root / "result.avi",
        )
        self.assertEqual(command[command.index("--mode") + 1], "vid_gen")
        self.assertEqual(command[command.index("--video-frames") + 1], "5")

    def test_manifest_rejects_unknown_fields_duplicates_and_bad_defaults(self) -> None:
        for mutation in [
            lambda d: d.update(schema_version=True),
            lambda d: d["models"].append(d["models"][0]),
            lambda d: d["models"][0].update(extra_args=["--output", "escape"]),
            lambda d: d["models"][0]["components"].update(unknown="x"),
            lambda d: d["models"][0]["runtime"].update(max_vram_gib=float("inf")),
            lambda d: d["models"][0]["runtime"].update(backend=[]),
            lambda d: d["models"][0]["runtime"].update(image_to_image_flag="unknown"),
            lambda d: d["models"][0]["defaults"].update(width=256),
            lambda d: d["models"][0]["defaults"].update(frames=1),
            lambda d: d["models"][0].update(tasks=["unknown"]),
        ]:
            document = model_document()
            mutation(document)
            self.manifest.write_text(json.dumps(document))
            with self.subTest(document=document), self.assertRaises(ValueError):
                load_models(self.manifest)

    def test_environment_requires_same_lock_and_valid_finite_timeouts(self) -> None:
        environment = {
            "SDCPP_BIN": str(self.root / "sd-cli"),
            "SDCPP_BIN_SHA256": "a" * 64,
            "GPU_LOCK_FILE": str(self.root / "gpu.lock"),
        }
        config = SdcppConfig.from_env(environment)
        self.assertEqual(config.queue_timeout, 900)
        self.assertEqual(config.run_timeout, 3600)
        for key, value in [
            ("GPU_LOCK_WAIT_TIMEOUT", "0"),
            ("SDCPP_RUN_TIMEOUT", "nan"),
            ("SDCPP_CLEANUP_TIMEOUT", "inf"),
            ("CUDA_RELEASE_DELAY", "-1"),
            ("SDCPP_BIN_SHA256", "bad"),
            ("GPU_LOCK_FILE", ""),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                SdcppConfig.from_env({**environment, key: value})

    def test_binary_digest_is_checked_without_executing_binary(self) -> None:
        binary = self.root / "sd-cli"
        binary.write_bytes(b"not an actual model runtime")
        binary.chmod(0o700)
        config = SdcppConfig.from_env(
            {
                "SDCPP_BIN": str(binary),
                "SDCPP_BIN_SHA256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "GPU_LOCK_FILE": str(self.root / "gpu.lock"),
            }
        )
        config.validate_binary()
        binary.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            config.validate_binary()
