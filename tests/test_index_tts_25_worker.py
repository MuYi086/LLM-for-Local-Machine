"""IndexTTS-2.5 worker 与本地模型视图的无模型合同测试。"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = ROOT / "model-audio" / "indexTTS-2.5"


def load_service_module(module_name: str, filename: str):
    """按文件路径加载服务模块，避免目录名中的连字符影响导入。"""
    spec = importlib.util.spec_from_file_location(module_name, SERVICE_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载服务模块：{filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_resources = load_service_module(
    "index_tts_25_model_resources_test", "index_tts25_model_resources.py"
)
worker = load_service_module("index_tts_25_worker_test", "worker.py")


def create_model_resources(root: Path, *, include_campplus: bool = True):
    """创建与 IndexTTS-2.5 本地布局一致的小型模型资源夹具。"""
    model_dir = root / "model"
    model_dir.mkdir()
    for filename in (
        "config.yaml",
        "gpt.pth",
        "codec.pth",
        "s2mel.pth",
        "wav2vec2bert_stats.pt",
        "feat1.pt",
        "feat2.pt",
        "multilingual_zh_ja_yue_char_del.tiktoken",
    ):
        if filename == "config.yaml":
            (model_dir / filename).write_text("version: 2.5\n", encoding="utf-8")
        else:
            (model_dir / filename).write_bytes(filename.encode("utf-8"))
    qwen_dir = model_dir / "qwen0.6bemo4-merge"
    qwen_dir.mkdir()
    (qwen_dir / "config.json").write_text("{}", encoding="utf-8")

    w2v_bert_dir = root / "w2v-bert-2.0"
    w2v_bert_dir.mkdir()
    for filename in (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "conformer_shaw.pt",
    ):
        (w2v_bert_dir / filename).write_bytes(filename.encode("utf-8"))

    campplus_path = root / "campplus_cn_common.bin"
    if include_campplus:
        campplus_path.write_bytes(b"campplus")

    bigvgan_dir = root / "bigvgan"
    bigvgan_dir.mkdir()
    (bigvgan_dir / "config.json").write_text("{}", encoding="utf-8")
    (bigvgan_dir / "bigvgan_generator.pt").write_bytes(b"vocoder")
    return model_dir, w2v_bert_dir, campplus_path, bigvgan_dir


class ModelViewTests(unittest.TestCase):
    """模型视图只链接文件，不向权重来源写入辅助资源。"""

    def test_build_model_view_links_checkpoint_and_existing_auxiliary_assets(self):
        with tempfile.TemporaryDirectory(prefix="index-tts-view-test-") as temporary_dir:
            root = Path(temporary_dir)
            model_dir, w2v_bert_dir, campplus_path, bigvgan_dir = create_model_resources(root)
            original_model_entries = sorted(path.name for path in model_dir.iterdir())

            view_dir = root / "runtime-view"
            model_resources.build_model_view(
                model_dir=model_dir,
                view_dir=view_dir,
                w2v_bert_dir=w2v_bert_dir,
                campplus_path=campplus_path,
                bigvgan_dir=bigvgan_dir,
            )

            self.assertTrue((view_dir / "config.yaml").is_symlink())
            self.assertEqual(
                (view_dir / "config.yaml").read_text(encoding="utf-8"), "version: 2.5\n"
            )
            self.assertTrue((view_dir / "qwen0.6bemo4-merge").is_symlink())
            self.assertTrue((view_dir / "multilingual_zh_ja_yue_char_del.tiktoken").is_symlink())
            self.assertTrue((view_dir / "hf_cache" / "w2v-bert-2.0").is_symlink())
            self.assertTrue((view_dir / "hf_cache" / "campplus_cn_common.bin").is_symlink())
            self.assertTrue((view_dir / "hf_cache" / "bigvgan").is_symlink())
            self.assertEqual(
                sorted(path.name for path in model_dir.iterdir()),
                original_model_entries,
            )

    def test_build_model_view_rejects_missing_auxiliary_asset(self):
        with tempfile.TemporaryDirectory(prefix="index-tts-view-missing-test-") as temporary_dir:
            root = Path(temporary_dir)
            model_dir, w2v_bert_dir, _, bigvgan_dir = create_model_resources(
                root, include_campplus=False
            )

            with self.assertRaises(FileNotFoundError):
                model_resources.build_model_view(
                    model_dir=model_dir,
                    view_dir=root / "runtime-view",
                    w2v_bert_dir=w2v_bert_dir,
                    campplus_path=root / "missing-campplus.bin",
                    bigvgan_dir=bigvgan_dir,
                )


class OfficialInferenceCallTests(unittest.TestCase):
    """worker 使用官方 IndexTTS2 构造器和 2.5 推理参数。"""

    def test_generate_audio_calls_official_index_tts_api(self):
        with tempfile.TemporaryDirectory(prefix="index-tts-worker-test-") as temporary_dir:
            root = Path(temporary_dir)
            model_dir = root / "model-view"
            model_dir.mkdir()
            (model_dir / "config.yaml").write_text("version: 2.5\n", encoding="utf-8")
            code_path = root / "index-tts-source"
            official_package = code_path / "indextts"
            official_package.mkdir(parents=True)
            (official_package / "infer_v2_5.py").write_text("", encoding="utf-8")
            reference_audio = root / "reference.wav"
            reference_audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
            output_path = root / "output.wav"
            observed: dict[str, object] = {}

            class FakeIndexTTS2:
                def __init__(self, **kwargs):
                    observed["constructor"] = kwargs
                    observed["offline_environment"] = (
                        os.environ.get("HF_HUB_OFFLINE"),
                        os.environ.get("TRANSFORMERS_OFFLINE"),
                        os.environ.get("HF_DATASETS_OFFLINE"),
                    )

                def infer(self, **kwargs):
                    observed["infer"] = kwargs
                    Path(kwargs["output_path"]).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
                    return kwargs["output_path"]

            package = types.ModuleType("indextts")
            package.__path__ = []
            infer_module = types.ModuleType("indextts.infer_v2_5")
            infer_module.IndexTTS2 = FakeIndexTTS2
            with (
                patch.dict(
                    sys.modules,
                    {"indextts": package, "indextts.infer_v2_5": infer_module},
                ),
                patch.dict(os.environ, {}, clear=False),
            ):
                worker.generate_audio(
                    {
                        "model_dir": str(model_dir),
                        "code_path": str(code_path),
                        "reference_audio_path": str(reference_audio),
                        "text": "你好，欢迎使用 IndexTTS。",
                        "lang": "ZH",
                        "duration_factor": 1.2,
                        "emo_vector": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "emo_alpha": 0.7,
                        "device": "cuda:0",
                        "use_bf16": True,
                    },
                    output_path,
                )

            self.assertEqual(
                observed["constructor"],
                {
                    "cfg_path": str(model_dir / "config.yaml"),
                    "model_dir": str(model_dir),
                    "use_bf16": True,
                    "device": "cuda:0",
                    "use_cuda_kernel": False,
                    "use_deepspeed": False,
                    "use_qwen_emo": False,
                },
            )
            self.assertEqual(observed["offline_environment"], ("1", "1", "1"))
            infer_kwargs = observed["infer"]
            self.assertIsInstance(infer_kwargs, dict)
            self.assertEqual(infer_kwargs["spk_audio_prompt"], str(reference_audio))
            self.assertEqual(infer_kwargs["text"], "你好，欢迎使用 IndexTTS。")
            self.assertEqual(infer_kwargs["lang"], "ZH")
            self.assertEqual(infer_kwargs["output_path"], str(output_path))
            self.assertEqual(infer_kwargs["duration_factor"], 1.2)
            self.assertEqual(infer_kwargs["emo_vector"][2], 0.8)
            self.assertEqual(infer_kwargs["emo_alpha"], 0.7)
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
