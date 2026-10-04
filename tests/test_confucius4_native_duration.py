"""原生感叹音长度控制的无模型测试，确认完整语义输入未被截断。"""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

WORKER_PATH = Path(__file__).parents[1] / "Confucius4_TTS" / "worker.py"
spec = importlib.util.spec_from_file_location("confucius4_native_duration_worker", WORKER_PATH)
assert spec and spec.loader
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Lengths:
    """模拟保持设备与 dtype 的长度 Tensor，不导入 Torch。"""

    shape = (1,)

    def __init__(self, value: int) -> None:
        self.value = value

    def new_full(self, shape: tuple[int, ...], value: int) -> "Lengths":
        assert shape == self.shape
        return Lengths(value)


class NativeDurationTests(unittest.TestCase):
    """只替换 S2A 长度，词汇文本、语义序列与 latent 对象必须完整传入。"""

    def test_keeps_complete_semantic_input_and_sets_target_frames(self) -> None:
        captured = []

        def inference(**kwargs):
            captured.append(kwargs)
            return "audio"

        model = SimpleNamespace(
            sample_rate=22050, hop_length=256, s2a_model=SimpleNamespace(inference=inference)
        )
        semantic, latent = object(), object()
        worker.configure_vocalization_duration(model, 0.441)
        result = model.s2a_model.inference(
            semantic_token=semantic, lm_latent=latent, target_feat_len=Lengths(101)
        )
        self.assertEqual(result, "audio")
        self.assertEqual(captured[0]["target_feat_len"].value, 38)
        self.assertIs(captured[0]["semantic_token"], semantic)
        self.assertIs(captured[0]["lm_latent"], latent)
        with self.assertRaisesRegex(ValueError, "单个合成块"):
            model.s2a_model.inference(target_feat_len=Lengths(101))

    def test_default_does_not_wrap_inference(self) -> None:
        def inference(**kwargs):
            return kwargs

        model = SimpleNamespace(s2a_model=SimpleNamespace(inference=inference))
        worker.configure_vocalization_duration(model, None)
        self.assertIs(model.s2a_model.inference, inference)

    def test_rejects_invalid_worker_input_without_gpu(self) -> None:
        for seconds in (float("nan"), float("inf"), -1, 0, 3, True):
            with (
                self.subTest(seconds=seconds),
                self.assertRaisesRegex(ValueError, "vocalization_duration_seconds"),
            ):
                worker.configure_vocalization_duration(SimpleNamespace(), seconds)
