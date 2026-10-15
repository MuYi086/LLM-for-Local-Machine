"""Nailong 官方 NativeLong 文档接口的无模型回归。"""

import struct
from unittest.mock import patch

from index_native_test_support import NativeServiceCase, load_module


class IndexNailongMigrationTests(NativeServiceCase):
    """验证四个固定方向、整篇输入和停止 token 映射。"""

    project = "Index-Nailong-9B-GGUF"
    prefix = "INDEX_NAILONG"
    route = "naiLong"

    def test_entire_document_is_preserved_without_tts_length_limit(self):
        text = "第一章\n\n" + "长" * 20_000 + "\n第二章\n结尾"
        response, worker = self.post_text({"text": text, "direction": "zh-en"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(worker.call_args.args[0]["text"], text)
        self.assertIsNone(worker.call_args.args[0]["max_new_tokens"])
        self.assertTrue(response.headers["content-type"].startswith("text/plain"))
        self.assertFalse(hasattr(self.main, "transcribe_audio"))
        self.assertFalse(hasattr(self.main, "synthesize_audio"))

    def test_fixed_directions_and_old_audio_contract(self):
        for direction in ("zh-en", "en-zh", "zh-ja", "ja-zh"):
            with self.subTest(direction=direction):
                response, _ = self.post_text({"text": "完整原文", "direction": direction})
                self.assertEqual(response.status_code, 200)
        for payload in (
            {"text": "原文", "direction": "zh-fr"},
            {"audio_path": "old.wav"},
            {"text": "原文", "glossary": {"词": "term"}},
            {"text": " "},
        ):
            with patch.object(self.main, "gpu_runtime_lock") as lock:
                self.assertEqual(
                    self.client.post("/v1/indexTranslate/naiLong", json=payload).status_code, 422
                )
                lock.assert_not_called()

    def test_native_long_sampling_and_template(self):
        payload, _ = self.check_worker_request(
            {"text": "第一章\n\n原文", "direction": "zh-en", "max_new_tokens": None}
        )
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["top_p"], 1)
        self.assertEqual(payload["top_k"], -1)
        self.assertEqual(payload["min_p"], 0)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["repeat_penalty"], 1)
        self.assertEqual(payload["stop"], ["<|endoftext|>", "<|im_end|>"])
        self.assertNotIn("max_tokens", payload)
        prompt = payload["messages"][0]["content"]
        self.assertTrue(prompt.startswith("请将以下中文小说翻译成英语"))
        self.assertIn("不要遗漏、概括、合并或添加内容", prompt)
        self.assertTrue(prompt.endswith("〖源文〗\n\n第一章\n\n原文\n"))

    def test_gguf_stop_tokens_are_read_from_actual_vocabulary(self):
        module = load_module("native_long_gguf_reader_test", self.service_dir / "gguf_tokens.py")

        def string(value):
            raw = value.encode()
            return struct.pack("<Q", len(raw)) + raw

        content = b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
        content += string("general.name") + struct.pack("<I", 8) + string("test")
        content += string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 4)
        content += b"".join(string(value) for value in ("x", "<stop1>", "y", "<stop2>"))
        self.model_path.write_bytes(content)
        self.assertEqual(module.load_stop_tokens(self.model_path, (1, 3)), ["<stop1>", "<stop2>"])
        with self.assertRaises(ValueError):
            module.load_stop_tokens(self.model_path, (99,))
        self.model_path.write_bytes(b"GGUF")
        with self.assertRaises(ValueError):
            module.load_stop_tokens(self.model_path)

    def test_worker_lifecycle(self):
        for failure in (None, "exit", "timeout"):
            with self.subTest(failure=failure):
                self.check_manager_cleanup(failure)

    def test_native_document_output_preserves_whitespace(self):
        """文档模型保留原生输出的段落和末尾换行。"""
        self.check_worker_request(
            {"text": "整篇原文", "direction": "zh-en"}, response_text="\n第一段\n\n第二段\n"
        )

    def test_all_templates_keep_document_structure(self):
        worker = load_module("nailong_native_template_test", self.service_dir / "worker.py")
        headers = {
            "zh-en": "中文小说翻译成英语",
            "en-zh": "英文小说翻译成中文",
            "zh-ja": "中文小说翻译成日语",
            "ja-zh": "日文小说翻译成中文",
        }
        for direction, header in headers.items():
            prompt = worker.build_translation_prompt("段落一\r\n\r\n段落二", direction)
            self.assertIn(header, prompt)
            self.assertIn("段落一\n\n段落二", prompt)
        with self.assertRaises(ValueError):
            worker.build_translation_prompt("原文", "zh-fr")
