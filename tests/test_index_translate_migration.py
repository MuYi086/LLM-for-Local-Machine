"""Index-Translate 原生文本、约束和图片翻译的无模型回归。"""

import base64
from unittest.mock import patch

from index_native_test_support import NativeServiceCase, load_module

IMAGE = "data:image/png;base64," + base64.b64encode(b"test image bytes").decode()


class IndexTranslateMigrationTests(NativeServiceCase):
    """验证原生消息、配套视觉组件及移除音频编排后的契约。"""

    project = "Index-Translate-9B-GGUF"
    prefix = "INDEX_TRANSLATE"
    route = "translate"

    def test_text_constraints_and_language_names(self):
        response, worker = self.post_text(
            {
                "text": "源文",
                "target_language": "ar",
                "source_language": "zh",
                "glossary": {"术语": "term"},
                "hard_constraints": ["保留 JSON"],
                "soft_constraints": ["自然语气"],
                "instruction": "保留人名",
                "genre": "字幕",
            }
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.text, "第一段译文\n\nSecond paragraph.")
        self.assertEqual(response.headers["content-type"], "text/plain; charset=utf-8")
        payload = worker.call_args.args[0]
        self.assertEqual(payload["text"], "源文")
        self.assertEqual(payload["source_language_name"], "中文")
        self.assertEqual(payload["target_language_name"], "阿拉伯语")
        self.assertEqual(payload["max_new_tokens"], 1024)
        self.assertEqual(payload["instruction"], "保留人名")
        self.assertFalse(hasattr(self.main, "synthesize_audio"))
        self.assertFalse(hasattr(self.main, "transcribe_audio"))

    def test_no_tts_length_or_language_restrictions(self):
        response, worker = self.post_text({"text": "长" * 12_001, "target_language": "斯瓦希里语"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(worker.call_args.args[0]["text"]), 12_001)
        self.assertEqual(worker.call_args.args[0]["target_language_name"], "斯瓦希里语")

    def test_invalid_and_old_audio_requests_reject_before_gpu(self):
        for payload in (
            {"audio_path": "old.wav"},
            {"text": " "},
            {"text": "源文", "image": "/etc/passwd"},
            {"image": "data:image/png;base64,***"},
            {"text": "源文", "hard_constraints": [" "]},
        ):
            with self.subTest(payload=payload), patch.object(self.main, "gpu_runtime_lock") as lock:
                self.assertEqual(
                    self.client.post("/v1/indexTranslate/translate", json=payload).status_code, 422
                )
                lock.assert_not_called()

    def test_image_requires_projector_before_gpu(self):
        with (
            patch.object(self.main, "resolve_executable", return_value="/usr/bin/true"),
            patch.object(self.main, "gpu_runtime_lock") as lock,
        ):
            response = self.client.post("/v1/indexTranslate/translate", json={"image": IMAGE})
        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn("mmproj", response.text)
        lock.assert_not_called()

    def test_image_is_forwarded_without_ocr(self):
        self.main.MMPROJ_PATH.touch()
        response, worker = self.post_text({"image": IMAGE, "target_language": "zh"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(worker.call_args.args[0]["image"], IMAGE)
        self.assertEqual(worker.call_args.args[0]["mmproj_path"], str(self.main.MMPROJ_PATH))

    def test_worker_image_message_and_projector(self):
        self.main.MMPROJ_PATH.touch()
        payload, command = self.check_worker_request(
            {
                "image": IMAGE,
                "source_language_name": "英文",
                "target_language_name": "中文",
                "mmproj_path": str(self.main.MMPROJ_PATH),
                "max_new_tokens": 1024,
            }
        )
        self.assertIn("--mmproj", command)
        self.assertEqual(command[command.index("--mmproj") + 1], str(self.main.MMPROJ_PATH))
        content = payload["messages"][0]["content"]
        self.assertIn("图片中的英文文本翻译为中文", content[0]["text"])
        self.assertEqual(content[1], {"type": "image_url", "image_url": {"url": IMAGE}})
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_json_constraint_template_matches_official(self):
        worker = load_module("translate_official_template_test", self.service_dir / "worker.py")
        prompt = worker.build_translation_prompt(
            '{"title":"源文"}',
            "英语",
            {"词": "term"},
            ["保留 JSON"],
            ["自然语气"],
            "结构化数据",
            "中文",
            "保留标题",
        )
        self.assertTrue(
            prompt.startswith("请将以下中文结构化数据翻译成英语，并且严格遵循所有约束要求。")
        )
        self.assertIn('〖源文〗\n{"title":"源文"}', prompt)
        self.assertIn("1. 〖硬性要求〗保留 JSON", prompt)
        self.assertIn("2. 〖硬性要求〗专名/术语对照: 词→term", prompt)
        self.assertIn("3. 〖注意〗自然语气", prompt)
        self.assertIn("4. 〖注意〗保留标题", prompt)
        self.assertIn("key 保持不变，value 为对应译文", prompt)

    def test_raw_prompt_has_no_automatic_template(self):
        payload, _ = self.check_worker_request(
            {"text": "原始提示词", "target_language_name": "英语", "raw_prompt": True}
        )
        self.assertEqual(payload["messages"], [{"role": "user", "content": "原始提示词"}])

    def test_plain_translation_uses_source_language(self):
        payload, _ = self.check_worker_request(
            {"text": "源文", "source_language_name": "中文", "target_language_name": "英语"}
        )
        self.assertEqual(
            payload["messages"],
            [
                {
                    "role": "user",
                    "content": "请将以下中文文本翻译为英语，直接输出翻译结果，不要进行任何解释。\n\n源文",
                }
            ],
        )
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 1024)

    def test_worker_lifecycle(self):
        for failure in (None, "exit", "timeout"):
            with self.subTest(failure=failure):
                self.check_manager_cleanup(failure)

    def test_health_distinguishes_text_and_image_readiness(self):
        with patch.object(self.main, "resolve_executable", return_value="/usr/bin/true"):
            result = self.main.health()
        self.assertTrue(result["ready"])
        self.assertFalse(result["image_ready"])
        self.assertNotIn("confucius4_tts", result["available"])
