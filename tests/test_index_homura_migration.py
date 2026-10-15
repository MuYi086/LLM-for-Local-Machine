"""Homura 原生音节约束文本接口的无模型回归。"""

from unittest.mock import patch

from index_native_test_support import NativeServiceCase, load_module


class IndexHomuraMigrationTests(NativeServiceCase):
    """验证直接文本输入、语言范围和官方采样。"""

    project = "Index-Homura-9B-GGUF"
    prefix = "INDEX_HOMURA"
    route = "homura"

    def test_text_in_text_out_and_native_constraints(self):
        response, worker = self.post_text(
            {
                "text": "我们今天去看电影吧",
                "target_language": "ar",
                "target_syllables": 8,
                "glossary": {"电影": "cinema"},
            }
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "text/plain; charset=utf-8")
        self.assertEqual(response.text, "第一段译文\n\nSecond paragraph.")
        payload = worker.call_args.args[0]
        self.assertEqual(payload["text"], "我们今天去看电影吧")
        self.assertEqual(payload["target_language_name"], "阿拉伯语")
        self.assertEqual(payload["target_syllables"], 8)
        self.assertEqual(payload["glossary"], {"电影": "cinema"})
        self.assertEqual(payload["max_new_tokens"], 512)
        self.assertFalse(hasattr(self.main, "transcribe_audio"))
        self.assertFalse(hasattr(self.main, "synthesize_audio"))

    def test_invalid_requests_do_not_acquire_gpu(self):
        for payload in (
            {"audio_path": "old.wav", "target_syllables": 8},
            {"text": "你好"},
            {"text": " ", "target_syllables": 8},
            {"text": "你好", "target_syllables": 0},
            {"text": "你好", "target_syllables": 8, "glossary": {" ": "name"}},
        ):
            with self.subTest(payload=payload), patch.object(self.main, "gpu_runtime_lock") as lock:
                self.assertEqual(
                    self.client.post("/v1/indexTranslate/homura", json=payload).status_code, 422
                )
                lock.assert_not_called()

    def test_missing_runtime_is_503_before_gpu(self):
        with patch.object(self.main, "gpu_runtime_lock") as lock:
            response = self.client.post(
                "/v1/indexTranslate/homura", json={"text": "你好", "target_syllables": 8}
            )
        self.assertEqual(response.status_code, 503)
        lock.assert_not_called()

    def test_worker_matches_official_prompt(self):
        payload, _ = self.check_worker_request(
            {
                "text": "原文",
                "target_language_name": "英语",
                "target_syllables": 8,
                "glossary": {"电影": "cinema"},
                "max_new_tokens": 512,
                "temperature": 0.3,
            }
        )
        self.assertEqual(payload["temperature"], 0.3)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(
            payload["messages"],
            [
                {
                    "role": "user",
                    "content": "请将以下文本翻译为英语，译文严格控制在 8 个音节。要求：严格遵守术语映射表〖电影 -> cinema〗。直接输出翻译结果，不要进行任何解释。\n\n原文",
                }
            ],
        )

    def test_worker_lifecycle(self):
        for failure in (None, "exit", "timeout"):
            with self.subTest(failure=failure):
                self.check_manager_cleanup(failure)

    def test_health_is_local_and_native(self):
        result = self.main.health()
        self.assertFalse(result["ready"])
        self.assertNotIn("qwen3_asr", result["available"])
        self.assertNotIn("confucius4_tts", result["available"])
        self.assertEqual(result["runtime"]["output"], "text/plain")

    def test_custom_language_name_is_preserved(self):
        response, worker = self.post_text(
            {"text": "Hello", "target_language": "斯瓦希里语", "target_syllables": 20}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(worker.call_args.args[0]["target_language_name"], "斯瓦希里语")

    def test_prompt_rejects_empty_input(self):
        worker = load_module("homura_prompt_validation", self.service_dir / "worker.py")
        with self.assertRaises(ValueError):
            worker.build_translation_prompt(" ", "英语", 8, {})
