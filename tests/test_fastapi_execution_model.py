"""阻塞式 FastAPI 处理函数的回归检查。

模型请求可能启动一次性 worker、获取共享 GPU 文件锁，或执行阻塞式上游
HTTP 调用。FastAPI 必须在线程池中执行这些处理函数，避免它们阻塞事件循环，
导致健康检查或上传接口无法响应。
"""

from __future__ import annotations

# 静态检查阻塞路由是否交给线程池，避免模型请求卡住健康检查和上传。
import ast
import unittest
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parents[1]


def route_handler_kinds(source_path: Path) -> dict[str, str]:
    """返回每个带装饰器的应用路由是同步函数还是异步函数。"""
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    handlers: dict[str, str] = {}

    for node in module.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            if not isinstance(decorator.func, ast.Attribute):
                continue
            if decorator.func.attr not in {"get", "post"}:
                continue
            route = decorator.args[0]
            if isinstance(route, ast.Constant) and isinstance(route.value, str):
                handlers[route.value] = type(node).__name__
    return handlers


class FastApiExecutionModelTests(unittest.TestCase):
    """确保阻塞任务不占用 FastAPI 事件循环，同时不改变 API 契约。"""

    def test_blocking_routes_use_sync_handlers(self) -> None:
        expected_routes = {
            "main/main.py": {"/v1/health", "/v1/check/audio"},
            "mimo_tts/main.py": {
                "/v1/health",
                "/v1/mimo/timbre",
            },
            "qwen3_tts/main.py": {
                "/v1/health",
                "/v1/check/audio",
                "/v1/qwen/clone",
            },
            "voxcpm2/main.py": {"/v1/health", "/v1/check/audio"},
            "LongCat_AudioDiT_3.5B_bf16/main.py": {
                "/v1/health",
                "/v1/check/audio",
                "/v1/longCat/clone",
            },
            "dots_tts_soar/main.py": {
                "/v1/health",
                "/v1/check/audio",
                "/v1/dotsTTS/clone",
            },
            "moss_soundEffect/main.py": {
                "/v1/health",
                "/v1/moss/soundEffect",
            },
            "stable_audio_3_medium/main.py": {
                "/v1/health",
                "/v1/stableAudio/soundEffect",
            },
            "ace_step_1_5/main.py": {
                "/v1/health",
                "/v1/aceStep/bgm",
            },
            "qwen3_voiceDesign/main.py": {"/v1/health"},
            "moss_voiceGenerator/main.py": {"/v1/health"},
            "moss_audio_4b_thinking/main.py": {"/v1/health"},
            "Confucius4_TTS/main.py": {
                "/v1/health",
                "/v1/check/audio",
                "/v1/confucius4TTS/generate",
            },
            "Step_Audio_EditX/main.py": {"/v1/health", "/v1/check/audio"},
            "TIGER-DnR/main.py": {"/v1/health"},
            "seed-vc/main.py": {"/v1/health", "/v1/check/audio", "/v1/seedVc/voiceConversion"},
            "SoulX-FlashHead-1_3B/main.py": {
                "/v1/health",
                "/v1/check/audio",
                "/v1/check/image",
                "/v1/soulX/flashHead",
            },
        }

        for relative_path, routes in expected_routes.items():
            with self.subTest(source=relative_path):
                handler_kinds = route_handler_kinds(REPOSITORY_DIR / relative_path)
                self.assertEqual(
                    {route: handler_kinds.get(route) for route in routes},
                    {route: "FunctionDef" for route in routes},
                )

        voxcpm2_source = (REPOSITORY_DIR / "voxcpm2/main.py").read_text(encoding="utf-8")
        self.assertIn(
            "return await run_in_threadpool(synthesize_voxcpm2_payload, data)", voxcpm2_source
        )

    def test_async_upload_routes_offload_filesystem_work(self) -> None:
        upload_sources = {
            "main/main.py",
            "qwen3_tts/main.py",
            "voxcpm2/main.py",
            "LongCat_AudioDiT_3.5B_bf16/main.py",
            "dots_tts_soar/main.py",
            "Step_Audio_EditX/main.py",
            "Confucius4_TTS/main.py",
            "seed-vc/main.py",
            "SoulX-FlashHead-1_3B/main.py",
        }

        for relative_path in upload_sources:
            with self.subTest(source=relative_path):
                source = (REPOSITORY_DIR / relative_path).read_text(encoding="utf-8")
                self.assertIn(
                    "return await run_in_threadpool(store_uploaded_audio,",
                    source,
                )

        thinking_source = (REPOSITORY_DIR / "moss_audio_4b_thinking" / "main.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "result = await run_in_threadpool(execute_understanding_payload, payload)",
            thinking_source,
        )


if __name__ == "__main__":
    unittest.main()
