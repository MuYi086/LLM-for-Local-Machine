"""四个 Index 原生服务的启动、依赖和进程组配置回归。"""

import ast
import signal
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch

from index_native_test_support import REPOSITORY_DIR, load_module

PROJECTS = (
    ("Index-Echo-S2ST-9B-GGUF", "INDEX_ECHO_S2ST", "index_echo", 8382),
    ("Index-Homura-9B-GGUF", "INDEX_HOMURA", "index_homura", 8383),
    ("Index-Nailong-9B-GGUF", "INDEX_NAILONG", "index_nailong", 8384),
    ("Index-Translate-9B-GGUF", "INDEX_TRANSLATE", "index_translate", 8385),
)


class IndexNativeConfigurationTests(unittest.TestCase):
    """防止后续改动重新恢复 ASR/TTS 编排或遗漏原生部署配置。"""

    def test_startup_keeps_ports_and_processes_without_asr_tts_configuration(self):
        startup = (REPOSITORY_DIR / "start.sh").read_text()
        quality = (REPOSITORY_DIR / "scripts/quality_gate.sh").read_text()
        for project, prefix, _, port in PROJECTS:
            with self.subTest(project=project):
                self.assertIn(f'{prefix}_PORT="${{{prefix}_PORT:-{port}}}"', startup)
                self.assertIn(f'setsid uv run --no-sync --project "${prefix}_PROJECT_DIR"', startup)
                self.assertNotIn(f"{prefix}_ASR_URL", startup)
                self.assertNotIn(f"{prefix}_TTS_URL", startup)
                self.assertIn(project, quality)
        self.assertIn("${INDEX_NAILONG_CONTEXT_SIZE:-229376}", startup)
        self.assertIn("${INDEX_HOMURA_CONTEXT_SIZE:-32768}", startup)
        self.assertIn("Index-Translate-9B.mmproj-Q8_0.gguf", startup)
        self.assertNotIn("INDEX_ECHO_S2ST_LLAMA_SERVER_BIN", startup)

    def test_api_processes_remain_lightweight(self):
        for project, _, _, _ in PROJECTS:
            path = REPOSITORY_DIR / project / "main.py"
            source = path.read_text()
            for node in ast.walk(ast.parse(source)):
                imports = (
                    [name.name for name in node.names]
                    if isinstance(node, ast.Import)
                    else ([node.module] if isinstance(node, ast.ImportFrom) else [])
                )
                self.assertFalse(
                    {"torch", "transformers", "modeling_dubbing", "httpx"}.intersection(imports),
                    path,
                )
            self.assertNotIn("AudioReferenceStore", source)
            self.assertNotIn("transcribe_audio", source)
            self.assertNotIn("synthesize_audio", source)

    def test_echo_optional_environment_is_exact_and_cuda129(self):
        config = tomllib.loads((REPOSITORY_DIR / PROJECTS[0][0] / "pyproject.toml").read_text())
        native = config["project"]["optional-dependencies"]["native"]
        self.assertIn("torch==2.11.0", native)
        self.assertIn("transformers==5.6.0", native)
        self.assertEqual(config["tool"]["uv"]["sources"]["torch"]["index"], "pytorch-cu129")
        self.assertNotIn("httpx==0.28.1", config["project"]["dependencies"])
        self.assertIn("python-multipart==0.0.32", config["project"]["dependencies"])

    def test_parent_exit_does_not_leave_descendants_alive(self):
        for project, _, slug, _ in PROJECTS:
            runtime = load_module(
                f"{slug}_group_cleanup_test", REPOSITORY_DIR / project / f"{slug}_runtime.py"
            )
            process = Mock(pid=12345)
            process.poll.return_value = 0
            with patch.object(runtime.os, "killpg") as kill:
                runtime.terminate_process_group(process, "test")
            self.assertEqual(kill.call_args_list[0].args, (12345, signal.SIGTERM))
            self.assertEqual(kill.call_args_list[-1].args, (12345, signal.SIGKILL))

    def test_disabled_gpu_queue_timeout_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for project, prefix, _, _ in PROJECTS:
                with patch.dict(
                    "os.environ",
                    {"STORAGE_DIR": temporary, "GPU_LOCK_WAIT_TIMEOUT": "0"},
                    clear=True,
                ):
                    with self.subTest(project=project), self.assertRaises(ValueError):
                        load_module(
                            f"{prefix.lower()}_invalid_queue", REPOSITORY_DIR / project / "main.py"
                        )
