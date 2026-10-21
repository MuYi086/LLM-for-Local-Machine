"""管理固定版本的宿主机 LocalAI 进程及其 audio-cpp 后端目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "localai/manifests/versions.json"
QWEN3_ASR_MODEL_NAME = "unitale.qwen3-asr.1p7b"
QWEN3_ASR_CONFIG_PATH = ROOT / "localai/manifests/qwen3-asr-candidate.yaml"
QWEN3_TTS_MODEL_NAME = "unitale.qwen3-tts.1p7b-base"
QWEN3_TTS_CONFIG_PATH = ROOT / "localai/manifests/qwen3-tts-base-candidate.yaml"
QWEN3_TTS_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
QWEN3_VOICEDESIGN_MODEL_NAME = "unitale.qwen3-tts.1p7b-voicedesign"
QWEN3_VOICEDESIGN_CONFIG_PATH = ROOT / "localai/manifests/qwen3-tts-voicedesign-candidate.yaml"
DOTS_TTS_SOAR_MODEL_NAME = "unitale.dots-tts-soar.q8-0"
DOTS_TTS_SOAR_CONFIG_PATH = ROOT / "localai/manifests/dots-tts-soar-candidate.yaml"
VOXCPM2_MODEL_NAME = "unitale.voxcpm2.q8-0"
VOXCPM2_CONFIG_PATH = ROOT / "localai/manifests/voxcpm2-candidate.yaml"
CONFUCIUS4_TTS_MODEL_NAME = "unitale.confucius4-tts.orig"
CONFUCIUS4_TTS_CONFIG_PATH = ROOT / "localai/manifests/confucius4-tts-candidate.yaml"
MOSS_VOICEGENERATOR_MODEL_NAME = "unitale.moss-voicegenerator.bf16"
MOSS_VOICEGENERATOR_CONFIG_PATH = ROOT / "localai/manifests/moss-voicegenerator-candidate.yaml"
FIREREDTTS3_BASE_MODEL_NAME = "unitale.fireredtts3.base.q8-0"
FIREREDTTS3_BASE_CONFIG_PATH = ROOT / "localai/manifests/fireredtts3-base-candidate.yaml"
FIREREDTTS3_INSTRUCT_MODEL_NAME = "unitale.fireredtts3.instruct.q8-0"
FIREREDTTS3_INSTRUCT_CONFIG_PATH = ROOT / "localai/manifests/fireredtts3-instruct-candidate.yaml"
ACE_STEP_MODEL_NAME = "unitale.ace-step15.xl-turbo.q8dit"
ACE_STEP_CONFIG_PATH = ROOT / "localai/manifests/ace-step-xl-turbo-candidate.yaml"
STABLE_AUDIO_MODEL_NAME = "unitale.stable-audio3-medium.q8-0"
STABLE_AUDIO_CONFIG_PATH = ROOT / "localai/manifests/stable-audio3-medium-candidate.yaml"
SEED_VC_VC_MODEL_NAME = "unitale.seed-vc.vc.f16"
SEED_VC_VC_CONFIG_PATH = ROOT / "localai/manifests/seed-vc-vc-candidate.yaml"
SEED_VC_SVC_MODEL_NAME = "unitale.seed-vc.svc.f16"
SEED_VC_SVC_CONFIG_PATH = ROOT / "localai/manifests/seed-vc-svc-candidate.yaml"
SEED_VC_ARTIFACT_STAMP = "unitale.seed-vc.artifact.json"
WORKER_BRIDGE_CANDIDATES = {
    "unitale.sheetsage2.worker.v1": ROOT / "localai/manifests/sheetsage2-worker-candidate.yaml",
    "unitale.mert-v2-fullsong.worker.v1": ROOT
    / "localai/manifests/mert-v2-fullsong-worker-candidate.yaml",
    "unitale.yue2-vae.worker.v1": ROOT / "localai/manifests/yue2-vae-worker-candidate.yaml",
    "unitale.soulx-flashhead.worker.v1": ROOT
    / "localai/manifests/soulx-flashhead-worker-candidate.yaml",
    "unitale.ditto.worker.v1": ROOT / "localai/manifests/ditto-worker-candidate.yaml",
    "unitale.yue2-3b.worker.v1": ROOT / "localai/manifests/yue2-3b-worker-candidate.yaml",
    "unitale.moss-soundeffect-v2.worker.v1": ROOT
    / "localai/manifests/moss-soundeffect-v2-worker-candidate.yaml",
    "unitale.step-audio-editx.worker.v1": ROOT
    / "localai/manifests/step-audio-editx-worker-candidate.yaml",
    "unitale.moss-audio-4b.thinking.worker.v1": ROOT
    / "localai/manifests/moss-audio-4b-thinking-worker-candidate.yaml",
    "unitale.moss-audio-4b.instruct.worker.v1": ROOT
    / "localai/manifests/moss-audio-4b-instruct-worker-candidate.yaml",
    "unitale.tiger-dnr.worker.v1": ROOT / "localai/manifests/tiger-dnr-worker-candidate.yaml",
}
WORKER_BRIDGE_MANIFEST_PATH = ROOT / "localai/manifests/worker-bridge.json"
WORKER_BRIDGE_PROJECTS = (
    "SheetSage2",
    "MERT-v2-FullSong",
    "YuE2-Vae",
    "SoulX-FlashHead-1_3B",
    "ditto",
    "YuE2-3B",
    "moss_soundEffect",
    "Step_Audio_EditX",
    "moss_audio_4b_thinking",
    "TIGER-DnR",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_COMMAND_TIMEOUT_SECONDS = 15
_STOP_TIMEOUT_SECONDS = 15


class LocalAIManagerError(RuntimeError):
    """LocalAI 宿主机进程或固定后端无法安全管理。"""


class LocalAIManager:
    """校验本机安装并管理 LocalAI 宿主机进程组。"""

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        readiness_timeout: float = 60.0,
        stop_timeout: float = _STOP_TIMEOUT_SECONDS,
    ) -> None:
        self.environ = dict(os.environ if environ is None else environ)
        self.readiness_timeout = readiness_timeout
        self.stop_timeout = stop_timeout
        self.manifest = self._load_manifest()
        self.data_dir = Path(
            self.environ.get(
                "UNITALE_LOCALAI_DATA_DIR",
                str(Path(self.environ.get("STORAGE_DIR", ROOT / "storage")) / ".cache/localai"),
            )
        ).expanduser()
        self.models_dir = Path(
            self.environ.get("UNITALE_LOCALAI_MODELS_DIR", str(self.data_dir / "models"))
        ).expanduser()
        self.backends_dir = Path(
            self.environ.get("UNITALE_LOCALAI_BACKENDS_DIR", str(self.data_dir / "backends"))
        ).expanduser()
        self.binary_path = Path(
            self.environ.get("UNITALE_LOCALAI_BINARY", str(self.data_dir / "bin/local-ai"))
        ).expanduser()
        self.logs_dir = self.data_dir / "logs"
        self.state_dir = self.data_dir / "run"
        self.state_path = self.state_dir / "localai.json"
        self.install_stamp_path = self.data_dir / ".unitale-install.json"

    @staticmethod
    def _load_manifest() -> dict[str, object]:
        """读取并校验 P0 中固定的 LocalAI/audio.cpp 源码版本。"""
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            localai = manifest["localai"]
            backend = manifest["audio_cpp_backend"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError("LocalAI 版本清单缺失或格式无效。") from exc

        if not isinstance(localai, dict) or not isinstance(backend, dict):
            raise LocalAIManagerError("LocalAI/audio.cpp 版本清单格式无效。")
        if localai.get("version") != "v4.11.0":
            raise LocalAIManagerError("当前 manager 只支持清单锁定的 LocalAI v4.11.0。")
        for label, value in (
            ("LocalAI", localai.get("source_commit")),
            ("audio.cpp", backend.get("source_commit")),
        ):
            if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
                raise LocalAIManagerError(f"{label} 缺少完整的固定源码 commit。")
        if backend.get("localai_source_commit") != localai["source_commit"]:
            raise LocalAIManagerError("audio.cpp backend 与 LocalAI 源码版本不匹配。")
        if backend.get("compute_variant") != "CUDA 12":
            raise LocalAIManagerError("当前 manager 只支持 P0 锁定的 CUDA 12 后端构建。")
        return manifest

    @property
    def localai_version(self) -> str:
        """返回锁定的 LocalAI 版本。"""
        return str(self.manifest["localai"]["version"])

    @property
    def localai_commit(self) -> str:
        """返回锁定的 LocalAI 源码 commit。"""
        return str(self.manifest["localai"]["source_commit"])

    @property
    def audio_cpp_commit(self) -> str:
        """返回 LocalAI v4.11.0 内嵌的 audio.cpp commit。"""
        return str(self.manifest["audio_cpp_backend"]["source_commit"])

    def verify_install(self) -> None:
        """校验预先准备的宿主机二进制和后端，并写入内容指纹。"""
        self._ensure_linux_amd64()
        self._ensure_directories()
        self._write_install_stamp()
        print("已校验宿主机 LocalAI/audio.cpp 文件；该检查不代表模型推理验收。")

    def register_worker_bridge_candidates(self) -> None:
        """将已固定的一次性 worker bridge 模型 YAML 注册到 LocalAI。"""
        self._verify_worker_bridge_manifest()
        self._ensure_directories()
        for model_name, template_path in WORKER_BRIDGE_CANDIDATES.items():
            rendered = template_path.read_text(encoding="utf-8")
            self._register_model_config(model_name, rendered)
        print("已注册 E0 one-shot worker bridge 候选；不代表真实模型推理已验收。")

    def verify_worker_bridge_candidates(self) -> None:
        """检查 worker registry 路径和 LocalAI 固定 YAML 是否完整。"""
        self._verify_worker_bridge_manifest()
        for model_name, template_path in WORKER_BRIDGE_CANDIDATES.items():
            try:
                expected = template_path.read_bytes()
                installed = (self.models_dir / f"{model_name}.yaml").read_bytes()
            except OSError as exc:
                raise LocalAIManagerError(
                    f"LocalAI worker bridge 模型 {model_name} 尚未注册；先运行 "
                    "manager.py register-worker-bridge-candidates。"
                ) from exc
            if expected != installed:
                raise LocalAIManagerError(f"LocalAI worker bridge YAML 不匹配：{model_name}")

    @staticmethod
    def _verify_worker_bridge_manifest() -> None:
        """验证版本化 bridge registry 只引用仓库内固定 worker 脚本。"""
        try:
            manifest = json.loads(WORKER_BRIDGE_MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError(
                "LocalAI worker bridge manifest 不可读或 JSON 无效。"
            ) from exc
        workers = manifest.get("workers") if isinstance(manifest, dict) else None
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise LocalAIManagerError("LocalAI worker bridge manifest schema_version 必须为 1。")
        if not isinstance(workers, dict) or set(workers) != set(WORKER_BRIDGE_CANDIDATES):
            raise LocalAIManagerError("LocalAI worker bridge registry 与固定候选模型集合不一致。")
        for model_name, entry in workers.items():
            if not isinstance(entry, dict):
                raise LocalAIManagerError(
                    f"LocalAI worker bridge registry entry 无效：{model_name}"
                )
            project_value = entry.get("project")
            script_value = entry.get("script")
            if (
                not isinstance(project_value, str)
                or Path(project_value).is_absolute()
                or ".." in Path(project_value).parts
                or not isinstance(script_value, str)
                or Path(script_value).is_absolute()
                or ".." in Path(script_value).parts
            ):
                raise LocalAIManagerError(f"LocalAI worker bridge 路径无效：{model_name}")
            project = (ROOT / project_value).resolve()
            script = (project / script_value).resolve()
            if not project.is_relative_to(ROOT) or not script.is_relative_to(project):
                raise LocalAIManagerError(f"LocalAI worker bridge 路径越界：{model_name}")
            if not project.is_dir() or not script.is_file():
                raise LocalAIManagerError(f"LocalAI worker bridge worker 不存在：{model_name}")

    def register_qwen3_asr_candidate(self) -> None:
        """校验本地候选 GGUF 后，将固定 YAML 原子复制到 LocalAI models 目录。"""
        self._verify_qwen3_asr_artifact()
        self._ensure_directories()
        try:
            candidate_yaml = QWEN3_ASR_CONFIG_PATH.read_bytes()
        except OSError as exc:
            raise LocalAIManagerError("Qwen3-ASR 候选 YAML 模板不可读。") from exc
        destination = self.models_dir / f"{QWEN3_ASR_MODEL_NAME}.yaml"
        if destination.exists():
            try:
                current_yaml = destination.read_bytes()
            except OSError as exc:
                raise LocalAIManagerError("现有 Qwen3-ASR YAML 不可读。") from exc
            if current_yaml != candidate_yaml:
                raise LocalAIManagerError(f"现有模型配置与候选模板不同，拒绝覆盖：{destination}")
            print(f"Qwen3-ASR 候选配置已存在：{destination}")
            return

        temporary_path = self._write_temporary_file(self.models_dir, ".qwen3-asr-", ".yaml")
        try:
            temporary_path.write_bytes(candidate_yaml)
            self._fsync_file(temporary_path)
            os.replace(temporary_path, destination)
            self._fsync_directory(self.models_dir)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"已注册 E0 候选配置（不是生产验收）：{destination}")

    def verify_qwen3_asr_candidate(self) -> None:
        """校验候选模型文件大小、SHA-256 和已注册 YAML。"""
        self._verify_qwen3_asr_artifact()
        destination = self.models_dir / f"{QWEN3_ASR_MODEL_NAME}.yaml"
        try:
            installed_yaml = destination.read_bytes()
            candidate_yaml = QWEN3_ASR_CONFIG_PATH.read_bytes()
        except OSError as exc:
            raise LocalAIManagerError(
                "Qwen3-ASR 固定 YAML 尚未注册；运行 manager.py register-qwen3-asr-candidate。"
            ) from exc
        if installed_yaml != candidate_yaml:
            raise LocalAIManagerError("LocalAI models 目录中的 Qwen3-ASR YAML 与固定模板不一致。")

    def _verify_qwen3_asr_artifact(self) -> None:
        """按 P0 记录的相对路径、文件长度和 hash 校验 Qwen3-ASR GGUF。"""
        artifacts = self.manifest.get("model_artifacts")
        packages = (
            artifacts.get("selected_localai_packages") if isinstance(artifacts, dict) else None
        )
        if not isinstance(packages, list):
            raise LocalAIManagerError("版本清单缺少 Qwen3-ASR 制品记录。")
        records = [
            record
            for record in packages
            if isinstance(record, dict) and record.get("capability") == "Qwen3-ASR-1.7B / 8371"
        ]
        if len(records) != 1:
            raise LocalAIManagerError("版本清单必须且只能包含一个 Qwen3-ASR 候选制品。")
        record = records[0]
        relative_path = record.get("model_file")
        expected_size = record.get("model_file_size_bytes")
        expected_hash = record.get("model_file_sha256")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or "\\" in relative_path
            or ".." in Path(relative_path).parts
        ):
            raise LocalAIManagerError("Qwen3-ASR 制品路径必须是 models 目录下的相对路径。")
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise LocalAIManagerError("Qwen3-ASR 制品缺少有效文件大小。")
        if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
            raise LocalAIManagerError("Qwen3-ASR 制品缺少有效 SHA-256。")
        artifact_path = (self.models_dir / relative_path).resolve()
        if not artifact_path.is_relative_to(self.models_dir.resolve()):
            raise LocalAIManagerError("Qwen3-ASR 制品路径越出了 LocalAI models 目录。")
        try:
            actual_size = artifact_path.stat().st_size
        except OSError as exc:
            raise LocalAIManagerError(f"Qwen3-ASR 候选权重不存在：{artifact_path}") from exc
        if actual_size != expected_size:
            raise LocalAIManagerError(
                f"Qwen3-ASR 候选权重大小不匹配：期望 {expected_size}，实际 {actual_size}。"
            )
        if self._sha256_file(artifact_path) != expected_hash:
            raise LocalAIManagerError("Qwen3-ASR 候选权重 SHA-256 与版本清单不匹配。")

    def register_qwen3_tts_candidate(self) -> None:
        """校验用户准备的 Base safetensors 快照并注册本地模型 YAML。"""
        model_dir = self._verify_qwen3_tts_model_files()
        try:
            template = QWEN3_TTS_CONFIG_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            raise LocalAIManagerError("Qwen3-TTS Base 候选 YAML 模板不可读。") from exc
        if template.count("__QWEN3_TTS_MODEL_PATH__") != 1:
            raise LocalAIManagerError("Qwen3-TTS Base YAML 模板必须且只能有一个模型路径占位符。")
        candidate_yaml = template.replace(
            "__QWEN3_TTS_MODEL_PATH__", json.dumps(str(model_dir))
        ).encode()
        self._ensure_directories()
        destination = self.models_dir / f"{QWEN3_TTS_MODEL_NAME}.yaml"
        if destination.exists():
            try:
                current_yaml = destination.read_bytes()
            except OSError as exc:
                raise LocalAIManagerError("现有 Qwen3-TTS Base YAML 不可读。") from exc
            if current_yaml != candidate_yaml:
                raise LocalAIManagerError(f"现有 Qwen3-TTS Base 配置不同，拒绝覆盖：{destination}")
            print(f"Qwen3-TTS Base 候选配置已存在：{destination}")
            return

        temporary_path = self._write_temporary_file(self.models_dir, ".qwen3-tts-", ".yaml")
        try:
            temporary_path.write_bytes(candidate_yaml)
            self._fsync_file(temporary_path)
            os.replace(temporary_path, destination)
            self._fsync_directory(self.models_dir)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"已注册 E0 候选配置（不是推理验收）：{destination}")

    def verify_qwen3_tts_candidate(self) -> None:
        """校验 Qwen3-TTS Base 所需权重文件及已注册模型配置。"""
        model_dir = self._verify_qwen3_tts_model_files()
        try:
            template = QWEN3_TTS_CONFIG_PATH.read_text(encoding="utf-8")
            installed_yaml = (self.models_dir / f"{QWEN3_TTS_MODEL_NAME}.yaml").read_bytes()
        except OSError as exc:
            raise LocalAIManagerError(
                "Qwen3-TTS Base 模型或固定 YAML 尚未就绪；先下载模型并运行 "
                "manager.py register-qwen3-tts-candidate。"
            ) from exc
        candidate_yaml = template.replace(
            "__QWEN3_TTS_MODEL_PATH__", json.dumps(str(model_dir))
        ).encode()
        if installed_yaml != candidate_yaml:
            raise LocalAIManagerError(
                "LocalAI models 目录中的 Qwen3-TTS Base YAML 与固定模板不一致。"
            )

    def _verify_qwen3_tts_model_files(self) -> Path:
        """检查官方 Base safetensors 布局，不下载或导入模型运行时。"""
        model_dir_value = self.environ.get("QWEN3_TTS_MODEL_DIR")
        if not model_dir_value:
            hf_mirror = Path(self.environ.get("HF_MIRROR_DIR", "~/hf-mirror")).expanduser()
            model_dir_value = str(hf_mirror / "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        model_dir = Path(os.path.expandvars(model_dir_value)).expanduser().resolve()
        missing_files = [
            relative_path
            for relative_path in QWEN3_TTS_MODEL_FILES
            if not (model_dir / relative_path).is_file()
            or (model_dir / relative_path).stat().st_size == 0
        ]
        if missing_files:
            missing = ", ".join(missing_files)
            raise LocalAIManagerError(
                f"Qwen3-TTS Base 模型文件缺失或为空：{model_dir}（{missing}）；"
                "该候选需要官方 Qwen3-TTS-12Hz-1.7B-Base safetensors 快照。"
            )
        return model_dir

    def register_qwen3_voicedesign_candidate(self) -> None:
        """校验手动准备的 VoiceDesign 文件并注册固定 task:vdes YAML。"""
        model_dir = self._verify_qwen3_voicedesign_model_files()
        template = QWEN3_VOICEDESIGN_CONFIG_PATH.read_text(encoding="utf-8")
        if template.count("__QWEN3_VOICEDESIGN_MODEL_PATH__") != 1:
            raise LocalAIManagerError("Qwen3 VoiceDesign 候选 YAML 模板占位符无效。")
        rendered = template.replace("__QWEN3_VOICEDESIGN_MODEL_PATH__", json.dumps(str(model_dir)))
        self._register_model_config(QWEN3_VOICEDESIGN_MODEL_NAME, rendered)

    def verify_qwen3_voicedesign_candidate(self) -> None:
        """检查 VoiceDesign 模型文件与已注册 task:vdes YAML。"""
        model_dir = self._verify_qwen3_voicedesign_model_files()
        try:
            template = QWEN3_VOICEDESIGN_CONFIG_PATH.read_text(encoding="utf-8")
            installed_yaml = (self.models_dir / f"{QWEN3_VOICEDESIGN_MODEL_NAME}.yaml").read_bytes()
        except OSError as exc:
            raise LocalAIManagerError(
                "Qwen3 VoiceDesign 模型或固定 YAML 尚未就绪；先手动准备模型并运行 "
                "manager.py register-qwen3-voicedesign-candidate。"
            ) from exc
        candidate_yaml = template.replace(
            "__QWEN3_VOICEDESIGN_MODEL_PATH__", json.dumps(str(model_dir))
        ).encode()
        if installed_yaml != candidate_yaml:
            raise LocalAIManagerError(
                "LocalAI models 目录中的 Qwen3 VoiceDesign YAML 与固定 task:vdes 模板不一致。"
            )

    def _verify_qwen3_voicedesign_model_files(self) -> Path:
        """检查官方 VoiceDesign safetensors 布局，不下载或导入模型运行时。"""
        model_dir_value = self.environ.get("QWEN_VOICEDESIGN_MODEL_DIR")
        if not model_dir_value:
            hf_mirror = Path(self.environ.get("HF_MIRROR_DIR", "~/hf-mirror")).expanduser()
            model_dir_value = str(hf_mirror / "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        model_dir = Path(os.path.expandvars(model_dir_value)).expanduser().resolve()
        missing_files = [
            relative_path
            for relative_path in QWEN3_TTS_MODEL_FILES
            if not (model_dir / relative_path).is_file()
            or (model_dir / relative_path).stat().st_size == 0
        ]
        if missing_files:
            missing = ", ".join(missing_files)
            raise LocalAIManagerError(
                f"Qwen3 VoiceDesign 模型文件缺失或为空：{model_dir}（{missing}）；"
                "该候选需要官方 Qwen3-TTS-12Hz-1.7B-VoiceDesign safetensors 快照。"
            )
        return model_dir

    def register_dots_tts_soar_candidate(self) -> None:
        """为手动准备的固定 DotTTS SOAR GGUF 注册 YAML 和本地 hash 记录。"""
        artifact_path = self._dots_tts_soar_artifact_path()
        record = self._dots_tts_candidate_record()
        self._ensure_directories()
        template = DOTS_TTS_SOAR_CONFIG_PATH.read_text(encoding="utf-8")
        relative_path = str(record["model_file"])
        if template.count(f"  model: {relative_path}\n") != 1:
            raise LocalAIManagerError("DotTTS SOAR YAML 路径与版本清单不一致。")
        destination = self.models_dir / f"{DOTS_TTS_SOAR_MODEL_NAME}.yaml"
        self._register_model_config(DOTS_TTS_SOAR_MODEL_NAME, template)
        stamp = {
            "schema_version": 1,
            "model": DOTS_TTS_SOAR_MODEL_NAME,
            "relative_path": relative_path,
            "size_bytes": artifact_path.stat().st_size,
            "sha256": self._sha256_file(artifact_path),
            "source_revision": record["repository_revision"],
        }
        self._write_json_atomically(
            self.models_dir / f"{DOTS_TTS_SOAR_MODEL_NAME}.artifact.json",
            stamp,
            prefix=".dots-tts-soar-artifact-",
        )
        print(
            "已注册 E0 DotTTS SOAR Q8 候选；本地 hash 仅用于检测后续文件变化，不等于上游制品验收："
            f"{destination}"
        )

    def verify_dots_tts_soar_candidate(self) -> None:
        """校验 DotTTS SOAR 候选路径、配置与注册时记录的本地 hash。"""
        artifact_path = self._dots_tts_soar_artifact_path()
        record = self._dots_tts_candidate_record()
        config_path = self.models_dir / f"{DOTS_TTS_SOAR_MODEL_NAME}.yaml"
        stamp_path = self.models_dir / f"{DOTS_TTS_SOAR_MODEL_NAME}.artifact.json"
        try:
            expected_config = DOTS_TTS_SOAR_CONFIG_PATH.read_bytes()
            installed_config = config_path.read_bytes()
            stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError(
                "DotTTS SOAR 权重或固定 YAML 尚未注册；手动准备模型后运行 "
                "manager.py register-dots-tts-soar-candidate。"
            ) from exc
        if expected_config.decode("utf-8").count(f"  model: {record.get('model_file')}\n") != 1:
            raise LocalAIManagerError("DotTTS SOAR YAML 路径与版本清单不一致。")
        if installed_config != expected_config:
            raise LocalAIManagerError("LocalAI DotTTS SOAR YAML 与固定模板不一致。")
        if (
            not isinstance(stamp, dict)
            or stamp.get("schema_version") != 1
            or stamp.get("model") != DOTS_TTS_SOAR_MODEL_NAME
            or stamp.get("relative_path") != record.get("model_file")
            or stamp.get("source_revision") != record.get("repository_revision")
        ):
            raise LocalAIManagerError("DotTTS SOAR 本地制品记录缺失或与固定 revision 不符。")
        if stamp.get("size_bytes") != artifact_path.stat().st_size or stamp.get(
            "sha256"
        ) != self._sha256_file(artifact_path):
            raise LocalAIManagerError("DotTTS SOAR GGUF 与注册时记录的本地 size/hash 不一致。")

    def _dots_tts_soar_artifact_path(self) -> Path:
        """解析 models 下固定相对路径并要求权重文件已由用户提供。"""
        record = self._dots_tts_candidate_record()
        relative_value = record.get("model_file")
        if not isinstance(relative_value, str) or not relative_value:
            raise LocalAIManagerError("DotTTS SOAR 版本清单缺少 GGUF 相对路径。")
        relative_path = Path(relative_value)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative_value
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", relative_value)
        ):
            raise LocalAIManagerError("DotTTS SOAR 模型路径必须位于 LocalAI models 目录内。")
        artifact_path = (self.models_dir / relative_path).resolve()
        if not artifact_path.is_relative_to(self.models_dir.resolve()):
            raise LocalAIManagerError("DotTTS SOAR 模型路径越出了 LocalAI models 目录。")
        try:
            size = artifact_path.stat().st_size
        except OSError as exc:
            raise LocalAIManagerError(
                "DotTTS SOAR GGUF 尚未准备；请手动放入 LocalAI models 目录的 "
                f"{relative_value}。本命令不会下载模型。"
            ) from exc
        if not artifact_path.is_file() or size <= 0:
            raise LocalAIManagerError("DotTTS SOAR GGUF 必须是非空常规文件。")
        return artifact_path

    def _dots_tts_candidate_record(self) -> dict[str, object]:
        """读取唯一 DotTTS SOAR 制品记录并核验 revision 形状。"""
        artifacts = self.manifest.get("model_artifacts")
        candidates = artifacts.get("dots_tts_candidates") if isinstance(artifacts, dict) else None
        records = (
            [
                item
                for item in candidates or []
                if isinstance(item, dict) and item.get("capability") == "dots.tts-soar clone / 8324"
            ]
            if isinstance(candidates, list)
            else []
        )
        if len(records) != 1:
            raise LocalAIManagerError("版本清单必须且只能有一个 DotTTS SOAR 候选制品记录。")
        record = records[0]
        revision = record.get("repository_revision")
        if not isinstance(revision, str) or not _COMMIT_RE.fullmatch(revision):
            raise LocalAIManagerError("DotTTS SOAR 候选缺少固定的 HF repository revision。")
        return record

    def register_voxcpm2_candidate(self) -> None:
        """为手动准备的 VoxCPM2 Q8 GGUF 注册固定 audio.cpp YAML。"""
        artifact_path = self._voxcpm2_artifact_path()
        record = self._voxcpm2_candidate_record()
        rendered = VOXCPM2_CONFIG_PATH.read_text(encoding="utf-8")
        if rendered.count(f"  model: {record['model_file']}\n") != 1:
            raise LocalAIManagerError("VoxCPM2 候选 YAML 与版本清单中的路径不一致。")
        self._register_model_config(VOXCPM2_MODEL_NAME, rendered)
        self._write_json_atomically(
            self.models_dir / f"{VOXCPM2_MODEL_NAME}.artifact.json",
            {
                "schema_version": 1,
                "model": VOXCPM2_MODEL_NAME,
                "relative_path": record["model_file"],
                "size_bytes": artifact_path.stat().st_size,
                "sha256": self._sha256_file(artifact_path),
                "source_revision": record["repository_revision"],
            },
            prefix=".voxcpm2-artifact-",
        )
        print("已注册 E0 VoxCPM2 Q8 候选；本地 hash 仅用于检测后续文件变化，不等于上游制品验收。")

    def verify_voxcpm2_candidate(self) -> None:
        """核对 VoxCPM2 候选路径、配置与注册时记录的本地 hash。"""
        artifact_path = self._voxcpm2_artifact_path()
        record = self._voxcpm2_candidate_record()
        try:
            expected_config = VOXCPM2_CONFIG_PATH.read_bytes()
            installed_config = (self.models_dir / f"{VOXCPM2_MODEL_NAME}.yaml").read_bytes()
            stamp = json.loads(
                (self.models_dir / f"{VOXCPM2_MODEL_NAME}.artifact.json").read_text()
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError(
                "VoxCPM2 权重或固定 YAML 尚未注册；手动准备模型后运行 "
                "manager.py register-voxcpm2-candidate。"
            ) from exc
        if expected_config.decode("utf-8").count(f"  model: {record['model_file']}\n") != 1:
            raise LocalAIManagerError("VoxCPM2 YAML 路径与固定版本清单不一致。")
        if installed_config != expected_config:
            raise LocalAIManagerError("LocalAI VoxCPM2 YAML 与固定模板不一致。")
        if (
            not isinstance(stamp, dict)
            or stamp.get("schema_version") != 1
            or stamp.get("model") != VOXCPM2_MODEL_NAME
            or stamp.get("relative_path") != record.get("model_file")
            or stamp.get("source_revision") != record.get("repository_revision")
        ):
            raise LocalAIManagerError("VoxCPM2 本地制品记录缺失或与固定 revision 不符。")
        if stamp.get("size_bytes") != artifact_path.stat().st_size or stamp.get(
            "sha256"
        ) != self._sha256_file(artifact_path):
            raise LocalAIManagerError("VoxCPM2 GGUF 与注册时记录的本地 size/hash 不一致。")

    def _voxcpm2_artifact_path(self) -> Path:
        """解析 LocalAI models 下的 VoxCPM2 文件，并拒绝越界路径。"""
        record = self._voxcpm2_candidate_record()
        relative_value = record.get("model_file")
        if not isinstance(relative_value, str) or not relative_value:
            raise LocalAIManagerError("VoxCPM2 清单缺少模型相对路径。")
        relative_path = Path(relative_value)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative_value
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", relative_value)
        ):
            raise LocalAIManagerError("VoxCPM2 模型路径必须位于 LocalAI models 目录内。")
        artifact_path = (self.models_dir / relative_path).resolve()
        if not artifact_path.is_relative_to(self.models_dir.resolve()):
            raise LocalAIManagerError("VoxCPM2 模型路径越出了 LocalAI models 目录。")
        try:
            size = artifact_path.stat().st_size
        except OSError as exc:
            raise LocalAIManagerError(
                "VoxCPM2 Q8 GGUF 尚未准备；请手动放入 LocalAI models 目录的 "
                f"{relative_value}。本命令不会下载模型。"
            ) from exc
        if not artifact_path.is_file() or size <= 0:
            raise LocalAIManagerError("VoxCPM2 GGUF 必须是非空常规文件。")
        return artifact_path

    def _voxcpm2_candidate_record(self) -> dict[str, object]:
        """读取唯一 VoxCPM2 制品记录并核验完整 HF commit。"""
        artifacts = self.manifest.get("model_artifacts")
        candidates = artifacts.get("voxcpm2_candidates") if isinstance(artifacts, dict) else None
        records = (
            [
                item
                for item in candidates or []
                if isinstance(item, dict)
                and item.get("capability") == "VoxCPM2 clone and controllable clone / 8322"
            ]
            if isinstance(candidates, list)
            else []
        )
        if len(records) != 1:
            raise LocalAIManagerError("版本清单必须且只能有一个 VoxCPM2 候选制品记录。")
        record = records[0]
        if not isinstance(record.get("repository_revision"), str) or not _COMMIT_RE.fullmatch(
            str(record["repository_revision"])
        ):
            raise LocalAIManagerError("VoxCPM2 候选缺少固定的 HF repository revision。")
        return record

    def register_confucius4_tts_candidate(self) -> None:
        """为手动准备的 Confucius4-TTS original GGUF 注册固定 audio.cpp YAML。"""
        artifact_path = self._confucius4_tts_artifact_path()
        record = self._confucius4_tts_candidate_record()
        rendered = CONFUCIUS4_TTS_CONFIG_PATH.read_text(encoding="utf-8")
        if rendered.count(f"  model: {record['model_file']}\n") != 1:
            raise LocalAIManagerError("Confucius4-TTS 候选 YAML 与版本清单中的路径不一致。")
        self._register_model_config(CONFUCIUS4_TTS_MODEL_NAME, rendered)
        self._write_json_atomically(
            self.models_dir / f"{CONFUCIUS4_TTS_MODEL_NAME}.artifact.json",
            {
                "schema_version": 1,
                "model": CONFUCIUS4_TTS_MODEL_NAME,
                "relative_path": record["model_file"],
                "size_bytes": artifact_path.stat().st_size,
                "sha256": self._sha256_file(artifact_path),
                "source_revision": record["repository_revision"],
            },
            prefix=".confucius4-tts-artifact-",
        )
        print(
            "已注册 E0 Confucius4-TTS original 候选；本地 hash 只检测后续文件变化，"
            "不等于上游制品验收。"
        )

    def verify_confucius4_tts_candidate(self) -> None:
        """核对 Confucius4-TTS 候选路径、配置和本地文件指纹。"""
        artifact_path = self._confucius4_tts_artifact_path()
        record = self._confucius4_tts_candidate_record()
        try:
            expected_config = CONFUCIUS4_TTS_CONFIG_PATH.read_bytes()
            installed_config = (self.models_dir / f"{CONFUCIUS4_TTS_MODEL_NAME}.yaml").read_bytes()
            stamp = json.loads(
                (self.models_dir / f"{CONFUCIUS4_TTS_MODEL_NAME}.artifact.json").read_text()
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError(
                "Confucius4-TTS 权重或固定 YAML 尚未注册；手动准备模型后运行 "
                "manager.py register-confucius4-tts-candidate。"
            ) from exc
        if expected_config.decode("utf-8").count(f"  model: {record['model_file']}\n") != 1:
            raise LocalAIManagerError("Confucius4-TTS YAML 与版本清单中的路径不一致。")
        if installed_config != expected_config:
            raise LocalAIManagerError("LocalAI Confucius4-TTS YAML 与固定模板不一致。")
        if (
            not isinstance(stamp, dict)
            or stamp.get("schema_version") != 1
            or stamp.get("model") != CONFUCIUS4_TTS_MODEL_NAME
            or stamp.get("relative_path") != record.get("model_file")
            or stamp.get("source_revision") != record.get("repository_revision")
        ):
            raise LocalAIManagerError("Confucius4-TTS 本地制品记录缺失或与固定 revision 不符。")
        if stamp.get("size_bytes") != artifact_path.stat().st_size or stamp.get(
            "sha256"
        ) != self._sha256_file(artifact_path):
            raise LocalAIManagerError("Confucius4-TTS GGUF 与注册时记录的本地 size/hash 不一致。")

    def _confucius4_tts_artifact_path(self) -> Path:
        """解析 LocalAI models 下的 Confucius4-TTS 文件并拒绝路径越界。"""
        record = self._confucius4_tts_candidate_record()
        relative_value = record.get("model_file")
        if not isinstance(relative_value, str) or not relative_value:
            raise LocalAIManagerError("Confucius4-TTS 清单缺少模型相对路径。")
        relative_path = Path(relative_value)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative_value
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", relative_value)
        ):
            raise LocalAIManagerError("Confucius4-TTS 模型路径必须位于 LocalAI models 目录内。")
        artifact_path = (self.models_dir / relative_path).resolve()
        if not artifact_path.is_relative_to(self.models_dir.resolve()):
            raise LocalAIManagerError("Confucius4-TTS 模型路径越出了 LocalAI models 目录。")
        try:
            size = artifact_path.stat().st_size
        except OSError as exc:
            raise LocalAIManagerError(
                "Confucius4-TTS original GGUF 尚未准备；请手动放入 LocalAI models 目录的 "
                f"{relative_value}。本命令不会下载模型。"
            ) from exc
        if not artifact_path.is_file() or size <= 0:
            raise LocalAIManagerError("Confucius4-TTS GGUF 必须是非空常规文件。")
        return artifact_path

    def _confucius4_tts_candidate_record(self) -> dict[str, object]:
        """读取唯一 Confucius4-TTS 制品记录并核验固定 HF commit。"""
        artifacts = self.manifest.get("model_artifacts")
        candidates = (
            artifacts.get("confucius4_tts_candidates") if isinstance(artifacts, dict) else None
        )
        records = (
            [
                item
                for item in candidates or []
                if isinstance(item, dict)
                and item.get("capability") == "Confucius4-TTS clone / 8361"
            ]
            if isinstance(candidates, list)
            else []
        )
        if len(records) != 1:
            raise LocalAIManagerError("版本清单必须且只能有一个 Confucius4-TTS 候选制品记录。")
        record = records[0]
        if not isinstance(record.get("repository_revision"), str) or not _COMMIT_RE.fullmatch(
            str(record["repository_revision"])
        ):
            raise LocalAIManagerError("Confucius4-TTS 候选缺少固定 HF repository revision。")
        return record

    def register_moss_voicegenerator_candidate(self) -> None:
        """为用户手动准备的 MOSS VoiceGenerator GGUF 注册固定 audio.cpp YAML。"""
        artifact_path = self._moss_voicegenerator_artifact_path()
        record = self._moss_voicegenerator_candidate_record()
        rendered = MOSS_VOICEGENERATOR_CONFIG_PATH.read_text(encoding="utf-8")
        if rendered.count(f"  model: {record['model_file']}\n") != 1:
            raise LocalAIManagerError("MOSS VoiceGenerator 候选 YAML 与清单中的路径不一致。")
        self._register_model_config(MOSS_VOICEGENERATOR_MODEL_NAME, rendered)
        self._write_json_atomically(
            self.models_dir / f"{MOSS_VOICEGENERATOR_MODEL_NAME}.artifact.json",
            {
                "schema_version": 1,
                "model": MOSS_VOICEGENERATOR_MODEL_NAME,
                "relative_path": record["model_file"],
                "size_bytes": artifact_path.stat().st_size,
                "sha256": self._sha256_file(artifact_path),
                "source_revision": record["repository_revision"],
            },
            prefix=".moss-voicegenerator-artifact-",
        )
        print(
            "已注册 E0 MOSS VoiceGenerator BF16 候选；本地 hash 仅检测后续变化，"
            "不等于上游制品验收。"
        )

    def verify_moss_voicegenerator_candidate(self) -> None:
        """核对 MOSS VoiceGenerator GGUF 路径、固定 YAML 和本地文件指纹。"""
        artifact_path = self._moss_voicegenerator_artifact_path()
        record = self._moss_voicegenerator_candidate_record()
        try:
            expected_config = MOSS_VOICEGENERATOR_CONFIG_PATH.read_bytes()
            installed_config = (
                self.models_dir / f"{MOSS_VOICEGENERATOR_MODEL_NAME}.yaml"
            ).read_bytes()
            stamp = json.loads(
                (self.models_dir / f"{MOSS_VOICEGENERATOR_MODEL_NAME}.artifact.json").read_text()
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError(
                "MOSS VoiceGenerator 权重或固定 YAML 尚未注册；手动准备模型后运行 "
                "manager.py register-moss-voicegenerator-candidate。"
            ) from exc
        if expected_config.decode("utf-8").count(f"  model: {record['model_file']}\n") != 1:
            raise LocalAIManagerError("MOSS VoiceGenerator YAML 与清单中的路径不一致。")
        if installed_config != expected_config:
            raise LocalAIManagerError("LocalAI MOSS VoiceGenerator YAML 与固定模板不一致。")
        if (
            not isinstance(stamp, dict)
            or stamp.get("schema_version") != 1
            or stamp.get("model") != MOSS_VOICEGENERATOR_MODEL_NAME
            or stamp.get("relative_path") != record.get("model_file")
            or stamp.get("source_revision") != record.get("repository_revision")
        ):
            raise LocalAIManagerError("MOSS VoiceGenerator 本地制品记录缺失或 revision 不一致。")
        if stamp.get("size_bytes") != artifact_path.stat().st_size or stamp.get(
            "sha256"
        ) != self._sha256_file(artifact_path):
            raise LocalAIManagerError("MOSS VoiceGenerator GGUF 与注册时记录的 size/hash 不一致。")

    def _moss_voicegenerator_artifact_path(self) -> Path:
        """解析固定 models 相对路径并拒绝 MOSS VoiceGenerator 权重越界。"""
        record = self._moss_voicegenerator_candidate_record()
        relative_value = record.get("model_file")
        if not isinstance(relative_value, str) or not relative_value:
            raise LocalAIManagerError("MOSS VoiceGenerator 清单缺少模型相对路径。")
        relative_path = Path(relative_value)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative_value
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", relative_value)
        ):
            raise LocalAIManagerError("MOSS VoiceGenerator 模型路径必须位于 models 目录内。")
        artifact_path = (self.models_dir / relative_path).resolve()
        if not artifact_path.is_relative_to(self.models_dir.resolve()):
            raise LocalAIManagerError("MOSS VoiceGenerator 模型路径越出了 models 目录。")
        try:
            size = artifact_path.stat().st_size
        except OSError as exc:
            raise LocalAIManagerError(
                "MOSS VoiceGenerator GGUF 尚未准备；请手动放入 LocalAI models 目录的 "
                f"{relative_value}。本命令不会下载模型。"
            ) from exc
        if not artifact_path.is_file() or size <= 0:
            raise LocalAIManagerError("MOSS VoiceGenerator GGUF 必须是非空常规文件。")
        return artifact_path

    def _moss_voicegenerator_candidate_record(self) -> dict[str, object]:
        """读取唯一 MOSS VoiceGenerator 候选并校验固定 HF commit。"""
        artifacts = self.manifest.get("model_artifacts")
        candidates = (
            artifacts.get("moss_voicegenerator_candidates") if isinstance(artifacts, dict) else None
        )
        records = (
            [
                item
                for item in candidates or []
                if isinstance(item, dict)
                and item.get("capability") == "MOSS-VoiceGenerator voice design / 8302"
            ]
            if isinstance(candidates, list)
            else []
        )
        if len(records) != 1:
            raise LocalAIManagerError("版本清单必须且只能有一个 MOSS VoiceGenerator 候选制品记录。")
        record = records[0]
        if not isinstance(record.get("repository_revision"), str) or not _COMMIT_RE.fullmatch(
            str(record["repository_revision"])
        ):
            raise LocalAIManagerError("MOSS VoiceGenerator 候选缺少固定 HF revision。")
        return record

    def register_fireredtts3_base_candidate(self) -> None:
        """注册用户手动准备的 FireRedTTS3 Base Q8 克隆模型。"""
        self._register_fireredtts3_candidate("base")

    def register_fireredtts3_instruct_candidate(self) -> None:
        """注册用户手动准备的 FireRedTTS3 Instruct Q8 音色设计模型。"""
        self._register_fireredtts3_candidate("instruct")

    def verify_fireredtts3_base_candidate(self) -> None:
        """校验已注册的 FireRedTTS3 Base Q8 克隆模型。"""
        self._verify_fireredtts3_candidate("base")

    def verify_fireredtts3_instruct_candidate(self) -> None:
        """校验已注册的 FireRedTTS3 Instruct Q8 音色设计模型。"""
        self._verify_fireredtts3_candidate("instruct")

    def register_ace_step_candidate(self) -> None:
        """注册用户手动准备的 ACE-Step XL Turbo Q8 DiT 候选。"""
        artifact_path = self._ace_step_artifact_path()
        record = self._ace_step_candidate_record()
        rendered = ACE_STEP_CONFIG_PATH.read_text(encoding="utf-8")
        if rendered.count(f"  model: {record['model_file']}\n") != 1:
            raise LocalAIManagerError("ACE-Step YAML 与固定版本清单中的模型路径不一致。")
        self._register_model_config(ACE_STEP_MODEL_NAME, rendered)
        self._write_json_atomically(
            self.models_dir / f"{ACE_STEP_MODEL_NAME}.artifact.json",
            {
                "schema_version": 1,
                "model": ACE_STEP_MODEL_NAME,
                "relative_path": record["model_file"],
                "size_bytes": artifact_path.stat().st_size,
                "sha256": self._sha256_file(artifact_path),
                "source_revision": record["repository_revision"],
            },
            prefix=".ace-step-artifact-",
        )
        print(
            "已注册 E0 ACE-Step XL Turbo Q8 DiT 候选；本地 hash 只检测后续文件变化，"
            "不等于上游制品验收。"
        )

    def verify_ace_step_candidate(self) -> None:
        """核对 ACE-Step Q8 DiT 候选路径、固定 YAML 和本地文件指纹。"""
        artifact_path = self._ace_step_artifact_path()
        record = self._ace_step_candidate_record()
        try:
            expected_config = ACE_STEP_CONFIG_PATH.read_bytes()
            installed_config = (self.models_dir / f"{ACE_STEP_MODEL_NAME}.yaml").read_bytes()
            stamp = json.loads(
                (self.models_dir / f"{ACE_STEP_MODEL_NAME}.artifact.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError(
                "ACE-Step XL Turbo 权重或固定 YAML 尚未注册；手动准备模型后运行 "
                "manager.py register-ace-step-candidate。"
            ) from exc
        if expected_config.decode("utf-8").count(f"  model: {record['model_file']}\n") != 1:
            raise LocalAIManagerError("ACE-Step YAML 与版本清单中的模型路径不一致。")
        if installed_config != expected_config:
            raise LocalAIManagerError("LocalAI ACE-Step YAML 与固定模板不一致。")
        if (
            not isinstance(stamp, dict)
            or stamp.get("schema_version") != 1
            or stamp.get("model") != ACE_STEP_MODEL_NAME
            or stamp.get("relative_path") != record.get("model_file")
            or stamp.get("source_revision") != record.get("repository_revision")
        ):
            raise LocalAIManagerError("ACE-Step 本地制品记录缺失或与固定 revision 不符。")
        if stamp.get("size_bytes") != artifact_path.stat().st_size or stamp.get(
            "sha256"
        ) != self._sha256_file(artifact_path):
            raise LocalAIManagerError("ACE-Step GGUF 与注册时记录的本地 size/hash 不一致。")

    def _ace_step_candidate_record(self) -> dict[str, object]:
        """读取唯一 ACE-Step XL Turbo Q8 DiT 制品并验证固定仓库 revision。"""
        artifacts = self.manifest.get("model_artifacts")
        candidates = artifacts.get("ace_step_candidates") if isinstance(artifacts, dict) else None
        records = (
            [
                item
                for item in candidates or []
                if isinstance(item, dict)
                and item.get("capability") == "ACE-Step 1.5 XL Turbo BGM / 8313"
            ]
            if isinstance(candidates, list)
            else []
        )
        if len(records) != 1:
            raise LocalAIManagerError("版本清单必须且只能有一个 ACE-Step XL Turbo 候选。")
        record = records[0]
        if not isinstance(record.get("repository_revision"), str) or not _COMMIT_RE.fullmatch(
            str(record["repository_revision"])
        ):
            raise LocalAIManagerError("ACE-Step 候选缺少固定模型仓库 commit。")
        return record

    def _ace_step_artifact_path(self) -> Path:
        """解析 models 内固定 ACE-Step GGUF 路径并拒绝路径越界。"""
        record = self._ace_step_candidate_record()
        relative_value = record.get("model_file")
        if not isinstance(relative_value, str) or not relative_value:
            raise LocalAIManagerError("ACE-Step 版本清单缺少模型相对路径。")
        relative_path = Path(relative_value)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative_value
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", relative_value)
        ):
            raise LocalAIManagerError("ACE-Step 模型路径必须位于 LocalAI models 目录内。")
        artifact_path = (self.models_dir / relative_path).resolve()
        if not artifact_path.is_relative_to(self.models_dir.resolve()):
            raise LocalAIManagerError("ACE-Step 模型路径越出了 LocalAI models 目录。")
        try:
            size = artifact_path.stat().st_size
        except OSError as exc:
            raise LocalAIManagerError(
                "ACE-Step XL Turbo Q8 DiT GGUF 尚未准备；请手动放入 LocalAI models 目录的 "
                f"{relative_value}。本命令不会下载模型。"
            ) from exc
        if not artifact_path.is_file() or size <= 0:
            raise LocalAIManagerError("ACE-Step GGUF 必须是非空常规文件。")
        return artifact_path

    def register_stable_audio_candidate(self) -> None:
        """注册用户手动准备的 Stable Audio 3 Medium Q8 GGUF 候选。"""
        artifact_path = self._stable_audio_artifact_path()
        record = self._stable_audio_candidate_record()
        template = STABLE_AUDIO_CONFIG_PATH.read_text(encoding="utf-8")
        if template.count(f"  model: {record['model_file']}\n") != 1:
            raise LocalAIManagerError("Stable Audio YAML 与固定版本清单中的模型路径不一致。")
        self._register_model_config(STABLE_AUDIO_MODEL_NAME, template)
        self._write_json_atomically(
            self.models_dir / f"{STABLE_AUDIO_MODEL_NAME}.artifact.json",
            {
                "schema_version": 1,
                "model": STABLE_AUDIO_MODEL_NAME,
                "relative_path": record["model_file"],
                "size_bytes": artifact_path.stat().st_size,
                "sha256": self._sha256_file(artifact_path),
                "source_revision": record["repository_revision"],
            },
            prefix=".stable-audio-artifact-",
        )
        print(
            "已注册 E0 Stable Audio 3 Medium Q8 候选；本地 hash 只检测文件后续变化，"
            "不证明上游制品来源。"
        )

    def verify_stable_audio_candidate(self) -> None:
        """核对 Stable Audio 固定 YAML、手动文件路径和本地 hash 指纹。"""
        artifact_path = self._stable_audio_artifact_path()
        record = self._stable_audio_candidate_record()
        try:
            expected_config = STABLE_AUDIO_CONFIG_PATH.read_bytes()
            installed_config = (self.models_dir / f"{STABLE_AUDIO_MODEL_NAME}.yaml").read_bytes()
            stamp = json.loads(
                (self.models_dir / f"{STABLE_AUDIO_MODEL_NAME}.artifact.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError(
                "Stable Audio 3 Medium 权重或固定 YAML 尚未注册；手动准备模型后运行 "
                "manager.py register-stable-audio-candidate。"
            ) from exc
        if expected_config.decode("utf-8").count(f"  model: {record['model_file']}\n") != 1:
            raise LocalAIManagerError("Stable Audio YAML 与版本清单中的模型路径不一致。")
        if installed_config != expected_config:
            raise LocalAIManagerError("LocalAI Stable Audio YAML 与固定模板不一致。")
        if (
            not isinstance(stamp, dict)
            or stamp.get("schema_version") != 1
            or stamp.get("model") != STABLE_AUDIO_MODEL_NAME
            or stamp.get("relative_path") != record.get("model_file")
            or stamp.get("source_revision") != record.get("repository_revision")
        ):
            raise LocalAIManagerError("Stable Audio 本地制品记录缺失或与固定 revision 不符。")
        if stamp.get("size_bytes") != artifact_path.stat().st_size or stamp.get(
            "sha256"
        ) != self._sha256_file(artifact_path):
            raise LocalAIManagerError("Stable Audio GGUF 与登记时记录的本地 size/hash 不一致。")

    def _stable_audio_candidate_record(self) -> dict[str, object]:
        """读取唯一 Stable Audio Medium 候选并核验固定包 revision。"""
        artifacts = self.manifest.get("model_artifacts")
        candidates = (
            artifacts.get("stable_audio_candidates") if isinstance(artifacts, dict) else None
        )
        records = (
            [
                item
                for item in candidates or []
                if isinstance(item, dict)
                and item.get("capability") == "Stable Audio 3 Medium generation / 8311"
            ]
            if isinstance(candidates, list)
            else []
        )
        if len(records) != 1:
            raise LocalAIManagerError("版本清单必须且只能有一个 Stable Audio Medium 候选。")
        record = records[0]
        if not isinstance(record.get("repository_revision"), str) or not _COMMIT_RE.fullmatch(
            str(record["repository_revision"])
        ):
            raise LocalAIManagerError("Stable Audio 候选缺少固定模型包 commit。")
        return record

    def _stable_audio_artifact_path(self) -> Path:
        """解析 Stable Audio Q8 GGUF 相对路径并拒绝 models 目录外路径。"""
        record = self._stable_audio_candidate_record()
        relative_value = record.get("model_file")
        if not isinstance(relative_value, str) or not relative_value:
            raise LocalAIManagerError("Stable Audio 版本清单缺少模型相对路径。")
        relative_path = Path(relative_value)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative_value
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", relative_value)
        ):
            raise LocalAIManagerError("Stable Audio 模型路径必须位于 LocalAI models 目录内。")
        artifact_path = (self.models_dir / relative_path).resolve()
        if not artifact_path.is_relative_to(self.models_dir.resolve()):
            raise LocalAIManagerError("Stable Audio 模型路径越出了 LocalAI models 目录。")
        try:
            size = artifact_path.stat().st_size
        except OSError as exc:
            raise LocalAIManagerError(
                "Stable Audio 3 Medium Q8 GGUF 尚未准备；请手动放入 LocalAI models 目录的 "
                f"{relative_value}。本命令不会下载模型。"
            ) from exc
        if not artifact_path.is_file() or size <= 0:
            raise LocalAIManagerError("Stable Audio GGUF 必须是非空常规文件。")
        return artifact_path

    def register_seed_vc_candidate(self) -> None:
        """登记用户手动准备的 Seed-VC 目录和 vc/svc 双任务配置。"""
        model_dir = self._seed_vc_model_dir()
        templates = (
            (SEED_VC_VC_MODEL_NAME, SEED_VC_VC_CONFIG_PATH),
            (SEED_VC_SVC_MODEL_NAME, SEED_VC_SVC_CONFIG_PATH),
        )
        rendered_configs = [(name, path.read_text(encoding="utf-8")) for name, path in templates]
        for model_name, rendered in rendered_configs:
            installed = self.models_dir / f"{model_name}.yaml"
            if installed.exists() and installed.read_text(encoding="utf-8") != rendered:
                raise LocalAIManagerError(f"拒绝覆盖不同的 Seed-VC LocalAI 配置：{installed}。")
        for model_name, rendered in rendered_configs:
            self._register_model_config(model_name, rendered)
        artifacts = self.manifest.get("model_artifacts")
        record = artifacts.get("seed_vc_candidate") if isinstance(artifacts, dict) else None
        if not isinstance(record, dict):
            raise LocalAIManagerError("版本清单缺少 Seed-VC 目录候选记录。")
        self._write_json_atomically(
            self.models_dir / SEED_VC_ARTIFACT_STAMP,
            {
                "schema_version": 1,
                "model_directory": record["model_directory"],
                "tree_sha256": self._sha256_tree(model_dir),
                "audio_cpp_source_commit": record["audio_cpp_source_commit"],
                "model_source_revision": record.get("model_source_revision"),
            },
            prefix=".seed-vc-artifact-",
        )
        print(
            "已登记 E0 Seed-VC 目录与 vc/svc 配置；目录 hash 只检测后续变更，"
            "不证明上游来源或推理兼容。"
        )

    def verify_seed_vc_candidate(self) -> None:
        """核对 Seed-VC 手动目录树指纹以及两种固定任务 YAML。"""
        model_dir = self._seed_vc_model_dir()
        artifacts = self.manifest.get("model_artifacts")
        record = artifacts.get("seed_vc_candidate") if isinstance(artifacts, dict) else None
        if not isinstance(record, dict):
            raise LocalAIManagerError("版本清单缺少 Seed-VC 目录候选记录。")
        try:
            stamp = json.loads(
                (self.models_dir / SEED_VC_ARTIFACT_STAMP).read_text(encoding="utf-8")
            )
            installed_vc = (self.models_dir / f"{SEED_VC_VC_MODEL_NAME}.yaml").read_bytes()
            installed_svc = (self.models_dir / f"{SEED_VC_SVC_MODEL_NAME}.yaml").read_bytes()
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError(
                "Seed-VC 模型目录或 vc/svc 配置尚未登记；手动准备模型后运行 "
                "manager.py register-seed-vc-candidate。本命令不会下载模型。"
            ) from exc
        if installed_vc != SEED_VC_VC_CONFIG_PATH.read_bytes():
            raise LocalAIManagerError("LocalAI Seed-VC vc YAML 与固定模板不一致。")
        if installed_svc != SEED_VC_SVC_CONFIG_PATH.read_bytes():
            raise LocalAIManagerError("LocalAI Seed-VC svc YAML 与固定模板不一致。")
        if (
            not isinstance(stamp, dict)
            or stamp.get("schema_version") != 1
            or stamp.get("model_directory") != record.get("model_directory")
            or stamp.get("audio_cpp_source_commit") != record.get("audio_cpp_source_commit")
            or stamp.get("model_source_revision") != record.get("model_source_revision")
        ):
            raise LocalAIManagerError("Seed-VC 本地目录指纹缺失或版本记录不匹配。")
        if stamp.get("tree_sha256") != self._sha256_tree(model_dir):
            raise LocalAIManagerError("Seed-VC 模型目录与登记时记录的 tree hash 不一致。")

    def _seed_vc_model_dir(self) -> Path:
        """解析 models 下的 Seed-VC 资源目录并检查其为非空安全目录。"""
        artifacts = self.manifest.get("model_artifacts")
        record = artifacts.get("seed_vc_candidate") if isinstance(artifacts, dict) else None
        relative_value = record.get("model_directory") if isinstance(record, dict) else None
        if not isinstance(relative_value, str) or not relative_value:
            raise LocalAIManagerError("Seed-VC 版本清单缺少模型目录。")
        relative_path = Path(relative_value)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative_value
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", relative_value)
        ):
            raise LocalAIManagerError("Seed-VC 模型目录必须位于 LocalAI models 目录内。")
        model_dir = (self.models_dir / relative_path).resolve()
        if not model_dir.is_relative_to(self.models_dir.resolve()):
            raise LocalAIManagerError("Seed-VC 模型目录越出了 LocalAI models 目录。")
        if not model_dir.is_dir():
            raise LocalAIManagerError(
                f"Seed-VC 资源目录尚未准备：{relative_value}。请手动准备，本命令不会下载模型。"
            )
        try:
            self._sha256_tree(model_dir)
        except (OSError, LocalAIManagerError) as exc:
            raise LocalAIManagerError(f"Seed-VC 资源目录不完整或包含不安全文件：{exc}") from exc
        if not any(path.is_file() for path in model_dir.rglob("*")):
            raise LocalAIManagerError("Seed-VC 资源目录必须包含模型文件。")
        return model_dir

    def _fireredtts3_spec(self, mode: str) -> tuple[str, Path, str]:
        if mode == "base":
            return (
                FIREREDTTS3_BASE_MODEL_NAME,
                FIREREDTTS3_BASE_CONFIG_PATH,
                "FireRedTTS3 Base clone / 8325",
            )
        if mode == "instruct":
            return (
                FIREREDTTS3_INSTRUCT_MODEL_NAME,
                FIREREDTTS3_INSTRUCT_CONFIG_PATH,
                "FireRedTTS3 Instruct voice design / 8304",
            )
        raise LocalAIManagerError(f"FireRedTTS3 模型模式无效：{mode}")

    def _fireredtts3_candidate_record(self, mode: str) -> dict[str, object]:
        """读取单个 FireRedTTS3 模式候选并验证固定 HF commit。"""
        _model_name, _config_path, capability = self._fireredtts3_spec(mode)
        artifacts = self.manifest.get("model_artifacts")
        candidates = (
            artifacts.get("fireredtts3_candidates") if isinstance(artifacts, dict) else None
        )
        records = (
            [
                item
                for item in candidates or []
                if isinstance(item, dict) and item.get("capability") == capability
            ]
            if isinstance(candidates, list)
            else []
        )
        if len(records) != 1:
            raise LocalAIManagerError(f"版本清单必须且只能有一个 FireRedTTS3 {mode} 候选制品记录。")
        record = records[0]
        if not isinstance(record.get("repository_revision"), str) or not _COMMIT_RE.fullmatch(
            str(record["repository_revision"])
        ):
            raise LocalAIManagerError(f"FireRedTTS3 {mode} 候选缺少固定 HF revision。")
        return record

    def _fireredtts3_artifact_path(self, mode: str) -> Path:
        """解析 models 内固定相对路径并拒绝 FireRedTTS3 权重越界。"""
        record = self._fireredtts3_candidate_record(mode)
        relative_value = record.get("model_file")
        if not isinstance(relative_value, str) or not relative_value:
            raise LocalAIManagerError(f"FireRedTTS3 {mode} 清单缺少模型相对路径。")
        relative_path = Path(relative_value)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative_value
            or not re.fullmatch(r"[A-Za-z0-9._/-]+", relative_value)
        ):
            raise LocalAIManagerError(f"FireRedTTS3 {mode} 模型路径必须位于 models 目录内。")
        artifact_path = (self.models_dir / relative_path).resolve()
        if not artifact_path.is_relative_to(self.models_dir.resolve()):
            raise LocalAIManagerError(f"FireRedTTS3 {mode} 模型路径越出了 models 目录。")
        try:
            size = artifact_path.stat().st_size
        except OSError as exc:
            raise LocalAIManagerError(
                f"FireRedTTS3 {mode} Q8 GGUF 尚未准备；请手动放入 LocalAI models 目录的 "
                f"{relative_value}。本命令不会下载模型。"
            ) from exc
        if not artifact_path.is_file() or size <= 0:
            raise LocalAIManagerError(f"FireRedTTS3 {mode} GGUF 必须是非空常规文件。")
        return artifact_path

    def _register_fireredtts3_candidate(self, mode: str) -> None:
        model_name, config_path, _capability = self._fireredtts3_spec(mode)
        artifact_path = self._fireredtts3_artifact_path(mode)
        record = self._fireredtts3_candidate_record(mode)
        rendered = config_path.read_text(encoding="utf-8")
        if rendered.count(f"  model: {record['model_file']}\n") != 1:
            raise LocalAIManagerError(f"FireRedTTS3 {mode} 候选 YAML 与版本清单路径不一致。")
        self._register_model_config(model_name, rendered)
        self._write_json_atomically(
            self.models_dir / f"{model_name}.artifact.json",
            {
                "schema_version": 1,
                "model": model_name,
                "relative_path": record["model_file"],
                "size_bytes": artifact_path.stat().st_size,
                "sha256": self._sha256_file(artifact_path),
                "source_revision": record["repository_revision"],
            },
            prefix=f".{model_name}-artifact-",
        )
        print(
            f"已注册 E0 FireRedTTS3 {mode} Q8 候选；本地 hash 仅检测后续文件变化，"
            "不等于上游制品验收。"
        )

    def _verify_fireredtts3_candidate(self, mode: str) -> None:
        model_name, config_path, _capability = self._fireredtts3_spec(mode)
        artifact_path = self._fireredtts3_artifact_path(mode)
        record = self._fireredtts3_candidate_record(mode)
        try:
            expected_config = config_path.read_bytes()
            installed_config = (self.models_dir / f"{model_name}.yaml").read_bytes()
            stamp = json.loads(
                (self.models_dir / f"{model_name}.artifact.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError(
                f"FireRedTTS3 {mode} 权重或固定 YAML 尚未注册；手动准备模型后运行对应 register 命令。"
            ) from exc
        if expected_config.decode("utf-8").count(f"  model: {record['model_file']}\n") != 1:
            raise LocalAIManagerError(f"FireRedTTS3 {mode} YAML 与版本清单路径不一致。")
        if installed_config != expected_config:
            raise LocalAIManagerError(f"LocalAI FireRedTTS3 {mode} YAML 与固定模板不一致。")
        if (
            not isinstance(stamp, dict)
            or stamp.get("schema_version") != 1
            or stamp.get("model") != model_name
            or stamp.get("relative_path") != record.get("model_file")
            or stamp.get("source_revision") != record.get("repository_revision")
        ):
            raise LocalAIManagerError(f"FireRedTTS3 {mode} 本地制品记录缺失或 revision 不一致。")
        if stamp.get("size_bytes") != artifact_path.stat().st_size or stamp.get(
            "sha256"
        ) != self._sha256_file(artifact_path):
            raise LocalAIManagerError(f"FireRedTTS3 {mode} GGUF 与注册时记录的 size/hash 不一致。")

    def _write_json_atomically(
        self, destination: Path, payload: dict[str, object], *, prefix: str
    ) -> None:
        """在指定目录原子保存 JSON 安装记录。"""
        temporary_path = self._write_temporary_file(destination.parent, prefix, ".json")
        try:
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            self._fsync_file(temporary_path)
            os.replace(temporary_path, destination)
            self._fsync_directory(destination.parent)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _register_model_config(self, model_name: str, rendered: str) -> None:
        """原子注册模型 YAML，拒绝覆盖内容不同的既有配置。"""
        destination = self.models_dir / f"{model_name}.yaml"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_text(encoding="utf-8") != rendered:
                raise LocalAIManagerError(
                    f"拒绝覆盖不同的 LocalAI 模型配置：{destination}；请先人工检查并移除旧配置。"
                )
            print(f"模型配置已匹配：{destination}")
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{model_name}-", suffix=".yaml.tmp", dir=self.models_dir
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(rendered)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, destination)
            self._fsync_directory(self.models_dir)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"已注册 E0 候选配置（不是推理验收）：{destination}")

    def start(self) -> None:
        """以独立进程组启动只监听 loopback 的 LocalAI，并等待 readiness。"""
        if not self._enabled():
            raise LocalAIManagerError("启动前必须显式设置 UNITALE_LOCALAI_ENABLED=true。")
        self._ensure_linux_amd64()
        self._ensure_install_stamp()
        self._ensure_directories()
        base_url, port = self._base_url_and_port()
        self._ensure_port_available(port)
        self._ensure_no_managed_processes()
        if self._read_state(required=False) is not None:
            raise LocalAIManagerError("受管 LocalAI 已有进程状态；先检查并显式执行 stop。")

        owner_pid: int | None = None
        owner_start_time: str | None = None
        raw_owner_pid = self.environ.get("UNITALE_LOCALAI_OWNER_PID")
        if raw_owner_pid:
            try:
                owner_pid = int(raw_owner_pid)
            except ValueError as exc:
                raise LocalAIManagerError("UNITALE_LOCALAI_OWNER_PID 必须是进程号。") from exc
            owner_start_time = self._process_start_time(owner_pid)
            if owner_start_time is None:
                raise LocalAIManagerError("LocalAI 启动所有者已退出。")

        run_id = uuid.uuid4().hex
        log_path = self.logs_dir / "localai.log"
        localai_environment = {
            **os.environ,
            **self.environ,
            "LOCALAI_ADDRESS": f"127.0.0.1:{port}",
            "LOCALAI_MODELS_PATH": str(self.models_dir.resolve()),
            "LOCALAI_BACKENDS_PATH": str(self.backends_dir.resolve()),
            "LOCALAI_MAX_ACTIVE_BACKENDS": "1",
            "LOCALAI_AUTOLOAD_GALLERIES": "false",
            "LOCALAI_AUTOLOAD_BACKEND_GALLERIES": "false",
            "LOCALAI_DISABLE_GALLERY_ENDPOINT": "true",
            "LOCALAI_DISABLE_WEBUI": "true",
            "UNITALE_LOCALAI_RUN_ID": run_id,
        }
        runtime_cache = Path(
            localai_environment.get(
                "RUNTIME_CACHE_DIR",
                str(
                    Path(localai_environment.get("STORAGE_DIR", ROOT / "storage"))
                    / ".cache/runtime"
                ),
            )
        ).expanduser()
        localai_environment.setdefault(
            "UNITALE_LOCALAI_ARTIFACT_DIR", str(runtime_cache / "localai_artifacts")
        )
        worker_bridge_enabled = self._worker_bridge_enabled()
        command = [str(self.binary_path), "run"]
        if worker_bridge_enabled:
            bridge_address, bridge_port = self._bridge_address_and_port()
            self._ensure_port_available(bridge_port)
            localai_environment["UNITALE_LOCALAI_WORKER_BRIDGE_ENABLED"] = "true"
            localai_environment["UNITALE_LOCALAI_BRIDGE_ADDRESS"] = bridge_address
            worker_manifest = Path(
                self.environ.get(
                    "UNITALE_LOCALAI_BRIDGE_WORKERS",
                    str(ROOT / "localai/manifests/worker-bridge.json"),
                )
            ).expanduser()
            if not worker_manifest.is_file():
                raise LocalAIManagerError(
                    f"LocalAI bridge worker manifest 不存在：{worker_manifest}"
                )
            localai_environment["UNITALE_LOCALAI_BRIDGE_WORKERS"] = str(worker_manifest.resolve())
            backend_entries = [
                entry
                for entry in localai_environment.get("LOCALAI_EXTERNAL_GRPC_BACKENDS", "").split(
                    ","
                )
                if entry and entry.split(":", 1)[0] != "unitale-worker"
            ]
            backend_entries.append(f"unitale-worker:{bridge_address}")
            localai_environment["LOCALAI_EXTERNAL_GRPC_BACKENDS"] = ",".join(backend_entries)
            command = [sys.executable, str(ROOT / "scripts/localai/host.py"), str(self.binary_path)]
        if api_key := self.environ.get("UNITALE_LOCALAI_API_KEY"):
            localai_environment["LOCALAI_API_KEY"] = api_key
        try:
            process_id = self._spawn_detached(command, localai_environment, stdout_path=log_path)
        except OSError as exc:
            raise LocalAIManagerError(f"无法启动宿主机 LocalAI：{exc}") from exc

        process_start_time = self._process_start_time(process_id)
        if process_start_time is None:
            self._reap_spawned_child(process_id)
            raise LocalAIManagerError("LocalAI 子进程启动后立即退出。")
        state: dict[str, object] = {
            "schema_version": 1,
            "boot_id": self._boot_id(),
            "pid": process_id,
            "pgid": process_id,
            "process_start_time": process_start_time,
            "owner_pid": owner_pid,
            "owner_start_time": owner_start_time,
            "run_id": run_id,
            "base_url": base_url,
            "command": command,
            "bridge_address": (
                localai_environment.get("UNITALE_LOCALAI_BRIDGE_ADDRESS")
                if worker_bridge_enabled
                else None
            ),
            "started_at": time.time(),
        }
        try:
            self._atomic_write_json(self.state_path, state)
            if owner_pid is not None and owner_start_time is not None:
                self._start_owner_watcher(owner_pid, owner_start_time, state)
            self._wait_until_ready(base_url)
        except BaseException:
            self._terminate_process_group(state)
            self.state_path.unlink(missing_ok=True)
            raise
        print(f"LocalAI 宿主机进程已就绪：{base_url}（PID {process_id}）")

    def stop(self) -> None:
        """终止受管进程组并确认所有组成员退出且 API 端口关闭。"""
        state = self._read_state(required=False)
        if state is None:
            self._ensure_no_managed_processes()
            self._ensure_port_closed()
            print("受管 LocalAI 未运行。")
            return
        self._terminate_process_group(state)
        self._ensure_bridge_port_closed(state.get("bridge_address"))
        self.state_path.unlink(missing_ok=True)
        self._fsync_directory(self.state_dir)
        print("LocalAI 进程组已停止并确认回收。")

    def status(self) -> None:
        """报告受管进程及监听端口状态，不触发模型加载。"""
        state = self._read_state(required=False)
        if state is None:
            self._ensure_no_managed_processes()
            self._ensure_port_closed()
            print("受管 LocalAI 未运行。")
            return
        pid = int(state["pid"])
        if self._process_start_time(pid) != str(state["process_start_time"]):
            raise LocalAIManagerError("LocalAI 状态文件中的 PID 已失效或发生复用；保留状态供排查。")
        print(f"LocalAI 正在运行：PID {pid}，地址 {state['base_url']}")

    def verify_managed_localai_stopped(self) -> None:
        """确认受管 LocalAI 进程组不存在且 loopback API 端口关闭。"""
        state = self._read_state(required=False)
        if state is not None:
            raise LocalAIManagerError("受管 LocalAI 状态仍存在；先运行 manager.py stop。")
        self._ensure_no_managed_processes()
        self._ensure_port_closed()
        if self._worker_bridge_enabled():
            address, _port = self._bridge_address_and_port()
            self._ensure_bridge_port_closed(address)

    @staticmethod
    def _ensure_port_available(port: int) -> None:
        """在启动前检测目标 loopback 端口是否已占用。"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise LocalAIManagerError(f"LocalAI loopback 端口 {port} 已被占用。") from exc

    def _worker_bridge_enabled(self) -> bool:
        """读取是否将固定的一次性 worker bridge 纳入受管 LocalAI 进程组。"""
        return self.environ.get(
            "UNITALE_LOCALAI_WORKER_BRIDGE_ENABLED", "false"
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _bridge_address_and_port(self) -> tuple[str, int]:
        """读取并限制 bridge gRPC 监听在本机回环。"""
        address = self.environ.get("UNITALE_LOCALAI_BRIDGE_ADDRESS", "127.0.0.1:50052")
        host, separator, raw_port = address.rpartition(":")
        if not separator or host != "127.0.0.1":
            raise LocalAIManagerError("UNITALE_LOCALAI_BRIDGE_ADDRESS 必须是 127.0.0.1 回环地址。")
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise LocalAIManagerError("LocalAI bridge gRPC 端口无效。") from exc
        if not 1 <= port <= 65535:
            raise LocalAIManagerError("LocalAI bridge gRPC 端口超出有效范围。")
        return address, port

    @staticmethod
    def _ensure_bridge_port_closed(address: object) -> None:
        """当实例启用了外部 backend 时，确认它的 gRPC 监听也已关闭。"""
        if not isinstance(address, str):
            return
        host, separator, raw_port = address.rpartition(":")
        if not separator or host != "127.0.0.1":
            raise LocalAIManagerError("状态文件中的 LocalAI bridge 地址无效。")
        try:
            port = int(raw_port)
            connection = socket.create_connection((host, port), timeout=0.5)
        except OSError:
            return
        connection.close()
        raise LocalAIManagerError(f"LocalAI worker bridge 端口仍可连接：{address}")

    def _spawn_detached(
        self,
        command: list[str],
        environment: Mapping[str, str],
        *,
        stdout_path: Path | None = None,
    ) -> int:
        """使用 posix_spawn 建立独立会话，避免长驻子进程的 Popen 析构告警。"""
        setsid_path = shutil.which("setsid")
        if setsid_path is None:
            raise FileNotFoundError("找不到 setsid；请按仓库部署要求安装 util-linux。")
        output_path = str(stdout_path) if stdout_path is not None else os.devnull
        output_flags = os.O_WRONLY
        if stdout_path is not None:
            output_flags |= os.O_CREAT | os.O_APPEND
        file_actions = [
            (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),
            (os.POSIX_SPAWN_OPEN, 1, output_path, output_flags, 0o600),
            (os.POSIX_SPAWN_DUP2, 1, 2),
        ]
        process_id = os.posix_spawn(
            setsid_path,
            [setsid_path, "--", *command],
            dict(environment),
            file_actions=file_actions,
        )
        try:
            self._ensure_detached_process_group(process_id)
        except BaseException:
            try:
                os.kill(process_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._reap_spawned_child(process_id)
            raise
        return process_id

    def _ensure_detached_process_group(self, process_id: int) -> None:
        """等 setsid exec 完成，确认 PID、PGID、SID 一致后再记录进程状态。"""
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if os.getpgid(process_id) == process_id and os.getsid(process_id) == process_id:
                    return
            except ProcessLookupError:
                break
            time.sleep(0.01)
        raise LocalAIManagerError("setsid 子进程未能建立独立会话和进程组。")

    def _reap_spawned_child(self, process_id: int) -> None:
        """有界回收本进程直接启动的子进程；独立 stop 命令会收到 ChildProcessError。"""
        deadline = time.monotonic() + self.stop_timeout
        while True:
            try:
                waited_pid, _ = os.waitpid(process_id, os.WNOHANG)
            except ChildProcessError:
                return
            if waited_pid == process_id:
                return
            if time.monotonic() >= deadline:
                raise LocalAIManagerError("LocalAI 子进程退出后未能在时限内回收。")
            time.sleep(0.05)

    def _localai_version_output(self) -> str:
        """读取本机二进制版本，并在启动前确认文件可执行。"""
        if not self.binary_path.is_file() or not os.access(self.binary_path, os.X_OK):
            raise LocalAIManagerError(
                f"LocalAI 二进制不存在或不可执行：{self.binary_path}；"
                "请按固定源码版本准备宿主机二进制后重试。"
            )
        try:
            result = subprocess.run(
                [str(self.binary_path), "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=_COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise LocalAIManagerError("LocalAI --version 检查超时。") from exc
        except OSError as exc:
            raise LocalAIManagerError(f"无法读取 LocalAI 二进制版本：{exc}") from exc
        output = f"{result.stdout}\n{result.stderr}".strip()
        if result.returncode != 0 or self.localai_version not in output:
            raise LocalAIManagerError(
                f"LocalAI 二进制版本检查失败；期望 {self.localai_version}，输出：{output}"
            )
        return output

    def _backend_run_script(self) -> Path:
        """查找并校验 LocalAI audio-cpp backend 的启动入口。"""
        backend_dir = self.backends_dir / "audio-cpp"
        run_script = backend_dir / "run.sh"
        try:
            resolved_dir = backend_dir.resolve(strict=True)
            resolved_script = run_script.resolve(strict=True)
        except OSError as exc:
            raise LocalAIManagerError(
                "缺少 audio-cpp 原生 backend 包；需准备 run.sh 到 LocalAI backends/audio-cpp。"
            ) from exc
        if not resolved_dir.is_relative_to(self.backends_dir.resolve()):
            raise LocalAIManagerError("audio-cpp backend 目录不能指向 backends 路径之外。")
        if not resolved_script.is_relative_to(resolved_dir):
            raise LocalAIManagerError("audio-cpp backend run.sh 不能逃逸其固定目录。")
        if not resolved_script.is_file() or not os.access(resolved_script, os.X_OK):
            raise LocalAIManagerError("audio-cpp backend run.sh 不存在或不可执行。")
        return resolved_script

    def _write_install_stamp(self) -> None:
        """校验 LocalAI 与后端文件后写入其内容指纹。"""
        output = self._localai_version_output()
        run_script = self._backend_run_script()
        stamp = {
            "localai_version": self.localai_version,
            "expected_localai_source_commit": self.localai_commit,
            "localai_binary": str(self.binary_path.resolve()),
            "localai_binary_sha256": self._sha256_file(self.binary_path),
            "localai_version_output": output,
            "expected_audio_cpp_source_commit": self.audio_cpp_commit,
            "audio_cpp_backend_run_script": str(run_script),
            "audio_cpp_backend_run_script_sha256": self._sha256_file(run_script),
            "audio_cpp_backend_tree_sha256": self._sha256_tree(self.backends_dir / "audio-cpp"),
            "compute_variant": "CUDA 12",
        }
        self._atomic_write_json(self.install_stamp_path, stamp)

    def _ensure_install_stamp(self) -> None:
        """拒绝运行版本未核验或文件内容发生变化的本机安装。"""
        try:
            stamp = json.loads(self.install_stamp_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError(
                "缺少 LocalAI/audio-cpp 校验记录；先运行 manager.py verify-install。"
            ) from exc
        current_version = self._localai_version_output()
        run_script = self._backend_run_script()
        expected = {
            "localai_version": self.localai_version,
            "expected_localai_source_commit": self.localai_commit,
            "localai_binary": str(self.binary_path.resolve()),
            "localai_binary_sha256": self._sha256_file(self.binary_path),
            "localai_version_output": current_version,
            "expected_audio_cpp_source_commit": self.audio_cpp_commit,
            "audio_cpp_backend_run_script": str(run_script),
            "audio_cpp_backend_run_script_sha256": self._sha256_file(run_script),
            "audio_cpp_backend_tree_sha256": self._sha256_tree(self.backends_dir / "audio-cpp"),
            "compute_variant": "CUDA 12",
        }
        if stamp != expected:
            raise LocalAIManagerError(
                "本机 LocalAI/audio-cpp 文件与校验记录不一致；重新运行 verify-install。"
            )

    def _ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.models_dir,
            self.backends_dir,
            self.logs_dir,
            self.state_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ensure_linux_amd64() -> None:
        if sys.platform != "linux" or os.uname().machine not in {"x86_64", "amd64"}:
            raise LocalAIManagerError("固定宿主机二进制方案仅支持 linux/amd64。")

    def _base_url_and_port(self) -> tuple[str, int]:
        raw_url = self.environ.get("UNITALE_LOCALAI_BASE_URL", "http://127.0.0.1:8400")
        parsed = urlsplit(raw_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise LocalAIManagerError("受管 LocalAI 仅允许无路径的 HTTP loopback 地址。")
        try:
            port = parsed.port or 80
        except ValueError as exc:
            raise LocalAIManagerError("UNITALE_LOCALAI_BASE_URL 端口无效。") from exc
        if not 1 <= port <= 65535:
            raise LocalAIManagerError("UNITALE_LOCALAI_BASE_URL 端口超出有效范围。")
        return f"http://{parsed.hostname}:{port}", port

    def _ensure_port_closed(self) -> None:
        base_url, port = self._base_url_and_port()
        try:
            connection = socket.create_connection(("127.0.0.1", port), timeout=0.5)
        except OSError:
            return
        connection.close()
        raise LocalAIManagerError(f"LocalAI 状态文件不存在，但 loopback 端口仍可连接：{base_url}")

    def _managed_process_ids(self) -> list[int]:
        """查找命令行或可执行路径仍指向本地安装目录的 LocalAI/backend 进程。"""
        markers = (
            str(self.binary_path.resolve()).encode(),
            str(self.backends_dir.resolve()).encode(),
            str((ROOT / "scripts/localai/host.py").resolve()).encode(),
            b"unitale_localai_bridge.server",
        )
        process_ids: list[int] = []
        for process_path in Path("/proc").glob("[0-9]*"):
            try:
                process_id = int(process_path.name)
            except ValueError:
                continue
            stat = self._read_process_stat(process_id)
            if stat is None or stat[0] in {"Z", "X"}:
                continue
            try:
                command_line = (process_path / "cmdline").read_bytes()
                executable = os.readlink(process_path / "exe").encode()
            except OSError:
                continue
            if any(marker in command_line or marker in executable for marker in markers):
                process_ids.append(process_id)
        return process_ids

    def _ensure_no_managed_processes(self) -> None:
        process_ids = self._managed_process_ids()
        if process_ids:
            raise LocalAIManagerError(
                f"状态文件缺失但仍发现受管 LocalAI/backend 进程：{process_ids}。"
            )

    def _read_state(self, *, required: bool) -> dict[str, object] | None:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if required:
                raise LocalAIManagerError("LocalAI 进程状态文件不存在。") from None
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalAIManagerError("LocalAI 状态文件损坏；保留隔离并人工检查。") from exc
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise LocalAIManagerError("LocalAI 状态文件格式无效；保留隔离并人工检查。")
        try:
            pid = int(state["pid"])
            pgid = int(state["pgid"])
            boot_id = state["boot_id"]
            start_time = state["process_start_time"]
            base_url = state["base_url"]
            run_id = state["run_id"]
        except (KeyError, TypeError, ValueError) as exc:
            raise LocalAIManagerError("LocalAI 状态文件缺少进程身份字段。") from exc
        if (
            pid <= 1
            or pgid != pid
            or not isinstance(boot_id, str)
            or not re.fullmatch(r"[0-9a-f-]{36}", boot_id)
            or not isinstance(start_time, str)
        ):
            raise LocalAIManagerError("LocalAI 状态文件中的进程身份无效。")
        if not isinstance(base_url, str) or not isinstance(run_id, str):
            raise LocalAIManagerError("LocalAI 状态文件中的地址或运行标识无效。")
        expected_url, _ = self._base_url_and_port()
        if base_url != expected_url or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise LocalAIManagerError("LocalAI 状态文件与当前实例配置不匹配。")
        return state

    def _terminate_process_group(self, state: Mapping[str, object]) -> None:
        """按 PID 启动时间和运行标识验证进程组，再执行 TERM/KILL 回收。"""
        if state.get("boot_id") not in {None, self._boot_id()}:
            self._ensure_no_managed_processes()
            self._ensure_port_closed()
            return
        pid = int(state["pid"])
        pgid = int(state["pgid"])
        start_time = str(state["process_start_time"])
        run_id = str(state["run_id"])
        current_start = self._process_start_time(pid)
        if current_start is not None and current_start != start_time:
            raise LocalAIManagerError("LocalAI PID 已被复用；拒绝向该进程组发送信号。")
        if self._process_group_alive(pgid):
            if current_start != start_time and not self._group_has_run_id(pgid, run_id):
                raise LocalAIManagerError("无法确认遗留进程组归属；保留状态供人工恢复。")
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self._wait_for_process_group(pgid, self.stop_timeout)
            if self._process_group_alive(pgid):
                if not self._group_has_run_id(pgid, run_id) and current_start != start_time:
                    raise LocalAIManagerError("进程组身份无法复核；不发送 SIGKILL。")
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._wait_for_process_group(pgid, self.stop_timeout)
        if self._process_group_alive(pgid):
            raise LocalAIManagerError("LocalAI 进程组在 SIGKILL 后仍有活动进程。")
        if state.get("bridge_address") is not None:
            self._reap_worker_bridge_processes()
        self._ensure_port_closed()
        self._ensure_bridge_port_closed(state.get("bridge_address"))
        self._reap_spawned_child(pid)

    def _reap_worker_bridge_processes(self) -> None:
        """回收 bridge 被强制终止后脱离宿主机进程组的一次性模型 worker。"""
        expected_scripts = {
            str((ROOT / project / "worker.py").resolve()) for project in WORKER_BRIDGE_PROJECTS
        }
        groups: dict[int, dict[int, str]] = {}
        for process_path in Path("/proc").glob("[0-9]*"):
            try:
                process_id = int(process_path.name)
                arguments = process_path.joinpath("cmdline").read_bytes().split(b"\0")
                decoded_arguments = [
                    argument.decode("utf-8", errors="replace") for argument in arguments
                ]
                process_environment = process_path.joinpath("environ").read_bytes().split(b"\0")
                stat = self._read_process_stat(process_id)
                process_group = os.getpgid(process_id)
            except (OSError, ProcessLookupError, ValueError):
                continue
            if (
                stat is None
                or stat[0] in {"Z", "X"}
                or "--input-json" not in decoded_arguments
                or not expected_scripts.intersection(decoded_arguments)
                or not any(
                    entry.startswith(b"UNITALE_LOCALAI_REQUEST_ID=")
                    for entry in process_environment
                )
                or process_group <= 1
                or process_group == os.getpgrp()
            ):
                continue
            groups.setdefault(process_group, {})[process_id] = stat[2]

        for process_group, process_starts in groups.items():
            if not any(
                self._process_start_time(process_id) == start_time
                for process_id, start_time in process_starts.items()
            ):
                continue
            try:
                os.killpg(process_group, signal.SIGTERM)
            except ProcessLookupError:
                continue
            self._wait_for_process_group(process_group, min(self.stop_timeout, 3.0))
            if self._process_group_alive(process_group):
                try:
                    os.killpg(process_group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._wait_for_process_group(process_group, min(self.stop_timeout, 3.0))
            if self._process_group_alive(process_group):
                raise LocalAIManagerError(
                    f"LocalAI worker bridge 子进程组 {process_group} 未能确认退出；保留运行状态。"
                )

    def _wait_for_process_group(self, pgid: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while self._process_group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _read_process_stat(pid: int) -> tuple[str, int, str] | None:
        try:
            raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return None
        closing_parenthesis = raw_stat.rfind(")")
        if closing_parenthesis < 0:
            return None
        fields = raw_stat[closing_parenthesis + 1 :].split()
        if len(fields) <= 19:
            return None
        return fields[0], int(fields[2]), fields[19]

    @staticmethod
    def _boot_id() -> str:
        """返回 Linux boot ID，避免重启后 PID/启动 tick 复用旧运行状态。"""
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        except OSError as exc:
            raise LocalAIManagerError(
                "无法读取 Linux boot ID；拒绝管理无法识别的进程状态。"
            ) from exc

    @classmethod
    def _process_start_time(cls, process_id: int) -> str | None:
        """返回 Linux 进程启动 tick；已退出或 zombie 时返回 None。"""
        stat = cls._read_process_stat(process_id)
        if stat is None or stat[0] in {"Z", "X"}:
            return None
        return stat[2]

    @classmethod
    def _process_group_alive(cls, pgid: int) -> bool:
        for process_path in Path("/proc").glob("[0-9]*"):
            try:
                process_id = int(process_path.name)
            except ValueError:
                continue
            stat = cls._read_process_stat(process_id)
            if stat is not None and stat[0] not in {"Z", "X"} and stat[1] == pgid:
                return True
        return False

    @classmethod
    def _group_has_run_id(cls, pgid: int, run_id: str) -> bool:
        expected = f"UNITALE_LOCALAI_RUN_ID={run_id}".encode()
        for process_path in Path("/proc").glob("[0-9]*"):
            try:
                process_id = int(process_path.name)
            except ValueError:
                continue
            stat = cls._read_process_stat(process_id)
            if stat is None or stat[0] in {"Z", "X"} or stat[1] != pgid:
                continue
            try:
                if expected in (process_path / "environ").read_bytes().split(b"\0"):
                    return True
            except OSError:
                continue
        return False

    def _start_owner_watcher(
        self, owner_pid: int, owner_start_time: str, state: Mapping[str, object]
    ) -> None:
        """启动独立 watcher，在 start.sh 异常退出时回收宿主机进程组。"""
        child_environment = {**os.environ, **self.environ}
        child_environment.pop("UNITALE_LOCALAI_API_KEY", None)
        child_environment.pop("LOCALAI_API_KEY", None)
        child_environment.pop("UNITALE_LOCALAI_OWNER_PID", None)
        try:
            self._spawn_detached(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "watch",
                    str(owner_pid),
                    owner_start_time,
                    str(state["pid"]),
                    str(state["process_start_time"]),
                ],
                child_environment,
            )
        except OSError as exc:
            self._terminate_process_group(state)
            self.state_path.unlink(missing_ok=True)
            raise LocalAIManagerError(f"无法启动 LocalAI 所有者 watcher：{exc}") from exc

    def watch_owner(
        self, owner_pid: int, owner_start_time: str, process_pid: int, process_start_time: str
    ) -> None:
        """监控启动脚本所有者；所有者退出后回收所记录的 LocalAI 进程组。"""
        while True:
            state = self._read_state(required=False)
            if state is None:
                return
            if (
                int(state["pid"]) != process_pid
                or str(state["process_start_time"]) != process_start_time
            ):
                return
            if self._process_start_time(process_pid) is None and not self._process_group_alive(
                process_pid
            ):
                return
            if self._process_start_time(owner_pid) != owner_start_time:
                self._terminate_process_group(state)
                self.state_path.unlink(missing_ok=True)
                return
            time.sleep(1)

    def _wait_until_ready(self, base_url: str) -> None:
        """以有界轮询确认 LocalAI HTTP API 已开始服务。"""
        deadline = time.monotonic() + self.readiness_timeout
        last_error: Exception | None = None
        health_url = f"{base_url}/readyz"
        headers: dict[str, str] = {}
        if api_key := self.environ.get("UNITALE_LOCALAI_API_KEY"):
            headers["Authorization"] = f"Bearer {api_key}"
        while True:
            state = self._read_state(required=False)
            if state is None or self._process_start_time(int(state["pid"])) is None:
                raise LocalAIManagerError("LocalAI 在 readiness 成功前退出。") from last_error
            try:
                with urlopen(Request(health_url, headers=headers), timeout=1.0) as response:
                    if 200 <= response.status < 300:
                        return
                    last_error = LocalAIManagerError(
                        f"LocalAI readiness 返回 HTTP {response.status}。"
                    )
            except (OSError, URLError, TimeoutError) as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                raise LocalAIManagerError(
                    f"等待 LocalAI 就绪超时（{self.readiness_timeout:g}s）。"
                ) from last_error
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """按固定块大小校验大文件，避免把权重或二进制整体读入内存。"""
        digest = hashlib.sha256()
        with path.open("rb") as artifact:
            while chunk := artifact.read(4 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _sha256_tree(cls, root: Path) -> str:
        """对 backend 目录内容做稳定 hash，并拒绝越界软链接和特殊文件。"""
        resolved_root = root.resolve(strict=True)
        entries: list[dict[str, object]] = []
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            relative_path = path.relative_to(root).as_posix()
            if path.is_symlink():
                try:
                    resolved_path = path.resolve(strict=True)
                except OSError as exc:
                    raise LocalAIManagerError(
                        f"audio-cpp backend 存在失效软链接：{relative_path}"
                    ) from exc
                if not resolved_path.is_relative_to(resolved_root):
                    raise LocalAIManagerError(f"audio-cpp backend 软链接越界：{relative_path}")
                if resolved_path.is_file():
                    entries.append(
                        {
                            "path": relative_path,
                            "kind": "symlink-file",
                            "target": os.readlink(path),
                            "size": resolved_path.stat().st_size,
                            "sha256": cls._sha256_file(resolved_path),
                        }
                    )
                elif resolved_path.is_dir():
                    entries.append(
                        {"path": relative_path, "kind": "symlink-dir", "target": os.readlink(path)}
                    )
                else:
                    raise LocalAIManagerError(
                        f"audio-cpp backend 软链接指向特殊文件：{relative_path}"
                    )
            elif path.is_dir():
                entries.append({"path": relative_path, "kind": "dir"})
            elif path.is_file():
                entries.append(
                    {
                        "path": relative_path,
                        "kind": "file",
                        "mode": path.stat().st_mode & 0o777,
                        "size": path.stat().st_size,
                        "sha256": cls._sha256_file(path),
                    }
                )
            else:
                raise LocalAIManagerError(f"audio-cpp backend 包含特殊文件：{relative_path}")
        serialized = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _write_temporary_file(directory: Path, prefix: str, suffix: str) -> Path:
        descriptor, temporary_name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
        os.close(descriptor)
        return Path(temporary_name)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as file_handle:
            os.fsync(file_handle.fileno())

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @classmethod
    def _atomic_write_json(cls, path: Path, value: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
            )
            cls._fsync_file(temporary_path)
            os.replace(temporary_path, path)
            cls._fsync_directory(path.parent)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _enabled(self) -> bool:
        value = self.environ.get("UNITALE_LOCALAI_ENABLED", "false").strip().lower()
        return value in {"1", "true", "yes", "on"}


def main() -> int:
    """解析固定版本校验、启停与状态子命令。"""
    parser = argparse.ArgumentParser(description="管理 Unitale 裸机 LocalAI 实例")
    parser.add_argument(
        "command",
        choices=(
            "verify-install",
            "register-qwen3-asr-candidate",
            "register-qwen3-tts-candidate",
            "verify-qwen3-tts-candidate",
            "register-qwen3-voicedesign-candidate",
            "register-dots-tts-soar-candidate",
            "register-voxcpm2-candidate",
            "verify-voxcpm2-candidate",
            "register-confucius4-tts-candidate",
            "verify-confucius4-tts-candidate",
            "register-moss-voicegenerator-candidate",
            "verify-moss-voicegenerator-candidate",
            "register-fireredtts3-base-candidate",
            "verify-fireredtts3-base-candidate",
            "register-fireredtts3-instruct-candidate",
            "verify-fireredtts3-instruct-candidate",
            "register-ace-step-candidate",
            "verify-ace-step-candidate",
            "register-stable-audio-candidate",
            "verify-stable-audio-candidate",
            "register-seed-vc-candidate",
            "verify-seed-vc-candidate",
            "register-worker-bridge-candidates",
            "verify-worker-bridge-candidates",
            "verify-dots-tts-soar-candidate",
            "verify-qwen3-voicedesign-candidate",
            "start",
            "stop",
            "status",
            "watch",
        ),
    )
    parser.add_argument("watch_args", nargs="*")
    arguments = parser.parse_args()
    try:
        manager = LocalAIManager()
        if arguments.command == "watch":
            if len(arguments.watch_args) != 4:
                raise LocalAIManagerError("watch 参数格式无效。")
            manager.watch_owner(
                int(arguments.watch_args[0]),
                arguments.watch_args[1],
                int(arguments.watch_args[2]),
                arguments.watch_args[3],
            )
        elif arguments.command == "register-qwen3-asr-candidate":
            manager.register_qwen3_asr_candidate()
        elif arguments.command == "register-qwen3-tts-candidate":
            manager.register_qwen3_tts_candidate()
        elif arguments.command == "verify-qwen3-tts-candidate":
            manager.verify_qwen3_tts_candidate()
        elif arguments.command == "register-qwen3-voicedesign-candidate":
            manager.register_qwen3_voicedesign_candidate()
        elif arguments.command == "register-dots-tts-soar-candidate":
            manager.register_dots_tts_soar_candidate()
        elif arguments.command == "register-voxcpm2-candidate":
            manager.register_voxcpm2_candidate()
        elif arguments.command == "verify-voxcpm2-candidate":
            manager.verify_voxcpm2_candidate()
        elif arguments.command == "register-confucius4-tts-candidate":
            manager.register_confucius4_tts_candidate()
        elif arguments.command == "verify-confucius4-tts-candidate":
            manager.verify_confucius4_tts_candidate()
        elif arguments.command == "register-moss-voicegenerator-candidate":
            manager.register_moss_voicegenerator_candidate()
        elif arguments.command == "verify-moss-voicegenerator-candidate":
            manager.verify_moss_voicegenerator_candidate()
        elif arguments.command == "register-fireredtts3-base-candidate":
            manager.register_fireredtts3_base_candidate()
        elif arguments.command == "verify-fireredtts3-base-candidate":
            manager.verify_fireredtts3_base_candidate()
        elif arguments.command == "register-fireredtts3-instruct-candidate":
            manager.register_fireredtts3_instruct_candidate()
        elif arguments.command == "verify-fireredtts3-instruct-candidate":
            manager.verify_fireredtts3_instruct_candidate()
        elif arguments.command == "register-ace-step-candidate":
            manager.register_ace_step_candidate()
        elif arguments.command == "verify-ace-step-candidate":
            manager.verify_ace_step_candidate()
        elif arguments.command == "register-stable-audio-candidate":
            manager.register_stable_audio_candidate()
        elif arguments.command == "verify-stable-audio-candidate":
            manager.verify_stable_audio_candidate()
        elif arguments.command == "register-seed-vc-candidate":
            manager.register_seed_vc_candidate()
        elif arguments.command == "verify-seed-vc-candidate":
            manager.verify_seed_vc_candidate()
        elif arguments.command == "register-worker-bridge-candidates":
            manager.register_worker_bridge_candidates()
        elif arguments.command == "verify-worker-bridge-candidates":
            manager.verify_worker_bridge_candidates()
        elif arguments.command == "verify-dots-tts-soar-candidate":
            manager.verify_dots_tts_soar_candidate()
        elif arguments.command == "verify-qwen3-voicedesign-candidate":
            manager.verify_qwen3_voicedesign_candidate()
        elif arguments.command == "verify-install":
            manager.verify_install()
        else:
            getattr(manager, arguments.command)()
    except LocalAIManagerError as exc:
        print(f"LocalAI 管理失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
