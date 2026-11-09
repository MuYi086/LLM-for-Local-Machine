"""串行调用八个本地 audio.cpp 服务并保存可播放的生成示例。"""

from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import os
import sys
import uuid
import wave
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "localai/manifests/audio-cpp-models.json"
OUTPUT_DIR = ROOT / "storage/audio_cpp_tests"
MAX_RESPONSE_BYTES = 512 * 1024 * 1024
TEXT_PROMPT = "你好，欢迎使用本地 audio.cpp 音频服务。"
_TTS_KINDS = {"tts_clone", "tts_optional_reference"}


def build_parser(*, include_models: bool = True) -> argparse.ArgumentParser:
    """构造顺序运行本地生成示例的命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    if include_models:
        parser.add_argument(
            "--models",
            nargs="+",
            default=["all"],
            help="要生成的模型 key；默认依清单顺序串行运行全部八个模型。",
        )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=os.getenv("AUDIO_CPP_TEST_REFERENCE_AUDIO"),
        help="TTS/ASR/SeedVC 使用的本地 WAV；原文件只读。",
    )
    parser.add_argument(
        "--source-audio",
        type=Path,
        default=os.getenv("AUDIO_CPP_TEST_SOURCE_AUDIO"),
        help="SeedVC 源 WAV；默认使用 --reference-audio。",
    )
    parser.add_argument("--reference-text", help="Qwen3-TTS Base 的参考 WAV 逐字文本。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def _model_candidates() -> list[dict[str, object]]:
    """读取含端口、路由和任务类型的候选清单。"""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [
        model
        for model in manifest["models"]
        if isinstance(model, dict)
        and isinstance(model.get("service"), dict)
        and model["service"].get("runtime") == "audio_cpp_runtime"
    ]


def _validate_wav(path: Path) -> None:
    """确认测试输入为带采样帧的 PCM WAV。"""
    try:
        with wave.open(str(path), "rb") as source:
            params = source.getparams()
            frames = source.readframes(params.nframes)
    except (EOFError, OSError, wave.Error) as exc:
        raise ValueError(f"音频不是有效 WAV：{path}") from exc
    expected_bytes = params.nframes * params.nchannels * params.sampwidth
    if params.comptype != "NONE" or params.nframes <= 0 or len(frames) != expected_bytes:
        raise ValueError(f"音频没有完整 PCM 采样帧：{path}")


def _multipart_body(
    fields: dict[str, str],
    files: dict[str, Path],
) -> tuple[str, int, Iterable[bytes]]:
    """构造分块读取文件的 multipart 请求体，不把输入 WAV 整体载入内存。"""
    boundary = f"llm-local-test-{uuid.uuid4().hex}"
    parts: list[bytes | Path] = []
    for name, value in fields.items():
        parts.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            )
        )
    for name, path in files.items():
        filename = path.name.replace('"', "")
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        parts.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                path,
                b"\r\n",
            )
        )
    parts.append(f"--{boundary}--\r\n".encode())
    content_length = sum(
        part.stat().st_size if isinstance(part, Path) else len(part) for part in parts
    )

    def iterate() -> Iterable[bytes]:
        for part in parts:
            if isinstance(part, bytes):
                yield part
                continue
            with part.open("rb") as source:
                while chunk := source.read(64 * 1024):
                    yield chunk

    return boundary, content_length, iterate()


def _post(
    host: str,
    port: int,
    route: str,
    *,
    timeout: float,
    json_body: dict[str, object] | None = None,
    form_fields: dict[str, str] | None = None,
    files: dict[str, Path] | None = None,
) -> tuple[bytes, str]:
    """向单个本地模型服务发送 JSON 或有界 multipart 请求。"""
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        if json_body is not None:
            body = json.dumps(json_body, ensure_ascii=False, separators=(",", ":")).encode()
            connection.request(
                "POST",
                route,
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
        else:
            boundary, content_length, body = _multipart_body(form_fields or {}, files or {})
            connection.request(
                "POST",
                route,
                body=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Content-Length": str(content_length),
                },
            )
        response = connection.getresponse()
        content = response.read(MAX_RESPONSE_BYTES + 1)
        if response.status < 200 or response.status >= 300:
            detail = content[:2_000].decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {response.status} {route}: {detail}")
        if len(content) > MAX_RESPONSE_BYTES:
            raise RuntimeError(f"响应超过 {MAX_RESPONSE_BYTES // (1024 * 1024)} MiB：{route}")
        return content, response.getheader("Content-Type", "")
    finally:
        connection.close()


def _save_wav(output_dir: Path, model_key: str, content: bytes) -> Path:
    """原子保存并校验模型返回的 WAV 示例。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_dir / f".{model_key}-{uuid.uuid4().hex}.tmp"
    destination = (
        output_dir / f"{model_key}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.wav"
    )
    try:
        with temporary_path.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        _validate_wav(temporary_path)
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def _generate_one(
    model: dict[str, object],
    *,
    host: str,
    timeout: float,
    reference_audio: Path | None,
    reference_text: str | None,
    source_audio: Path | None,
    output_dir: Path,
) -> Path:
    """调用一个模型，校验返回合同并保存输出。"""
    service = model["service"]
    assert isinstance(service, dict)
    key = str(model["key"])
    kind = str(service["kind"])
    route = str(service["route"])
    port = int(service["port"])
    if kind in _TTS_KINDS:
        fields = {"text": TEXT_PROMPT, "language": "zh"}
        files: dict[str, Path] = {}
        if kind == "tts_clone":
            if reference_audio is None:
                raise ValueError(f"{key} 需要通过 --reference-audio 指定 WAV。")
            files["reference_audio"] = reference_audio
        if key == "qwen3_tts_base":
            if not reference_text:
                raise ValueError("Qwen3-TTS Base 需要通过 --reference-text 提供参考音频逐字稿。")
            fields["reference_text"] = reference_text
        if key == "voxcpm2":
            fields["instruction"] = "温和、清晰、自然的中文播音声线。"
            if reference_audio is not None:
                files["reference_audio"] = reference_audio
        content, _ = _post(host, port, route, timeout=timeout, form_fields=fields, files=files)
        return _save_wav(output_dir, key, content)

    if kind == "asr":
        if reference_audio is None:
            raise ValueError("Qwen3-ASR 需要通过 --reference-audio 指定 WAV。")
        content, _ = _post(
            host,
            port,
            route,
            timeout=timeout,
            form_fields={},
            files={"audio": reference_audio},
        )
        result = json.loads(content)
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise ValueError(f"{key} 没有返回转写文本 JSON。")
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{key}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        return destination

    if kind == "voice_conversion":
        if reference_audio is None:
            raise ValueError("SeedVC 需要通过 --reference-audio 指定目标音色 WAV。")
        if source_audio is None:
            raise ValueError("SeedVC 需要通过 --source-audio 指定源 WAV。")
        content, _ = _post(
            host,
            port,
            route,
            timeout=timeout,
            form_fields={},
            files={"audio": source_audio, "reference": reference_audio},
        )
        return _save_wav(output_dir, key, content)

    if kind == "sound_generation":
        prompts = {
            "midashenglm_gen": (
                "<|caption|>A quiet room on a rainy night. "
                "<|asr|>The rain outside is fading away. "
                "<|speech|>A calm, clear narrator. "
                "<|music|>Sparse, soft piano chords. "
                "<|sfx|>Gentle rain outside the window. <|env|>A quiet room."
            ),
            "stable_audio_sfx": (
                "Footsteps on gravel, close perspective, crisp natural stone texture."
            ),
        }
        content, _ = _post(
            host,
            port,
            route,
            timeout=timeout,
            json_body={"text": prompts[key], "duration_seconds": 4},
        )
        return _save_wav(output_dir, key, content)

    raise ValueError(f"测试脚本未支持的 audio.cpp 服务类型：{kind}")


def generate_examples(
    *,
    model_keys: list[str],
    host: str,
    timeout: float,
    reference_audio: Path | None,
    reference_text: str | None,
    source_audio: Path | None,
    output_dir: Path,
) -> list[Path]:
    """依候选清单顺序串行生成所选模型的示例。"""
    candidates = _model_candidates()
    by_key = {str(model["key"]): model for model in candidates}
    selected_keys = list(by_key) if model_keys == ["all"] else model_keys
    unknown = sorted(set(selected_keys) - set(by_key))
    if unknown:
        raise ValueError(f"未知模型 key：{', '.join(unknown)}")
    required_reference_models = {
        "confucius4_tts",
        "index_tts25",
        "qwen3_asr",
        "qwen3_tts_base",
        "seed_vc",
    }
    if reference_audio is None and required_reference_models.intersection(selected_keys):
        raise ValueError("所选任务需要通过 --reference-audio 指定本地 PCM WAV。")
    if "qwen3_tts_base" in selected_keys and not reference_text:
        raise ValueError("Qwen3-TTS Base 需要通过 --reference-text 提供参考音频逐字稿。")
    if reference_audio is not None:
        _validate_wav(reference_audio)
    if source_audio is not None and source_audio != reference_audio:
        _validate_wav(source_audio)
    if "seed_vc" in selected_keys and source_audio is None:
        source_audio = reference_audio

    output_paths = []
    for key in selected_keys:
        model = by_key[key]
        path = _generate_one(
            model,
            host=host,
            timeout=timeout,
            reference_audio=reference_audio,
            reference_text=reference_text,
            source_audio=source_audio,
            output_dir=output_dir,
        )
        output_paths.append(path)
        print(f"{key} 生成完成：{path}")
    return output_paths


def _execute(model_keys: list[str], arguments: argparse.Namespace) -> int:
    """解析路径、调用模型并统一呈现真实 API 测试结果。"""
    candidates = _model_candidates()
    available_keys = {str(model["key"]) for model in candidates}
    if model_keys != ["all"] and set(model_keys) - available_keys:
        print(
            f"未知模型 key：{', '.join(sorted(set(model_keys) - available_keys))}",
            file=sys.stderr,
        )
        return 2
    reference_audio = (
        Path(arguments.reference_audio).expanduser().resolve()
        if arguments.reference_audio is not None
        else None
    )
    source_audio = (
        Path(arguments.source_audio).expanduser().resolve()
        if arguments.source_audio is not None
        else reference_audio
    )
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    try:
        generate_examples(
            model_keys=model_keys,
            host=arguments.host,
            timeout=arguments.timeout,
            reference_audio=reference_audio,
            reference_text=arguments.reference_text,
            source_audio=source_audio,
            output_dir=output_dir,
        )
    except (
        OSError,
        http.client.HTTPException,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"audio.cpp 模型生成测试失败：{exc}", file=sys.stderr)
        return 1
    return 0


def run_single_model(model_key: str, argv: list[str] | None = None) -> int:
    """通过真实 HTTP API 运行固定模型的生成冒烟测试。"""
    arguments = build_parser(include_models=False).parse_args(argv)
    return _execute([model_key], arguments)


def main() -> int:
    """按顺序运行已选 audio.cpp 模型示例。"""
    arguments = build_parser().parse_args()
    return _execute(arguments.models, arguments)


if __name__ == "__main__":
    raise SystemExit(main())
