"""治理 Steam Audio 对象级任务、预处理资产并调用独立 C++ renderer。"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from spatial_audio import (
    AudioMasteringProcessor,
    SpatialAudioExportError,
    SpatialAudioExportResult,
)
from spatial_schema import SpatialRenderManifest

from llm_local_runtime import StagedUpload

MAX_ERROR_LINES = 40
RENDERER_PROGRESS_PATTERN = re.compile(r"^LLM_LOCAL_PROGRESS\s+(\d+)\s+(\d+)$")
LOGGER = logging.getLogger("uvicorn.error")
ProgressCallback = Callable[[str, int, str], None]


class SteamAudioRenderError(SpatialAudioExportError):
    """Steam Audio 配置、预处理、CLI 执行或输出合同不符合要求。"""


def _format_duration(seconds: float) -> str:
    """把进度估算时长格式化为紧凑的中文文本。"""
    total_seconds = max(1, math.ceil(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} 小时 {minutes} 分"
    if minutes:
        return f"{minutes} 分 {remaining_seconds} 秒"
    return f"{remaining_seconds} 秒"


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """终止任务自己的进程组，避免超时后遗留 renderer 或其子进程。"""
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


class SteamAudioRenderProcessor:
    """以隔离任务目录执行 FFmpeg 标准化、Steam 渲染和最终母带。"""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        renderer_bin: str,
        ffmpeg_bin: str = "ffmpeg",
        timeout_seconds: float = 900,
        mastering_timeout_seconds: float = 600,
        threads: int = 4,
        sdk_dir: str | Path | None = None,
        hrtf_path: str | Path | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0。")
        if threads <= 0:
            raise ValueError("threads 必须为正整数。")
        self.cache_dir = Path(cache_dir)
        self.renderer_bin = renderer_bin
        self.ffmpeg_bin = ffmpeg_bin
        self.timeout_seconds = timeout_seconds
        self.threads = threads
        self.sdk_dir = Path(sdk_dir).expanduser() if sdk_dir else None
        self.hrtf_path = Path(hrtf_path).expanduser() if hrtf_path else None
        self.mastering = AudioMasteringProcessor(
            cache_dir,
            ffmpeg_bin=ffmpeg_bin,
            timeout_seconds=mastering_timeout_seconds,
        )

    def _resolved_renderer(self) -> Path | None:
        """解析绝对/相对路径或 PATH 中的 renderer，不修改进程环境。"""
        configured = Path(self.renderer_bin).expanduser()
        if configured.parent != Path(".") or configured.is_absolute():
            return configured.resolve()
        resolved = shutil.which(self.renderer_bin)
        return Path(resolved).resolve() if resolved else None

    def diagnostics(self) -> dict[str, Any]:
        """返回健康检查所需的 SDK、动态库和 renderer 可用性。"""
        renderer = self._resolved_renderer()
        renderer_ready = bool(renderer and renderer.is_file() and os.access(renderer, os.X_OK))
        sdk_header = self.sdk_dir / "include/phonon.h" if self.sdk_dir else None
        if sys.platform == "win32":
            sdk_library = self.sdk_dir / "lib/windows-x64/phonon.dll" if self.sdk_dir else None
        elif sys.platform == "darwin":
            sdk_library = self.sdk_dir / "lib/osx/libphonon.dylib" if self.sdk_dir else None
        else:
            sdk_library = self.sdk_dir / "lib/linux-x64/libphonon.so" if self.sdk_dir else None
        hrtf_ready = self.hrtf_path is None or self.hrtf_path.is_file()
        return {
            "available": renderer_ready and hrtf_ready,
            "renderer_bin": str(renderer) if renderer else self.renderer_bin,
            "renderer_executable": renderer_ready,
            "sdk_dir": str(self.sdk_dir) if self.sdk_dir else None,
            "sdk_header": str(sdk_header) if sdk_header else None,
            "sdk_header_available": bool(sdk_header and sdk_header.is_file()),
            "sdk_library": str(sdk_library) if sdk_library else None,
            "sdk_library_available": bool(sdk_library and sdk_library.is_file()),
            "hrtf_path": str(self.hrtf_path) if self.hrtf_path else None,
            "hrtf_available": hrtf_ready,
            "threads": self.threads,
            "timeout_seconds": self.timeout_seconds,
            "gpu_lock": False,
        }

    def _run_renderer(
        self,
        command: list[str],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> str:
        """运行 Steam CLI，并在超时或异常退出时提供可诊断错误。"""
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            raise SteamAudioRenderError(f"无法启动 Steam Audio renderer: {exc}") from exc

        stdout_lines: list[str] = []
        stderr_lines: deque[str] = deque(maxlen=MAX_ERROR_LINES)

        def consume_stdout(stream: TextIO) -> None:
            stdout_lines.extend(stream.readlines())

        def consume_stderr(stream: TextIO) -> None:
            for line in stream:
                stripped = line.strip()
                match = RENDERER_PROGRESS_PATTERN.fullmatch(stripped)
                if match:
                    completed, total = (int(value) for value in match.groups())
                    if progress_callback and total > 0:
                        progress = 60 + round(25 * min(completed, total) / total)
                        progress_callback(
                            "rendering",
                            progress,
                            f"Steam Audio 正在渲染对象 {completed}/{total}",
                        )
                    continue
                if stripped:
                    stderr_lines.append(stripped)
                    LOGGER.info("[Steam Audio renderer] %s", stripped)

        assert process.stdout is not None and process.stderr is not None
        stdout_thread = threading.Thread(
            target=consume_stdout,
            args=(process.stdout,),
            name="steam-renderer-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=consume_stderr,
            args=(process.stderr,),
            name="steam-renderer-stderr",
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            process.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            process.stdout.close()
            process.stderr.close()
            raise SteamAudioRenderError(
                f"Steam Audio 渲染超过 {self.timeout_seconds:g} 秒。"
            ) from exc
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        process.stdout.close()
        process.stderr.close()
        if process.returncode != 0:
            excerpt = "\n".join(stderr_lines)
            raise SteamAudioRenderError(
                f"Steam Audio renderer 退出码 {process.returncode}:\n"
                f"{excerpt or 'renderer 未返回错误详情。'}"
            )
        return "".join(stdout_lines)

    def _normalize_source(
        self,
        *,
        staged: StagedUpload,
        output_path: Path,
        trim_start_ms: int,
        duration_ms: int,
        playback_rate: float,
        preserve_stereo: bool,
        loop: bool,
    ) -> None:
        """把单个不可信上传解码为 renderer 可读取的 48 kHz float WAV。"""
        command = [self.ffmpeg_bin, "-hide_banner", "-nostdin", "-nostats", "-y"]
        if loop:
            command.extend(["-stream_loop", "-1"])
        command.extend(["-i", str(staged.path)])
        if trim_start_ms:
            command.extend(["-ss", f"{trim_start_ms / 1000:.6f}"])
        layout = "stereo" if preserve_stereo else "mono"
        channels = "2" if preserve_stereo else "1"
        audio_filters: list[str] = []
        # FFmpeg 6.1.1 在 atempo=1 后接 SoXR 时可能无法结束滤镜链；1× 本就是无操作。
        if not math.isclose(playback_rate, 1.0, rel_tol=0, abs_tol=1e-9):
            audio_filters.append(f"atempo={playback_rate:.8g}")
        audio_filters.extend(
            [
                "aresample=48000:resampler=soxr:precision=28",
                f"aformat=sample_fmts=flt:sample_rates=48000:channel_layouts={layout}",
            ]
        )
        audio_filter = ",".join(audio_filters)
        command.extend(
            [
                "-t",
                f"{duration_ms / 1000:.6f}",
                "-af",
                audio_filter,
                "-map_metadata",
                "-1",
                "-ar",
                "48000",
                "-ac",
                channels,
                "-c:a",
                "pcm_f32le",
                str(output_path),
            ]
        )
        try:
            self.mastering._run_ffmpeg(command)
        except SpatialAudioExportError as exc:
            raise SteamAudioRenderError(f"资产标准化失败: {exc}") from exc
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise SteamAudioRenderError("FFmpeg 没有生成有效的规范化资产。")

    @staticmethod
    def _parse_renderer_report(stdout: str, expected_sources: int) -> dict[str, Any]:
        """验证 CLI stdout 的单行机器合同，日志只能写入 stderr。"""
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise SteamAudioRenderError("renderer stdout 必须只包含一行 JSON。")
        try:
            report = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise SteamAudioRenderError("renderer stdout 不是有效 JSON。") from exc
        if not isinstance(report, dict) or report.get("ok") is not True:
            raise SteamAudioRenderError("renderer 未返回 ok=true。")
        if report.get("sample_rate") != 48000 or report.get("channels") != 2:
            raise SteamAudioRenderError("renderer 输出必须是 48 kHz 双声道。")
        if report.get("sources") != expected_sources:
            raise SteamAudioRenderError("renderer 报告的 sources 数量不一致。")
        return report

    def render(
        self,
        manifest: SpatialRenderManifest,
        staged_assets: dict[str, StagedUpload],
        *,
        profile: str,
        output_format: str,
        progress_callback: ProgressCallback | None = None,
    ) -> SpatialAudioExportResult:
        """执行一个完整 CPU 空间音频任务，并保证所有异常路径清理。"""
        renderer = self._resolved_renderer()
        if not renderer or not renderer.is_file() or not os.access(renderer, os.X_OK):
            raise SteamAudioRenderError(
                f"Steam Audio renderer 不可用: {self.renderer_bin}。请先按文档构建并配置。"
            )
        if self.hrtf_path and not self.hrtf_path.is_file():
            raise SteamAudioRenderError(f"Steam Audio HRTF 不可读: {self.hrtf_path}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        job_dir = Path(tempfile.mkdtemp(prefix="steam_audio_", dir=self.cache_dir))
        assets_dir = job_dir / "assets"
        assets_dir.mkdir()
        manifest_path = job_dir / "render-manifest.json"
        temporary_manifest_path = job_dir / ".render-manifest.json.tmp"
        pre_master_path = job_dir / "pre-master.wav"
        succeeded = False
        try:
            normalized_manifest = manifest.model_copy(deep=True)
            normalized_manifest.scene.acoustic_quality = profile
            source_count = len(normalized_manifest.sources)
            normalization_started = time.perf_counter()
            for index, source in enumerate(normalized_manifest.sources):
                if progress_callback:
                    if index == 0:
                        timing_message = "完成首个对象后估算剩余时间"
                    else:
                        elapsed_seconds = time.perf_counter() - normalization_started
                        remaining_seconds = elapsed_seconds / index * (source_count - index)
                        timing_message = f"预计剩余 {_format_duration(remaining_seconds)}"
                    progress_callback(
                        "normalizing",
                        20 + round(35 * index / source_count),
                        f"正在标准化音频对象 {index + 1}/{source_count}（{timing_message}）",
                    )
                try:
                    staged = staged_assets[source.asset_filename]
                except KeyError as exc:
                    raise SteamAudioRenderError(
                        f"Manifest 引用的上传资产缺失: {source.asset_filename}"
                    ) from exc
                normalized_filename = f"source_{index:04d}.wav"
                self._normalize_source(
                    staged=staged,
                    output_path=assets_dir / normalized_filename,
                    trim_start_ms=source.trim_start_ms,
                    duration_ms=source.duration_ms,
                    playback_rate=source.playback_rate,
                    preserve_stereo=source.spatial.mode == "preserve_stereo",
                    loop=source.loop,
                )
                source.asset_filename = normalized_filename
                source.trim_start_ms = 0
                source.playback_rate = 1.0
                source.loop = False

            if progress_callback:
                normalization_elapsed = time.perf_counter() - normalization_started
                progress_callback(
                    "normalizing",
                    55,
                    f"音频对象标准化完成 {source_count}/{source_count}，"
                    f"耗时 {_format_duration(normalization_elapsed)}",
                )

            payload = normalized_manifest.model_dump(mode="json", exclude_none=True)
            with temporary_manifest_path.open("w", encoding="utf-8") as destination:
                json.dump(payload, destination, ensure_ascii=False, separators=(",", ":"))
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary_manifest_path, manifest_path)

            command = [
                str(renderer),
                "--manifest",
                str(manifest_path),
                "--assets-dir",
                str(assets_dir),
                "--output",
                str(pre_master_path),
                "--threads",
                str(self.threads),
            ]
            if self.hrtf_path:
                command.extend(["--hrtf", str(self.hrtf_path)])
            if progress_callback:
                progress_callback("rendering", 60, "Steam Audio renderer 已启动")
            report = self._parse_renderer_report(
                self._run_renderer(command, progress_callback=progress_callback),
                expected_sources=len(normalized_manifest.sources),
            )
            if report.get("duration_ms") != manifest.timeline_duration_ms:
                raise SteamAudioRenderError("renderer 输出时长与 Manifest 不一致。")
            if not pre_master_path.is_file() or pre_master_path.stat().st_size == 0:
                raise SteamAudioRenderError("renderer 没有生成有效的 pre-master。")

            if progress_callback:
                progress_callback("mastering", 90, "正在执行响度归一化与最终编码")
            result = self.mastering.master_pre_master(
                pre_master_path,
                job_dir,
                profile=profile,
                output_format=output_format,
                download_prefix="llm-local_steam",
            )
            pre_master_path.unlink(missing_ok=True)
            if progress_callback:
                progress_callback("finalizing", 98, "成品已生成，正在准备下载")
            succeeded = True
            return result
        finally:
            for staged in staged_assets.values():
                staged.path.unlink(missing_ok=True)
            temporary_manifest_path.unlink(missing_ok=True)
            if not succeeded:
                shutil.rmtree(job_dir, ignore_errors=True)
