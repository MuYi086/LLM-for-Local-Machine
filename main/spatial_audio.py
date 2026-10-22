"""使用 FFmpeg 完成 48 kHz 空间音频导出。"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from llm_local_runtime import StagedUpload

SPATIAL_EXPORT_PROFILES = ("standard", "balanced", "immersive")
SPATIAL_EXPORT_FORMATS = ("wav", "mp3")


class SpatialAudioExportError(RuntimeError):
    """空间音频请求、FFmpeg 执行或测量结果不符合导出约束。"""


@dataclass(frozen=True)
class LoudnormMeasurement:
    """FFmpeg 第一遍 loudnorm 输出的安全数值。"""

    input_i: str
    input_tp: str
    input_lra: str
    input_thresh: str
    target_offset: str


@dataclass(frozen=True)
class SpatialAudioExportResult:
    """可由 FileResponse 流式返回的临时导出结果。"""

    path: Path
    media_type: str
    download_name: str
    profile: str
    output_format: str

    def cleanup(self) -> None:
        """响应结束后删除整个任务目录，避免长音频长期占用缓存。"""
        shutil.rmtree(self.path.parent, ignore_errors=True)


def _safe_measurement_value(payload: dict[str, object], key: str) -> str:
    """把 loudnorm 字段约束为有限浮点数，避免拼接任意滤镜参数。"""
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise SpatialAudioExportError(f"loudnorm 缺少有效字段: {key}") from exc
    if not math.isfinite(value):
        raise SpatialAudioExportError(f"loudnorm 字段不是有限数值: {key}")
    return str(value)


def parse_loudnorm_measurement(report: str) -> LoudnormMeasurement:
    """从 FFmpeg stderr 中提取最后一个完整的 loudnorm JSON 对象。"""
    matches = re.findall(r"\{.*?\}", report, flags=re.DOTALL)
    if not matches:
        raise SpatialAudioExportError("loudnorm 没有返回 JSON 测量结果。")
    try:
        payload = json.loads(matches[-1])
    except json.JSONDecodeError as exc:
        raise SpatialAudioExportError("loudnorm 返回的 JSON 无法解析。") from exc
    if not isinstance(payload, dict):
        raise SpatialAudioExportError("loudnorm 测量结果必须是 JSON 对象。")
    return LoudnormMeasurement(
        input_i=_safe_measurement_value(payload, "input_i"),
        input_tp=_safe_measurement_value(payload, "input_tp"),
        input_lra=_safe_measurement_value(payload, "input_lra"),
        input_thresh=_safe_measurement_value(payload, "input_thresh"),
        target_offset=_safe_measurement_value(payload, "target_offset"),
    )


def build_pre_master_filter(profile: str) -> str:
    """构建保留现有立体声信息的 48 kHz pre-master 滤镜图。"""
    if profile not in SPATIAL_EXPORT_PROFILES:
        raise SpatialAudioExportError(f"不支持的 profile: {profile}")
    if profile == "standard":
        return (
            "[0:a]aresample=48000:resampler=soxr:precision=28,"
            "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            "alimiter=limit=0.891:attack=5:release=50:level=0:latency=1[out]"
        )

    if profile == "balanced":
        dry_level = "0.82"
        wide_highpass = "180"
        wide_lowpass = "8000"
        left_delay = "4"
        right_delay = "8"
        side_gain = "1.20"
        wide_level = "0.38"
        echo_out_gain = "0.70"
        echo_delays = "19|37|61"
        echo_decays = "0.24|0.14|0.08"
        room_delay = "13"
        room_lowpass = "6500"
        room_level = "0.16"
    else:
        dry_level = "0.78"
        wide_highpass = "160"
        wide_lowpass = "8500"
        left_delay = "7"
        right_delay = "13"
        side_gain = "1.45"
        wide_level = "0.58"
        echo_out_gain = "0.75"
        echo_delays = "17|35|71"
        echo_decays = "0.28|0.17|0.09"
        room_delay = "17"
        room_lowpass = "7000"
        room_level = "0.24"

    # 两条湿声只保留 Side，既能显著增加耳机宽度，也会在单声道折叠时自动抵消。
    side_only_pan = "pan=stereo|c0=0.5*c0-0.5*c1|c1=0.5*c1-0.5*c0"
    return (
        "[0:a]aresample=48000:resampler=soxr:precision=28,"
        "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        "asplit=3[dry_in][wide_in][room_in];"
        f"[dry_in]volume={dry_level}[dry];"
        f"[wide_in]highpass=f={wide_highpass},lowpass=f={wide_lowpass},"
        f"haas=left_delay={left_delay}:right_delay={right_delay}:side_gain={side_gain},"
        f"{side_only_pan},volume={wide_level}[wide];"
        "[room_in]pan=mono|c0=0.5*c0+0.5*c1,"
        f"aecho=0.8:{echo_out_gain}:'{echo_delays}':'{echo_decays}',"
        "pan=stereo|c0=c0|c1=c0,"
        f"adelay='0|{room_delay}',lowpass=f={room_lowpass},"
        f"{side_only_pan},volume={room_level}[room];"
        "[dry][wide][room]amix=inputs=3:normalize=0,highpass=f=70,"
        "alimiter=limit=0.891:attack=5:release=50:level=0:latency=1[out]"
    )


class AudioMasteringProcessor:
    """对 48 kHz 立体声 pre-master 执行双遍响度归一化和最终编码。"""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        ffmpeg_bin: str = "ffmpeg",
        timeout_seconds: float = 600,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0。")
        self.cache_dir = Path(cache_dir)
        self.ffmpeg_bin = ffmpeg_bin
        self.timeout_seconds = timeout_seconds

    def _run_ffmpeg(self, command: list[str]) -> str:
        """运行 FFmpeg；超时时终止独立进程组并保留错误尾部。"""
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise SpatialAudioExportError(f"无法启动 FFmpeg: {exc}") from exc

        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
            raise SpatialAudioExportError(
                f"FFmpeg 空间音频处理超过 {self.timeout_seconds:g} 秒。"
            ) from exc

        if process.returncode != 0:
            error_lines = stderr.strip().splitlines()
            excerpt = "\n".join(error_lines[-30:]) or "FFmpeg 未返回错误详情。"
            raise SpatialAudioExportError(f"FFmpeg 空间音频处理失败:\n{excerpt}")
        return "\n".join(part for part in (stdout, stderr) if part)

    def master_pre_master(
        self,
        pre_master_path: str | Path,
        job_dir: str | Path,
        *,
        profile: str,
        output_format: str,
        download_prefix: str = "llm-local",
    ) -> SpatialAudioExportResult:
        """只做母带与编码，不增加 Haas、声像或房间反射。

        Steam Audio 已经完成对象级空间化，因此该入口故意不调用
        :func:`build_pre_master_filter`，避免正式导出发生二次空间处理。
        """
        if profile not in SPATIAL_EXPORT_PROFILES:
            raise SpatialAudioExportError(f"不支持的 profile: {profile}")
        if output_format not in SPATIAL_EXPORT_FORMATS:
            raise SpatialAudioExportError(f"不支持的 output_format: {output_format}")

        pre_master = Path(pre_master_path)
        output_dir = Path(job_dir)
        temporary_output_path = output_dir / f"result.part.{output_format}"
        output_path = output_dir / f"result.{output_format}"
        true_peak = "-2.5" if output_format == "mp3" else "-2"

        analyze_filter = f"loudnorm=I=-18:TP={true_peak}:LRA=7:print_format=json"
        analyze_command = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-i",
            str(pre_master),
            "-af",
            analyze_filter,
            "-f",
            "null",
            "-",
        ]
        measurement = parse_loudnorm_measurement(self._run_ffmpeg(analyze_command))

        master_filter = (
            f"loudnorm=I=-18:TP={true_peak}:LRA=7:"
            f"measured_I={measurement.input_i}:"
            f"measured_TP={measurement.input_tp}:"
            f"measured_LRA={measurement.input_lra}:"
            f"measured_thresh={measurement.input_thresh}:"
            f"offset={measurement.target_offset}:"
            "linear=true:print_format=summary"
        )
        master_command = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-y",
            "-i",
            str(pre_master),
            "-af",
            master_filter,
            "-ar",
            "48000",
            "-ac",
            "2",
        ]
        if output_format == "wav":
            master_command.extend(["-c:a", "pcm_s24le"])
            media_type = "audio/wav"
        else:
            master_command.extend(["-c:a", "libmp3lame", "-b:a", "192k"])
            media_type = "audio/mpeg"
        master_command.append(str(temporary_output_path))
        self._run_ffmpeg(master_command)
        if not temporary_output_path.is_file() or temporary_output_path.stat().st_size == 0:
            raise SpatialAudioExportError("FFmpeg 没有生成有效的导出音频。")
        os.replace(temporary_output_path, output_path)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        return SpatialAudioExportResult(
            path=output_path,
            media_type=media_type,
            download_name=f"{download_prefix}_{profile}_{timestamp}.{output_format}",
            profile=profile,
            output_format=output_format,
        )


class SpatialAudioProcessor(AudioMasteringProcessor):
    """以一次性 FFmpeg 进程保留 legacy 总线空间导出。"""

    def export(
        self,
        staged: StagedUpload,
        *,
        profile: str,
        output_format: str,
    ) -> SpatialAudioExportResult:
        """生成 48 kHz 双声道母带，并在成功或失败后清理上传暂存文件。"""
        if profile not in SPATIAL_EXPORT_PROFILES:
            raise SpatialAudioExportError(f"不支持的 profile: {profile}")
        if output_format not in SPATIAL_EXPORT_FORMATS:
            raise SpatialAudioExportError(f"不支持的 output_format: {output_format}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        job_dir = Path(tempfile.mkdtemp(prefix="spatial_export_", dir=self.cache_dir))
        pre_master_path = job_dir / "pre-master.wav"
        succeeded = False
        try:
            pre_master_command = [
                self.ffmpeg_bin,
                "-hide_banner",
                "-nostdin",
                "-nostats",
                "-y",
                "-i",
                str(staged.path),
                "-filter_complex",
                build_pre_master_filter(profile),
                "-map",
                "[out]",
                "-ar",
                "48000",
                "-c:a",
                "pcm_s24le",
                str(pre_master_path),
            ]
            self._run_ffmpeg(pre_master_command)
            result = self.master_pre_master(
                pre_master_path,
                job_dir,
                profile=profile,
                output_format=output_format,
            )
            succeeded = True
            return result
        finally:
            staged.path.unlink(missing_ok=True)
            pre_master_path.unlink(missing_ok=True)
            if not succeeded:
                shutil.rmtree(job_dir, ignore_errors=True)
