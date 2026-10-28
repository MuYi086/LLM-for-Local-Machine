"""定义 Steam Audio 对象级渲染 Manifest v1。"""

from __future__ import annotations

import math
import re
from pathlib import PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from spatial_trajectory import Trajectory

MANIFEST_VERSION = "1.0"
MAX_TIMELINE_MS = 2 * 60 * 60 * 1000
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SpatialManifestError(ValueError):
    """Manifest 跨字段约束不成立。"""


class StrictModel(BaseModel):
    """拒绝未声明字段，防止 DSL 或客户端静默扩展底层 DSP 参数。"""

    model_config = ConfigDict(extra="forbid")


class SpatialScene(StrictModel):
    """任务级声学场景；首版只把房间作为稳定 preset 标签。"""

    room: Literal[
        "dry_studio",
        "bedroom",
        "living_room",
        "narrow_corridor",
        "stairwell",
        "basement",
        "warehouse",
        "cave",
        "outdoor_open",
    ] = "dry_studio"
    listener_pose: Literal["center_forward"] = "center_forward"
    acoustic_quality: Literal["standard", "balanced", "immersive"] = "balanced"


class SpatialPlan(StrictModel):
    """LLM 可生成、编译器可确定映射的有限语义空间计划。"""

    mode: Literal["dry_center", "point", "diffuse", "preserve_stereo"] = "point"
    location: Literal[
        "front_center",
        "front_left",
        "front_right",
        "side_left",
        "side_right",
        "rear_left",
        "rear_center",
        "rear_right",
        "above_front",
        "above_rear",
        "below_front",
    ] = "front_center"
    distance: Literal["intimate", "near", "conversational", "mid", "far"] = "conversational"
    movement: Literal[
        "static",
        "approaching",
        "receding",
        "left_to_right",
        "right_to_left",
        "front_to_rear",
        "rear_to_front",
        "rising",
        "falling",
    ] = "static"
    occlusion: Literal[
        "none",
        "wooden_door",
        "solid_door",
        "drywall",
        "concrete_wall",
        "floor_ceiling",
    ] = "none"
    spatial_blend: Literal["subtle", "medium", "strong", "full"] = "medium"
    source: Literal["default", "rule", "llm", "user"] = "default"
    trajectory: Trajectory | None = None

    @model_validator(mode="after")
    def validate_trajectory_mode(self) -> SpatialPlan:
        """显式轨迹仅用于点声源，不能与旧运动预设同时指定。"""
        if self.trajectory is not None and (self.mode != "point" or self.movement != "static"):
            raise ValueError("trajectory 要求 mode=point 且 movement=static")
        return self


class SpatialSource(StrictModel):
    """一个尚未混入总线的音频对象及其时间线位置。"""

    id: str
    kind: Literal["narrator", "dialogue", "sfx", "ambience", "bgm"]
    asset_id: str
    asset_filename: str
    start_ms: int = Field(ge=0, le=MAX_TIMELINE_MS)
    trim_start_ms: int = Field(default=0, ge=0, le=MAX_TIMELINE_MS)
    duration_ms: int = Field(gt=0, le=MAX_TIMELINE_MS)
    playback_rate: float = Field(default=1.0, ge=0.5, le=2.0)
    gain_db: float = Field(default=0.0, ge=-60.0, le=12.0)
    loop: bool = False
    spatial: SpatialPlan

    @field_validator("id", "asset_id")
    @classmethod
    def validate_safe_id(cls, value: str) -> str:
        """限制 ID 为可安全写入日志和内部文件名的 ASCII 子集。"""
        if not SAFE_ID_PATTERN.fullmatch(value):
            raise ValueError("必须以字母或数字开头，且只能包含字母、数字、点、下划线或连字符")
        return value

    @field_validator("asset_filename")
    @classmethod
    def validate_asset_filename(cls, value: str) -> str:
        """资产只能引用本次 multipart 上传中的安全 basename。"""
        if not value or len(value) > 255:
            raise ValueError("必须是 1 到 255 个字符")
        if (
            value != PurePath(value).name
            or value in {".", ".."}
            or any(separator in value for separator in ("/", "\\", ":"))
        ):
            raise ValueError("必须是安全 basename，不能包含目录")
        lowered = value.lower()
        if "://" in lowered or lowered.startswith(("file:", "data:", "blob:")):
            raise ValueError("不能是 URL 或 URI")
        if "\x00" in value or any(ord(char) < 32 for char in value):
            raise ValueError("不能包含控制字符")
        return value

    @field_validator("playback_rate", "gain_db")
    @classmethod
    def validate_finite_number(cls, value: float) -> float:
        """FFmpeg 和 renderer 参数不接受 NaN 或无穷值。"""
        if not math.isfinite(value):
            raise ValueError("必须是有限数值")
        return value

    @model_validator(mode="after")
    def validate_kind_and_mode(self) -> SpatialSource:
        """稳定立体声床不能被误当作点声源，点声源也不能声明循环。"""
        if self.kind == "bgm" and self.spatial.mode != "preserve_stereo":
            raise ValueError("bgm 必须使用 preserve_stereo")
        if self.loop and self.kind not in {"bgm", "ambience"}:
            raise ValueError("只有 bgm 或 ambience 可以循环")
        if self.spatial.mode == "preserve_stereo" and self.kind not in {"bgm", "ambience"}:
            raise ValueError("只有 bgm 或 ambience 可以使用 preserve_stereo")
        if self.spatial.mode == "diffuse":
            raise ValueError("首版 renderer 尚未实现 diffuse，mode 必须使用 point 或 dry_center")
        if self.spatial.occlusion != "none":
            raise ValueError("首版 renderer 尚未实现遮挡，occlusion 必须为 none")
        return self


class SpatialRenderManifest(StrictModel):
    """8300 与 Steam Audio CLI 共用的 Render Manifest v1。"""

    version: Literal["1.0"] = MANIFEST_VERSION
    sample_rate: Literal[48000] = 48000
    timeline_duration_ms: int = Field(gt=0, le=MAX_TIMELINE_MS)
    scene: SpatialScene = Field(default_factory=SpatialScene)
    sources: list[SpatialSource] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_manifest_consistency(self) -> SpatialRenderManifest:
        """拒绝重复对象、资产映射冲突和越过总时间线的对象。"""
        source_ids: set[str] = set()
        filename_by_asset_id: dict[str, str] = {}
        for source in self.sources:
            if source.id in source_ids:
                raise SpatialManifestError(f"source id 重复: {source.id}")
            source_ids.add(source.id)

            previous_filename = filename_by_asset_id.setdefault(
                source.asset_id, source.asset_filename
            )
            if previous_filename != source.asset_filename:
                raise SpatialManifestError(f"asset_id {source.asset_id} 映射到多个上传文件")

            if source.start_ms + source.duration_ms > self.timeline_duration_ms:
                raise SpatialManifestError(f"source {source.id} 超出 timeline_duration_ms")
        return self

    def referenced_filenames(self) -> set[str]:
        """返回 Manifest 实际引用的 multipart 文件名集合。"""
        return {source.asset_filename for source in self.sources}
