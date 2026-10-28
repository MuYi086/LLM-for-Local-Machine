"""通用轨迹合同；坐标为 x 右、y 上、z 后，单位米与秒。"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Duration = Annotated[Number, Field(gt=0, le=7200)]
Distance = Annotated[Number, Field(ge=0.25, le=1000)]
Angle = Annotated[Number, Field(ge=-360000, le=360000)]


def validate_position(value: tuple[float, float, float]) -> tuple[float, float, float]:
    """限制点到听者的距离，避免未定义方向与无界声学输入。"""
    if not 0.25 - 1e-6 <= math.hypot(*value) <= 1000 + 1e-6:
        raise ValueError("位置到听者的距离必须为 0.25–1000 米")
    return value


Position = Annotated[tuple[Number, Number, Number], AfterValidator(validate_position)]


def validate_segment(start: tuple, end: tuple) -> None:
    """验证整条线段，不只验证端点，以免轨迹穿过听者。"""
    delta = [b - a for a, b in zip(start, end, strict=True)]
    squared = sum(v * v for v in delta)
    progress = (
        max(0, min(1, -sum(a * d for a, d in zip(start, delta, strict=True)) / squared))
        if squared
        else 0
    )
    nearest = [a + progress * d for a, d in zip(start, delta, strict=True)]
    if math.hypot(*nearest) < 0.25 - 1e-6:
        raise ValueError("轨迹线段必须与听者保持至少 0.25 米距离")


class TrajectoryModel(BaseModel):
    """拒绝无关参数，避免拼错字段后悄悄使用默认轨迹。"""

    model_config = ConfigDict(extra="forbid")


class StaticTrajectory(TrajectoryModel):
    """固定在听者坐标中的一点。"""

    type: Literal["static"]
    position_m: Position


class MotionTrajectory(TrajectoryModel):
    """运动的时间外推和每段插值约定。"""

    repeat: Literal["once", "loop", "ping_pong"] = "once"
    interpolation: Literal["linear", "smoothstep"] = "linear"

    def validate_closed(self, start: tuple, end: tuple) -> None:
        """循环必须首尾相接；往返可以用于开放路径。"""
        if self.repeat == "loop" and math.dist(start, end) > 1e-6:
            raise ValueError("loop 轨迹必须首尾闭合；开放路径请使用 ping_pong")


class LinearTrajectory(MotionTrajectory):
    """在三维坐标中按直线运动，结束后按 repeat 处理。"""

    type: Literal["linear"]
    start_m: Position
    end_m: Position
    duration_seconds: Duration

    @model_validator(mode="after")
    def validate_path(self) -> LinearTrajectory:
        """验证直线路径和循环接缝。"""
        validate_segment(self.start_m, self.end_m)
        self.validate_closed(self.start_m, self.end_m)
        return self


class RadialTrajectory(MotionTrajectory):
    """沿固定方向靠近或远离，正方位角指向右侧。"""

    type: Literal["approach", "recede"]
    azimuth_degrees: Angle = 0
    elevation_degrees: Annotated[Number, Field(ge=-90, le=90)] = 0
    start_distance_m: Distance
    end_distance_m: Distance
    duration_seconds: Duration

    @model_validator(mode="after")
    def validate_path(self) -> RadialTrajectory:
        """名称与距离变化必须一致，循环回程应使用 ping_pong。"""
        delta = self.end_distance_m - self.start_distance_m
        if (self.type == "approach" and delta >= 0) or (self.type == "recede" and delta <= 0):
            raise ValueError("approach 必须距离减小，recede 必须距离增大")
        if self.repeat == "loop":
            raise ValueError("靠近/远离不闭合，请使用 once 或 ping_pong")
        return self


class CircularTrajectory(MotionTrajectory):
    """围绕听者竖直轴的水平圆弧，角度保持连续累加。"""

    type: Literal["circular"]
    radius_m: Distance
    height_m: Annotated[Number, Field(ge=-1000, le=1000)] = 0
    start_angle_degrees: Angle = 0
    end_angle_degrees: Angle = 360
    duration_seconds: Duration

    @model_validator(mode="after")
    def validate_path(self) -> CircularTrajectory:
        """验证三维距离和整圈循环；允许跨越 ±180 度的连续圆弧。"""
        validate_position((self.radius_m, self.height_m, 0))
        delta = self.end_angle_degrees - self.start_angle_degrees
        if self.repeat == "loop" and abs(math.remainder(delta, 360)) > 1e-6:
            raise ValueError("loop 圆周的起终角必须相差整圈")
        return self


class Waypoint(TrajectoryModel):
    """相对于本声源播放起点的空间节点。"""

    time_seconds: Annotated[Number, Field(ge=0, le=7200)]
    position_m: Position


class WaypointTrajectory(MotionTrajectory):
    """按严格递增的局部时间依次经过节点。"""

    type: Literal["waypoints"]
    points: list[Waypoint] = Field(min_length=2, max_length=1024)

    @model_validator(mode="after")
    def validate_path(self) -> WaypointTrajectory:
        """检查节点时间、每段最近距离和可选循环接缝。"""
        if self.points[0].time_seconds != 0:
            raise ValueError("首节点 time_seconds 必须为 0")
        for first, second in zip(self.points, self.points[1:], strict=False):
            if second.time_seconds <= first.time_seconds:
                raise ValueError("节点时间必须严格递增")
            validate_segment(first.position_m, second.position_m)
        self.validate_closed(self.points[0].position_m, self.points[-1].position_m)
        return self


Trajectory = Annotated[
    StaticTrajectory
    | LinearTrajectory
    | RadialTrajectory
    | CircularTrajectory
    | WaypointTrajectory,
    Field(discriminator="type"),
]
