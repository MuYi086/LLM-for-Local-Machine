# 通用空间轨迹 v1

轨迹在 CPU Steam Audio renderer 中求值，适用于音乐、对白和音效；不修改或运行生成模型。HTTP Manifest 保持 `version: "1.0"`，新增可选 `sources[].spatial.trajectory`，省略时沿用原有预设。

## 坐标与时间

听者固定在原点，x 向右、y 向上、z 向后；前方为负 z。位置/距离单位米，时间单位秒。方位角 0° 在前方、+90° 在右侧，仰角为正时向上。轨迹时间从本声源裁剪后的播放起点开始，与 Manifest 的 `start_ms` 无关。

每个音频块按中点时间求值，运动方向使用双线性 HRTF 插值。`once` 到终点后保持；`loop` 重复闭合路径；`ping_pong` 原路往返。`interpolation` 支持 `linear`（默认）和 `smoothstep`（每段缓入缓出）。这些设置仅用于运动轨迹；静态轨迹不接受它们。

位置距离限制 0.25–1000 米，持续时间不超过 7200 秒，节点数 2–1024。拒绝非有限值、未知字段、非递增节点时间和穿过听者 0.25 米范围的线段。开放路径不允许 `loop`，可使用 `ping_pong`。圆周角度保持连续，例如 170→190 表示跨越后方的短弧，0→720 表示两圈。

## 六类 JSON

```json
{"type":"static","position_m":[1,0,-1]}
```
```json
{"type":"linear","start_m":[-4,0,-2],"end_m":[4,0,-2],"duration_seconds":12,"repeat":"ping_pong","interpolation":"smoothstep"}
```
```json
{"type":"approach","azimuth_degrees":30,"elevation_degrees":0,"start_distance_m":8,"end_distance_m":1,"duration_seconds":12}
```
```json
{"type":"recede","azimuth_degrees":-30,"start_distance_m":1,"end_distance_m":8,"duration_seconds":12}
```
```json
{"type":"circular","radius_m":1,"height_m":0,"start_angle_degrees":0,"end_angle_degrees":360,"duration_seconds":24,"repeat":"loop"}
```
```json
{"type":"waypoints","points":[{"time_seconds":0,"position_m":[-2,0,-2]},{"time_seconds":6,"position_m":[0,1,-2]},{"time_seconds":12,"position_m":[2,0,-2]}],"repeat":"ping_pong","interpolation":"smoothstep"}
```

径向轨迹的方位角和仰角默认 0；圆周高度/起角默认 0，终角默认 360。节点必须从 0 秒开始，最后一个节点决定时长。完整合同见 `main/spatial_trajectory.py`；几何实现见 `model-audio/spatial_steam-audio/src/trajectory.cpp`。

## HTTP 接入

在 `/v1/audio/spatial/render` 的原 Manifest 单个声源中设置：

```json
"spatial": {
  "mode": "point", "movement": "static",
  "trajectory": {"type":"circular","radius_m":1,"duration_seconds":24,"repeat":"loop"}
}
```

显式轨迹要求 `mode=point`、`movement=static`，避免两种运动定义冲突。选择 `balanced` 或 `immersive` profile；`standard` 仍按原约定直接混音，不应用空间轨迹。其他输入、裁剪、声源增益和 HTTP 最终响度处理不变。

## 原生 CLI（音乐适配接口）

先按 renderer README 构建，再调用：

```bash
model-audio/spatial_steam-audio/build/steam-audio-render --info
model-audio/spatial_steam-audio/build/steam-audio-render --validate-trajectory trajectory.json
model-audio/spatial_steam-audio/build/steam-audio-render \
  --input mono-48k.wav --trajectory trajectory.json \
  --output binaural.wav --trajectory-report trajectory.tsv
```

输入为 48 kHz 单声道 WAV，轨迹文件至多 1 MiB。输出和轨迹报告必须是新路径。`--trajectory` 与旧 `--position`、`--listener`、`--manifest` 互斥。新模式返回单位混合增益 float32 双耳 WAV，不做峰值归一化，允许中间结果超过 ±1；最终交付方负责限幅和响度。保留 SDK 查询的 HRTF 尾长，并额外保留 4096 帧供距离/空气吸收 IIR 衰减；最后半块残留须低于 max(1e-8, 输入峰值×1e-7)，否则报错。`direct_settle_samples` 单独记录这一补零预算，不能视为 SDK 查询的尾长。回执 `output_frames` 包含尾部，`tail_samples` 为尾部帧数；`duration_ms` 仍表示源时间线长度。TSV 记录实际块中点的局部时间和坐标。

旧单点 CLI 和 HTTP Manifest 保留既有混音增益与峰值处理；不因新增模式改变历史结果。

## 验证与边界

```bash
bash scripts/test_steam_audio_renderer_core.sh
uv run --project qa --locked python -m unittest discover -s tests -p 'test_steam*.py' -v
STEAM_AUDIO_TEST_BIN="$PWD/model-audio/spatial_steam-audio/build/steam-audio-render" \
  uv run --project qa --locked python -m unittest discover \
  -s model-audio/spatial_steam-audio/tests -p 'test_native_trajectory.py' -v
```

真实 SDK 测试检查左右方向、距离能量变化、单位增益线性、源局部时钟，以及整块/非整块 EOF 冲激尾部。SDK 4.8.1 的 [DirectEffect 尾部接口](https://github.com/ValveSoftware/steam-audio/blob/v4.8.1/core/src/core/direct_effect.cpp) 返回静音，因此显式零输入通过原滤波链完成衰减。定位感受仍需耳机试听；此实现不包含房间几何、反射、遮挡、头部追踪或多普勒。

已知低优先级限制：Python 使用双精度验证，原生坐标进入 SDK 前转换为 float；距离/闭合约束在约 1e-6 米边界附近可能出现不同接受结果。样例不涉及这些边界，原生 `--validate-trajectory` 可作为实际运行前的最终预检。


## 歌曲距离动态控制

原生单声源 `--trajectory` 模式可追加 `--distance-exponent 0.15`。允许有限数值 0–1，默认 1；实际距离增益为 Steam 默认距离增益的该次幂。0 关闭距离音量衰减，1 保留物理模型；空气吸收和 HRTF 不变。0.15 使 1–8 米的纯距离变化从约 18 dB 缩小到约 2.7 dB，适合已混好、需要保留固定声部的歌曲。

`--info` 的 `distance_exponent_control=true` 声明此能力，渲染回执包含实际 `distance_exponent`。参数不允许用于旧 `--position` 或 `--manifest` 模式，HTTP Manifest 继续使用原先的物理默认。歌曲端必须记录该参数，避免把旧成品当作新版复用。
