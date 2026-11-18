# 控制面与 Steam Audio 空间渲染

## Purpose

覆盖 8300 控制面的职责边界（不加载模型、阻塞任务处理方式）、共享 CPU 导出，
以及 Steam Audio 空间渲染的调用方式、对象边界与未实现能力口径。

## 实现依据

- [main/main.py](../../../main/main.py)
- [main/gpu_runtime.py](../../../main/gpu_runtime.py)
- [main/spatial_audio.py](../../../main/spatial_audio.py)
- [main/spatial_schema.py](../../../main/spatial_schema.py)
- [main/spatial_trajectory.py](../../../main/spatial_trajectory.py)
- [main/steam_audio_render.py](../../../main/steam_audio_render.py)
- [docs/spatial-trajectories.md](../../../docs/spatial-trajectories.md)
- [model-audio/spatial_steam-audio/](../../../model-audio/spatial_steam-audio/)

## Requirements

### Requirement: 控制面不加载模型

main/ 控制面 MUST 只负责健康/控制、共享上传检查、CPU 导出和空间渲染编排，MUST NOT 加载模型；
主要接口为 `GET /v1/health`、`GET /v1/control`、`POST /v1/audio/export`、
`POST /v1/audio/spatial/render`。

#### Scenario: 诊断请求

- **WHEN** 客户端调用控制面诊断接口
- **THEN** 响应反映服务与配置状态，控制面进程的显存占用为零

#### Scenario: 共享 CPU 导出

- **WHEN** 客户端请求 `POST /v1/audio/export`
- **THEN** 控制面在 CPU 上完成导出编排，不触发任何模型加载

### Requirement: 阻塞任务不进入事件循环

阻塞锁、subprocess 和网络任务 MUST 使用同步路由或线程池，MUST NOT 阻塞异步事件循环。

#### Scenario: 渲染期间到达轻量请求

- **WHEN** 空间渲染或导出阻塞任务执行中
- **THEN** 健康检查等轻量请求仍可及时响应

### Requirement: Steam Audio 按需调用且不取 GPU 锁

Steam Audio renderer 由控制面按需调用，MUST NOT 作为 HTTP 服务暴露；它是使用仓库外 Steam Audio
SDK 与动态库的 C++17 CPU renderer，MUST NOT 获取 GPU 锁。

#### Scenario: 发起空间渲染

- **WHEN** 客户端调用 `POST /v1/audio/spatial/render`
- **THEN** 控制面调用 CPU renderer 并回报进度，GPU 队列不出现该渲染的锁请求

### Requirement: 正式路径保留对象边界

Steam Audio 正式 balanced/immersive 路径 MUST 保留对象边界，使用 Manifest v1 和重复 assets
字段，MUST NOT 在浏览器预混。

#### Scenario: 提交多对象场景

- **WHEN** 场景 Manifest 包含多个对象与重复 assets 字段
- **THEN** renderer 按对象边界渲染，不把对象预先混合成单轨

### Requirement: pre-master 后只允许响度归一与编码

Steam Audio pre-master 之后 MUST 只允许 loudnorm/编码；MUST NOT 叠加 Haas/aecho 等后处理效果。

#### Scenario: 导出阶段处理

- **WHEN** 渲染产物进入导出
- **THEN** 导出链只执行 loudnorm 与编码，不追加效果器

### Requirement: 未实现能力不写入可用口径

diffuse、非 none 遮挡和几何反射尚未实现；保留的语义字段 MUST NOT 被描述为可用能力。

#### Scenario: 请求未实现的遮挡语义

- **WHEN** 渲染请求包含非 none 遮挡或几何反射字段
- **THEN** 该请求被拒绝或忽略并明确说明能力未实现，不产生伪造的渲染结果

### Requirement: 空间轨迹合同以文档为准

空间轨迹的坐标系、时间语义与六类轨迹 JSON 合同 MUST 以
[spatial-trajectories 文档](../../../docs/spatial-trajectories.md) 为准；实现变化时同步该文档。

#### Scenario: 轨迹校验

- **WHEN** 渲染请求携带轨迹 JSON
- **THEN** 控制面按文档合同校验并返回进度或明确的校验错误
