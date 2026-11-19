# Easel 网关安全边界

## Purpose

覆盖 `easel_bridge` 网关（127.0.0.1:8500）的监听与鉴权约束、OpenAI-compatible 路由的白名单转发，
以及不得绕过既有模型服务与 GPU 锁的边界。

## 实现依据

- [easel_bridge/app.py](../../../easel_bridge/app.py)
- [docs/easel-bridge.md](../../../docs/easel-bridge.md)
- [.env.easel-bridge.example](../../../.env.easel-bridge.example)

## Requirements

### Requirement: 只监听 loopback 且强制随机 Key

网关 MUST 只监听 127.0.0.1，默认端口 8500，可由 `EASEL_BRIDGE_PORT` 覆盖为 1–65535 的合法端口，
MUST NOT 绑定非 loopback 地址；启动 MUST 设置至少 32 字节的随机 `EASEL_BRIDGE_API_KEY`；
MUST NOT 匿名运行，MUST NOT 把该 Key 转发到上游。

#### Scenario: 缺少网关 Key

- **WHEN** 启动配置缺少 `EASEL_BRIDGE_API_KEY`
- **THEN** 网关拒绝启动或启动预检失败，不进入匿名可用状态

#### Scenario: 客户端携带 Bearer Key 请求

- **WHEN** 客户端以正确 Bearer Key 访问网关路由
- **THEN** 请求按白名单转发；转发请求中只含上游模型 API 所需字段，不含网关自身 Key

#### Scenario: 覆盖监听端口

- **WHEN** 部署显式设置合法范围内的 `EASEL_BRIDGE_PORT`
- **THEN** 网关只在该端口的 127.0.0.1 上监听，仍不绑定外部网卡，鉴权要求不变

### Requirement: 路由只代理既有模型 API

网关 MUST 只提供 `GET /v1/models`、`POST /v1/images/generations`、`POST /v1/audio/speech`、
`POST /v1/chat/completions`，并分别转发给既有 MOSS-Image-SFT（8501）、VoxCPM2（8313）与
Ornith MTP 9B（8504）API；MUST NOT 直接加载模型或绕过共享 GPU 锁。

#### Scenario: 语音合成请求

- **WHEN** 客户端经 `POST /v1/audio/speech` 提交文本
- **THEN** 网关把请求转发给 VoxCPM2 既有 API 并透传响应，自身不触发模型加载

#### Scenario: 越白名单的模型请求

- **WHEN** 请求参数引用白名单之外的模型或能力
- **THEN** 网关拒绝该请求，不尝试构造对未接入服务的调用

### Requirement: 兼容请求字段有明确边界

OpenAI-compatible 请求 MUST 按既有文档合同处理：图片 `n` 最多 4 且使用 `WIDTHxHEIGHT` 尺寸，每张依次
生成；语音接受 `voice=alloy` 占位值，仅支持默认 `speed=1`，输出 WAV 或 MP3，MP3 MUST 由本机 FFmpeg
从 VoxCPM2 的 WAV 响应转换，MUST NOT 写临时音频文件；请求体上限 12 MiB、响应体上限 64 MiB、
上游超时（`EASEL_BRIDGE_TIMEOUT`，默认 900 秒）、FFmpeg 二进制（`EASEL_BRIDGE_FFMPEG_BIN`）与转码
超时（`EASEL_BRIDGE_FFMPEG_TIMEOUT`，默认 120 秒）MUST 为正数配置，等待上游或转码超时 MUST 返回
504；HTTP 重定向 MUST 关闭。越界参数 MUST 在转发前拒绝，MUST NOT 向上游伪造不支持的能力。

#### Scenario: 请求速度调节

- **WHEN** 客户端对 `POST /v1/audio/speech` 提交非默认 `speed`
- **THEN** 请求按仅支持默认 `speed=1` 的合同处理，不向 VoxCPM2 声称支持速度调节

#### Scenario: 请求 MP3 输出

- **WHEN** 客户端请求 MP3
- **THEN** 网关用本机 FFmpeg 转换 WAV 响应，不写临时音频文件

### Requirement: 上游默认绑定 loopback

网关转发的上游地址 MUST 默认指向 loopback（`EASEL_BRIDGE_MOSS_URL`、`EASEL_BRIDGE_VOXCPM_URL`、
`EASEL_BRIDGE_MTP_CHAT_URL`），避免从网关之外绕过 Bearer 鉴权；覆盖绑定属于显式部署决策。

#### Scenario: 默认部署

- **WHEN** 按默认环境变量启动
- **THEN** 三个上游 URL 均指向本机回环地址，外部无法直接访问被代理的模型 API

### Requirement: 配置说明与网关实现同步

网关的配置项、模型填写项与兼容限制 MUST 以 [docs/easel-bridge.md](../../../docs/easel-bridge.md)
为准；实现变化时同步该文档与 `.env.easel-bridge.example`。

#### Scenario: 新增或调整网关路由

- **WHEN** 网关路由或转发行为变化
- **THEN** docs/easel-bridge.md 与示例环境文件同步更新，避免文档落后于实现
