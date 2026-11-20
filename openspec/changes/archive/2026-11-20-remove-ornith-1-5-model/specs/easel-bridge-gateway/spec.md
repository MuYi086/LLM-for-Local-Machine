## MODIFIED Requirements

### Requirement: 路由只代理既有模型 API

网关 MUST 只提供 `GET /v1/models`、`POST /v1/images/generations`、`POST /v1/audio/speech`，
并分别转发给既有 MOSS-Image-SFT（8501）与 VoxCPM2（8313）API；MUST NOT 直接加载模型或绕过共享 GPU 锁。
网关 MUST NOT 提供聊天路由，也 MUST NOT 转发任何本地聊天模型请求。

#### Scenario: 语音合成请求

- **WHEN** 客户端经 `POST /v1/audio/speech` 提交文本
- **THEN** 网关把请求转发给 VoxCPM2 既有 API 并透传响应，自身不触发模型加载

#### Scenario: 越白名单的模型请求

- **WHEN** 请求参数引用白名单之外的模型或能力
- **THEN** 网关拒绝该请求，不尝试构造对未接入服务的调用

#### Scenario: 聊天路由已移除

- **WHEN** 客户端请求 `POST /v1/chat/completions`
- **THEN** 网关返回 404，不连接任何已下线的聊天服务

### Requirement: 上游默认绑定 loopback

网关转发的上游地址 MUST 默认指向 loopback（`EASEL_BRIDGE_MOSS_URL`、`EASEL_BRIDGE_VOXCPM_URL`），
避免从网关之外绕过 Bearer 鉴权；覆盖绑定属于显式部署决策。聊天上游配置 `EASEL_BRIDGE_MTP_CHAT_URL`
MUST NOT 存在。

#### Scenario: 默认部署

- **WHEN** 按默认环境变量启动
- **THEN** 两个上游 URL 均指向本机回环地址，外部无法直接访问被代理的模型 API
