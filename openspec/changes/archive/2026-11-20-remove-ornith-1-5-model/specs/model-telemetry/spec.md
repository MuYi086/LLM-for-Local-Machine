## MODIFIED Requirements

### Requirement: 简报出口只含固定指标

收尾简报 MUST 只输出固定白名单指标，MUST NOT 携带请求正文、逐进程完整记录或任意额外字段。
资源指标保留采样数、整卡显存峰值、guest 内存峰值；MUST NOT 输出 `ornith_backend` 后端 RSS 峰值
或任何 llama.cpp 专属字段。简报 MUST 为可被 `json.loads` 解析的单行 JSON 且不超过 4096 UTF-8 字节。

#### Scenario: 归档简报

- **WHEN** 对含大量原始记录的报告执行 `end --brief`
- **THEN** 简报只含 `test_id`、`test_state`、`window_seconds`、服务固定指标与资源白名单，无原始记录泄漏

### Requirement: 记录最小且写入隔离

采集器 MUST NOT 记录请求正文、环境变量或命令行等敏感字段；进程角色仅用固定标签。
MUST NOT 记录 `ornith_backend`/`ornith_adapter` 角色、`ORNITH_MTP_9B_*` 配置快照、
Ornith 模型文件信息或 MTP 原生日志解析结果。LocalAI、sd.cpp 与其余模型的采集口径不变。

#### Scenario: 记录敏感字段

- **WHEN** 采集器读取进程、日志与环境变量
- **THEN** 不持久化 Ornith 专属角色/配置/模型文件字段，不把命令行或环境变量原文写入记录
