# Dagu 独立作业调度

## Purpose

覆盖可选 Dagu 调度层以 HTTP 调用 MOSS 与 VoxCPM2 的作业合同：固定版本与 loopback 边界、唯一 job_id
与入队、四类状态文件、失败不重试、队列/超时限制，以及不调用真实模型的验证边界。
Dagu 不改变模型服务边界，见 [service-lifecycle-and-startup](../service-lifecycle-and-startup/spec.md)。

## 实现依据

- [scripts/dagu/start.sh](../../../scripts/dagu/start.sh)
- [scripts/dagu/http_worker.py](../../../scripts/dagu/http_worker.py)
- [scripts/dagu/job.py](../../../scripts/dagu/job.py)
- [dagu/config.yaml](../../../dagu/config.yaml)
- [dagu/dags/moss.yaml](../../../dagu/dags/moss.yaml)
- [dagu/dags/voxcpm.yaml](../../../dagu/dags/voxcpm.yaml)
- [docs/dagu.md](../../../docs/dagu.md)
- [tests/dagu_smoke.py](../../../tests/dagu_smoke.py)
- [tests/test_dagu_jobs.py](../../../tests/test_dagu_jobs.py)

## Requirements

### Requirement: Dagu 只经 HTTP 调用既有模型 API

Dagu 是可选后台调度层，MUST 先由 `bash start.sh` 启动既有模型服务；Dagu MUST 只经 HTTP 调用 MOSS
与 VoxCPM2，MUST NOT 加载模型、MUST NOT 提前获取 GPU 锁。GPU 锁、backend 收尾与 quarantine 仍由模型
服务负责；Dagu 故障 MUST NOT 改变直接调用入口，Dagu 以外的直接请求也仍受共享 GPU 锁保护。

#### Scenario: Dagu 队列触发任务

- **WHEN** Dagu 队列触发一次生成任务
- **THEN** 任务通过既有 HTTP API 完成，锁与生命周期仍由既有服务管理

### Requirement: 固定版本且只监听 loopback

Dagu MUST 使用固定 2.26.0 Linux amd64 发布包并核验其 SHA-256；二进制 MUST 放在约定的
`~/.local/opt/llm-dagu/2.26.0/dagu` 或由 `DAGU_BIN` 指定同版本；服务默认 MUST 只监听
`127.0.0.1:18673`，状态与认证数据默认位于 `~/.local/share/llm-dagu/server`，MUST NOT 向不可信网络开放
可执行命令的 Web UI。

#### Scenario: 版本不匹配

- **WHEN** 发布包摘要不符或 `DAGU_BIN` 指向其他版本
- **THEN** 启动不被视为合法部署，需先核验同版本二进制

### Requirement: 入队使用全局唯一 job_id 并校验输入

每个任务 MUST 使用全局唯一 job_id：首字符为字母或数字，后续允许字母、数字、下划线与短横线，最多
80 字符；输入 JSON 每任务一个且最大 1 MiB。MOSS MUST 接受 `prompt`，VoxCPM2 MUST 接受 `text` 及可选
`language`/`instruction`，本阶段 MUST NOT 支持参考音频。长度等合同限制 MUST 在登记前校验，校验失败
MUST NOT 占用任务 ID；已存在的 ID MUST NOT 被覆盖。

#### Scenario: 校验失败的入队

- **WHEN** 输入超过 1 MiB、字段不合法或 ID 已被占用
- **THEN** 任务不被登记，已有用户输入与任务目录保持不变

### Requirement: 作业状态机区分登记、提交与完成

`request.json` MUST 保存本次输入快照；`submitted.json` MUST 保存快照 SHA-256、模型端点、Dagu run ID
与登记时间，仅代表请求可能已提交，MUST NOT 作为成功证明；`completed.json` MUST 只在媒体通过数据块、
CRC 与 FFmpeg CPU 解码（PNG）或完整非空 PCM（WAV）校验且产物发布后原子写入，包含产物 SHA-256。
记录与产物 MUST 用独占链接发布并同步文件与目录，同名产物 MUST NOT 被覆盖。

#### Scenario: 成功发布

- **WHEN** 媒体校验通过且产物原子发布
- **THEN** `completed.json` 写入产物 SHA-256，结果位于 `jobs/<job_id>/result.png` 或 `result.wav`

#### Scenario: 仅存在 submitted.json

- **WHEN** 任务目录只有 `submitted.json` 而无 `completed.json`
- **THEN** 状态视为未确认成功，MUST NOT 推断完成或自动重试

### Requirement: 失败不重试且需人工核对

两个 DAG MUST NOT 启用自动重试；任务目录 MUST 在网络请求前独占创建，同一 ID MUST NOT 再次执行，即使
首次失败或超时。捕获错误时 MUST 尽力原子写入 `failure.json`，记录 registering、request、validation 或
publishing 阶段与错误类型，request 阶段一律保守标记为 unknown；进程被强杀或磁盘故障可能没有失败记录，
缺少失败记录 MUST NOT 视为成功。停止 Dagu 步骤或 HTTP 客户端 MUST NOT 被当作服务端推理已停止；恢复时
MUST 检查日志、任务目录、模型服务与 GPU quarantine，并用新的 ID 人工提交。

#### Scenario: 网络请求超时

- **WHEN** HTTP 客户端等待超过上限
- **THEN** 只终止客户端，不标记服务端推理已取消，不自动用同 ID 重跑

### Requirement: 队列与传输限制显式

`gpu-heavy` 队列 MUST 同时只运行一个已入队 DAG，`cpu-post` 最多三个并发且当前无 CPU 工作流；HTTP 客户端
MUST 经独立子进程流式接收、响应上限 64 MiB、总等待上限 7200 秒（含建连、提交与接收），媒体校验另有
最长 30 秒 FFmpeg 解码时限。默认服务根地址 MUST 指向 loopback，可由 `DAGU_MOSS_BASE_URL`、
`DAGU_VOXCPM_BASE_URL` 在启动前显式覆盖；脚本日志 MUST NOT 写入提示词，快照原文需保护与备份。

#### Scenario: 响应超过上限

- **WHEN** 上游响应超过 64 MiB 或等待超过 7200 秒
- **THEN** 客户端终止并报告错误，不把该错误写成模型已完成

### Requirement: 验证不调用真实模型

作业状态机 MUST 由 `tests/test_dagu_jobs.py` 覆盖，已安装固定版本 Dagu 时可另用 `tests/dagu_smoke.py`
在独立临时目录与假 HTTP 服务验证真实入队；联调脚本 MUST NOT 启动 Web UI、MUST NOT 调用真实模型。
无模型验证通过 MUST NOT 写成已通过真实推理联调。

#### Scenario: 无 Dagu 环境执行测试

- **WHEN** 环境未安装固定版本 Dagu
- **THEN** 无模型作业状态测试仍执行，真实入队联调明确跳过且不视为已完成
