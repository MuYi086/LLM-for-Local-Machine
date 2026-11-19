# 全局 GPU 锁与隔离恢复

## Purpose

覆盖跨进程共享 GPU 锁的获取与释放顺序、请求队列超时、`.quarantine` 隔离标记的语义与解除流程，
以及不取锁的执行边界。

## 实现依据

- [llm_local_runtime/src/llm_local_runtime/gpu.py](../../../llm_local_runtime/src/llm_local_runtime/gpu.py)
- [docs/global-gpu-lock.md](../../../docs/global-gpu-lock.md)
- [scripts/localai/recover_quarantine.py](../../../scripts/localai/recover_quarantine.py)
- [inference_gateway/src/inference_gateway/execution.py](../../../inference_gateway/src/inference_gateway/execution.py)
- [inference_gateway/src/inference_gateway/config.py](../../../inference_gateway/src/inference_gateway/config.py)
- [inference_gateway/src/inference_gateway/errors.py](../../../inference_gateway/src/inference_gateway/errors.py)

## Requirements

### Requirement: 全部 GPU provider 共用同一把锁

所有 GPU provider MUST 使用同一个 `GPU_LOCK_FILE`，保证跨进程串行执行，避免显存 OOM；纯
CPU/文件预检 MUST 在获取锁前完成，持锁时间只保留给真正需要 GPU 的阶段。

#### Scenario: 两个服务同时请求 GPU

- **WHEN** 两个 GPU 请求并发到达
- **THEN** 后到者在队列中等待锁，先到者完成并释放后才开始执行，全程不并行占用 GPU

#### Scenario: 纯 CPU 预检

- **WHEN** 请求进入仅涉及文件、清单或配置的预检阶段
- **THEN** 预检在获取锁之前完成，不占用 GPU 队列等待时间

### Requirement: Ornith 请求持锁期间完成生命周期确认

Ornith 请求在持锁后 MUST 先确认 backend `/props.is_sleeping`，再提交推理；结束后 MUST 确认模型
卸载并等待 `CUDA_RELEASE_DELAY`，最终在 finally 中释放锁。

#### Scenario: 正常推理请求

- **WHEN** Ornith 请求获得 GPU 锁
- **THEN** 依次完成 sleep 状态确认、推理、卸载确认与延迟等待后释放锁

#### Scenario: 推理中途异常

- **WHEN** 推理或卸载确认抛出异常
- **THEN** 锁仍由 finally 释放，不因异常路径造成锁泄漏

### Requirement: 队列等待有界且显式

GPU 队列默认等待 900 秒；部署值 MUST 为正数；超时请求 MUST 返回 503，不得无限排队。

#### Scenario: 队列超时

- **WHEN** 请求在队列中等待超过部署的等待上限
- **THEN** 请求返回 503，客户端可感知并自行重试

#### Scenario: 非法部署值

- **WHEN** 队列等待被配置为非正数
- **THEN** 启动或配置校验失败，拒绝以非法值运行

### Requirement: 隔离标记阻止一切 GPU 推理

`.quarantine` 标记存在时，任何 GPU provider MUST NOT 推理；LocalAI MUST 确认 backend 已关闭；
Ornith MUST 确认 llama.cpp 进程组和各自 loopback API 已退出；标记 MUST 仅在
`scripts/localai/recover_quarantine.py` 完成上述确认后移除。

#### Scenario: backend 停止状态不能确认

- **WHEN** provider 无法确认 backend 已退出
- **THEN** 保留锁旁的 `.quarantine` 标记，所有后续 GPU 请求被拒绝

#### Scenario: 执行恢复流程

- **WHEN** 维护者运行 `scripts/localai/recover_quarantine.py`
- **THEN** 脚本核验进程组与 loopback API 退出状态，确认后才允许移除隔离标记

### Requirement: CLI provider 按进程身份核验

sdcpp CLI provider 无 HTTP 端口；其隔离与恢复 MUST 按 PID/starttime 与进程组核验，
并统一使用既有 `recover_quarantine.py` 入口，不引入第二套恢复工具。

#### Scenario: worker 退出未确认

- **WHEN** 图片生成的 worker 进程未确认退出
- **THEN** 保留 quarantine 标记，按统一恢复流程处理

### Requirement: 网关配置在启动前拒绝不安全值

`inference_gateway.config` MUST 校验 provider 只取 `native` 或 `localai`，`localai` 必须显式设置
`LLM_LOCALAI_ENABLED=true` 才可用；`LLM_LOCALAI_BASE_URL` MUST 是不带凭据、无 query/fragment、
无无效端口的本机 HTTP 回环地址；LocalAI model 名称 MUST 只含字母、数字、点、下划线和短横线；
连接/请求/清理超时与 `GPU_LOCK_WAIT_TIMEOUT` MUST 为有限正数，`CUDA_RELEASE_DELAY` MUST 为有限
非负数。非法配置 MUST 抛出配置错误并阻止启动，MUST NOT 进入半可用状态。

#### Scenario: 配置外部 LocalAI 地址

- **WHEN** `LLM_LOCALAI_BASE_URL` 指向非 loopback 地址或包含凭据
- **THEN** 网关拒绝该配置，不向外部地址发起任何推理请求

#### Scenario: provider 或超时非法

- **WHEN** provider 非 `native`/`localai`，localai 未显式启用，或任一超时非正数
- **THEN** 启动失败并报告具体变量，不进入部分启动状态

### Requirement: 网关错误分类保持隔离语义

网关 MUST 使用稳定错误类型区分配置错误、provider 能力不支持、LocalAI 不可用、LocalAI 请求或
响应不符合约定，以及 backend 停止无法确认的隔离错误（`BackendQuarantinedError`）。隔离类错误
MUST 阻止后续 GPU 推理，并只在恢复流程确认后解除；普通上游错误 MUST NOT 被误报为隔离恢复成功。

#### Scenario: 后端停止无法确认

- **WHEN** provider 无法确认 backend 已退出
- **THEN** 返回隔离类错误并保留 `.quarantine` 标记，后续 GPU 请求被拒绝

### Requirement: 不取锁的执行保持边界

Steam Audio renderer 是 CPU 实现，MUST NOT 获取 GPU 锁；默认 MUST NOT 并行执行多个 GPU 音频
模型；训练代码 MUST NOT 自动取得服务 GPU 锁，执行真实训练/预测需单独安排。

#### Scenario: CPU 渲染与 GPU 推理并发

- **WHEN** Steam Audio 渲染进行中另有 GPU 推理请求
- **THEN** 渲染不申请 GPU 锁，GPU 排队不受渲染影响

#### Scenario: 触发训练脚本

- **WHEN** 维护者运行文本润色 LoRA 实验代码
- **THEN** 该代码不获取服务 GPU 锁，也不影响在线服务的排队行为
