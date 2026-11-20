## MODIFIED Requirements

### Requirement: 隔离标记阻止一切 GPU 推理

`.quarantine` 标记存在时，任何 GPU provider MUST NOT 推理。LocalAI MUST 确认 backend 已关闭；
sd.cpp MUST 按进程身份核验；仅使用 `scripts/localai/recover_quarantine.py` 完成确认后移除标记。
llama.cpp provider 的隔离恢复分支 MUST NOT 存在，因为对应 backend 已下线。

#### Scenario: backend 停止状态不能确认

- **WHEN** provider 无法确认 backend 已退出
- **THEN** 保留锁旁的 `.quarantine` 标记，所有后续 GPU 请求被拒绝

#### Scenario: 执行恢复流程

- **WHEN** 维护者运行 `scripts/localai/recover_quarantine.py`
- **THEN** 脚本按 LocalAI 或 sd.cpp 既有流程核验 backend 退出状态，确认后才允许移除隔离标记

## REMOVED Requirements

### Requirement: Ornith 请求持锁期间完成生命周期确认

**Reason**: Ornith 适配器与 llama.cpp backend 已删除，不再存在持锁的 Ornith 请求及其 sleep-idle/quarantine 生命周期收尾。
