## MODIFIED Requirements

### Requirement: 启动预检完备后才启动服务

start.sh MUST 自动读取 `.env.easel-bridge` 并校验网关 Key，检查全部十九个 HTTP 端口、
sd.cpp 可执行文件摘要和 MOSS、FLUX.3 Klein 各自四个模型组件，注册十五个 LocalAI 候选并执行 readiness 预检；
任一预检失败 MUST NOT 继续启动。start.sh MUST NOT 预检或启动 Ornith MTP 服务、llama-server、GGUF、mmproj
或 sleep-cache 补丁。

#### Scenario: 预检通过

- **WHEN** 端口无冲突、权重与可执行文件齐全
- **THEN** 预检通过并进入服务启动阶段

#### Scenario: 端口被占用

- **WHEN** 预检发现任一服务端口已被占用
- **THEN** 启动终止并报告冲突端口，不进入部分启动状态

#### Scenario: 网关 Key 缺失或过短

- **WHEN** `.env.easel-bridge` 缺少至少 32 字节的随机 `EASEL_BRIDGE_API_KEY`
- **THEN** 启动预检失败，网关不以匿名或弱 Key 状态运行

### Requirement: 退出清理确认进程组

任一服务退出时 MUST 清理全部服务进程组；统一启动依赖 setsid、进程组信号和 fcntl.flock；使用 Ctrl+C
等待正常退出。Ornith llama.cpp backend 的停止确认与相关 quarantine 分支 MUST NOT 存在；
LocalAI 与 sd.cpp 的隔离恢复语义保持不变。

#### Scenario: 正常退出

- **WHEN** 维护者以 Ctrl+C 结束 start.sh
- **THEN** 全部服务进程组收到清理信号，剩余 GPU provider 的隔离状态确认后进程退出

#### Scenario: backend 停止状态不能确认

- **WHEN** 退出时共享 GPU 锁旁仍存在 `.quarantine` 标记
- **THEN** 仅按 LocalAI 或 sd.cpp 既有恢复流程处理，不假设退出成功

## REMOVED Requirements

### Requirement: Ornith 预热先于全部 API

**Reason**: Ornith backend 已下线，start.sh 不再预热任何 llama.cpp backend，预热顺序要求整体移除。
