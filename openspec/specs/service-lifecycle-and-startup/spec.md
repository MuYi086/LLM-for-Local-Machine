# 服务生命周期与启动

## Purpose

覆盖 `start.sh` 的启动预检、Ornith backend 预热顺序、就绪确认、退出清理与进程组管理，
以及 Dagu 可选调度层的服务边界。

## 实现依据

- [start.sh](../../../start.sh)
- [scripts/localai/manager.py](../../../scripts/localai/manager.py)
- [scripts/localai/preflight.py](../../../scripts/localai/preflight.py)
- [main/check_ports.py](../../../main/check_ports.py)
- [.env.easel-bridge.example](../../../.env.easel-bridge.example)
- [scripts/llama_cpp/build_sleep_cache.py](../../../scripts/llama_cpp/build_sleep_cache.py)
- [docs/dagu.md](../../../docs/dagu.md)
- [README.md](../../../README.md)
- [scripts/dagu/start.sh](../../../scripts/dagu/start.sh)

## Requirements

### Requirement: 启动预检完备后才启动服务

start.sh MUST 自动读取 `.env.easel-bridge` 并校验网关 Key，检查全部二十一个 HTTP 端口、
sd.cpp 可执行文件摘要和 MOSS、FLUX.3 Klein 各自四个模型组件、Ornith MTP 主模型、MTP projector
和 CUDA llama-server，注册十五个 LocalAI 候选并执行 readiness 预检；任一预检失败 MUST NOT
继续启动。

#### Scenario: 预检通过

- **WHEN** 端口无冲突、权重与可执行文件齐全
- **THEN** 预检通过并进入 backend 预热阶段

#### Scenario: 端口被占用

- **WHEN** 预检发现任一服务端口已被占用
- **THEN** 启动终止并报告冲突端口，不进入部分启动状态

#### Scenario: 网关 Key 缺失或过短

- **WHEN** `.env.easel-bridge` 缺少至少 32 字节的随机 `EASEL_BRIDGE_API_KEY`
- **THEN** 启动预检失败，网关不以匿名或弱 Key 状态运行

### Requirement: 启动日志实时同步且权限受控

每次 `start.sh` MUST 把完整控制台输出实时同步到 `$RUNTIME_CACHE_DIR/logs/` 下的独立日志文件
（默认 `storage/.cache/runtime/logs/`），并在启动时打印本次日志完整路径；日志文件 MUST 由 `mktemp`
创建且默认仅当前用户可读写。日志进程 MUST 避开终端退出信号，由主脚本统一清理服务；日志管道异常时
清理过程 MUST 追加到同一文件。Ctrl+C、终端挂断及清理期间再次 Ctrl+C MUST 继续完成已启动进程组的
退出确认。

#### Scenario: 启动后收集诊断日志

- **WHEN** 服务异常需要排查
- **THEN** 终端打印的本次启动日志文件存在且仅当前用户可读，包含完整控制台输出

#### Scenario: 终端挂断或中断

- **WHEN** 清理期间终端挂断或再次按下 Ctrl+C
- **THEN** 不丢失日志，继续完成进程组退出确认，不把清理日志写入另一文件

### Requirement: Ornith 预热先于全部 API

start.sh MUST 通过共享 GPU 锁预热 Ornith MTP backend，确认模型进入 sleep-idle 后再启动
控制面和全部 API；预热未完成 MUST NOT 对外提供 API。

#### Scenario: 预热成功

- **WHEN** backend 首次加载完成并进入 sleep-idle
- **THEN** 控制面与全部模型 API 依次启动并纳入就绪检查

#### Scenario: 预热超时

- **WHEN** backend 在 `ORNITH_MTP_9B_STARTUP_TIMEOUT` 内未进入 sleep-idle
- **THEN** 启动失败并清理已启动进程，不进入半可用状态

### Requirement: 退出清理确认进程组

任一服务退出时 MUST 清理全部服务进程组并确认 Ornith MTP llama.cpp backend 停止；统一启动依赖
setsid、进程组信号和 fcntl.flock；使用 Ctrl+C 等待正常退出。

#### Scenario: 正常退出

- **WHEN** 维护者以 Ctrl+C 结束 start.sh
- **THEN** 全部服务进程组收到清理信号，backend 停止状态被确认后进程退出

#### Scenario: backend 停止状态不能确认

- **WHEN** 退出时无法确认 llama.cpp backend 已停止
- **THEN** 保留 quarantine 标记并按恢复流程处理，不假设退出成功

### Requirement: 参数变更须重启生效

已运行进程 MUST 重启才会采用新的默认参数；文档 MUST NOT 把旧运行进程的参数当作当前部署值。

#### Scenario: 修改默认参数

- **WHEN** 维护者调整默认环境变量后查看运行中服务
- **THEN** 运行中服务保持旧参数，重启后新默认值才生效

### Requirement: Dagu 调度层不改变服务边界

可选的独立 Dagu 调度器 MUST 只经 HTTP 调用既有模型 API（如 VoxCPM2、MOSS），不改变现有模型
API 与 start.sh 行为，不持 GPU 锁。

#### Scenario: Dagu 后台任务执行

- **WHEN** Dagu 队列触发一次生成任务
- **THEN** 任务通过既有 HTTP API 完成请求，锁与生命周期仍由既有服务管理
