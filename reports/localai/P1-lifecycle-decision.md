# P1：LocalAI backend 生命周期决策

- 状态：`implemented_unverified`
- 日期：2026-10-21
- 适用范围：Qwen3-TTS Base/VoiceDesign 直接 provider 与五个 one-shot worker bridge
- 部署路线：LocalAI 和 gRPC bridge 使用宿主机进程，不使用 Docker
- 固定版本：LocalAI `v4.11.0`、audio.cpp `9a02e61326aaaf9d462b584ca5e0daba22c0abfc`

## 决策

LocalAI 使用宿主机独立进程组运行；固定 CUDA backend 已在本机完成构建、打包和安装。管理器校验
LocalAI 版本、backend 入口、文件树 SHA-256 和路径边界，不负责下载模型或第三方源码。LocalAI
只绑定 loopback，模型注册限于当前两个 Qwen3-TTS 候选和五个登记的 worker bridge ID。

请求方在进入 LocalAI 前获取共享 `GPU_LOCK_FILE`。直接模型候选在请求结束时调用
`/backend/shutdown`；worker bridge 请求通过 LocalAI 调度对应项目的一次性 worker，worker 退出后
执行 `CUDA_RELEASE_DELAY`。两条路径在停止无法确认时都必须保留 quarantine 标记，拒绝后续 GPU 请求。
native provider 仍为默认值。

## 请求与进程回收

1. 服务先完成 CPU 和本地文件预检，再获取 GPU 文件锁。
2. LocalAI 请求期间写入原子 quarantine 标记，记录模型 ID 和请求标识。
3. 推理成功、失败或超时后均执行 backend shutdown；只有清理确认后才等待 CUDA 回收并移除标记。
4. shutdown 失败时保留标记并返回失败；不能在未确认后端停止时释放 GPU 租约并继续接单。
5. `start.sh` 退出时先回收业务进程组，再停止受管 LocalAI 与 bridge。管理器使用 PID 启动 tick、运行标识、
   TERM/KILL、有界等待、进程组扫描和端口关闭检查，确认进程退出后才清除状态文件。
6. 受管进程的 owner watcher 在启动 shell 异常退出时执行回收。quarantine 恢复工具只有在受管进程组退出、
   状态文件消失且 loopback 端口关闭后才能清理标记；不会猜测性终止外部托管实例。

## 构建与无模型验证

用户提供的 LocalAI/audio.cpp 源码归档和 Ubuntu 24.04 amd64 `.deb` 已在仓库外私有前缀使用，
未安装系统包。audio.cpp CUDA `grpc-server` 构建、standalone 打包、LocalAI 空模型目录 readiness/start/stop、
独立 backend SIGTERM 收尾以及 `manager.py verify-install` 均已通过。版本、SHA-256 与命令记录见
[`versions.json`](../../localai/manifests/versions.json) 和 [`P1-P2.md`](P1-P2.md)。

无模型测试覆盖文件指纹、loopback 配置、启动 readiness、owner watcher、PID 重用拒绝、TERM/KILL 进程组清理、
端口关闭判定、bridge 合同、候选预检与 GPU 锁互斥。它们不验证真实模型 RPC、模型输出、backend 子进程行为或
`/backend/shutdown` 后 GPU 显存回落；当前尚无 E2/E3 模型证据。

## 回滚

保持服务 provider 默认值 `native`，并禁用 `LLM_LOCALAI_ENABLED`、`LLM_LOCALAI_MANAGED` 与
`LLM_LOCALAI_WORKER_BRIDGE_ENABLED`。需停止 LocalAI 时使用
`python3 scripts/localai/manager.py stop`，确认进程组和 loopback 端口已关闭；不要在 backend 仍运行时
手动删除 quarantine 标记。
