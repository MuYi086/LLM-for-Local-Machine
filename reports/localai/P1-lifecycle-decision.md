# P1：LocalAI backend 生命周期决策

- 状态：`implemented_unverified`
- 日期：2026-10-21
- 适用范围：Qwen3-ASR 试点及固定版本 LocalAI/audio.cpp
- 部署路线：宿主机 LocalAI 二进制与本机 CUDA backend，不使用 Docker
- 版本组合：见 [`manifests/versions.json`](../../localai/manifests/versions.json)

## 决策

LocalAI 以宿主机独立进程组运行；audio.cpp 按 LocalAI v4.11.0 源码中的固定 commit 编译为 CUDA 12
backend。`manager.py verify-install` 检查 LocalAI `--version`、backend `run.sh`、整棵 backend 文件树的 SHA-256 和路径边界，
随后将实际内容指纹写入本机状态目录。管理器不下载或编译第三方源码。LocalAI 使用 `LOCALAI_BACKENDS_PATH`
和 `LOCALAI_MODELS_PATH` 读取本机路径。

业务服务仍在入口获取共享 `GPU_LOCK_FILE`。该锁覆盖请求、`/backend/shutdown` 和停止确认；LocalAI backend
不重复获取同一锁。原生 backend 只有在固定版本 smoke 证明 `/backend/shutdown` 后模型进程与显存退出，才能满足
一次性推理要求；否则本阶段不扩大迁移范围，继续保留 native provider，并评估一次性外部 gRPC adapter。

## 请求和停止顺序

1. 服务完成 CPU 预检，获取 GPU 文件锁并创建原子 `.quarantine` 请求标记。
2. 通过 LocalAI transcription API 发请求；成功、失败和超时后均调用 `/backend/shutdown`。
3. 只有 shutdown 返回成功，且后续进程/显存 smoke 证明 backend 已退出，才允许清除标记并释放锁；失败时保留标记，
   后续 native/LocalAI GPU 准入均被拒绝。
4. `start.sh` 退出时先终止并回收业务进程组，再调用 manager 停止 LocalAI 进程组。管理器先发 TERM，等待有界时间，
   再发 KILL；只有进程组无活动成员且 loopback API 端口关闭后才删除 PID 状态文件。
5. 管理器通过 `posix_spawn` 执行 util-linux `setsid`，并等待 PID、PGID、SID 一致后再记录状态。状态文件记录 boot ID、PID、PGID、`/proc` 启动 tick、随机运行标识和 loopback 地址；
   启动后立即拉起 watcher。watcher 通过 owner PID 与启动 tick 判断父 shell 是否退出，避免 PID 重用误判；主机重启后仅在进程扫描和端口检查均通过时清除旧状态。
6. API 失联时，quarantine 恢复工具持有 GPU 锁并停止受管 LocalAI 进程组；只有进程组退出、状态文件消失且 loopback API
   端口关闭，才允许清理隔离标记。外部托管实例不会被该恢复脚本猜测性终止。

请求超时、客户端断开或 backend crash 后由清理调用卸载模型。若不能确认则隔离，不切回 native、不释放锁后继续接单。
本地 manager 停止失败时保留进程状态文件，提示运维运行 `status` 并复核进程组。

## 2026-10-21 执行进度

- 用户提供的 LocalAI `v4.11.0` Linux amd64 二进制已在忽略的运行目录通过 `--version` 检查；其 SHA-256 记录在 `P1-P2.md`。
- LocalAI 与 audio.cpp 固定 commit 源码归档已展开，audio.cpp 位于 Makefile 预期的 `audio.cpp/` 子目录，避免 GitHub clone。
- audio.cpp CUDA engine 已使用 CUDA 12.0.140、CMake 3.28.3 和 `sm_89` 成功配置；`engine_runtime` 完整构建退出码为 0。上游若干 warning 未阻止目标完成。
- 用户提供的 20 个 Ubuntu 24.04 amd64 gRPC/protobuf `.deb` 全部通过包名、版本与架构校验，并用 `dpkg-deb -x` 解压到私有前缀；没有安装系统包。固定 backend `grpc-server` 的 CUDA `89-real` 构建与 `package.sh` standalone 打包通过，backend 文件树已放入忽略运行目录并通过 `manager.py verify-install`。
- 无模型检查已运行：LocalAI 空模型目录 readiness/start/stop 通过；打包后的 gRPC server 独立监听回环端口，并在进程组 SIGTERM 后以退出码 0 结束。以上不覆盖实际模型加载、`/backend/shutdown` 后 CUDA/GPU 回落或异常故障回收。
- 模型权重未下载。

## 尚未验收

- audio.cpp CUDA `engine_runtime` 与 LocalAI gRPC wrapper 均已于 2026-10-21 使用 CUDA 12.0.140、CMake 3.28.3、GCC 13.3.0 和 `sm_89` 编译通过；standalone backend 已打包和安装。LocalAI 无模型启停通过，但没有执行模型请求。
- 无模型用例覆盖版本与文件指纹、loopback 绑定、最大活动 backend、启动超时收尾、owner watcher、PID 复用拒绝、TERM/KILL 进程组停止
  和端口关闭判定。它们不能证明 LocalAI CLI、audio-cpp wrapper、backend child process 或 `/backend/shutdown` 的实际行为。
- P1 仍需在准备好权重后验证：LocalAI 启动 backend 的 PID 与进程组归属、shutdown 后 30 秒内模型进程和 VRAM 回落、
  LocalAI crash、业务 shell SIGKILL、子进程残留、磁盘满及断连恢复。
- 若 shutdown 2xx 后模型进程或显存未退出，停止 P2 放量，继续保留 quarantine；再决定是否实现一次性 audio.cpp 子进程 gRPC adapter。

## 回滚

保持 `UNITALE_QWEN3_ASR_PROVIDER=native`，禁用 `UNITALE_LOCALAI_ENABLED` 与 `UNITALE_LOCALAI_MANAGED`，并使用
`scripts/localai/manager.py stop` 停止受管 LocalAI 宿主机进程组。仅在模型进程已退出后按隔离恢复脚本清除标记。
旧 Qwen3-ASR worker、依赖和 uv lock 保持保留。
