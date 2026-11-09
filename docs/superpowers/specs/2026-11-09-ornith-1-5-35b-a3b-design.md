# Ornith-1.5-35B-A3B 服务设计

## 目标

通过 `start.sh` 管理本地 Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K。llama.cpp 使用 CUDA 承载可放入显存的层，将前 20 层 MoE 专家放在 CPU，启用 mmproj 视觉输入，并在 8401 暴露 `POST /v1/AtomicChat/Ornith-1.5-35B-A3B-AD-Q5`。真实生成样例写入运行数据目录。

## 约束与非目标

- 不复制或提交模型、mmproj、llama.cpp 源码或生成样例。
- GGUF 和 mmproj 默认从 `$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/` 加载。
- llama-server 只绑定 `127.0.0.1:8402`；公开适配器绑定 `${HOST:-0.0.0.0}:8401`。
- 复用 `GPU_LOCK_FILE`，并在锁旁使用 `.quarantine`。GPU 请求期间持锁；请求后必须通过 `/props` 确认 llama.cpp 进入 sleep-idle 状态，等待 `CUDA_RELEASE_DELAY` 后才移除隔离标记并释放锁。
- llama-server 空闲时通过 `--sleep-idle-seconds` 卸载模型及 KV cache，避免常驻占用 GPU 并允许其他 provider 安全使用共享 GPU。
- 启动时先通过 `gpu_runtime_lock` 申请共享租约和隔离标记，加载模型并确认 sleep-idle，再启动其他 GPU HTTP 服务。
- 不能确认模型卸载或 backend 停止时保留 quarantine。`scripts/localai/recover_quarantine.py` 必须识别 Ornith 标记，并确认其进程组及 loopback API 已退出后才解除隔离。
- 本次不增加 LocalAI 候选，也不改变其他服务合同。

## 方案

1. 仅用 llama-server 暴露 API：不能满足任务要求的完整路由。
2. 在 8401 使用 FastAPI 适配器，将固定模型请求转发到 loopback llama-server：采用此方案。适配器直接使用共享 GPU 锁和 quarantine 协议，并在每个请求后等待后端睡眠。
3. 每个请求都启动和停止 llama-server：虽然容易释放 GPU，但每次重新载入约 22 GB 的权重会造成高等待；采用 sleep-idle 保留进程、卸载模型的方案。

## 实现范围

- `start.sh` 验证 CUDA llama-server、模型和 mmproj，预检 8401/8402，并在共享 GPU 锁内启动 backend。使用 `--device CUDA0`、`--gpu-layers 99`、`--n-cpu-moe 20`、`--flash-attn on`、`--jinja`、32K context 和 sleep-idle。模型进入休眠后才启动其他 HTTP 服务。
- API 适配器只绑定到固定 Ornith API 路由。预检在锁外完成；请求取得锁并写标记后，先等 `/props.is_sleeping == true` 才发送请求，避免在 sleep 转换窗口唤醒模型。请求结束后再次确认 sleep-idle，等待释放缓冲并清除本次标记。
- Startup helper 用带 gate 文件的 exec 包装进程：backend 在写入 PID/PGID 标记前不能开始加载，消除 marker 与 server 启动之间的进程身份空窗。
- 发生进程崩溃或不能确认休眠时保留 `.quarantine`。恢复脚本通过标记中的 PID/PGID 和地址确认进程组及 loopback 端口退出。
- 正常关闭先停止 API provider，再通过共享 GPU 租约写入 marker、停止 llama-server，并确认进程组和端口都已退出。
- `tests/ornith_1_5_35b_a3b/` 提供模型无关合同测试、操作说明和真实样例生成入口，输出放入 `storage/`。
- README 和 AGENTS 补充混合推理、环境变量、GPU 锁及恢复流程。

## 验收

- 启动脚本语法、仓库质量门禁和无模型测试通过。
- 启动实际 CUDA 构建的 llama-server 后，指定路由可返回聊天补全；视觉投影作为 backend mmproj 加载。
- 样例脚本将真实生成结果写入运行数据目录。
