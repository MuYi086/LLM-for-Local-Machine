# 全局 GPU 锁

仓库的 GPU 模型服务必须共用一个文件锁，避免多个模型进程同时申请显存而触发 OOM。锁实现在
`llm_local_runtime/src/llm_local_runtime/gpu.py`，公共接口为 `gpu_runtime_lock` 和
`gpu_quarantine_path`；它不是由某个模型项目私有持有的锁。

## 共享路径

`start.sh` 默认设置：

```text
GPU_LOCK_FILE=$RUNTIME_CACHE_DIR/gpu-runtime.lock
```

若服务通过不同启动脚本运行，必须显式传入同一个 `GPU_LOCK_FILE`。每个 provider 都需要在开始
GPU 推理前调用 `gpu_runtime_lock(lock_path, label, timeout_seconds=...)`。传入不同文件路径会形成
不同锁，不能保护共享 GPU。

## 执行顺序

1. 先完成不使用 GPU 的请求、文件和权重检查。
2. 取得同一路径的排他 `flock` 后才启动模型或调用 backend。
3. 成功、异常和超时路径都必须回收显存：LocalAI 确认 backend shutdown；Ornith 和 Qwen 确认各自
   `llama-server /props` 返回 `is_sleeping: true`，并继续持锁等待 `CUDA_RELEASE_DELAY`。
   清理异常不能覆盖原始推理错误。
4. 在 `finally` 中释放锁。

VoiceDesign 和 audio.cpp provider 通过 `inference_gateway.execution.execute_localai_operation` 走此流程；
Ornith、Qwen 的 backend startup、请求与 shutdown 通过 `llm_local_runtime.gpu_runtime_lock` 申请同一租约。
每个文本模型请求在 POST 前确认对应 backend 已处于 sleep-idle，避免撞入 sleep 转换窗口。今后增加其他 GPU provider 时，也必须复用共享锁；
不得另建模型专属锁，也不得绕过锁直接启动 GPU worker。Steam Audio renderer 是 CPU 路径，健康信息中报告
`gpu_lock: false`，不加入队列。

## 排队和指标

- `GPU_LOCK_WAIT_TIMEOUT` 默认 900 秒。部署应设置正数；等待超时由 HTTP provider 转成可恢复的
  `503`。共享底层函数保留非正数表示无限等待的兼容行为，新部署和 provider 不应使用该行为。
- 持锁期间以 `nvidia-smi` 采集峰值，采样间隔由 `GPU_METRICS_SAMPLE_INTERVAL` 控制。指标中的
  `peak_vram_mib` 是所有可见 GPU 的已用显存总和，不是当前模型独占的显存。
- 锁基于 Linux `fcntl.flock`，所有服务必须运行于可访问同一锁文件的本机文件系统。

## Quarantine 与恢复

进程崩溃或 backend 停止无法确认时，服务在锁旁保留 `${GPU_LOCK_FILE}.quarantine`。锁释放后，后续
provider 取得文件锁会先检查该标记；标记存在时立即拒绝推理，避免不确定的残留模型继续占用显存。
不能仅因 HTTP shutdown 返回成功就删除标记，必须确认受管模型/backend 进程组已退出。

LocalAI、Ornith 与 Qwen 共用恢复入口：

```bash
python3 scripts/localai/recover_quarantine.py
```

恢复脚本先确认没有活动 GPU 请求，在同一锁下读取 `provider` 并分派处理。LocalAI 未确认 shutdown，或
llama.cpp 进程组或 loopback API 仍活动时，会保留隔离标记。只有确认对应 backend 进程组和端口退出后才原子删除标记。
新增 GPU provider 也必须提供相同强度的进程回收与恢复流程，不能自行绕过 quarantine。
