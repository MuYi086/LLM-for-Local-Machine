# Dagu 独立任务调度

Dagu 是可选的后台调度层。先用 `bash start.sh` 启动既有模型服务；Ornith、图片和音频 API 仍可直接调用。Dagu 通过 HTTP 调用 MOSS 与 VoxCPM2，既不加载模型，也不提前获取 GPU 锁。GPU 锁、backend 收尾和 quarantine 仍由模型服务负责。Dagu 故障不会改变直接调用入口。

## 安装与启动

使用 Dagu 2.26.0 Linux amd64 的 [官方发布包](https://github.com/dagucloud/dagu/releases/tag/v2.26.0)，文件 `dagu_2.26.0_linux_amd64.tar.gz` 的 SHA-256 为 `2b1cab5dbab7c7dc6607d807e2cbf6597f26d5a5d09442e2c2abcfe1acbf9e68`。核验后将 `dagu` 放在 `~/.local/opt/llm-dagu/2.26.0/dagu`；也可以通过 `DAGU_BIN` 指定同版本二进制。

```bash
bash scripts/dagu/start.sh
```

该命令在前台独立运行 Dagu server 和 scheduler，默认只监听 `127.0.0.1:18673`，首次访问 Web UI 按 `/setup` 提示创建管理员账号。`DAGU_PORT` 可改变端口；状态和认证数据默认存放在 `~/.local/share/llm-dagu/server`，可用 `DAGU_HOME` 覆盖。请备份该目录并保护本机账号；UI 可执行命令，不要向不可信网络开放。Dagu 停止后排队任务不能继续调度，重新启动前应检查已有任务状态。

## 提交任务

输入和产物默认位于 `~/.local/share/llm-dagu`，可以在启动 Dagu 前用 `DAGU_JOB_ROOT` 指向其他目录。使用 CLI 时必须沿用启动服务时的相同环境。每个任务使用全局唯一 ID，首字符为字母或数字，后续允许字母、数字、下划线和短横线，最多 80 个字符。

先准备输入文件（每个任务一个 JSON，最大 1 MiB）：

```bash
mkdir -p ~/.local/share/llm-dagu/inputs
( set -C; printf '%s\n' '{"prompt":"A quiet lake at sunrise"}' > ~/.local/share/llm-dagu/inputs/image-001.json )
( set -C; printf '%s\n' '{"text":"你好，欢迎收听。","language":"zh"}' > ~/.local/share/llm-dagu/inputs/speech-001.json )
```

MOSS 接受 `prompt`；VoxCPM2 接受 `text` 和可选 `language`、`instruction`，本阶段不支持参考音频。使用新的 ID，避免覆盖已有用户输入。

```bash
bash scripts/dagu/start.sh enqueue moss -- job_id=image-001
bash scripts/dagu/start.sh enqueue voxcpm -- job_id=speech-001
```

也可在 Web UI 选择工作流、填写 `job_id`，选择 **Enqueue**。不要使用直接 Start 代替入队来承诺队列并发；Dagu 以外的直接 API 请求也不受 Dagu 队列排序影响，但仍受已有共享 GPU 锁保护。`gpu-heavy` 同时运行一个已入队 DAG；`cpu-post` 配置为三个并发，当前没有 CPU 工作流。

结果位于 `jobs/<job_id>/result.png` 或 `result.wav`。PNG 必须通过数据块、CRC 和 FFmpeg CPU 解码检查；WAV 必须包含完整的非空 PCM 采样，因此运行图片任务需要 FFmpeg。输入参数会在登记任务前校验长度等合同限制，校验失败不会占用任务 ID。

`request.json` 保存本次输入快照；`submitted.json` 保存快照 SHA-256、模型端点、Dagu run ID 和登记时间，代表请求可能已提交，并不是成功证明。`completed.json` 在媒体通过校验且产物发布后原子写入，包含产物 SHA-256。记录与产物使用独占链接发布，并同步文件与目录；同名产物不会被覆盖。脚本不把提示词写入自己的日志，快照仍包含原文，需要保护和备份。

## 失败与恢复

两个 DAG 不启用自动重试。任务目录在网络请求前独占创建，同一 ID 不能再次执行，即使第一次失败或超时；这是为了避免请求状态未知时重复生成。捕获到错误时会尽力原子写入 `failure.json`，记录 registering、request、validation 或 publishing 阶段及错误类型；request 阶段一律保守标记为 unknown。进程被强制终止或磁盘故障可能来不及记录失败，缺少失败记录不是成功证明。启动失败、超时、取消、Dagu 或服务退出后，请先检查 Dagu 日志、任务目录、模型服务状态与 GPU quarantine，再决定是否用**新的** ID 人工提交。停止 Dagu 步骤或 HTTP 客户端不保证服务端推理已经停止。产物存在但缺少 `completed.json` 时也应人工核对，不要推断完成或自动重试。

`DAGU_MOSS_BASE_URL` 和 `DAGU_VOXCPM_BASE_URL` 可覆盖默认 loopback 服务根地址及端口；必须在启动 Dagu 前设置。HTTP 客户端通过独立子进程流式接收响应，响应最大 64 MiB，总等待上限 7200 秒（包含建连、提交和接收）；超时只终止客户端，不代表模型推理已取消。媒体校验另有最长 30 秒的 FFmpeg 解码时限。队列与现有 GPU 锁有各自的等待时间；直接请求可能比 Dagu 队列中的任务先获得锁。Windows ComfyUI、跨系统 GPU 互斥和业务资产管理尚未接入。

## 无模型验证

```bash
uv run --project qa --locked python -m unittest discover -s tests -p test_dagu_jobs.py -v
# 已安装固定版本 Dagu 时，独立临时目录与假 HTTP 服务验证真实入队：
uv run --project qa --locked python tests/dagu_smoke.py
```

联调脚本使用独立 scheduler，不启动 Web UI，也不调用真实模型。完整质量门禁仍使用 `bash scripts/quality_gate.sh`。
