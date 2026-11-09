# Python 工程规范

当前 Python 运行面由轻量控制面、Qwen3-TTS VoiceDesign API、共享 runtime/gateway 和无模型 QA 组成。
控制面与 VoiceDesign 有独立 uv 项目，不把模型依赖合并到统一环境。退役模型目录和旧文档是归档，不纳入
当前安装、Ruff 或锁文件检查。

## 可复现环境

- 当前维护的 HTTP 项目固定使用 Python `3.12.13`；`.python-version` 与 `requires-python` 保持一致。
- 当前项目锁检查范围为 `main/`、`model-audio/design_qwen3-tts-1.7b-voicedesign/` 和 `qa/`。
- `llm_local_runtime/` 与 `inference_gateway/` 是本地 editable 包；消费者项目锁定其版本，不合并到服务环境。
- 通用依赖由锁文件解析。VoiceDesign 的模型推理在仓库外固定版本的 LocalAI/audio.cpp backend 中运行；不在控制面或 Python API 环境加载 Torch。

同步依赖：

```bash
uv sync --project main --locked
uv sync --project model-audio/design_qwen3-tts-1.7b-voicedesign --locked
uv sync --project qa --locked
```

修改任一有锁项目的 `pyproject.toml` 后，在相同项目运行 `uv lock`，提交 `pyproject.toml` 与 `uv.lock`，
并通过 `uv lock --check`。修改共享 runtime/gateway 时，同步对应 editable 包消费者并运行质量门禁。

## 格式化与静态检查

Ruff 基线为 `E`、`F`、`I`、`UP`、`B`，目标版本 Python 3.12，行宽 100。`B008` 是 FastAPI
`File(...)`/`Form(...)` 路由签名的框架约定；`E501` 交由 formatter 处理。

```bash
bash scripts/quality_gate.sh
```

质量门禁运行 shell 语法、Steam Audio 无 SDK C++ 核心测试、三个活跃 uv lock 检查、Ruff、能力清单校验和
根目录无模型 unittest。GitHub Actions 使用同一脚本，不需要模型权重、CUDA 或 Steam Audio SDK；需要 `uv` 与 `g++`。
锁检查可能读取 Git/wheel 元数据；已有完整缓存时可设置 `UV_OFFLINE=1`，离线模式不能替代首次准备依赖元数据。

针对性检查示例：

```bash
uv run --project qa --locked ruff check main llm_local_runtime inference_gateway tests
uv run --project qa --locked ruff format --check main llm_local_runtime inference_gateway tests
```

运行 Ruff 修复时只使用 `ruff check --fix` 的安全修复并审阅行为差异。不要使用无规则号的 `noqa`；必要例外
需说明原因并限制到最小文件或行范围。

## 路由、并发与进程边界

- HTTP 服务入口负责参数验证、响应、存储和运行时编排。阻塞锁、subprocess 和网络任务使用同步路由或线程池，
  不阻塞异步事件循环。
- 控制面不导入模型包或推理 runtime。VoiceDesign 通过 `inference_gateway` 调用固定 LocalAI/audio.cpp，
  不常驻 Python 模型或 CUDA 上下文。
- 所有 GPU provider 必须共用同一个 `GPU_LOCK_FILE`，按跨进程 `flock` 串行运行。CPU/文件预检放在锁外；
  backend shutdown、`CUDA_RELEASE_DELAY` 和 GPU 资源收尾在锁内完成，`finally` 释放锁。
- 部署时 `GPU_LOCK_WAIT_TIMEOUT` 必须是正值，默认等待 900 秒；等待超时返回 `503`。backend 停止无法确认时，
  保留 `.quarantine` 并拒绝新的 GPU 推理。恢复前必须确认 LocalAI backend 已停止。
- Steam Audio 空间 renderer 是独立 C++17 CPU 路径，不获取 GPU 锁。renderer 子进程需设定超时，成功/失败后清理
  请求临时文件。
- 共享锁实现、隔离和新 provider 接入规范见[全局 GPU 锁说明](global-gpu-lock.md)。

## 上传与存储

- 文件上传使用 `llm_local_runtime` 流式暂存 helper，按路由限制大小、扩展名和媒体类型，计算 SHA-256，使用临时文件
  与 `os.replace` 原子提交。不整文件 `read()` 上传，不直接覆盖 sidecar。
- Manifest 中的空间资产文件名必须与重复 `assets` 上传字段一一对应。输入、暂存产物与临时输出按请求生命周期清理。
- VoiceDesign 成功 WAV 写入 `TIMBRE_STORAGE_DIR`，默认 `storage/timbre/`。保留策略默认关闭，只能通过维护脚本预览并显式
  `--apply` 删除符合生成文件名前缀的 WAV。
- `storage/` 和模型/renderer 外部运行数据由用户管理，代码清理不得触碰这些数据。

## 测试

测试不得下载权重、依赖 CUDA、调用真实模型服务或修改用户文件。GPU 锁行为使用真实临时锁文件与独立子进程覆盖；
LocalAI、subprocess、网络和模型执行边界使用替身。

```bash
uv run --project qa --locked python -m unittest discover -s tests -v
```

FastAPI TestClient 回归使用 QA 锁定的 `httpx2` 兼容依赖，不恢复旧客户端组合。真实模型调用和文本润色训练/预测
由操作者单独安排，不属于无模型门禁。

## Python 风格

Python 注释和 Docstring 默认简体中文；标识符、类型名、库/API 名及通用技术术语保留英文。公共函数、公共类和重要
业务入口提供 Google Style / PEP 257 Docstring。并发、进程组、GPU/CUDA 与 dtype/device 选择需解释原因；不为
自解释赋值和条件写逐行注释。
