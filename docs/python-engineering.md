# Python 工程规范

本仓库的模型服务是独立的 uv 项目，而非一个可发布的统一 Python
包。每个服务的顶层 `main.py` 与 `worker.py` 都需要以该服务目录为导入
根目录运行，因此保持 `[tool.uv] package = false`；`src/*/__init__.py`
仅用作元数据包标记，不能把这些 HTTP 服务误当作 PyPI 发布物。

## 可复现环境

- 19 个服务项目固定使用 Python `3.12.13`。服务的 `.python-version` 与
  `requires-python` 必须保持一致；`qa` 和共享运行时也声明相同 Python 版本。
- 各服务独立锁定 Torch、CUDA wheel 和上游运行时，不能合并到一个环境。
  核心依赖尽量使用精确版本；范围依赖、开发工具和 Git 依赖由对应 `uv.lock` 锁定。
- 通用 Python 包从官方 PyPI 解析。CUDA wheel 和模型专用包仍必须通过
  `[tool.uv.sources]` 中显式声明的来源获取。`moss_voiceGenerator` 是部署
  例外：它依赖预先准备的 `/home/muyi086/tts-depency/MOSS-TTS` editable
  checkout，因此同步前必须确认该目录存在；不要让 uv 改为联网拉取它。

完整服务清单、模型/外部源码路径与安装循环见 [根目录 README](../README.md)。
统一启动不会安装依赖；从仓库根目录提前同步需要的服务，例如：

```bash
uv sync --project qwen3_tts --locked
uv sync --project qa --locked
```

修改任一 `pyproject.toml` 后，必须在相同项目目录运行 `uv lock`，提交
`pyproject.toml` 和 `uv.lock`，再用 `uv lock --check` 验证。
`unitale_runtime/` 是本地 editable 包，其依赖随消费者锁定，没有独立锁文件。
MOSS-Audio 两个变体与 FireRed 两个模式分别共用一个服务项目，不重复创建环境。

## 格式化与静态检查

每个服务都在 `pyproject.toml` 中配置同一套 Ruff 基线：`E`、`F`、`I`、
`UP`、`B`，目标版本 Python 3.12，行宽 100。`B008` 是 FastAPI
`File(...)`/`Form(...)` 路由签名的框架约定；`E501` 交给 Ruff formatter
处理，因为 URL、协议文本和框架签名不应为了机械折行而损失可读性。

```bash
bash scripts/quality_gate.sh
```

统一入口先运行 `bash -n start.sh` 和不依赖 Steam Audio SDK 的三个 C++ 核心测试，
再检查 19 个服务项目与 QA 的锁文件，使用轻量 QA 环境执行 Ruff、格式检查和三组单元测试。
GitHub Actions 调用同一脚本，不需要模型环境或 CUDA；需要 `uv` 与 `g++`。
锁检查仍可能访问 Git/wheel 元数据；已有完整缓存时可用 `UV_OFFLINE=1`，
离线模式不能代替首次准备依赖元数据。

只检查改动文件时使用锁定的 QA Ruff，例如：

```bash
uv run --project qa --locked ruff check main unitale_runtime tests
uv run --project qa --locked ruff format --check main unitale_runtime tests
```

完整 Ruff 路径以 `scripts/quality_gate.sh` 为准；当前不包含 `scripts/training/` 与
`soundEffect/` 示例，修改这些脚本时需显式加入文件路径检查。

修复格式时使用 `ruff format .`；只使用 `ruff check --fix` 的安全修复，
并审阅每项行为相关改动。不要使用无规则号的 `noqa`；必要的例外必须写明
原因，并限制到最小文件或行范围。

## 日志、异常和 worker 边界

- HTTP 服务代码使用 `logging.getLogger(__name__)`；在把未知异常转换为
  `HTTPException` 的边界使用 `logger.exception(...)`，以保留 traceback。
- one-shot worker 的 stdout/stderr 是父进程提取错误摘要的协议边界，保留其
  错误输出；新增 worker 诊断应写到 stderr，不应污染成功 WAV 的输出路径。
- 异常转换必须保留原因链（`raise ... from exc`）。超时后先终止进程组，
  再抛出带原始 `TimeoutExpired` 原因的业务异常。
- `main.py` 只做 HTTP 校验、存储和 worker 生命周期管理；不得在其中导入
  或加载重型模型。`worker.py` 负责模型导入和推理；父进程在成功、失败、
  超时后均回收 worker 进程组与临时文件，并在持锁期间等待 CUDA 回收后释放 GPU 锁。
- GPU 文件锁、subprocess 和云端请求等阻塞操作通过同步路由或线程池执行；
  `async` 路由不能直接阻塞事件循环。纯 CPU/文件预检在获取 GPU 锁前完成。

## 测试

静态检查和单元测试必须不下载权重、不依赖 CUDA，也不调用真实 MiMo。
先运行根目录迁移测试，再运行含有专属测试的服务：

```bash
bash -n start.sh
uv run --project qa --locked python -m unittest discover -s tests -v
(cd stable_audio_3_medium && uv run --project ../qa --locked python -m unittest discover -s tests -v)
(cd ace_step_1_5 && uv run --project ../qa --locked python -m unittest discover -s tests -v)
```

FastAPI TestClient 回归依赖 QA 的 `httpx2` 兼容组合；不得恢复已弃用的旧客户端版本。
从服务目录运行两组专属测试，可避免同名 `runtime` 模块导入冲突。
真实模型演示、MOSS GPU smoke test 和文本润色训练/预测均由操作者单独运行，
不属于无模型质量门禁；训练脚本的纯文本逻辑由根目录测试覆盖。

Python 注释与 Docstring 默认简体中文，公共入口遵循 Google Style 与 PEP 257。
并发、进程组、GPU/CUDA 与 dtype/device 选择应解释原因，不给显然操作添加逐行注释。
