# MOSS-SoundEffect v2.0 服务设计

## 意图

在 `model-audio/soundEffect-v2.0/` 新增独立 HTTP 服务，按文本提示词生成环境音效，监听 8362，并提供 `POST /v1/moss/soundEffect-v2.0`。模型已在 `$HF_MIRROR_DIR/OpenMOSS/MOSS-SoundEffect-v2.0` 准备；服务应接入仓库现有启动、共享 GPU 排队、一次性 worker 和无模型质量门禁。

## 目标与边界

- 使用官方 `MossSoundEffectPipeline`，仅从本地模型目录加载，不在请求期下载权重。
- 接收提示词、时长、采样步数、CFG、sigma shift 和 seed；输出单声道 48 kHz WAV，最长 30 秒。
- API 进程不导入 Torch 或模型包；每个推理请求启动单独 worker，退出后回收 CUDA 上下文。
- 生成物原子保存到 `storage/sound_effect_v2/`，并以 `audio/wav` 返回。
- 不加入上传、参考音频、训练、母带或其他 MOSS-TTS 能力。

## 约束

- Python `3.12.13`，独立 uv 项目和锁文件；使用官方 MOSS-TTS 源码固定 revision `934d6826b084c46a0d033402174d5f8ac4ed2519` 及官方 CUDA 12.8 推理依赖。
- 模型必须从本地 `$HF_MIRROR_DIR` 路径读取；健康检查只检查文件存在与大小，不加载权重。
- 纯请求校验和模型文件检查在 GPU 锁之前完成；GPU 锁等待使用正的共享超时，worker 超时与失败均清理整组进程及临时文件。
- 新增服务须纳入 `start.sh`、服务速查表、README、Ruff/锁质量门禁及无模型测试。

## 实现方案

1. **独立 FastAPI 服务和一次性 worker（采用）**：遵循 Confucius4-TTS 的 HTTP/worker 边界，利用共享 GPU 锁，并将官方 pipeline 放在锁定的服务环境中。符合仓库现有的模型隔离和进程生命周期约束。
2. **在 API 进程中常驻模型**：启动后加载大模型，减少重复加载；违反一次性 worker 约束并长期占用显存。
3. **把上游模型源码复制进仓库**：便于离线导入，但重复维护第三方实现，违反不提交第三方源码的仓库规则。

## 交给实现者的选择

采用方案 1；用本地 Hugging Face 权重和固定的官方源码 revision。字段边界采用模型卡默认值：时长 10 秒（0.1–30 秒）、100 步、CFG 4.0、sigma shift 5.0、seed 0。部署者先对新 uv 项目执行 `uv sync --locked`，统一启动仍使用 `uv run --no-sync`。
