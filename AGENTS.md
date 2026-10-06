# 仓库开发指南

## 适用范围与依据

本仓库是 Unitale 的本地音视频后端，包含语音克隆、音色设计、音效/BGM、编辑、理解/转写、
音源分离、音色转换、人像视频及 CPU 空间音频导出。服务采用独立 uv 项目和一次性 worker。
文档遵循 [CONSTITUTION.md](CONSTITUTION.md) 的中文优先要求；接口和配置以实际源码为准。
[README.md](README.md) 提供接入、配置和运维说明，本文件只记录开发边界与命令。

## 项目结构与服务速查

当前有 19 个服务 uv 项目，`start.sh` 启动 22 个 HTTP 进程。表中业务入口均为 `POST`，
`/v1/control` 为 `GET`；各进程另有 `GET /v1/health`。路径大小写必须与代码一致。

| 目录 | 默认端口 | 能力与业务入口 |
| --- | --- | --- |
| `main/` | 8300 | 控制面 `/v1/control`；母带 `/v1/audio/export`；空间导出 `/v1/audio/spatial/render`；MiMo 同路径代理 |
| `qwen3_voiceDesign/` | 8301 | 音色设计 `/v1/qwen/timbre` |
| `moss_voiceGenerator/` | 8302 | 音色设计 `/v1/moss/timbre` |
| `mimo_tts/` | 8303 | 云端音色设计 `/v1/mimo/timbre` |
| `firered_tts3/` | 8304、8325 | Instruct `/v1/FireRedTTS3/timbre`；Base `/v1/FireRedTTS3/clone` |
| `stable_audio_3_medium/` | 8311 | 音效/音乐 `/v1/stableAudio/soundEffect` |
| `moss_soundEffect/` | 8312 | 音效 `/v1/moss/soundEffect` |
| `ace_step_1_5/` | 8313 | BGM `/v1/aceStep/bgm` |
| `qwen3_tts/` | 8321 | 克隆 `/v1/qwen/clone` |
| `voxcpm2/` | 8322 | 克隆与可控克隆 `/v1/voxcpm2/clone` |
| `LongCat_AudioDiT_3.5B_bf16/` | 8323 | 克隆 `/v1/longCat/clone` |
| `dots_tts_soar/` | 8324 | 克隆 `/v1/dotsTTS/clone` |
| `Step_Audio_EditX/` | 8331 | 编辑 `/v1/stepAudioEditx/edit` |
| `moss_audio_4b_thinking/` | 8341、8342 | Thinking/Instruct 共用 `/v1/mossAudioThinking/understand` |
| `TIGER-DnR/` | 8351 | 对白/音效/音乐分离 `/v1/tigerDnr/separate` |
| `Confucius4_TTS/` | 8361 | 零样本克隆 `/v1/confucius4TTS/generate` |
| `Qwen3_ASR_1.7B/` | 8371 | 转写 `/v1/qwen3/asr` |
| `seed-vc/` | 8381 | 音色转换 `/v1/seedVc/voiceConversion` |
| `SoulX-FlashHead-1_3B/` | 8391 | 人像视频 `/v1/soulX/flashHead` |
| `ditto/` | 8392 | 人像视频 `/v1/ditto/talkingHead` |

- MOSS-Audio 的变体由 `MOSS_AUDIO_4B_VARIANT=thinking|instruct` 明确指定，两者共用一个项目。
  不要把 `moss_audio_4b_instruct/` 当作独立服务。FireRed 使用
  `FIRERED_TTS3_MODE=timbre|clone`，每个进程只注册其模式对应的业务路由。
- `main/main.py` 只处理健康/控制、共享上传/检查、MiMo 代理和 CPU 导出，不加载模型包或推理。
  空间任务进度为 `GET /v1/audio/spatial/render/progress/{job_id}`，是进程内状态。
- `unitale_runtime/` 是共享轻量包，负责流式上传、引用、原子提交、GPU 队列及存储容量/保留策略；
  不得导入 FastAPI、Torch 或模型包。它是其他项目的 editable 依赖，没有独立 `uv.lock`。
- `steam_audio_renderer/` 是 C++17 Steam Audio CPU renderer，不获取 GPU 锁；SDK 和动态库在仓库外。
- `qa/` 是无模型测试/Ruff 项目；`scripts/quality_gate.sh` 与 GitHub Actions 使用同一质量门禁。
  `tests/` 为根目录回归，ACE-Step 和 Stable Audio 的服务内测试须从各自目录执行。
- `scripts/training/`、`training/configs/` 是纯文本润色 LoRA 实验，不是语音训练或 HTTP 服务。
  模型训练需要独立外部环境；数据、adapter、预测保存到 `storage/training/`。
- `soundEffect/` 和 `tests/testConfucius4TTS/`、`tests/testSoulX_FlashHead/`、`tests/testDitto/`
  含真实模型演示，不能纳入自动无模型推理测试。`流行*模型功能简介.md`、`task*.md` 和
  `special-audio-effect/` 是调研/任务/方案资料，不能据此推断已实现能力。

## 安装、启动与测试

部署使用 Linux、Python `3.12.13`、`uv`、FFmpeg/ffprobe 和 CUDA/NVIDIA 驱动。
统一启动依赖 `setsid`、进程组信号与 `fcntl.flock`；不同服务必须保留各自 Torch/CUDA 环境。
Step-Audio-EditX 另需系统 `sox`，Ditto 需要 GCC，renderer 构建需要 CMake 3.17+ 和 C++17。

按照 README 准备权重与外部源码，再执行 `uv sync --project <dir> --locked`。
`moss_voiceGenerator` 的 `moss-tts` 是唯一固定路径例外：必须使用预先准备的本地 editable
源码 `/home/muyi086/tts-depency/MOSS-TTS`，同步前确认该目录存在，不得改为 Git/PyPI 下载。
修改依赖后在对应项目执行 `uv lock`，提交 `pyproject.toml` 与 `uv.lock`，并用 `uv lock --check` 验证。

`bash start.sh` 在 `qwen3_tts` 环境启动 8300，其他进程使用各自服务环境；端口/路径可通过
环境变量覆盖。启动统一使用 `uv run --no-sync`，并将 `unitale_runtime/src` 加入 `PYTHONPATH`
作为共享包离线兜底。该兜底不能代替新增依赖后的 `uv sync --locked`。
脚本先检查整组端口，任一服务退出时终止其他进程组；重启前用 `Ctrl+C` 等待旧进程退出。
只部署部分模型时直接启动对应 `main.py`；统一脚本没有服务启停开关。
健康检查不会加载模型，HTTP 200 不能代替响应内的依赖/权重可用性检查。

```bash
bash -n start.sh
bash scripts/quality_gate.sh
curl -fsS http://127.0.0.1:8300/v1/control
```

质量门禁包含无 SDK 的三个 C++ 核心测试、20 个项目的锁检查、Ruff 与格式检查及三组无模型
测试。锁校验可能读取 Git/wheel 元数据；已有完整缓存时可设置 `UV_OFFLINE=1` 运行。
单独执行测试必须用 QA，不能通过模型环境安装依赖来运行根目录回归：

```bash
uv run --project qa --locked python -m unittest discover -s tests -v
(cd ace_step_1_5 && uv run --project ../qa --locked python -m unittest discover -s tests -v)
(cd stable_audio_3_medium && uv run --project ../qa --locked python -m unittest discover -s tests -v)
```

测试不得下载权重、依赖 CUDA、调用 MiMo 或执行真实模型；应 mock worker、subprocess、
文件系统边界和网络。FastAPI TestClient 使用 QA 的 `httpx2` 兼容依赖，不恢复旧客户端组合。
真实 MOSS GPU smoke test 单独使用 `bash soundEffect/run_moss_soundeffect_v2.sh`。

## 架构与接口约束

- 服务 `main.py` 负责校验、兼容字段、存储、响应与 worker 生命周期，`worker.py` 负责模型加载
  和推理。阻塞锁、subprocess 和网络任务必须使用同步路由或线程池，不能阻塞异步事件循环。
- 本地重型请求只启动一个 worker，使用该服务 uv 解释器。成功、失败、超时均须终止整个进程组
  并清理临时 JSON/WAV/MP4；不得让模型或 CUDA 上下文跨请求常驻。
- 本地模型使用共享 `GPU_LOCK_FILE` 串行执行，纯 CPU/文件预检应在获取锁前完成。保留
  `finally` 锁释放和持锁期间的 `CUDA_RELEASE_DELAY`；默认排队 900 秒，超时返回 `503`。
  整套部署应保持 `GPU_LOCK_WAIT_TIMEOUT` 为正数，新增代码不得恢复无限期等待；共享实现
  尚有非正值兼容分支，Seed-VC 已明确拒绝该配置。指标采样用 `GPU_METRICS_SAMPLE_INTERVAL`。
- 模型默认值集中在服务模块顶部；新增/修改 `start.sh` 配置只负责路由、路径、端口、环境和
  共享运行参数，不应新增静默推理覆盖。现有部分服务的默认参数还由脚本导出，改默认值须核对两处。
- 不恢复 IndexTTS2、Ming、OmniVoice、旧 MOSS-TTS 克隆、Stable Audio 3 Small SFX、VoxCPM2
  音色设计、集中式 `api/` 或 Conda 回退；不新增旧接口别名。8300 的 MiMo 同路径代理是既有边界。
- Qwen/VoxCPM2/LongCat/dots/FireRed 克隆拒绝 `style_prompt`。VoxCPM2 `controllable` 模式使用
  `control_instruction`，与 `prompt_text` 互斥且忽略 sidecar；`nonverbal_tags` 最多一个白名单标签。
  LongCat 与 FireRed Base 要求准确参考转写；Confucius4 默认 `lang=zh`，参考 sidecar 不参与推理。
- MOSS-Audio、Qwen3-ASR、TIGER 直接接收 multipart `audio`，输入只暂存；前两者返回 JSON，
  TIGER 返回 ZIP 并原子保存分离批次。SoulX/Ditto 返回带声音 MP4，不使用 WAV 响应约定。
  两个人像视频服务提供 `POST /v1/upload_image` 和 `GET /v1/check/image`，同样使用逻辑标识。
- 正式 `balanced`/`immersive` 必须保留对象边界，走 `/v1/audio/spatial/render` 的 Manifest v1
  与重复 `assets` 字段，不在浏览器预混、不获取 GPU 锁。Steam Audio pre-master 后只能
  loudnorm/编码，禁止再叠加 Haas/aecho。`/v1/audio/export` 调用只用于显式 `profile=standard`
  的预混母带；其实现仍兼容旧空间滤镜，不扩展该旧路径。
- `diffuse`、非 `none` 遮挡、几何反射尚未实现；不得将保留的语义字段描述为已可用能力。
  SoulX/Ditto 仅提供单 GPU 离线文件生成，不承诺实时流或多 GPU。

## 上传、存储与清理

- 音频上传经 `unitale_runtime` 按块暂存，默认 64 MiB，由 `UPLOAD_MAX_BYTES` 覆盖；校验扩展名/
  Content-Type、计算 SHA-256，使用临时文件和 `os.replace` 原子提交。不整文件 `read()` 上传，
  不直接覆盖 sidecar 或引用映射。图片使用独立扩展名/MIME 策略，沿用共享暂存和原子提交。
- `full_path` 是客户端逻辑标识，不能用于直接读取服务器任意文件。参考文本可保存 sidecar；
  8300/Step 上传不接收 `prompt_text`，MOSS-Audio/ASR/TIGER 没有独立上传/检查接口。
- 默认目录：音色 `storage/timbre/`，音效 `storage/soundEffect/`，BGM `storage/bgm/`，普通
  参考音频、克隆、编辑、Seed-VC 结果 `storage/clone/`，分离批次 `storage/separation/`，
  人像视频 `storage/video/`；SoulX/Ditto 图片各保存在 `storage/flashhead/images/`、
  `storage/ditto/images/`。目录均可按对应配置覆盖。
- 设计 WAV 只保留在 `storage/timbre/`。通过共享 `AudioReferenceStore` 同步到其他服务时，
  同内容上传只在 `storage/timbre/.references/` 保存小型引用映射/文本，不在 clone 再复制 WAV。
  普通用户参考上传仍保存在 clone；不得破坏引用校验和内容哈希。
- 保留策略默认关闭；运维显式设置 `STORAGE_RETENTION_HOURS` 或 `STORAGE_RETENTION_MAX_BYTES`
  后才可运行 `uv run --project qwen3_tts python main/storage_maintenance.py [--apply]`。
  默认预览，不能自动删除普通上传或仍被引用的音色。该工具只扫描列出的 WAV 前缀，不覆盖
  FireRed/Confucius4/Seed-VC、新视频/分离/训练产物或任意自定义输出目录；详见 README。
- 训练脚本不自动获取服务 GPU 锁，执行真实训练/预测需独立安排 GPU；教师候选必须保留
  `research-only`、`automated-poc` 标记与按来源隔离的数据划分，不能当作生产 Gold 数据。

## 编码、文档与安全

Python 使用 4 空格、`snake_case` 函数/变量、`PascalCase` Pydantic 模型，遵循现有类型标注、
import 和 Ruff 风格。路由、校验、存储或生命周期变更须补针对性无模型测试并同步 README；
新增服务须同时核对 `start.sh`、质量门禁、配置回归测试及本文件速查表。
详细工程规范见 [docs/python-engineering.md](docs/python-engineering.md)。

不得提交权重、第三方源码、上传素材、生成音视频、训练数据/adapter、虚拟环境、缓存、密钥或
机器专用绝对路径；MOSS editable 路径是上述唯一例外。`MIMO_API_KEY` 仅从环境读取。
Commit subject 使用简洁的 Conventional Commit，如 `feat:`、`fix:`、`docs:`。

- Python 注释与 Docstring 默认简体中文，标识符、类型名、库/API 名和通用技术术语保留英文。
- 公共函数、公共类和重要业务入口提供 Google Style / PEP 257 Docstring。
- 复杂算法、进程生命周期、并发、GPU/CUDA、dtype/device 和兼容 workaround 说明“为什么”。
- 不为显然赋值、简单条件或自解释代码写逐行注释；AGENTS 保持规则手册，不追加变更流水账。
