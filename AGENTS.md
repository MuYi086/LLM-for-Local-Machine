# 仓库开发指南

## 适用范围

本仓库是 Unitale 的本地语音后端，提供语音合成、音色设计、语音编辑和音效生成能力。
项目采用多项目 uv 结构；每个服务独立维护自己的 `pyproject.toml`、`uv.lock`、
HTTP 入口和 worker。

## 项目结构与服务速查

- `main/main.py` 是轻量的 `8300` 控制面，负责控制路由、共享上传/检查工具和 MiMo 兼容代理；
  不得在其中加载模型包或执行推理。
- `steam_audio_renderer/` 是独立 C++17 Steam Audio CPU renderer；正式空间导出必须保留对象
  边界并走 `/v1/audio/spatial/render`，不得先在浏览器预混，也不得获取 GPU 锁。
- `unitale_runtime/` 是所有服务共用的轻量运行时包，提供流式上传、内容寻址引用、原子提交、
  GPU 队列和存储容量/保留策略；不得在其中导入 FastAPI、Torch 或模型包。
- `qa/` 是无模型质量门禁环境，`scripts/quality_gate.sh` 是统一的锁文件、Ruff、格式化和测试入口。
- 模型服务位于 `mimo_tts/`、`qwen3_tts/`、`voxcpm2/`、
  `LongCat_AudioDiT_3.5B_bf16/`、`dots_tts_soar/`、`moss_soundEffect/`、
  `stable_audio_3_medium/`、`ace_step_1_5/`、`qwen3_voiceDesign/`、`moss_voiceGenerator/` 和
  `moss_audio_4b_thinking/`、`Step_Audio_EditX/`、`firered_tts3/`、`Qwen3_ASR_1.7B/`。其中 MOSS-Audio 通过同一目录的两个独立进程分别提供
  Thinking 和 Instruct，FireRedTTS3 通过两个独立进程分别提供
  Instruct 音色设计和 Base 参考音频克隆。
- `Confucius4_TTS/` 是独立的 Confucius4-TTS 零样本语音克隆服务，依赖仓库外的官方推理源码和
  Wav2Vec2-BERT、BigVGAN、CAMPPlus 外部权重。
- `Qwen3_ASR_1.7B/` 是独立的 Qwen3-ASR-1.7B 语音识别服务，使用 `$HF_MIRROR_DIR/Qwen/Qwen3-ASR-1.7B`
  本地权重和 `qwen-asr` Transformers 后端。
- `seed-vc/` 是独立的 Seed-VC 音色转换服务，使用 `$HF_MIRROR_DIR/Plachta/Seed-VC` 本地权重、
  仓库外官方源码、Whisper-small、BigVGAN 和 CAMPPlus；可选 F0 模式另需 RMVPE 和 44.1 kHz BigVGAN。
- `SoulX-FlashHead-1_3B/` 是独立的音频驱动人像视频服务，使用 Soul-AILab 本地 Lite/Pro 权重、
  仓库外官方源码和 `facebook/wav2vec2-base-960h`；返回 MP4，成品默认写入 `storage/video/`。
- 最终默认端口：`8300` 控制面、`8301` Qwen VoiceDesign、`8302` MOSS VoiceGenerator、
  `8303` MiMo、`8304` FireRedTTS3 Instruct、`8311` Stable Audio 3 Medium、
  `8312` MOSS-SoundEffect、`8313` ACE-Step BGM、`8321` Qwen3-TTS、`8322` VoxCPM2、
  `8323` LongCat、`8324` dots.tts-soar、`8325` FireRedTTS3 Base、`8331` Step-Audio-EditX、
  `8341` MOSS-Audio-4B-Thinking、`8342` MOSS-Audio-4B-Instruct、`8361` Confucius4-TTS、
  `8371` Qwen3-ASR-1.7B、`8381` Seed-VC、`8391` SoulX-FlashHead；另有 `8351` TIGER-DnR。
- `tests/` 存放无模型 `unittest` 迁移回归测试；ACE-Step 和 Stable Audio 的服务内测试独立存放。
  `soundEffect/` 存放 MOSS GPU 示例；`storage/` 存放运行音频、sidecar、缓存和 GPU 锁，
  不得提交其内容。

## 安装、启动与测试

本地模型推理需要 Python `3.12.13`、`uv` 和 CUDA 可用的主机。按照 `README.md` 准备模型权重
与外部源码，然后使用 `uv sync --project <dir> --locked` 同步需要的项目。
`moss_voiceGenerator` 的 `moss-tts` 是唯一例外：必须使用预先准备的本地 editable
源码 `/home/muyi086/tts-depency/MOSS-TTS`，不得改为 Git/PyPI 下载；执行该项目的
`uv sync` 前先确认该目录存在。
`bash start.sh` 会在 `qwen3_tts` uv 项目中启动轻量的 8300 控制面，并在各自项目中启动其余
20 个 HTTP 进程（MOSS-Audio 占用 8341 和 8342 两个变体进程，FireRedTTS3 占用 8304 和 8325 两个模式进程）；共启动 21 个进程。
端口、路径、项目和运行参数均通过环境变量覆盖。
启动脚本使用 `uv run --no-sync`，并将 `unitale_runtime/src` 放入 `PYTHONPATH` 作为共享包的
离线兜底；新增或变更项目依赖仍必须提前执行对应项目的 `uv sync --locked`。

```bash
bash -n start.sh
bash scripts/quality_gate.sh
```

需要启动本地模型服务时，先准备 README 中列出的权重和外部源码，再执行 `bash start.sh`。
服务启动后可用 `curl -fsS http://127.0.0.1:8300/v1/control` 检查控制面。
无模型测试也可以单独从 QA 环境运行：

```bash
uv run --project qa --locked python -m unittest discover -s tests -v
(cd ace_step_1_5 && uv run --project ../qa --locked python -m unittest discover -s tests -v)
(cd stable_audio_3_medium && uv run --project ../qa --locked python -m unittest discover -s tests -v)
```

测试不得下载权重、依赖 CUDA、调用 MiMo 或执行真实模型。应 mock worker、subprocess、
文件系统边界和网络调用。只有真实 MOSS GPU smoke test 才使用
`bash soundEffect/run_moss_soundeffect_v2.sh`。
常规验证使用 `bash scripts/quality_gate.sh`；该脚本从服务目录运行 ACE-Step 和 Stable Audio
测试，避免同名 `runtime` 模块解析错误。

## 架构约束

- 各服务的 `main.py` 负责 HTTP 校验、兼容逻辑、存储和响应；`worker.py` 负责模型加载与推理。
- 重型本地服务每个请求只启动一个 worker。worker 必须使用该服务的 uv 解释器，在成功、失败
  或超时时终止自己的进程组，并清理临时 JSON/WAV 文件。
- 本地 GPU 服务通过共享的 `GPU_LOCK_FILE` 串行执行；必须保留 `finally` 中的锁释放逻辑和
  CUDA 释放等待时间。获取锁前必须完成纯 CPU/文件预检；等待超过 `GPU_LOCK_WAIT_TIMEOUT`
  时返回 `503`，不得无限期阻塞。持锁期间记录排队时间、worker 执行时间和峰值显存，采样间隔
  可由 `GPU_METRICS_SAMPLE_INTERVAL` 调整。
- 参考音频上传必须经 `unitale_runtime` 流式暂存，默认上限 `64 MiB`（可由 `UPLOAD_MAX_BYTES`
  调整），校验音频扩展名/Content-Type、计算 SHA-256，并用临时文件加 `os.replace` 原子提交。
  不得使用整文件 `read()` 或直接覆盖 sidecar、引用映射。
- 参考音频使用 WebUI 的 `full_path` 标识，并可保存 `prompt_text` sidecar。生成结果按用途
  保存：音色写入 `storage/timbre/`，音效写入 `storage/soundEffect/`，克隆/编辑音频写入
  `storage/clone/`；BGM 写入 `storage/bgm/`；这些目录都可覆盖。
- 音色设计返回的 WAV 只能保存在 `storage/timbre/`。当 WebUI 为克隆预览把设计音频同步到
  Qwen3-TTS、VoxCPM2、LongCat、dots 或 FireRedTTS3 服务时，只能在 `storage/timbre/.references/` 写入
  小型引用映射和文本 sidecar，不得在 `storage/clone/` 再复制一份设计 WAV；普通用户上传的
  参考音频仍保存到 `storage/clone/`。
- 参考音频克隆使用 `/v1/qwen/clone`、`/v1/voxcpm2/clone`、`/v1/longCat/clone` 和
  `/v2/dotsTTS/clone`、`/v1/FireRedTTS3/clone`；音色设计使用 `/v1/qwen/timbre`、
  `/v1/moss/timbre`、`/v1/mimo/timbre` 和 `/v1/FireRedTTS3/timbre`；音效使用
  `/v1/stableAudio/soundEffect`、`/v1/moss/soundEffect`；BGM 使用 `/v1/aceStep/bgm`；
  语音编辑使用 `/v1/stepAudioEditx/edit`；音频理解使用
  `/v1/mossAudioThinking/understand`；Qwen3-ASR 转写使用 `/v1/qwen3/asr`；Seed-VC 音色转换使用
  `/v1/seedVc/voiceConversion`。Seed-VC 输入通过上传后的 `full_path` 解析，结果写入 `storage/clone/`。
  SoulX-FlashHead 人像视频使用 `/v1/soulX/flashHead`，图片和音频均通过上传后的 `full_path` 解析。
- 后端只注册并使用上述最终接口；不得新增或保留任何旧接口兼容别名。
- `/v1/audio/export` 只保留预混总线的标准母带兼容；正式 `balanced`/`immersive` 使用
  `/v1/audio/spatial/render` 的 Manifest v1 和重复 `assets` 字段。正式 Steam Audio pre-master
  之后只能执行 loudnorm/编码，禁止再叠加 Haas、aecho 或旧空间滤镜。
- 模型默认值集中放在各服务模块顶部。`start.sh` 只负责路由、路径、端口、环境和共享运行参数，
  不应静默替换推理默认值。
- `GET /v1/control` 和各服务健康接口可报告存储容量；历史生成文件保留策略默认关闭。只有运维
  显式设置 `STORAGE_RETENTION_HOURS` 或 `STORAGE_RETENTION_MAX_BYTES` 后，才能运行
  `uv run --project qwen3_tts python main/storage_maintenance.py [--apply]`；该工具不得删除
  普通上传或仍被 `.references` 引用的音色 WAV。

## 编码规范与安全

Python 使用 4 个空格缩进，函数和变量使用 `snake_case`，Pydantic 模型使用 `PascalCase`。
遵循现有的类型标注、import、docstring 和换行风格；Ruff 规则按各项目 `pyproject.toml`
配置，并由 `scripts/quality_gate.sh` 实际执行 `ruff check` 与 `ruff format --check`。
路由、校验、存储或 worker 生命周期发生变化时，补充针对性的无模型测试，并同步更新
`README.md` 中的 endpoint 或字段说明；根目录测试必须通过 `qa` 项目运行。
- FastAPI TestClient 回归使用轻量 QA 环境的 `httpx2` 兼容依赖，不要恢复已弃用的旧客户端组合。

不得加入模型权重、上传/参考音频、生成 WAV、虚拟环境、缓存、密钥或机器专用绝对路径；
`moss_voiceGenerator` 的 `moss-tts` 本地源码路径是上文明确的唯一例外。
`MIMO_API_KEY` 必须通过环境变量提供。Commit subject 使用简洁的 Conventional Commit 格式，
例如 `feat:`、`fix:` 或 `docs:`。

## Python 注释规范

- Python 注释和 Docstring 默认使用简体中文。
- Python 标识符、类型名、库名、API 名称和业内通用技术术语保持英文。
- 公共函数、公共类及重要业务入口应提供 Docstring。
- 复杂算法、进程生命周期、并发控制、GPU/CUDA 资源管理、
  dtype/device 选择及兼容性 workaround 应说明“为什么这样做”。
- 不为显而易见的赋值、简单条件判断和自解释代码添加逐行注释。
- Docstring 遵循 Google Style 与 PEP 257。
