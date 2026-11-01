# 仓库开发指南


# Music Postprocessing Rules

## Skill routing

- audio-analyzer:
  仅负责音频分析，输出 JSON，不修改源音频。

- mix-engineer:
  仅负责分轨 EQ、动态处理和混音。
  不执行最终母带标准化。

- mastering-engineer:
  唯一负责最终母带、响度及峰值控制。

- ffmpeg-skill:
  仅负责音频裁剪、探测、格式转换和通用导出。
  不主动接管混音与母带任务。

- music-evaluate:
  仅负责质量评估和生成报告。
  未经明确授权不得重新处理音频。

- batch-processor:
  负责批量任务执行，不决定音乐处理策略。

## Execution rules

1. 所有输入源文件只读。
2. 每个处理阶段写入独立输出目录。
3. 前一阶段通过验证后才能执行下一阶段。
4. 同一任务不得由两个 Skill 重复执行响度归一化。
5. 禁止多个进程同时写入同一输出文件。
6. 默认不并行执行多个 GPU 音频模型。
7. 全部阶段完成后执行最终质量检查。
8. 不得把 AI 分离的估计分轨当作原生录音分轨。

## 适用范围与依据

本仓库提供面向本地部署的 LLM 音视频后端，当前保留音乐理解、语音克隆、文本音色设计、音效生成、语音转写、
音源分离、图生视频、CPU 母带和 Steam Audio 空间导出。服务采用独立 uv 项目和一次性 worker。
文档遵循 CONSTITUTION.md 的中文优先要求；接口和配置以实际源码为准。
README.md 提供接入、配置和运维说明，本文件记录开发边界与命令。

## 项目结构与服务速查

当前维护 11 个模型服务 uv 项目与一个轻量控制面，start.sh 合计启动 12 个 HTTP 进程；
localai_worker_bridge 是额外的非 HTTP uv 包，不计入服务数，也不常驻导入 Torch。
原生能力和输出见 README 服务总览；端口、方法及路径以源码为准。
区分模型原生能力、已接入接口与服务编排；各 HTTP 进程提供 GET /v1/health。

| 模型/服务 | 默认端口 | 主业务接口 | 职责 | 原生输出 |
| --- | --- | --- | --- | --- |
| main 控制面 | 8300 | GET /v1/control；POST /v1/audio/export；POST /v1/audio/spatial/render | 控制、共享上传/检查、CPU 母带及空间导出 | 非模型服务 |
| `model-audio/design_qwen3-tts-1.7b-voicedesign/` | 8301 | POST /v1/qwen/timbre | 文本音色设计与 TTS | 语音波形 |
| `model-audio/analyse_moss-music-8b-Instruct/` | 8310 | POST /v1/openMoss/moss-music | 音乐描述、问答与歌词转写 | JSON 分析文本 |
| `model-audio/soundEffect-v2.0/` | 8362 | POST /v1/moss/soundEffect-v2.0 | 根据文本提示词生成最长 30 秒的环境音效 | 48 kHz 单声道 WAV |
| `model-audio/clone_qwen3-tts-1.7b-base/` | 8321 | POST /v1/qwen/clone | 参考音频零样本多语言 TTS 克隆 | 语音波形 |
| `model-audio/indexTTS-2.5/` | 8323 | POST /v1/index-team/indexTTS-2.5 | 参考音频零样本克隆，支持五种语言、时长和情感向量控制 | 22.05 kHz WAV |
| `model-audio/separate_tiger-dnr/` | 8351 | POST /v1/tigerDnr/separate | 对白/音效/音乐三路 Stem 分离 | 三路音频波形 |
| `model-audio/clone_confucius-4/` | 8361 | POST /v1/confucius4TTS/generate | 14 语种跨语言零样本 TTS 克隆 | 语音波形 |
| `model-audio/transcribe_qwen3-asr-1.7b/` | 8371 | POST /v1/qwen3/asr | 语种识别与语音转写 | 转写文本与语种 |
| `model-audio/separate_msst-mega53/` | 8380 | POST /v1/msst/separate | 按调用方选择分离音乐声部 | FLOAT WAV 与元数据 ZIP |
| `model-video/LongCat-Video-Avatar-1.5-Quanto-Int8/` | 8390 | POST /v1/meituan/LongCat-Video-Avatar-1.5-Quanto-Int8 | 参考图、音频与文本驱动单人视频，按时长延续生成 | 带输入音轨的 MP4 |
| `model-video/MiniMax-H3-16G-HiFi/` | 8392 | POST /v1/minimax/H3-16G-HiFi | MiniMax H3 FL2VA/Ref2VA 图生视频 | 默认无音轨；`include_audio=true` 时输出原生音轨 MP4 |

- main/main.py 只处理健康/控制、共享上传/检查和 CPU 导出，不加载模型包或推理。
  空间任务进度 GET /v1/audio/spatial/render/progress/{job_id} 是进程内状态。
- llm_local_runtime/ 是共享轻量包，负责流式上传、引用、原子提交、GPU 队列及存储容量/保留策略；
  不得导入 FastAPI、Torch 或模型包。它是其他项目的 editable 依赖，没有独立 uv.lock。
- inference_gateway/ 只被 Qwen3-TTS Base/VoiceDesign LocalAI provider 与 worker bridge 使用，
  只能依赖标准库和 llm_local_runtime。provider 默认 native。LocalAI backend 停止无法确认时，
  GPU 锁旁的 .quarantine 标记必须保留；运维需通过 scripts/localai/recover_quarantine.py
  解除隔离，并确认受管进程组退出、loopback API 端口关闭。
- scripts/localai/manager.py 校验宿主机 LocalAI 固定版本、audio.cpp backend 文件指纹、
  Qwen3-TTS 候选与 TIGER-DnR worker bridge；本项目使用 Linux 裸机，不使用 Docker。
  更新服务 API 后须同步 localai/manifests/capabilities.json 并运行能力清单校验。
- localai_worker_bridge/ 只依赖轻量 gateway/runtime 与 gRPC；按
  localai/manifests/worker-bridge.json 固定的 TIGER-DnR 模型 ID、项目和 worker.py 入口启动一次性 worker，
  只监听 loopback，不导入 FastAPI、Torch 或模型包。新增 worker 须覆盖合同、超时、大小、
  环境白名单和异常退出清理。
- model-audio/separate_msst-mega53/ 使用外部 MSST 源码和 hf-mirror 下的完整 Mega 53 权重；
  调用方必须通过 stems 选择模型支持的声部，并始终保留数学 residual。
  不加入 LocalAI worker bridge，默认 storage/msst/ ZIP 不由 WAV 保留策略清理。
- model-audio/indexTTS-2.5/ 使用仓库外官方 v2.5.0 推理源码与本地完整主/辅助权重；HTTP 进程
  只负责上传、检查、共享 GPU 排队和一次性 worker 编排。
- model-audio/soundEffect-v2.0/ 使用仓库外固定 revision 的官方 MOSS-TTS 推理源码与完整本地快照；
  HTTP 进程只负责请求校验、共享 GPU 排队与一次性 worker 编排。
- model-audio/spatial_steam-audio/ 是 C++17 Steam Audio CPU renderer，不获取 GPU 锁；SDK 和动态库在仓库外。
- qa/ 是无模型测试/Ruff 项目；scripts/quality_gate.sh 与 GitHub Actions 使用同一质量门禁。
  根目录与 MiniMax H3 基准的无模型回归使用 QA。
- model-text/scripts/training/、model-text/training/configs/ 是纯文本润色 LoRA 实验，不是语音训练或 HTTP 服务。
- model-video/MiniMax-H3-16G-HiFi/ 使用 `~/hf-mirror/minimax/MiniMax-H3-16G-HiFi` 的完整 HiFi 权重，
  以及仓库外固定版本 ComfyUI 和权重仓库配套自定义节点；8392 提供 POST /v1/minimax/H3-16G-HiFi。
  轻量 HTTP 层使用共享 GPU 锁和空闲显存门槛，每次请求由一次性 worker 启动 loopback ComfyUI，
  最多生成 960×544、5 秒视频；参考图只通过上传逻辑标识读取。
- model-video/LongCat-Video-Avatar-1.5-Quanto-Int8/ 使用 `$HF_MIRROR_DIR/meituan/LongCat-Video-Avatar-1.5` 的 Quanto DiT/T5、DMD LoRA、Whisper，
  以及 `$HF_MIRROR_DIR/meituan/LongCat-Video` 的基础 VAE；8390 通过 WanGP `shared.api.init()` 与 `submit_task()` 调用 `longcat_avatar_v1_5`，使用 WanGP 自带 MMGP Profile 4 和 SDPA attention。
  `start.sh` 先运行 `configure_wangp.py`，只在 WanGP `ckpts/` 与 `loras/` 下建立指向已有权重的软链接，并维护 WanGP `wgp_config.json`；推理 worker 使用 WanGP 独立 Python 环境。
  WanGP 强制启用 `KI` 参考图模式，API 必须提供已上传图片的逻辑标识。HTTP 层共享 GPU 锁，默认单卡至少留有 10,000 MiB 显存；每个请求启动一次性 worker，最多覆盖约 16.52 秒音频，结果通过 ffprobe 确认含视频与音轨后原子保存。
- 流行*模型功能简介.md、task*.md 和 `空间音频讨论/` 是调研/任务/方案资料，
  不能据此推断已实现能力。

## 安装、启动与测试

部署使用 Linux、Python `3.12.13`、`uv`、FFmpeg/ffprobe、CUDA/NVIDIA 驱动和固定版本 LocalAI/audio.cpp。
统一启动依赖 `setsid`、进程组信号与 `fcntl.flock`；不同服务必须保留各自 Torch/CUDA 环境。
renderer 构建需要 CMake 3.17+ 和 C++17。

按照 README 准备权重与外部源码，再执行 `uv sync --project <dir> --locked`。
修改依赖后在对应项目执行 `uv lock`，提交 `pyproject.toml` 与 `uv.lock`，并用 `uv lock --check` 验证。

MOSS-Music 使用本地 Q8/F16 GGUF 与固定 HOT-Step-CPP 的 `ace-caption`，不依赖 Torch。
Python worker 通过 exec 替换为 C++ 进程，仍由父进程统一回收进程组。CPU 预检拒绝超过 420 秒输入，
转为 16 kHz 单声道；上传与文本结果仅暂存，不长期保存。构建命令见该服务 README。

`bash start.sh` 在 `model-audio/clone_qwen3-tts-1.7b-base` 环境启动 8300，其他进程使用各自服务环境；端口/路径可通过
环境变量覆盖。启动统一使用 `uv run --no-sync`，并将 `llm_local_runtime/src` 加入 `PYTHONPATH`
作为共享包离线兜底。该兜底不能代替新增依赖后的 `uv sync --locked`。
脚本先检查整组端口，任一服务退出时终止其他进程组；重启前用 `Ctrl+C` 等待旧进程退出。
只部署部分模型时直接启动对应 `main.py`；统一脚本没有服务启停开关。
健康检查不会加载模型，HTTP 200 不能代替响应内的依赖/权重可用性检查。

```bash
bash -n start.sh
bash scripts/quality_gate.sh
curl -fsS http://127.0.0.1:8300/v1/control
```

质量门禁包含无 SDK 的三个 C++ 核心测试、全部服务/bridge/QA 锁检查、Ruff 与格式检查及两组无模型测试。缺失任何当前维护项目都会使门禁失败。
锁校验可能读取 Git/wheel 元数据；已有完整缓存时可设置 `UV_OFFLINE=1` 运行。
单独执行测试必须用 QA，不能通过模型环境安装依赖来运行根目录回归：

```bash
uv run --project qa --locked python -m unittest discover -s tests -v
uv run --project qa --locked python -m unittest discover -s tests/i2v-benchmark/tests -v
```

测试不得下载权重、依赖 CUDA、调用外部模型服务或执行真实模型；应 mock worker、subprocess、
文件系统边界和网络。FastAPI TestClient 使用 QA 的 `httpx2` 兼容依赖，不恢复旧客户端组合。

## 架构与接口约束

- 服务 `main.py` 负责校验、兼容字段、存储、响应与 worker 生命周期，`worker.py` 负责模型加载
  和推理。阻塞锁、subprocess 和网络任务必须使用同步路由或线程池，不能阻塞异步事件循环。
- 本地重型请求只启动一个 worker，使用该服务 uv 解释器。成功、失败、超时均须终止整个进程组
  并清理临时 JSON/WAV/MP4；不得让模型或 CUDA 上下文跨请求常驻。
- 本地模型使用共享 `GPU_LOCK_FILE` 串行执行，纯 CPU/文件预检应在获取锁前完成。保留
  `finally` 锁释放和持锁期间的 `CUDA_RELEASE_DELAY`；默认排队 900 秒，超时返回 `503`。
  整套部署应保持 `GPU_LOCK_WAIT_TIMEOUT` 为正数，新增代码不得恢复无限期等待；共享实现
  尚有非正值兼容分支。指标采样用 `GPU_METRICS_SAMPLE_INTERVAL`。
- 模型默认值集中在服务模块顶部；新增/修改 `start.sh` 配置只负责路由、路径、端口、环境和
  共享运行参数，不应新增静默推理覆盖。现有部分服务的默认参数还由脚本导出，改默认值须核对两处。
- `~/hf-mirror` 是模型维护范围的本地依据；新服务及现有服务的可选 provider 必须具有完整主模型和必需辅助权重，缺失时不得保留等待用户下载的运行代码。LocalAI/audio.cpp 的上游 family 支持声明不等于本地模型可用。
- 不恢复已从仓库移除的服务，不新增旧接口别名、集中式 `api/` 或 Conda 回退。若模型目录和依赖重新准备，先单独评估维护范围及完整运行合同。
- Qwen3-TTS 克隆拒绝 `style_prompt`。
- Confucius4 默认 `lang=zh`，参考 sidecar 不参与推理。Qwen3-ASR/TIGER 直接接收 multipart `audio`，
  输入只暂存；ASR 返回 JSON，TIGER 返回 ZIP 并原子保存分离批次。
- 正式 `balanced`/`immersive` 必须保留对象边界，走 `/v1/audio/spatial/render` 的 Manifest v1
  与重复 `assets` 字段，不在浏览器预混、不获取 GPU 锁。Steam Audio pre-master 后只能
  loudnorm/编码，禁止再叠加 Haas/aecho。`/v1/audio/export` 调用只用于显式 `profile=standard`
  的预混母带；其实现仍兼容旧空间滤镜，不扩展该旧路径。
- `diffuse`、非 `none` 遮挡、几何反射尚未实现；不得将保留的语义字段描述为已可用能力。

## 上传、存储与清理

- 音频上传经 `llm_local_runtime` 按块暂存，默认 64 MiB，由 `UPLOAD_MAX_BYTES` 覆盖；校验扩展名/
  Content-Type、计算 SHA-256，使用临时文件和 `os.replace` 原子提交。不整文件 `read()` 上传，
  不直接覆盖 sidecar 或引用映射。
- `full_path` 是客户端逻辑标识，不能用于直接读取服务器任意文件。参考文本可保存 sidecar；
  8300 上传不接收 `prompt_text`，ASR/TIGER 没有独立上传/检查接口。
- 默认目录：音色 `storage/timbre/`、视频 `storage/video/`、普通参考音频与克隆结果
  `storage/clone/`、分离批次 `storage/separation/`。
  目录均可按对应配置覆盖。
- 设计 WAV 只保留在 `storage/timbre/`。通过共享 `AudioReferenceStore` 同步到其他服务时，
  同内容上传只在 `storage/timbre/.references/` 保存小型引用映射/文本，不在 clone 再复制 WAV。
  普通用户参考上传仍保存在 clone；不得破坏引用校验和内容哈希。
- 保留策略默认关闭；运维显式设置 `STORAGE_RETENTION_HOURS` 或 `STORAGE_RETENTION_MAX_BYTES`
  后才可运行 `uv run --project model-audio/clone_qwen3-tts-1.7b-base python main/storage_maintenance.py [--apply]`。
  默认预览，不能自动删除普通上传或仍被引用的音色。该工具只扫描列出的 WAV 前缀，不覆盖
  Confucius4、分离/训练产物或任意自定义输出目录；详见 README。
- 训练脚本不自动获取服务 GPU 锁，执行真实训练/预测需独立安排 GPU；教师候选必须保留
  `research-only`、`automated-poc` 标记与按来源隔离的数据划分，不能当作生产 Gold 数据。

## 编码、文档与安全

Python 使用 4 空格、`snake_case` 函数/变量、`PascalCase` Pydantic 模型，遵循现有类型标注、
import 和 Ruff 风格。路由、校验、存储或生命周期变更须补针对性无模型测试并同步 README；
新增服务须同时核对 `start.sh`、质量门禁、配置回归测试及本文件速查表。
详细工程规范见 [docs/python-engineering.md](docs/python-engineering.md)。

不得提交权重、第三方源码、上传素材、生成音视频、训练数据/adapter、虚拟环境、缓存、密钥或
机器专用绝对路径。
Commit subject 使用简洁的 Conventional Commit，如 `feat:`、`fix:`、`docs:`。

- Python 注释与 Docstring 默认简体中文，标识符、类型名、库/API 名和通用技术术语保留英文。
- 公共函数、公共类和重要业务入口提供 Google Style / PEP 257 Docstring。
- 复杂算法、进程生命周期、并发、GPU/CUDA、dtype/device 和兼容 workaround 说明“为什么”。
- 不为显然赋值、简单条件或自解释代码写逐行注释；AGENTS 保持规则手册，不追加变更流水账。
