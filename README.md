# LLM Local Backend

本仓库当前提供九个音频模型 HTTP API、一个 Ornith MTP 文本模型 API、MOSS 与 FLUX.3 Klein 两个图片 API、
一个轻量控制面和一个 CPU 空间音频 renderer。
所有 GPU API 共用同一个跨进程文件锁，接入约束、隔离和恢复步骤见[全局 GPU 锁说明](docs/global-gpu-lock.md)。

| 组件 | 端口 | 接口/职责 |
| --- | ---: | --- |
| [控制面 `main/`](main/) | 8300 | `GET /v1/health`、`GET /v1/control`、CPU 导出、Steam Audio 空间渲染与进度 |
| [Qwen3-TTS VoiceDesign](model-audio/design_qwen3-tts-1.7b-voicedesign/) | 8301 | `POST /v1/qwen/Qwen3-TTS-1.7B-VoiceDesign` |
| [Confucius4-TTS](model-audio/tts_confucius4-tts/) | 8331 | `POST /v1/netease/Confucius4-TTS-GGUF`，参考音频克隆 |
| [IndexTTS2.5](model-audio/tts_indextts2.5/) | 8312 | `POST /v1/indexTeam/IndexTTS2.5-GGUF`，参考音频克隆 |
| [MiDashengLM-Gen](model-audio/gen_midashenglm/) | 8341 | `POST /v1/mi/MiDashengLM-Gen-GGUF`，文本提示音频生成 |
| [Qwen3-ASR](model-audio/asr_qwen3-1.7b/) | 8321 | `POST /v1/qwen/Qwen3-ASR-1.7B-GGUF`，语音转写 |
| [Qwen3-TTS Base](model-audio/tts_qwen3-1.7b-base/) | 8311 | `POST /v1/qwen/Qwen3-TTS-12Hz-1.7B-Base-GGUF`，参考音频克隆 |
| [SeedVC](model-audio/vc_seed-vc/) | 8342 | `POST /v1/SeedVC/SeedVC-MLX-GGUF`，音色转换 |
| [Stable Audio SFX](model-audio/gen_stable-audio-3-small-sfx/) | 8343 | `POST /v1/stabilityai/Stable-Audio-3-Small-SFX-GGUF`，音效生成 |
| [VoxCPM2](model-audio/tts_voxcpm2/) | 8313 | `POST /v1/OpenBMB/VoxCPM2-GGUF`，TTS、音色设计与可选参考克隆 |
| [Ornith-1.5-9B-MTP-GGUF-Q8](model-text/Ornith-1.5-9B-MTP-GGUF/) | 8504 | `POST /v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8`，CUDA MTP 文本与图片理解 |
| [MOSS-Image-SFT-Q8_0](model-image/MOSS-Image-SFT-Q8_0/) | 8501 | `POST /v1/openmoss/MOSS-Image-SFT-Q8_0`，stable-diffusion.cpp CUDA 文生图 |
| [FLUX-3-Image-Klein-GGUF-Q8](model-image/FLUX-3-Image-Klein-GGUF-Q8/) | 8502 | `POST /v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8` 文生图；`POST /v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8/edit` 图片编辑 |
| [Steam Audio renderer](model-audio/spatial_steam-audio/) | 无 HTTP 端口 | C++17 CPU renderer，由 8300 控制面调用 |
| [Easel API bridge](easel_bridge/) | 8500 | 受 Bearer Key 保护的 OpenAI-compatible 图片、语音和MTP 9B 聊天接口 |

共十五个 HTTP 进程由 `start.sh` 管理，包括仅绑定 loopback 的 Ornith llama-server backend `127.0.0.1:8506`。LocalAI 保持在 `127.0.0.1:8400`。文本 API 适配器复用
控制面 Python 项目；八个新增 audio.cpp API 共用轻量 Python 运行环境，不额外安装
Torch。`storage/` 中的用户运行数据不会由代码清理。

Easel/OpenCode 桥接服务由 `start.sh` 一并启动在 `127.0.0.1:8500`，只代理 MOSS-Image-SFT、VoxCPM2 和 Ornith 9B Chat API（8504 MTP 模型）。
它不会启动或绕过模型服务，底层 GPU 锁与 sleep-idle 生命周期仍由既有服务管理。MOSS、VoxCPM2 与 Ornith 的聊天桥接上游默认绑定 `127.0.0.1`，避免从网关之外绕过 Bearer 鉴权；可在本机环境变量中显式覆盖绑定地址。
配置和 Easel 中的模型填写项见 [Easel 本地模型网关](docs/easel-bridge.md)。

可选的独立 Dagu 调度器提供 VoxCPM2 与 MOSS 后台队列，不改变现有模型 API 和 `start.sh`；部署与任务提交见 [Dagu 调度说明](docs/dagu.md)。

## 安装与启动

部署需要 Linux、Python `3.12.13`、`uv`、FFmpeg/ffprobe、`setsid`、CUDA/NVIDIA 驱动、仓库外准备的
固定版本 LocalAI/audio.cpp，以及带 CUDA、CUDA0 和 sleep-idle 支持的 llama.cpp `llama-server`。九个音频模型 GGUF 从
`$HF_MIRROR_DIR/audio.cpp-gguf/` 读取；固定文件名、family/task 和服务路由见
[`localai/manifests/audio-cpp-models.json`](localai/manifests/audio-cpp-models.json)。Ornith 9B GGUF
主模型与 BF16 vision projector 从
`$HF_MIRROR_DIR/ornith-ai/Ornith-1.5-9B-MTP-GGUF/` 只读加载，不复制进仓库。
版本和宿主制品指纹见[`localai/manifests/versions.json`](localai/manifests/versions.json)。
运行与恢复步骤见[LocalAI/audio.cpp 运维说明](docs/localai-audiocpp-upgrade.md)。
图片服务使用固定 commit 的 stable-diffusion.cpp CUDA `sd-cli`。MOSS-Image-SFT 组件从
`$HF_MIRROR_DIR/stable-diffusion-cpp-gguf/MOSS-Image-SFT-GGUF/` 读取；FLUX.3 Klein Q8 diffusion、
Qwen3-VL 4B、mmproj 和 VAE 从 `$HF_MIRROR_DIR/stable-diffusion-cpp-gguf/FLUX-3-Image-Klein-GGUF/`
读取。模型权重、tokenizer、源码和二进制保持在仓库外；运行参数、GPU 锁和 quarantine 流程见
[sd.cpp 运行说明](docs/sdcpp-runtime.md)。

先准备 LocalAI binary、`audio-cpp` backend 和模型文件，再同步三个 Python 项目及 QA：

```bash
uv sync --project main --locked
uv sync --project model-audio/design_qwen3-tts-1.7b-voicedesign --locked
uv sync --project model-audio/audio_cpp_runtime --locked
uv sync --project qa --locked
python3 scripts/localai/manager.py verify-install
```

正式空间渲染还需要 Steam Audio SDK 4.8.1 和 C++17 renderer。SDK 保持在仓库外：

```bash
bash scripts/build_steam_audio_renderer.sh
```

从仓库根目录启动：

```bash
export SDCPP_BIN="$HOME/.local/src/stable-diffusion.cpp/build/bin/sd-cli"
export SDCPP_BIN_SHA256="$(sha256sum "$SDCPP_BIN" | cut -d ' ' -f 1)"
bash start.sh
```

首次启动前按 [网关配置说明](docs/easel-bridge.md#配置和启动) 创建 `.env.easel-bridge` 和随机 Key。
脚本自动读取该文件；也可用 `EASEL_BRIDGE_ENV_FILE` 指定其他配置文件，网关端口由 `EASEL_BRIDGE_PORT` 设置，默认 8500。
网关 Key 缺失、过短或使用模板占位符时，脚本在启动模型前退出，不会匿名运行。

脚本先检查全部十五个监听端口、sd.cpp 二进制摘要、MOSS 与 FLUX.3 Klein 的模型组件、CUDA 构建的 llama.cpp、Ornith MTP 主模型和 MTP projector，注册九个 GGUF 候选，
启动受管 LocalAI/audio.cpp 并执行 readiness 检查；随后通过共享 GPU 锁预热 Ornith MTP backend，并确认模型进入 sleep-idle，
再启动控制面与所有 API。文本 API 每次请求结束后持锁等待模型卸载，再释放 GPU 给其他 provider。
任一服务进程退出时会清理其他进程组和 llama-server，再停止受管 LocalAI。按 `Ctrl+C`
正常等待退出；遇到进程异常时，检查 `scripts/localai/manager.py status` 和 GPU quarantine 标记。

每次运行 `start.sh` 都会把完整控制台输出实时同步到 `$RUNTIME_CACHE_DIR/logs/` 下的独立日志文件，
终端启动时会打印本次日志的完整路径。默认路径为 `storage/.cache/runtime/logs/`；出现 Codex 或服务错误时，
把对应启动批次的日志文件发回即可。日志文件由 `mktemp` 创建，默认仅当前用户可读写。
日志进程会避开终端退出信号，由主脚本统一清理服务；日志管道异常时，清理过程直接追加到同一日志文件。
Ctrl+C、终端挂断及清理期间再次按 Ctrl+C 都会继续完成已启动进程组的退出确认。

默认模型文件路径：

```text
~/hf-mirror/audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF/qwen3-tts-12hz-1.7b-voicedesign-bf16.gguf
```

`start.sh` 的 Ornith 默认参数采用本机实测选定的配置；复测步骤、性能及资源结论见
[文本模型对比测试](tests/model_web_comparison/README.md)。归档中的 Qwen3.6 数据仅供历史追溯。所有参数仍可通过环境变量覆盖。

主要环境变量：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `HOST` / `PORT` | `0.0.0.0` / `8300` | 控制面监听地址与端口 |
| `QWEN_VOICEDESIGN_HOST` / `QWEN_VOICEDESIGN_PORT` | 继承 `HOST` / `8301` | VoiceDesign 监听地址与端口 |
| `AUDIO_CPP_*_PORT` | 各服务默认端口 | 八个 audio.cpp 服务端口，可分别覆盖 |
| `VOXCPM2_HOST` | `127.0.0.1` | Easel 桥接使用的 VoxCPM2 上游监听地址 |
| `AUDIO_CPP_*_MODEL_FILE` | `$HF_MIRROR_DIR/audio.cpp-gguf/<候选目录>/<固定文件名>` | 各服务 GGUF 路径，完整变量名见模型清单 |
| `LLAMA_SERVER_BIN` | PATH 中的 `llama-server`，否则依次尝试 `$HOME/.local/src/llama.cpp/build-sleep-cache/bin/llama-server`、`build-cuda/bin/llama-server` | llama.cpp HTTP server；默认启用缓存时会核验补丁能力，也可显式设置路径 |
| `ORNITH_MTP_9B_LLAMA_BASE_URL` | `http://127.0.0.1:$ORNITH_MTP_9B_LLAMA_CPP_PORT` | 9B backend 根地址，必须与 loopback host 和端口一致 |
| `ORNITH_MTP_9B_HOST` / `ORNITH_MTP_9B_PORT` | 继承 `HOST` / `8504` | Ornith 9B 对外 API 监听地址与端口 |
| `ORNITH_MTP_9B_MODEL_FILE` | `$HF_MIRROR_DIR/ornith-ai/Ornith-1.5-9B-MTP-GGUF/Ornith-1.5-9B-MTP-Q8_0.gguf` | MTP Q8 主模型，只读加载 |
| `ORNITH_MTP_9B_MMPROJ_FILE` | `$HF_MIRROR_DIR/ornith-ai/Ornith-1.5-9B-MTP-GGUF/mmproj-Ornith-1.5-9B-BF16.gguf` | MTP vision projector，只读加载 |
| `ORNITH_MTP_9B_LLAMA_CPP_PORT` | `8506` | MTP llama-server loopback 端口，不绑定外部网卡 |
| `ORNITH_MTP_9B_CONTEXT_SIZE` / `ORNITH_MTP_9B_BATCH_SIZE` | `98304` / `1024` | Ornith 9B 上下文与 logical batch size；历史普通9B的96k性能分析见 [调优记录](tests/ornith_1_5_9b_mtp_benchmarks/performance.md) |
| `ORNITH_MTP_9B_UBATCH_SIZE` | `1024` | physical batch size，与 MTP 4 组合实测；内存不足可与 logical batch 一起回退到 512 |
| `ORNITH_MTP_9B_THREADS` / `ORNITH_MTP_9B_THREADS_BATCH` | `6` / `8` | Ornith 9B CPU 推理与批处理线程数 |
| `ORNITH_MTP_9B_N_GPU_LAYERS` | `99` | 尝试将可用 Ornith 9B 模型层卸载到 CUDA0 |
| `ORNITH_MTP_9B_SPEC_TYPE` / `ORNITH_MTP_9B_DRAFT_MAX` | `draft-mtp` / `4` | 使用主 GGUF 内置 MTP 头，主/草稿 KV 保持 F16；设 `SPEC_TYPE=none` 可回退。更换 GGUF 须确认含 MTP 张量 |
| `ORNITH_MTP_9B_SLEEP_CACHE` / `ORNITH_MTP_9B_CACHE_RAM_MIB` | `1` / `4096` | MTP backend 的跨 sleep CPU 缓存开关与预算；上限 4096 MiB，仍每轮释放 GPU；需 [后端缓存补丁](docs/llama-cpp-sleep-cache.md)，设开关 `0` 回退 |
| `ORNITH_MTP_9B_SLEEP_IDLE_SECONDS` | `1` | Ornith 9B llama.cpp 空闲后卸载模型和 KV cache 的秒数 |
| `ORNITH_MTP_9B_STARTUP_TIMEOUT` | `1800` 秒 | 等待 Ornith 9B 首次加载并确认 sleep-idle 的最长时间 |
| `ORNITH_MTP_9B_SLEEP_TIMEOUT` | `300` 秒 | Ornith 9B 单个请求结束后等待模型卸载的最长时间 |
| `SDCPP_BIN` | `$HOME/.local/src/stable-diffusion.cpp/build/bin/sd-cli` | 固定版本 stable-diffusion.cpp CLI |
| `SDCPP_BIN_SHA256` | 必须显式设置 | 启动前校验 sd-cli 文件摘要 |
| `SDCPP_MODEL_MANIFEST` | `sdcpp/manifests/models.json` | MOSS 模型组件、参数和资源清单 |
| `MOSS_IMAGE_HOST` / `MOSS_IMAGE_PORT` | `127.0.0.1` / `8501` | MOSS 图片 API 监听地址 |
| `FLUX3_IMAGE_HOST` / `FLUX3_IMAGE_PORT` | 继承 `HOST` / `8502` | FLUX.3 Klein 图片 API 监听地址 |
| `FLUX3_IMAGE_UPLOAD_DIR` | `$RUNTIME_CACHE_DIR/sdcpp_uploads` | 图片编辑参考图的临时暂存目录 |
| `FLUX3_IMAGE_MAX_UPLOAD_BYTES` | `67108864` | 编辑参考图上传上限，单位为字节 |
| `SDCPP_RUN_TIMEOUT` / `SDCPP_CLEANUP_TIMEOUT` | `3600` / `30` 秒 | 单次图片生成和 worker 回收时限 |
| `HF_MIRROR_DIR` | `~/hf-mirror` | 本地模型镜像根目录 |
| `QWEN_VOICEDESIGN_MODEL_FILE` | 上述 GGUF 路径 | VoiceDesign 模型文件 |
| `LLM_LOCALAI_DATA_DIR` | `$STORAGE_DIR/.cache/localai` | LocalAI binary、models、backend 和状态数据根目录 |
| `LLM_LOCALAI_BASE_URL` | `http://127.0.0.1:8400` | LocalAI loopback API |
| `GPU_LOCK_FILE` | `$RUNTIME_CACHE_DIR/gpu-runtime.lock` | 所有 GPU provider 必须共用的锁文件 |
| `GPU_LOCK_WAIT_TIMEOUT` | `900` 秒 | GPU 队列最长等待时间，部署值必须为正数 |
| `CUDA_RELEASE_DELAY` | `2.0` 秒 | 请求结束后等待 CUDA 资源回收 |
| `STORAGE_DIR` | 仓库内 `storage/` | 运行数据根目录 |
| `TIMBRE_STORAGE_DIR` | `$STORAGE_DIR/timbre` | VoiceDesign WAV 输出目录 |
| `STEAM_AUDIO_RENDERER_BIN` | `model-audio/spatial_steam-audio/build/steam-audio-render` | CPU renderer 可执行文件 |
| `STEAM_AUDIO_SDK_DIR` | `$HOME/.local/share/steam-audio/4.8.1` | renderer 使用的仓库外 SDK |

各模型的完整默认权重路径、变量名、端口和路由见模型清单；上传上限、超时及空间 renderer 的其他设置
可从 `start.sh` 和对应服务模块读取。`GET /v1/health` 不触发推理，会检查 LocalAI 与固定模型注册状态。

Ornith API 使用 CUDA 推理；空闲后 llama.cpp 通过 sleep-idle 卸载模型，API 请求共享 `GPU_LOCK_FILE`，直到确认卸载才归还 GPU。调用示例见下方 API 章节。

## API

### 控制面与标准音频导出

`GET /v1/control` 和 `GET /v1/health` 返回服务诊断、存储空间、GPU 可见性和 renderer 状态。
`POST /v1/audio/export` 是单一预混音轨的兼容导出接口。母带标准路径需显式传 `profile=standard`：

```bash
curl -X POST http://127.0.0.1:8300/v1/audio/export \
  -F 'audio=@timeline-mix.wav;type=audio/wav' \
  -F 'profile=standard' \
  -F 'output_format=wav' \
  -o llm-local-standard.wav
```

该接口不执行对象级空间定位。`balanced`/`immersive` 的正式空间任务应使用下方 Manifest API；不得在浏览器
预混对象，也不得在 Steam Audio pre-master 后叠加 Haas/aecho。

### Steam Audio 对象级渲染

`POST /v1/audio/spatial/render` 接收 Manifest v1 和重复的 `assets` multipart 字段。Manifest 中每个
`asset_filename` 必须与一个上传文件名对应；renderer 逐对象应用 Steam Audio Direct Effect 与双耳 HRTF，
生成 48 kHz 双声道 pre-master，再执行响度处理和编码。此路径为 CPU 任务，不获取 GPU 锁。

```bash
curl -X POST http://127.0.0.1:8300/v1/audio/spatial/render \
  -F 'manifest=<render-manifest.json' \
  -F 'assets=@narrator.wav;filename=asset_narrator.wav' \
  -F 'assets=@door.wav;filename=asset_door.wav' \
  -F 'profile=balanced' \
  -F 'output_format=wav' \
  -o llm-local-steam-balanced.wav
```

可选 `job_id` 用于轮询 `GET /v1/audio/spatial/render/progress/{job_id}`。进度状态只保留在当前控制面进程内，
成功或失败终态默认保留一小时。Manifest、位置、移动轨迹及支持范围见
[Steam Audio renderer 文档](model-audio/spatial_steam-audio/README.md)和[通用空间轨迹](docs/spatial-trajectories.md)。

当前不支持 `diffuse`、非 `none` 遮挡和几何反射；服务会拒绝这些字段，不会静默降级。

### Ornith-1.5-9B-MTP-GGUF-Q8 聊天补全

9B API 接受 OpenAI Chat Completions 格式，路由固定模型别名；每次请求使用共享 GPU 锁，并等待 backend 确认 sleep-idle 后释放：

```bash
curl -X POST http://127.0.0.1:8504/v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8 \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"请用中文介绍一下你自己。"}],"max_tokens":128}'
uv run --project qa --locked python tests/ornith_1_5_9b_mtp_benchmarks/generate_example.py
```

真实模型样例默认保存在 `storage/llm_tests/ornith_1_5_9b_mtp_benchmarks/`；运行选项见
[`tests/ornith_1_5_9b_mtp_benchmarks/README.md`](tests/ornith_1_5_9b_mtp_benchmarks/README.md)。

### Ornith-1.5-9B-MTP-GGUF-Q8 聊天与图片理解

MTP API 接受 OpenAI Chat Completions 格式，固定路由和模型别名。llama-server 加载 MTP Q8 主模型及 BF16 vision projector；每次请求仍使用共享 GPU 锁：

```bash
curl -X POST http://127.0.0.1:8504/v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8 \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"请用中文介绍一下你自己。"}],"max_tokens":128}'
uv run --project qa --locked python tests/ornith_1_5_9b_mtp_gguf_q8/generate_example.py \
  --image /path/to/image.png --prompt '请描述图片中的主要内容。'
```

真实样例默认保存在 `storage/llm_tests/ornith_1_5_9b_mtp_gguf_q8/`；运行方式和无模型合同测试见
[`tests/ornith_1_5_9b_mtp_gguf_q8/README.md`](tests/ornith_1_5_9b_mtp_gguf_q8/README.md)。

### Qwen3-TTS VoiceDesign

`POST /v1/qwen/Qwen3-TTS-1.7B-VoiceDesign` 接受 JSON，成功响应为 WAV，并将副本原子写入
`storage/timbre/`（可由 `TIMBRE_STORAGE_DIR` 覆盖）：

```bash
curl -X POST http://127.0.0.1:8301/v1/qwen/Qwen3-TTS-1.7B-VoiceDesign \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o qwen-voice.wav
```

请求字段：`voice_description` 必填，最长 2000 字；`text` 最长 12000 字；可选 `language`、
`max_chars_per_chunk`、`pause_ms`、`max_new_tokens`、`top_p` 和 `temperature`。其他字段会被拒绝。
生成请求通过共享 GPU 锁串行运行；队列超时或隔离状态返回 `503`。成功响应包含
`X-Audio-Sample-Rate`。

### audio.cpp 模型 API

八个新增 API 各自暴露 `GET /v1/health` 和清单中对应的 POST 路由。TTS/SeedVC 使用 multipart WAV 输入，
ASR 返回 JSON，其余生成任务返回 PCM WAV。参考文本、提示词、目标音色和时长字段按模型任务提供。
TTS/ASR/转换的上传会流式暂存到唯一临时文件，单文件上限 64 MiB，请求结束后清理；这些 API 不长期保存输入或输出。

例如 Qwen3-TTS Base 克隆：

```bash
curl -X POST http://127.0.0.1:8311/v1/qwen/Qwen3-TTS-12Hz-1.7B-Base-GGUF \
  -F 'text=你好，这是本地语音合成。' \
  -F 'reference_text=请替换为参考音频中的逐字稿。' \
  -F 'language=zh' \
  -F 'reference_audio=@reference.wav;type=audio/wav' \
  -o qwen3-base.wav
```

Qwen3-TTS Base 必须提供与 `reference_audio` 逐字匹配的 `reference_text`；`language=zh` 会映射到 audio.cpp
接受的 `Chinese`。

MiDashengLM-Gen 与 Stable Audio SFX 接受 JSON `text` 和可选 `duration_seconds`（默认 8 秒，最多 30 秒）。
Qwen3-ASR 接受 multipart `audio`，可选 `language` 与 `prompt`。SeedVC 接受 multipart `audio`（源音频）
与 `reference`（目标音色参考），两者都必须是 WAV。VoxCPM2 接受 `text`，可选 `instruction`、
`reference_audio`、`reference_text` 和 `language`。

真实模型串行生成示例和输入要求见[`tests/audio_cpp/README.md`](tests/audio_cpp/README.md)。
所有服务共享同一个 GPU 锁，因此并发 HTTP 请求会排队执行。

### 路由范围

不提供已退役模型 API、`POST /v1/upload_audio` 或 `GET /v1/check/audio`。当前每个模型只暴露清单中的业务路由和
`GET /v1/health`；主控制面的音频上传与空间渲染接口仍单独由 8300 提供。

## 存储与清理

- VoiceDesign 生成 WAV 默认保存在 `storage/timbre/`。
- renderer 上传及导出临时文件位于 `$RUNTIME_CACHE_DIR` 下，请求完成后清理。
- GPU 锁、LocalAI 临时产物和空间任务缓存位于 `$RUNTIME_CACHE_DIR`；文本模型样例保存在 `storage/llm_tests/`。
- 保留策略默认关闭。设置 `STORAGE_RETENTION_HOURS` 或 `STORAGE_RETENTION_MAX_BYTES` 后，可先预览：

  ```bash
  uv run --project main --locked python main/storage_maintenance.py
  ```

  确认清理计划后才传 `--apply`。脚本只匹配 `storage/timbre/` 下的 `qwen_voicedesign_*.wav`。

## 全局 GPU 锁

锁实现位于 `llm_local_runtime`，使用同一个 `GPU_LOCK_FILE` 和跨进程 `flock` 串行执行所有 GPU 模型。
请求需设置有限且为正的等待时间，并在 `finally` 中释放锁。LocalAI backend 无法确认停止时，锁旁的
`.quarantine` 标记会阻止后续模型进入；不要手动删除标记。运维确认服务退出后按
[全局 GPU 锁说明](docs/global-gpu-lock.md)执行恢复。

## 质量门禁

根目录回归不下载权重、不需要 CUDA，也不调用真实模型服务：

```bash
bash -n start.sh
bash scripts/quality_gate.sh
```

门禁检查控制面、VoiceDesign、Ornith 适配器、共享 audio.cpp 运行环境和 QA 的锁文件，运行 Steam Audio 无 SDK 核心测试、Ruff、能力清单校验和
无模型 unittest。单独运行测试：

```bash
uv run --project qa --locked python -m unittest discover -s tests -v
```

FastAPI 测试使用 QA 环境中的 `httpx2` 兼容依赖；不得通过执行真实模型或下载权重来运行根目录回归。
维护项目的 Python 约定见[工程规范](docs/python-engineering.md)。

## 其他归档资料

旧音频服务源码、历史 LocalAI 报告、任务和设计文档保留作追溯，不代表当前支持的接口或依赖。
stable-diffusion.cpp 通用 CLI 执行层及 MOSS、FLUX.3 Klein 图片 API 已接入；调用 demo 见
[`tests/MOSS-Image-SFT-Q8_0/`](tests/MOSS-Image-SFT-Q8_0/) 和
[`tests/FLUX-3-Image-Klein-GGUF-Q8/`](tests/FLUX-3-Image-Klein-GGUF-Q8/)。FLUX.3 Klein 的真实推理尚待本机 demo 验证；视频模型尚未接入，
退役视频实现继续保留在 `model-video/`。共享 GPU 锁、超时和恢复流程见 [sd.cpp 接入说明](docs/sdcpp-runtime.md)。

纯文本训练实验见 `model-text/`；原视频集成已退役，相关源码留在 `model-video/`。第三方模型、源码、
Steam Audio SDK 和权重许可各自适用，不由仓库 Apache-2.0 许可覆盖。

### 模型测试性能记录

`bash start.sh` 默认开启逐请求性能与资源记录，输出独立的性能目录。测试完成后可将
[`config/model-telemetry.env`](config/model-telemetry.env) 中 `MODEL_TELEMETRY_ENABLED` 改为 `false` 并重启。
每轮测试使用 `python3 scripts/model_telemetry.py begin --label 96k-run1` 创建标记，回复结束后运行
`python3 scripts/model_telemetry.py end <测试ID>` 生成报告。采集范围、操作步骤与指标口径见
[模型性能记录说明](docs/model-telemetry.md)。

### Ornith 单模型迁移说明

当前只维护 MTP Q8（API 8504、backend 8506），启动栈不再占用8503或8505。
部署参数统一使用 `ORNITH_MTP_9B_*`，旧的 `ORNITH_9B_*` 环境变量不再控制服务。
网关仅接受 `Ornith-1.5-9B-MTP-GGUF-Q8` 聊天模型 ID，上游地址使用 `EASEL_BRIDGE_MTP_CHAT_URL`。
本机 `.env.easel-bridge` 与 OpenCode 已设96k，output4096、buffer16384、keep10240，
并沿用 draft3、纯文本、休眠1秒与MTP CUDA缓冲0.5秒。历史测试报告保留原始模型身份。
旧服务目录和调用入口已移除，仓库外权重及历史生成结果不清理。须正常退出旧服务栈再启动新版本。
