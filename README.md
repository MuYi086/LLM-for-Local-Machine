# LLM Local Backend

本仓库当前提供九个音频模型 HTTP API、两个文本模型 API、一个轻量控制面和一个 CPU 空间音频 renderer。
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
| [Ornith-1.5-35B-A3B](model-text/ornith_1_5_35b_a3b/) | 8401 | `POST /v1/AtomicChat/Ornith-1.5-35B-A3B-AD-Q5`，CUDA 推理，前 23 层 MoE 专家放 CPU，mmproj 支持图片 |
| [Qwen3.6-35B-A3B-MTP](model-text/qwen3_6_35b_a3b/) | 8402 | `POST /v1/qwen/Qwen3.6-35B-A3B-MTP`，CUDA + CPU MoE 混合推理，启用 MTP |
| [Steam Audio renderer](model-audio/spatial_steam-audio/) | 无 HTTP 端口 | C++17 CPU renderer，由 8300 控制面调用 |

共十四个 HTTP 进程由 `start.sh` 管理，包括两个仅绑定 loopback 的 llama-server 后端：Ornith 在 `127.0.0.1:8403`，
Qwen 在 `127.0.0.2:8400`。LocalAI 保持在 `127.0.0.1:8400`，两者通过不同 loopback 地址并存。文本 API 适配器复用
控制面 Python 项目；八个新增 audio.cpp API 共用轻量 Python 运行环境，不额外安装
Torch。`storage/` 中的用户运行数据不会由代码清理。

## 安装与启动

部署需要 Linux、Python `3.12.13`、`uv`、FFmpeg/ffprobe、`setsid`、CUDA/NVIDIA 驱动、仓库外准备的
固定版本 LocalAI/audio.cpp，以及带 CUDA、CUDA0、`--spec-type draft-mtp` 和 sleep-idle 支持的 llama.cpp `llama-server`。九个音频模型 GGUF 从
`$HF_MIRROR_DIR/audio.cpp-gguf/` 读取；固定文件名、family/task 和服务路由见
[`localai/manifests/audio-cpp-models.json`](localai/manifests/audio-cpp-models.json)。Ornith 的 GGUF 与 mmproj
从 `$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/` 读取，Qwen GGUF 从
`$HF_MIRROR_DIR/unsloth/Qwen3.6-35B-A3B-MTP/` 读取，不复制进仓库。
版本和宿主制品指纹见[`localai/manifests/versions.json`](localai/manifests/versions.json)。
运行与恢复步骤见[LocalAI/audio.cpp 运维说明](docs/localai-audiocpp-upgrade.md)。

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
bash start.sh
```

脚本先检查全部十四个监听端口、CUDA 构建的 llama.cpp、两个文本模型文件和必需参数，注册九个 GGUF 候选，
启动受管 LocalAI/audio.cpp 并执行 readiness 检查；随后使用共享 GPU 锁依次预热 Ornith 和 Qwen，并确认模型进入 sleep-idle，
再启动控制面与所有 API。两个文本 API 每次请求结束后持锁等待对应模型卸载，再释放 GPU 给其他 provider。
任一服务进程退出时会清理其他进程组和 llama-server，再停止受管 LocalAI。按 `Ctrl+C`
正常等待退出；遇到进程异常时，检查 `scripts/localai/manager.py status` 和 GPU quarantine 标记。

默认模型文件路径：

```text
~/hf-mirror/audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF/qwen3-tts-12hz-1.7b-voicedesign-bf16.gguf
```

`start.sh` 的 Ornith 与 Qwen 默认参数已采用本机三次完整实测选定的配置；复测步骤、性能及资源结论见
[文本模型对比测试](tests/model_web_comparison/README.md)。所有参数仍可通过环境变量覆盖。

主要环境变量：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `HOST` / `PORT` | `0.0.0.0` / `8300` | 控制面监听地址与端口 |
| `QWEN_VOICEDESIGN_HOST` / `QWEN_VOICEDESIGN_PORT` | 继承 `HOST` / `8301` | VoiceDesign 监听地址与端口 |
| `AUDIO_CPP_*_PORT` | 各服务默认端口 | 八个 audio.cpp 服务端口，可分别覆盖 |
| `AUDIO_CPP_*_MODEL_FILE` | `$HF_MIRROR_DIR/audio.cpp-gguf/<候选目录>/<固定文件名>` | 各服务 GGUF 路径，完整变量名见模型清单 |
| `LLAMA_SERVER_BIN` | PATH 中的 `llama-server`，否则尝试 `$HOME/.local/src/llama.cpp/build-cuda/bin/llama-server` | llama.cpp HTTP server 可执行文件；也可显式设置路径 |
| `ORNITH_MODEL_FILE` | `$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf` | Ornith 主模型，只读加载 |
| `ORNITH_MMPROJ_FILE` | `$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/mmproj-Ornith-1.5-35B-A3B-F16.gguf` | 多模态 projector，只读加载 |
| `ORNITH_MTP_FILE` | `$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/mtp-Ornith-1.5-35B-A3B-Q8_0.gguf` | 独立 MTP draft 模型，通过 `--spec-type draft-mtp --spec-draft-model` 只读加载 |
| `ORNITH_SPEC_TYPE` | `draft-mtp` | Ornith 推测解码类型：`draft-mtp` 或 `none`；`none` 不要求或加载独立 MTP 文件 |
| `ORNITH_SPEC_DRAFT_N_MAX` | `4` | 最大草稿长度，非负整数 |
| `ORNITH_SPEC_DRAFT_CACHE_TYPE_K` / `ORNITH_SPEC_DRAFT_CACHE_TYPE_V` | `f16` / `f16` | 独立 MTP 的 K/V 缓存类型 |
| `ORNITH_HOST` / `ORNITH_PORT` | 继承 `HOST` / `8401` | Ornith 对外 API 监听地址 |
| `LLAMA_CPP_PORT` | `8403` | Ornith llama-server loopback 端口，不绑定外部网卡 |
| `ORNITH_CONTEXT_SIZE` | `32768` | llama.cpp 上下文长度，最大 `32768` |
| `ORNITH_BATCH_SIZE` | `512` | llama.cpp logical batch size |
| `ORNITH_THREADS` | `6` | llama.cpp CPU 推理线程数 |
| `ORNITH_THREADS_BATCH` | `8` | 提示处理及 MTP 验证的 CPU 批量线程数 |
| `ORNITH_CACHE_TYPE_K` / `ORNITH_CACHE_TYPE_V` | `q8_0` / `q8_0` | 主模型 K/V 缓存 |
| `ORNITH_LOAD_MODE` | `none` | llama.cpp 加载方式，采用本机实测配置 |
| `ORNITH_N_GPU_LAYERS` | `99` | 尝试将可用模型层卸载到 CUDA0 |
| `ORNITH_N_CPU_MOE` | `23` | 前 23 个 MoE 层的专家权重放在 CPU |
| `ORNITH_SLEEP_IDLE_SECONDS` | `1` | llama.cpp 空闲后卸载模型和 KV cache 的秒数 |
| `ORNITH_STARTUP_TIMEOUT` | `1800` 秒 | 等待首次加载并确认 sleep-idle 的最长时间 |
| `ORNITH_SLEEP_TIMEOUT` | `300` 秒 | 单个请求结束后等待模型卸载的最长时间 |
| `QWEN36_MODEL_FILE` | `$HF_MIRROR_DIR/unsloth/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf` | Qwen GGUF，只读加载 |
| `QWEN36_HOST` / `QWEN36_PORT` | 继承 `HOST` / `8402` | Qwen 对外 API 监听地址 |
| `QWEN36_BACKEND_HOST` / `QWEN36_LLAMA_CPP_PORT` | `127.0.0.2` / `8400` | Qwen llama-server loopback 地址，与 LocalAI 的 `127.0.0.1:8400` 并存 |
| `QWEN36_CONTEXT_SIZE` / `QWEN36_BATCH_SIZE` | `32768` / `512` | Qwen llama.cpp 上下文和 batch 大小 |
| `QWEN36_THREADS` / `QWEN36_THREADS_BATCH` | `8` / `8` | 生成及 CPU 批量线程数 |
| `QWEN36_CACHE_TYPE_K` / `QWEN36_CACHE_TYPE_V` | `q8_0` / `q8_0` | 主模型 K/V 缓存 |
| `QWEN36_LOAD_MODE` | `none` | llama.cpp 加载方式 |
| `QWEN36_SPEC_DRAFT_N_MAX` | `4` | 最大 MTP 草稿长度 |
| `QWEN36_SPEC_DRAFT_CACHE_TYPE_K` / `QWEN36_SPEC_DRAFT_CACHE_TYPE_V` | `f16` / `f16` | MTP 草稿 K/V 缓存 |
| `QWEN36_N_GPU_LAYERS` / `QWEN36_N_CPU_MOE` | `99` / `23` | CUDA0 层卸载与 CPU MoE 层数，默认混合推理配置 |
| `QWEN36_SLEEP_IDLE_SECONDS` | `1` | llama.cpp 空闲后卸载模型和 KV cache 的秒数 |
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

Ornith API 使用 CUDA 计算，并将配置的前 23 个 MoE 专家层放在 CPU；空闲后 llama.cpp 通过 sleep-idle 卸载模型，
API 请求共享 `GPU_LOCK_FILE`，直到确认卸载才归还 GPU。混合推理接口和示例：

```bash
curl -X POST http://127.0.0.1:8401/v1/AtomicChat/Ornith-1.5-35B-A3B-AD-Q5 \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好，请用简洁的中文介绍一下自己。"}],"max_tokens":128}'
uv run --project qa --locked python tests/ornith_1_5_35b_a3b/generate_example.py
```

样例默认保存在 `storage/llm_tests/ornith_1_5_35b_a3b/`，也可通过 `ORNITH_TEST_OUTPUT_DIR` 指定目录。

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

### Ornith-1.5-35B-A3B 聊天补全

API 适配器将 OpenAI Chat Completions 格式转发给本机 llama-server；模型由 URL 路由固定：

```bash
curl -X POST http://127.0.0.1:8401/v1/AtomicChat/Ornith-1.5-35B-A3B-AD-Q5 \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"用中文解释什么是混合量化。"}],"max_tokens":256}'
```

真实模型样例生成入口见[`tests/ornith_1_5_35b_a3b/README.md`](tests/ornith_1_5_35b_a3b/README.md)。

### Qwen3.6-35B-A3B-MTP 聊天补全

Qwen 使用本机 `Qwen3.6-35B-A3B-UD-Q5_K_M.gguf`，固定 alias 为 `Qwen3.6-35B-A3B-MTP`。默认启用 CUDA0、
25 个 CPU MoE 层和 `draft-mtp`；backend 绑定 `127.0.0.2:8400`，可与 `127.0.0.1:8400` 的 LocalAI 并存。
对外接口为 `POST /v1/qwen/Qwen3.6-35B-A3B-MTP`：

```bash
curl -X POST http://127.0.0.1:8402/v1/qwen/Qwen3.6-35B-A3B-MTP \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"用中文简要解释 MTP。"}],"max_tokens":128}'
uv run --project qa --locked python tests/qwen3_6_35b_a3b/generate_example.py
```

真实生成的非空文本默认写入 `storage/llm_tests/qwen3_6_35b_a3b/`。样例请求关闭 thinking，普通 API 请求保留客户端参数。

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
纯文本训练实验见 `model-text/`；视频集成已退役，相关源码留在 `model-video/`。第三方模型、源码、
Steam Audio SDK 和权重许可各自适用，不由仓库 Apache-2.0 许可覆盖。
