# Unitale 本地 AI 音视频后端

Unitale 前端使用的本地后端，提供参考音频语音克隆、音色设计、语音编辑、音效与 BGM
生成、音频理解与转写、三路音源分离、音色转换、音频驱动人像视频，以及 CPU 音频母带和
Steam Audio 对象级空间导出。MiMo 音色设计调用云端，其余模型使用本地权重。

仓库包含 **19 个独立服务 uv 项目，统一启动 22 个 HTTP 进程**：FireRedTTS3 和 MOSS-Audio
各启动两个模式进程，另有一个轻量控制面。HTTP 进程不加载重型模型，模型推理由对应目录
中的一次性 worker 完成；共享运行时和无模型 QA 环境分别位于 `unitale_runtime/`、`qa/`。
接口、路径和默认值以当前 `main.py`、`worker.py`、`pyproject.toml` 与 `start.sh` 为准。

导航：[服务总览](#服务总览) · [已移除能力](#已移除与兼容边界) ·
[安装与启动](#安装与启动) · [模型与配置](#模型路径与主要配置) ·
[空间导出](#48-khz-母带与-steam-audio-正式导出) · [语音克隆](#参考音频克隆) ·
[训练实验](#文本润色训练实验) · [测试与开发](#测试与开发)。

## 服务总览

| 服务 | 端口 | 主要用途 | 主要路由 |
| --- | ---: | --- | --- |
| 控制面 | 8300 | 控制面、共享上传/检查、48 kHz 母带与 Steam Audio 正式导出、MiMo 兼容代理 | `/v1/control`、`/v1/audio/export`、`/v1/audio/spatial/render` |
| Qwen3-TTS VoiceDesign | 8301 | 本地音色设计 | `/v1/qwen/timbre` |
| MOSS VoiceGenerator | 8302 | 本地音色设计 | `/v1/moss/timbre` |
| MiMo TTS VoiceDesign | 8303 | 云端音色设计 | `/v1/mimo/timbre` |
| FireRedTTS3 Instruct | 8304 | 指令式音色设计 | `/v1/FireRedTTS3/timbre` |
| Stable Audio 3 Medium | 8311 | 文本生成音乐或声效 | `/v1/stableAudio/soundEffect` |
| MOSS-SoundEffect v2 | 8312 | 文本生成声效 | `/v1/moss/soundEffect` |
| ACE-Step 1.5 XL Turbo | 8313 | 有声小说 BGM、主题音乐和 underscore | `/v1/aceStep/bgm` |
| Qwen3-TTS Base | 8321 | 参考音频语音克隆 | `/v1/qwen/clone` |
| VoxCPM2 | 8322 | 语音克隆 | `/v1/voxcpm2/clone` |
| LongCat-AudioDiT-3.5B | 8323 | 参考音频语音克隆 | `/v1/longCat/clone` |
| dots.tts-soar | 8324 | 参考音频语音克隆 | `/v1/dotsTTS/clone` |
| FireRedTTS3 Base | 8325 | 参考音频语音克隆 | `/v1/FireRedTTS3/clone` |
| Step-Audio-EditX | 8331 | 语音编辑 | `/v1/stepAudioEditx/edit` |
| MOSS-Audio-4B-Thinking | 8341 | 音频转写、描述与问答 | `/v1/mossAudioThinking/understand` |
| MOSS-Audio-4B-Instruct | 8342 | 音频转写、描述与问答 | `/v1/mossAudioThinking/understand` |
| TIGER-DnR | 8351 | 电影混音的对白、音效、音乐三 Stem 分离 | `/v1/tigerDnr/separate` |
| Confucius4-TTS | 8361 | 参考音频零样本、多语言语音克隆 | `/v1/confucius4TTS/generate` |
| Qwen3-ASR-1.7B | 8371 | 本地多语言音频识别 | `/v1/qwen3/asr` |
| Seed-VC | 8381 | 原始语音到参考音色的转换，可选 F0 歌声转换 | `/v1/seedVc/voiceConversion` |
| SoulX-FlashHead-1_3B | 8391 | 图片与音频驱动人像视频，支持 Lite/Pro | `/v1/soulX/flashHead` |
| Ditto | 8392 | 图片与音频驱动人像视频，ONNX CUDA 后端 | `/v1/ditto/talkingHead` |

表中为业务入口，方法均为 `POST`，控制面的 `/v1/control` 为 `GET`。每个进程提供
`GET /v1/health`，FastAPI 的 `/docs` 和 `/openapi.json` 可查看该进程实际注册的字段与路由。
健康接口成功表示 HTTP 服务可访问，还须检查响应内的 `available` 和模型文件状态。

| 能力 | 成功响应 | 服务端保存行为 |
| --- | --- | --- |
| 克隆、音色设计、编辑、音效、BGM、音色转换 | `audio/wav` | 按用途保存 WAV |
| MOSS-Audio Thinking/Instruct、Qwen3-ASR | JSON 文本及上传摘要 | 请求结束后删除上传暂存和 worker 结果，不保留输入音频 |
| TIGER-DnR | `application/zip`，含 `dialog.wav`、`effects.wav`、`music.wav` | 原子保存三路 Stem 与 ZIP 批次 |
| SoulX-FlashHead、Ditto | 带声音的 `video/mp4` | 原子保存 MP4 |
| 控制面音频导出 | WAV 或 MP3 | 响应完成后删除临时成品 |

FireRedTTS3 的 `8304` 只注册音色设计，`8325` 只注册克隆及其上传/检查接口。
MOSS-Audio 的两个变体都从 `moss_audio_4b_thinking/` 启动，共用
`/v1/mossAudioThinking/understand` 路径，通过端口区分；没有独立的 Instruct uv 项目。

## 已移除与兼容边界

以下功能没有现行服务入口，不能按旧教程安装或调用：

| 已移除项 | 当前接入方式 |
| --- | --- |
| IndexTTS2、Ming/Ming-Omni、OmniVoice、旧 MOSS-TTS 克隆 | 使用服务表中的克隆模型；MOSS VoiceGenerator 仍提供音色设计 |
| Stable Audio 3 Small SFX | 声效使用 MOSS-SoundEffect v2 或 Stable Audio 3 Medium |
| VoxCPM2 音色设计及 `voice_design_worker.py` | 音色设计使用 Qwen、MOSS、MiMo 或 FireRedTTS3 Instruct；VoxCPM2 保留参考音频克隆和可控克隆 |
| 集中式 `api/`、旧 Conda API/worker 和运行时回退 | 从各服务 uv 项目的 `main.py` 启动 |
| 旧通用合成、音色设计和声效路由别名 | 使用服务总览中的最终路由，路径大小写需一致 |

8300 仅保留 MiMo 的同路径代理 `/v1/mimo/timbre`，不转发其他模型请求。
Qwen/VoxCPM2/LongCat/dots/FireRed 克隆接口的 `style_prompt` 已禁用；
VoxCPM2 可控克隆使用 `control_instruction`。
`/v1/audio/export` 仍兼容旧总线滤镜，但正式空间导出使用
`/v1/audio/spatial/render`，详见音频导出章节。

## 目录与运行数据

```text
main/                         8300 控制面，不包含模型推理
steam_audio_renderer/         独立 C++17 Steam Audio CPU renderer
qwen3_tts/                    Qwen3-TTS Base 的 HTTP 服务和 worker
voxcpm2/                      VoxCPM2 的 HTTP 服务和克隆 worker
LongCat_AudioDiT_3.5B_bf16/  LongCat-AudioDiT 服务和 worker
dots_tts_soar/                dots.tts-soar 服务和 worker
moss_soundEffect/             MOSS-SoundEffect v2 服务和 worker
stable_audio_3_medium/        Stable Audio 3 Medium 服务和 worker
ace_step_1_5/                  ACE-Step 1.5 BGM 服务和 worker
qwen3_voiceDesign/            Qwen VoiceDesign 服务和 worker
moss_voiceGenerator/          MOSS VoiceGenerator 服务和 worker
moss_audio_4b_thinking/       MOSS-Audio-4B-Thinking/Instruct 服务和 worker
Confucius4_TTS/               Confucius4-TTS 服务和一次性 worker
mimo_tts/                     MiMo 云端编排服务
Step_Audio_EditX/             Step-Audio-EditX 服务和 worker
firered_tts3/                 FireRedTTS3 Instruct/Base 服务和 worker
TIGER-DnR/                    TIGER-DnR 三 Stem 分离服务和 worker
Qwen3_ASR_1.7B/               Qwen3-ASR-1.7B 语音识别服务和 worker
seed-vc/                      Seed-VC 音色转换服务和一次性 worker
SoulX-FlashHead-1_3B/          音频驱动人像视频服务和一次性 worker
ditto/                        Ditto 人像视频服务和一次性 worker
tests/                        根目录无模型回归测试
qa/                           锁定的轻量测试与 Ruff 环境，不安装模型依赖
unitale_runtime/              共享流式上传、引用存储、GPU 队列与容量工具
scripts/                      质量门禁、renderer 构建与训练辅助工具
training/configs/             有声书纯文本润色 LoRA 的实验配置
docs/                         Python 工程规范
soundEffect/                  MOSS GPU 示例和提示词说明
storage/                      上传素材、生成音视频、sidecar、训练产物、缓存和 GPU 锁
```

根目录的 `流行*模型功能简介.md` 是调研资料，`special-audio-effect/` 是方案与验收记录，
`task*.md` 是任务资料；其中提到的模型或计划不表示已注册服务。实际服务清单以本 README
和 `start.sh` 为准。

默认运行数据目录为：

| 目录 | 内容 | 覆盖变量 |
| --- | --- | --- |
| `storage/timbre/` | Qwen、MOSS、MiMo、FireRedTTS3 生成的音色参考音频 | `TIMBRE_STORAGE_DIR` |
| `storage/soundEffect/` | MOSS 和 Stable Audio 生成的声效 | `SOUNDEFFECT_STORAGE_DIR`、`STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR` |
| `storage/bgm/` | ACE-Step 有声小说 BGM 和 OST | `BGM_STORAGE_DIR`、`ACESTEP_OUTPUT_DIR` |
| `storage/clone/` | 参考音频、克隆结果、Step 编辑结果和 Seed-VC 转换结果 | `CLONE_STORAGE_DIR`、各服务的 `*_OUTPUT_DIR` |
| `storage/separation/` | TIGER-DnR 每次分离的 dialog、effects、music Stem 与 ZIP | `TIGER_DNR_OUTPUT_DIR` |
| `storage/video/` | SoulX-FlashHead 和 Ditto 生成的 MP4 | `SOULX_FLASHHEAD_OUTPUT_DIR`、`DITTO_OUTPUT_DIR` |
| `storage/ditto/images/` | Ditto 上传的参考图片 | `DITTO_IMAGE_DIR` |
| `storage/flashhead/images/` | SoulX-FlashHead 上传的参考图片 | `SOULX_FLASHHEAD_IMAGE_DIR` |
| `storage/training/` | 文本润色实验的数据集、adapter 和预测结果 | 训练脚本的 `--output-dir` 与训练配置 |
| `storage/.cache/runtime/` | worker 临时文件、母带/Steam Audio 任务缓存、库缓存和共享 GPU 锁 | `RUNTIME_CACHE_DIR`、`SPATIAL_EXPORT_CACHE_DIR`、`STEAM_AUDIO_RENDER_CACHE_DIR`、`GPU_LOCK_FILE` |

如果上传音频的内容与 `storage/timbre/` 中已有的设计音色一致，Qwen3-TTS、VoxCPM2、
LongCat、dots.tts-soar、FireRedTTS3、Confucius4-TTS、Seed-VC，以及人像视频服务会在
`storage/timbre/.references/` 保存带 SHA-256 和相对路径的
小型 JSON 引用映射，不再把同一 WAV 复制到 `storage/clone/`；普通用户上传的参考音频仍保存到
`storage/clone/`。上传按块暂存并通过原子替换提交，默认上限为 64 MiB，可用
`UPLOAD_MAX_BYTES` 覆盖。这些目录是运行数据，不要提交到 Git。

## 安装与启动

只用于本地部署时，建议浅克隆主分支，只下载最新版本：

```bash
git clone --depth 1 --single-branch --branch main https://github.com/MuYi086/LLM-for-Local-Machine.git
cd LLM-for-Local-Machine
```

普通 `git clone` 会下载可达的历史对象，过去提交过的大型音频即使已从当前版本删除，仍会
随完整历史下载。浅克隆可避开这些旧对象；以后需要完整历史时再执行 `git fetch --unshallow`，
届时也会下载历史中的素材。删除当前文件或在本地运行 `git gc` 都不能清除远端可达历史。

各项目的 `pyproject.toml`、`uv.lock` 和 `.python-version` 必须保留，用于复现锁定环境；
`.venv/`、uv 缓存、安装包、构建产物、模型权重和音视频素材不提交，由本地安装或另行准备。
`special-audio-effect/` 中的音频文件属于手动研究素材，不是服务或自动测试依赖；文档保留
实测记录，复现示例时请自备素材并修改输入路径。

统一启动和 GPU 锁使用 Linux 的 `setsid`、进程组信号与 `fcntl.flock`，部署目标为 Linux。
运行要求：Python `3.12.13`、`uv`、FFmpeg/ffprobe、可用的 CUDA/NVIDIA 驱动（本地模型服务），
以及下方列出的模型权重和外部源码目录。Step-Audio-EditX 还需要系统 `sox`；Ditto 需要
GCC 编译上游 Cython 扩展；正式 Steam Audio renderer 需要 CMake 3.17+、C++17 编译器和本地 SDK。
不同模型锁定不同 Torch/CUDA 组合，必须使用各自环境。权重与第三方源码不放进本仓库。

先为需要的服务同步锁定依赖；部署全部服务时可以执行：

```bash
test -d /home/muyi086/tts-depency/MOSS-TTS &&
for project in qwen3_tts mimo_tts voxcpm2 LongCat_AudioDiT_3.5B_bf16 \
  dots_tts_soar moss_soundEffect stable_audio_3_medium ace_step_1_5 \
  qwen3_voiceDesign moss_voiceGenerator moss_audio_4b_thinking Confucius4_TTS \
  Qwen3_ASR_1.7B seed-vc SoulX-FlashHead-1_3B ditto Step_Audio_EditX \
  firered_tts3 TIGER-DnR; do
  uv sync --project "$project" --locked
done
```

`moss_voiceGenerator` 的 `moss-tts` 必须来自已准备的本地 editable 源码
`/home/muyi086/tts-depency/MOSS-TTS`；该固定路径是本仓库唯一明确保留的部署例外，
不能用 Git/PyPI 来源替换。上面的检查须成功后才同步该项目。
ACE-Step 的 Diffusers 和 dots 的 `dots-tts` 使用锁定的 Git revision，Step-Audio-EditX
使用指定的 vLLM wheel；首次安装和锁文件校验可能需要访问这些依赖来源。
`LOCAL_FILES_ONLY` 只控制模型加载，不代替依赖安装。启动前应完成 `uv sync --locked`；
`start.sh` 不执行依赖同步，也不下载权重或外部源码。

MiMo 是云端服务，必须配置密钥：

```bash
export MIMO_API_KEY='...'
```

默认情况下 `LOCAL_FILES_ONLY=1`，本地 worker 不会从 Hugging Face 下载权重。确认模型、
Tokenizer 和上游源码就绪后启动全部服务：

```bash
bash start.sh
```

`start.sh` 会启动 8300、8301、8302、8303、8304、8311、8312、8313、8321、8322、8323、8324、8325、
8331、8341、8342、8351、8361、8371、8381、8391 和 8392 共 22 个进程；8300 使用 `qwen3_tts` uv 项目中的轻量 HTTP 依赖，其余服务使用
各自的 uv 项目。启动命令统一使用 `uv run --no-sync`，不会在运行阶段联网解析依赖；
本地 GPU 服务通过 `GPU_LOCK_FILE` 串行访问 GPU。默认最多排队 900 秒，超过时返回
`503`；用正数 `GPU_LOCK_WAIT_TIMEOUT` 调整。共享锁实现仍兼容非正值无限等待，
但 Seed-VC 明确拒绝该配置，整套部署应保持正数。持锁期间用 `nvidia-smi` 采样，
`peak_vram_mib` 是所有可见 GPU 的已用显存总和峰值，并非某个 worker 的独立分配量。
任一子进程退出时脚本会终止其余进程组并清理 worker。
启动前会检查全部服务的监听地址；端口已被占用或配置相互冲突时，列出服务名和地址后退出。
更新代码后重新启动，应先在旧 `start.sh` 终端按 `Ctrl+C`，等待旧进程退出，再执行启动命令。
可用 `ss -lntp` 查看占用端口的进程。

控制面诊断与 HTTP 健康检查：

```bash
curl -fsS http://127.0.0.1:8300/v1/control
for port in 8300 8301 8302 8303 8304 8311 8312 8313 8321 8322 8323 8324 \
  8325 8331 8341 8342 8351 8361 8371 8381 8391 8392; do
  curl -fsS "http://127.0.0.1:${port}/v1/health" >/dev/null && echo "${port}: ok"
done
```

单独调试服务时，从仓库根目录执行，例如：

```bash
HOST=127.0.0.1 PORT=8321 \
  uv run --no-sync --project qwen3_tts python qwen3_tts/main.py
```

统一脚本没有按模型启停的开关；只部署部分模型时，准备并直接启动相应目录。
MOSS-Audio 单独启动要显式设置 `MOSS_AUDIO_4B_VARIANT=thinking` 或 `instruct`；
FireRedTTS3 设置 `FIRERED_TTS3_MODE=timbre` 或 `clone`，例如：

```bash
MOSS_AUDIO_4B_VARIANT=instruct MOSS_AUDIO_4B_INSTRUCT_PORT=8342 \
  uv run --no-sync --project moss_audio_4b_thinking python moss_audio_4b_thinking/main.py
FIRERED_TTS3_MODE=clone FIRERED_TTS3_PORT=8325 \
  uv run --no-sync --project firered_tts3 python firered_tts3/main.py
```

## SoulX-FlashHead 人像视频

独立服务以 `8391` 提供 `POST /v1/soulX/flashHead`，接收已上传的图片和驱动音频逻辑路径，
默认使用 Lite，也支持 Pro。先调用该服务的 `/v1/upload_image` 和 `/v1/upload_audio`，
再发送 `{"image_path":"portrait.png","audio_path":"speech.wav","model_type":"lite","seed":42}`。
成功返回带声音的 MP4，成品保存在 `storage/video/`。

默认 `model_type=lite`、`seed=42`、`use_face_crop=false`；驱动音频默认上限 300 秒，
worker 超时 1800 秒。图片支持 PNG/JPG/JPEG/WebP，上传使用 64 MiB 共享上限。
该接口生成完整文件，使用单 GPU，不提供实时视频流或 Pro 多 GPU 推理。

SoulX 与 Ditto 均提供 `GET /v1/check/image?file_name=portrait.png` 和
`GET /v1/check/audio?file_name=speech.wav` 检查已上传输入；`file_name` 使用原 `full_path`。
图片上传表单字段为 `image`、`full_path`，成功返回 `filename`、`sha256`、`size_bytes`。

需要额外准备官方 `Soul-AILab/SoulX-FlashHead` 源码和 `facebook/wav2vec2-base-960h` 权重，
再执行 `uv sync --project SoulX-FlashHead-1_3B --locked`。
完整安装命令、上传/生成示例、字段、状态码和配置见
[SoulX-FlashHead 服务文档](SoulX-FlashHead-1_3B/README.md)。

## Ditto 人像视频

独立 `ditto/` 服务以 `8392` 提供 `POST /v1/ditto/talkingHead`。先调用该服务的
`/v1/upload_image` 和 `/v1/upload_audio`，再发送
`{"image_path":"portrait.png","audio_path":"speech.wav","seed":42,"sampling_timesteps":50,"max_size":1920}`。
返回带声音的 MP4，成品默认保存在 `storage/video/`。

`sampling_timesteps` 为 1–100，`max_size` 为 256–4096；音频默认最长 300 秒，worker
超时 1800 秒。输入图片与 SoulX 一样支持 PNG/JPG/JPEG/WebP。

使用 `$HF_MIRROR_DIR/thewintersun/ditto-talkinghead` 的本地 ONNX 权重和官方外部源码，
沿用 Confucius4-TTS 的一次性 worker、共享 GPU 锁、超时与进程组清理方式。
同步 `uv sync --project ditto --locked` 后即可由 `start.sh` 启动。
Python 3.12 环境采用 ONNX Runtime CUDA 后端；配置映射和三维采样适配只写入临时目录。
完整安装、上传/生成示例、字段、状态码和环境变量见 [Ditto 服务文档](ditto/README.md)。
使用本地图片和音频生成视频的手动演示见 [tests/testDitto](tests/testDitto/README.md)。

## 模型路径与主要配置

`start.sh` 默认使用 `HF_MIRROR_DIR`（默认为 `$HOME/hf-mirror`）和 `$HOME/tts-depency`；
模型和运行数据路径可在启动前用环境变量覆盖；MOSS VoiceGenerator 的 editable 依赖路径
仍遵循安装章节中的固定路径例外。

| 服务 | 默认权重 | 权重覆盖变量与额外配置 |
| --- | --- | --- |
| Qwen3-TTS Base | `$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `QWEN3_TTS_MODEL_DIR` |
| Qwen VoiceDesign | `$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `QWEN_VOICEDESIGN_MODEL_DIR` |
| MOSS VoiceGenerator | `$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-VoiceGenerator` | `MOSS_VOICEGENERATOR_MODEL_DIR`、`MOSS_AUDIO_TOKENIZER_PATH` |
| MOSS-Audio-4B-Thinking | `$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-4B-Thinking` | `MOSS_AUDIO_4B_THINKING_MODEL_DIR`、`MOSS_AUDIO_4B_THINKING_DEPENDENCY_PATH` |
| MOSS-Audio-4B-Instruct | `$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-4B-Instruct` | `MOSS_AUDIO_4B_INSTRUCT_MODEL_DIR`、`MOSS_AUDIO_4B_INSTRUCT_DEPENDENCY_PATH` |
| Confucius4-TTS | `$HF_MIRROR_DIR/netease-youdao/Confucius4-TTS` | `CONFUCIUS4_TTS_MODEL_DIR`、`CONFUCIUS4_TTS_CODE_PATH`、`CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR`、`CONFUCIUS4_TTS_VOCODER_MODEL_DIR`、`CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT` |
| Qwen3-ASR-1.7B | `$HF_MIRROR_DIR/Qwen/Qwen3-ASR-1.7B` | `QWEN3_ASR_MODEL_DIR` |
| Seed-VC | `$HF_MIRROR_DIR/Plachta/Seed-VC` | `SEED_VC_MODEL_DIR`、`SEED_VC_CODE_PATH`、`SEED_VC_WHISPER_MODEL_DIR`、`SEED_VC_VOCODER_MODEL_DIR`、`SEED_VC_STYLE_ENCODER_CHECKPOINT`；F0 模式另需 `SEED_VC_F0_VOCODER_MODEL_DIR`、`SEED_VC_RMVPE_CHECKPOINT` |
| SoulX-FlashHead | `$HF_MIRROR_DIR/Soul-AILab/SoulX-FlashHead-1_3B` | `SOULX_FLASHHEAD_MODEL_DIR`、`SOULX_FLASHHEAD_CODE_PATH`、`SOULX_FLASHHEAD_WAV2VEC_DIR`（默认 `$HF_MIRROR_DIR/facebook/wav2vec2-base-960h`） |
| Ditto | `$HF_MIRROR_DIR/thewintersun/ditto-talkinghead` | `DITTO_MODEL_DIR`、`DITTO_CODE_PATH`、`DITTO_DATA_ROOT`（`$DITTO_MODEL_DIR/ditto_onnx`）、`DITTO_CONFIG_PATH`（`$DITTO_MODEL_DIR/ditto_cfg/v0.4_hubert_cfg_trt.pkl`） |
| MOSS-SoundEffect | `$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-SoundEffect-v2.0` | `MOSS_SOUNDEFFECT_CODE_PATH`、`MOSS_SOUNDEFFECT_MODEL_DIR` |
| Stable Audio 3 Medium | `$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium` | `STABLE_AUDIO_3_REPO_PATH`、`STABLE_AUDIO_3_MEDIUM_MODEL_DIR` |
| ACE-Step 1.5 XL Turbo | `$HF_MIRROR_DIR/ACE-Step/acestep-v15-xl-turbo-diffusers` | `ACESTEP_MODEL_DIR`、`ACESTEP_OFFLOAD`、`ACESTEP_VAE_TILING` |
| VoxCPM2 | `$HF_MIRROR_DIR/openbmb/VoxCPM2` | `VOXCPM2_MODEL_DIR`、仓库内 `voxcpm2/voxcpm2_helpers.py` |
| LongCat-AudioDiT | `$HF_MIRROR_DIR/drbaph/LongCat-AudioDiT-3.5B-bf16` | `LONGCAT_AUDIODIT_MODEL_DIR`、`LONGCAT_AUDIODIT_REPO_PATH`、`LONGCAT_AUDIODIT_TOKENIZER_PATH`（`$HF_MIRROR_DIR/google/umt5-base`） |
| dots.tts-soar | `$HF_MIRROR_DIR/rednote-hilab/dots.tts-soar` | `DOTS_TTS_SOAR_MODEL_DIR` |
| Step-Audio-EditX | `$HF_MIRROR_DIR/stepfun-ai/Step-Audio-EditX` | `STEP_AUDIO_EDITX_MODEL_DIR`、`STEP_AUDIO_TOKENIZER_PATH`（`$HF_MIRROR_DIR/stepfun-ai/Step-Audio-Tokenizer`）、`STEP_AUDIO_EDITX_CODE_PATH` |
| FireRedTTS3 Base/Instruct | `$HF_MIRROR_DIR/drbaph/FireRedTTS3-bf16` | `FIRERED_TTS3_MODEL_DIR`、`FIRERED_TTS3_CODE_PATH` |
| TIGER-DnR | `$HF_MIRROR_DIR/JusperLee/TIGER-DnR` | `TIGER_DNR_MODEL_DIR`、`TIGER_DNR_SOURCE_DIR` |

仓库外推理源码的默认位置如下，须在请求前准备；`start.sh` 不负责拉取源码：

| 服务 | 外部源码默认路径 | 覆盖变量 |
| --- | --- | --- |
| MOSS-SoundEffect | `$HOME/tts-depency/MOSS-TTS` | `MOSS_SOUNDEFFECT_CODE_PATH` |
| MOSS-Audio 两个变体 | `$HOME/tts-depency/MOSS-Audio` | `MOSS_AUDIO_4B_THINKING_DEPENDENCY_PATH`、`MOSS_AUDIO_4B_INSTRUCT_DEPENDENCY_PATH` |
| Stable Audio 3 | `$HOME/tts-depency/stable-audio-3` | `STABLE_AUDIO_3_REPO_PATH` |
| LongCat | `$HOME/tts-depency/LongCat-AudioDiT` | `LONGCAT_AUDIODIT_REPO_PATH` |
| Step-Audio-EditX | `$HOME/tts-depency/Step-Audio-EditX` | `STEP_AUDIO_EDITX_CODE_PATH` |
| FireRedTTS3 | `$HOME/tts-depency/FireRedTTS3` | `FIRERED_TTS3_CODE_PATH` |
| Confucius4-TTS | `$HOME/tts-depency/Confucius4-TTS` | `CONFUCIUS4_TTS_CODE_PATH` |
| Seed-VC | `$HOME/tts-depency/seed-vc` | `SEED_VC_CODE_PATH` |
| SoulX-FlashHead | `$HOME/tts-depency/SoulX-FlashHead` | `SOULX_FLASHHEAD_CODE_PATH` |
| Ditto | `$HOME/tts-depency/ditto-talkinghead` | `DITTO_CODE_PATH` |
| TIGER-DnR | `$HOME/.local/share/tiger-dnr/TIGER` | `TIGER_DNR_SOURCE_DIR` |

MOSS VoiceGenerator 的 editable 源码位置见安装章节，音频 tokenizer 默认为
`$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-Tokenizer`（v1）。

通用配置包括 `HOST`、`PORT`、`STORAGE_DIR`、`PROMPTS_DIR`、`RUNTIME_CACHE_DIR`、
`GPU_LOCK_FILE`、`LOCAL_FILES_ONLY` 和 `CUDA_RELEASE_DELAY`。48 kHz 导出可用
`SPATIAL_EXPORT_CACHE_DIR`、`SPATIAL_EXPORT_MAX_BYTES`、`SPATIAL_EXPORT_TIMEOUT` 和
`SPATIAL_EXPORT_FFMPEG_BIN` 覆盖旧总线母带的缓存目录、上传上限、处理超时和 FFmpeg 命令。
正式对象级导出使用 `STEAM_AUDIO_RENDERER_BIN`、`STEAM_AUDIO_SDK_DIR`、
`STEAM_AUDIO_HRTF_PATH`、`STEAM_AUDIO_RENDER_CACHE_DIR`、`STEAM_AUDIO_RENDER_TIMEOUT`、
`STEAM_AUDIO_RENDER_MAX_ASSETS`、`STEAM_AUDIO_RENDER_MAX_MANIFEST_BYTES`、
`STEAM_AUDIO_RENDER_MAX_BYTES` 和
`STEAM_AUDIO_RENDER_THREADS`。服务专用配置使用对应
前缀，例如 `QWEN3_TTS_*`、`QWEN3_ASR_*`、`VOXCPM2_*`、`LONGCAT_AUDIODIT_*`、`DOTS_TTS_SOAR_*`、
`MOSS_SOUNDEFFECT_*`、`STABLE_AUDIO_3_MEDIUM_*`、`ACESTEP_*`、`STEP_AUDIO_EDITX_*`、
`QWEN_VOICEDESIGN_*`、`MOSS_VOICEGENERATOR_*`、`MOSS_AUDIO_4B_THINKING_*`、
`MOSS_AUDIO_4B_INSTRUCT_*` 和
`CONFUCIUS4_TTS_*`、
`FIRERED_TTS3_*`、`TIGER_DNR_*`、`SEED_VC_*`、`SOULX_FLASHHEAD_*`、`DITTO_*`。每个服务的 `/v1/health` 会报告
生效的路径、运行时和可用性。
FireRedTTS3 的官方源码默认位于 `$HOME/tts-depency/FireRedTTS3`，通过
`FIRERED_TTS3_CODE_PATH` 覆盖；8304 以 `timbre` 模式加载 Instruct，8325 以 `clone` 模式加载
Base，两者不会同时在 worker 中常驻显存。

重要共享配置如下；运行数据路径默认位于 `STORAGE_DIR`（仓库内 `storage/`）下：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HF_MIRROR_DIR` | `$HOME/hf-mirror` | 本地权重根目录 |
| `HOST`、`PORT` | `0.0.0.0`、`8300` | 统一启动时的控制面地址；其他服务有独立端口 |
| `PROMPTS_DIR` | `$CLONE_STORAGE_DIR` | 普通参考音频和 sidecar；多服务须共享该目录才能复用逻辑标识 |
| `UPLOAD_MAX_BYTES` | `67108864` | 参考音频、理解/转写/分离输入与人像图片的单文件上限 |
| `GPU_LOCK_FILE` | `$RUNTIME_CACHE_DIR/gpu-runtime.lock` | 本地模型共用排他锁 |
| `GPU_LOCK_WAIT_TIMEOUT` | `900` 秒 | GPU 排队时限；整套启动须为正数 |
| `GPU_METRICS_SAMPLE_INTERVAL` | `0.5` 秒 | 显存采样间隔，实际不低于 0.1 秒 |
| `CUDA_RELEASE_DELAY` | `2.0` 秒 | worker 退出后持锁等待显存回收 |
| `LOCAL_FILES_ONLY` | `1` | 本地模型离线加载；不控制 MiMo 云端请求 |
| `SPATIAL_EXPORT_MAX_BYTES` | `536870912` | 预混总线上传上限，512 MiB |
| `SPATIAL_EXPORT_TIMEOUT` | `600` 秒 | FFmpeg 子进程超时 |
| `STEAM_AUDIO_RENDER_TIMEOUT` | `900` 秒 | renderer 子进程超时 |
| `STEAM_AUDIO_RENDER_MAX_BYTES` | `2147483648` | 正式空间导出上传总量上限，2 GiB |
| `STEAM_AUDIO_RENDER_MAX_ASSETS` | `500` | 上传资产数量上限；Manifest 同时最多 500 个对象 |
| `STEAM_AUDIO_RENDER_MAX_MANIFEST_BYTES` | `8388608` | Manifest 大小上限，8 MiB |
| `STEAM_AUDIO_RENDER_THREADS` | `4` | renderer 线程数 |
| `STEAM_AUDIO_PROGRESS_RETENTION_SECONDS` | `3600` 秒 | 空间任务终态在进程内保留时间 |
| `STORAGE_RETENTION_HOURS`、`STORAGE_RETENTION_MAX_BYTES` | `0`、`0` | 默认不启用生成 WAV 清理 |

`start.sh` 的监听配置通常使用 `<服务前缀>_HOST` / `<服务前缀>_PORT`，例外是
MOSS-SoundEffect 使用 `SOUNDEFFECT_HOST` / `SOUNDEFFECT_PORT`，Qwen VoiceDesign 使用
`QWEN_VOICEDESIGN_HOST` / `QWEN_VOICEDESIGN_PORT`，FireRedTTS3 使用
`FIRERED_TTS3_TIMBRE_*` 与 `FIRERED_TTS3_CLONE_*` 区分两个地址。
`*_PROJECT_DIR` 用于覆盖启动脚本中的 uv 项目路径；直接启动服务时使用该服务实际读取的
环境变量。省略 JSON 推理字段时，生效值由模块默认值和显式环境配置决定；
`start.sh` 当前也导出部分音效、ACE-Step、Step-Audio 和 MiMo 默认参数，修改模块后需同时核对脚本。

MiMo 默认 `MIMO_MODEL=mimo-v2.5-tts-voicedesign`、
`MIMO_BASE_URL=https://api.xiaomimimo.com/v1`、`MIMO_TIMEOUT=300` 秒，
`MIMO_AUTH_HEADER=api-key`；密钥仅从 `MIMO_API_KEY` 环境读取。
8300 代理可用 `MIMO_TTS_PROXY_URL` 和 `MIMO_TTS_PROXY_TIMEOUT`（默认 310 秒）配置。
客户端超时应覆盖排队、worker 执行和显存释放时间，代理超时也需覆盖 MiMo 分段和重试。

TIGER-DnR 的官方推理代码不包含在本仓库。默认使用已准备的
`$HOME/.local/share/tiger-dnr/TIGER`；也可以通过 `TIGER_DNR_SOURCE_DIR` 指向作者的
`JusperLee/TIGER` 克隆。worker 只使用其中的 `look2hear` DnR 模型代码，并在
`LOCAL_FILES_ONLY=1` 下从本地 `config.json` 与 `model.safetensors` 加载权重。

## Confucius4-TTS 零样本克隆

Confucius4-TTS 的官方推理代码不包含在本仓库。默认使用
`$HOME/tts-depency/Confucius4-TTS`，模型目录默认读取
`$HF_MIRROR_DIR/netease-youdao/Confucius4-TTS`。除 Confucius4-TTS 自身权重外，还需准备
Wav2Vec2-BERT、BigVGAN 和 CAMPPlus 权重；当前默认目录分别是
`$HF_MIRROR_DIR/netease-youdao/facebook/w2v-bert-2.0`、
`$HF_MIRROR_DIR/netease-youdao/nv-community/bigvgan_v2_22khz_80band_256x` 和
`$HF_MIRROR_DIR/netease-youdao/funasr/campplus/campplus_cn_common.bin`，也可通过
`CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR`、`CONFUCIUS4_TTS_VOCODER_MODEL_DIR` 和
`CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT` 覆盖；`LOCAL_FILES_ONLY=1` 时 worker 不会隐式下载。
Confucius4-TTS 项目固定使用 TorchAudio 2.11，需先按锁文件同步其中声明的 `torchcodec` 依赖。

Confucius4-TTS 使用与其他克隆服务相同的上传和检查流程：先调用 `POST /v1/upload_audio`，
再以保存后的 `audio_path` 调用 `POST http://127.0.0.1:8361/v1/confucius4TTS/generate`。
请求 JSON 必须包含 `text` 和 `audio_path`，`lang` 默认 `zh`。成功返回 `audio/wav`，并将生成结果保存到
`storage/clone/`。官方默认生成参数可通过 `CONFUCIUS4_TTS_*` 环境变量覆盖。

```bash
curl -fsS http://127.0.0.1:8361/v1/upload_audio \
  -F 'audio=@reference.wav;type=audio/wav' -F 'full_path=reference.wav'
curl -fsS http://127.0.0.1:8361/v1/confucius4TTS/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，欢迎使用。","lang":"zh","audio_path":"reference.wav"}' \
  -o confucius4-clone.wav
```

可选字段包括 `raw`、`temperature`、`top_p`、`top_k`、`num_beams`、`repetition_penalty`、
`max_length`、`n_timesteps`、`inference_cfg_rate`、`max_text_tokens_per_segment`、
`cross_fade_duration`、`edge_fade_duration`、`edge_pad_duration` 和 `verbose`。
参考音频的可选 `prompt_text` 会保存为 sidecar，本服务推理不读取该转写。

对单个极短非语言感叹音，可选传入 `vocalization_duration_seconds`（有限正数，最多 2 秒）。
worker 使用官方 S2A 的 `target_feat_len` 指定 Mel 帧数，完整保留 T2S 语义序列；不截断
最终 WAV，不用于压缩普通词汇台词。多合成块请求会明确失败。省略该字段时保持官方默认
生成长度，仍须对输出执行语义和时长验收。`GET /v1/health` 的
`runtime.native_vocalization_duration=true` 表示已加载支持此字段的 HTTP 版本；更新后需要
重新运行 `bash start.sh`，仅修改 worker 文件不能更新已启动的接口校验。

## Qwen3-ASR 语音识别

Qwen3-ASR 使用官方 `qwen-asr` Transformers 后端和本地模型目录。项目锁定了 `qwen-asr`
及其 Transformers 依赖；`LOCAL_FILES_ONLY=1` 时不会隐式下载模型。服务通过一次性 worker
执行识别，并使用与其他本地 GPU 服务共享的排他锁。

调用 `POST http://127.0.0.1:8371/v1/qwen3/asr` 时直接以 multipart/form-data 发送 `audio`，
无需先上传、也不使用 `full_path`。可选 `language`（最多 64 字符）强制指定识别语言，
`context`（最多 2000 字符）提供上下文，`max_new_tokens`（1–4096，默认 256）设置生成上限。
省略 `language` 时自动识别。成功 JSON 包含 `code`、`text`、`language`、`elapsed_seconds`
和 `audio`（`sha256`、`size_bytes`、`suffix`）；输入只暂存，请求完成后删除：

```bash
curl -fsS http://127.0.0.1:8371/v1/qwen3/asr \
  -F 'audio=@speech.wav;type=audio/wav' \
  -F 'language=Chinese' \
  -F 'context=人名：张三' \
  -F 'max_new_tokens=512'
```

## Seed-VC 音色转换

`seed-vc` 沿用 Confucius4-TTS 的独立 uv 项目、轻量 HTTP 进程和一次性 worker 结构，
默认监听 `8381`。worker 复用 [官方 `inference.py`](https://github.com/Plachtaa/seed-vc/blob/main/inference.py)
的离线转换流程，保留长音频分块和交叉淡化；参考音色按官方流程最多使用前 25 秒。
它接收原始音频与目标音色参考音频，不需要 TTS 文本。

官方源码在仓库外准备，通过 `SEED_VC_CODE_PATH` 覆盖默认目录：

```bash
git clone https://github.com/Plachtaa/seed-vc.git "$HOME/tts-depency/seed-vc"
uv sync --project seed-vc --locked
```

默认 22.05 kHz 语音转换需要以下本地文件；模型目录中的 `v2/ar_base.pth` 和
`v2/cfm_small.pth` 属于另一套上游 V2 流程，本接口使用下表的离线 DiT 模型。

| 用途 | 默认路径 | 覆盖变量 |
| --- | --- | --- |
| DiT 权重 | `$HF_MIRROR_DIR/Plachta/Seed-VC/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` | `SEED_VC_CHECKPOINT_PATH`、`SEED_VC_MODEL_DIR` |
| 配置 | `$HF_MIRROR_DIR/Plachta/Seed-VC/config_dit_mel_seed_uvit_whisper_small_wavenet.yml` | `SEED_VC_CONFIG_PATH` |
| Whisper-small | `$HF_MIRROR_DIR/openai/whisper-small`，含 `config.json`、`preprocessor_config.json` 与 `model.safetensors` 或 `pytorch_model.bin` | `SEED_VC_WHISPER_MODEL_DIR` |
| BigVGAN | `$HF_MIRROR_DIR/netease-youdao/nv-community/bigvgan_v2_22khz_80band_256x`，含 `config.json` 与 `bigvgan_generator.pt` | `SEED_VC_VOCODER_MODEL_DIR` |
| CAMPPlus | `$HF_MIRROR_DIR/netease-youdao/funasr/campplus/campplus_cn_common.bin` | `SEED_VC_STYLE_ENCODER_CHECKPOINT` |

先分别上传两路音频，`full_path` 是 WebUI 逻辑标识，不是任意服务端文件路径：

```bash
curl -fsS http://127.0.0.1:8381/v1/upload_audio \
  -F 'audio=@source.wav;type=audio/wav' -F 'full_path=source.wav'
curl -fsS http://127.0.0.1:8381/v1/upload_audio \
  -F 'audio=@reference.wav;type=audio/wav' -F 'full_path=reference.wav'
curl -fsS http://127.0.0.1:8381/v1/check/audio?file_name=reference.wav
curl -fsS http://127.0.0.1:8381/v1/seedVc/voiceConversion \
  -H 'Content-Type: application/json' \
  -d '{"source_audio_path":"source.wav","reference_audio_path":"reference.wav","diffusion_steps":30,"length_adjust":1.0,"inference_cfg_rate":0.7}' \
  -o converted.wav
```

成功响应为 `audio/wav`，并原子保存到 `storage/clone/`，可通过 `SEED_VC_OUTPUT_DIR`
覆盖。上传复用共享流式暂存、SHA-256、64 MiB 上限和设计音色引用机制。
可选 `prompt_text` 仅用于上传 sidecar，不参与转换。

| JSON 字段 | 默认值 | 范围或含义 |
| --- | --- | --- |
| `source_audio_path` | 必填 | 已上传的原始音频 `full_path` |
| `reference_audio_path` | 必填 | 已上传的参考音色 `full_path`，可引用设计音色 |
| `diffusion_steps` | `30` | 1–200 |
| `length_adjust` | `1.0` | 0.5–2，输出时长比例 |
| `inference_cfg_rate` | `0.7` | 0–1 |
| `f0_condition` | `false` | 切换到 44.1 kHz F0 模型 |
| `auto_f0_adjust` | `false` | 按参考音频自动调整音高，需启用 F0 |
| `semi_tone_shift` | `0` | -24–24 半音，非零时需启用 F0 |
| `fp16` | `true` | CUDA 上 DiT 使用 FP16 autocast；非 CUDA 设备关闭此选项 |

F0 模式另需模型目录内的
`DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth` 与
`config_dit_mel_seed_uvit_whisper_base_f0_44k.yml`，分别通过 `SEED_VC_F0_CHECKPOINT_PATH`
和 `SEED_VC_F0_CONFIG_PATH` 覆盖；另准备
`$HF_MIRROR_DIR/nvidia/bigvgan_v2_44khz_128band_512x` 与
`$HF_MIRROR_DIR/lj1995/VoiceConversionWebUI/rmvpe.pt`，分别通过
`SEED_VC_F0_VOCODER_MODEL_DIR` 和 `SEED_VC_RMVPE_CHECKPOINT` 覆盖。F0 配置同样使用 Whisper-small。

所有推理默认值集中在 `seed-vc/main.py` 顶部，可通过对应的 `SEED_VC_*` 环境变量覆盖；
`start.sh` 仅配置路径、端口和运行参数。`SEED_VC_HOST`、`SEED_VC_PORT`、
`SEED_VC_PROJECT_DIR`、`SEED_VC_DEVICE`（默认 `cuda:0`）、`SEED_VC_REQUEST_TIMEOUT`
（默认 900 秒）和 `SEED_VC_WORKER_TMP_DIR` 均可覆盖。
即使关闭 `LOCAL_FILES_ONLY`，本接口仍要求部署时准备所有本地模型，不在请求中下载。
`GET /v1/health` 分别报告普通/F0 模式缺失的文件、存储容量和最近一次 GPU 指标。
输入不存在返回 `404`，模型文件缺失或 GPU 排队超时返回 `503`，非法参数返回 `422`。
Seed-VC 要求 `GPU_LOCK_WAIT_TIMEOUT` 为正数，worker 成功、失败和超时均终止进程组、
清理临时 JSON/WAV，并在释放共享 GPU 锁前等待 `CUDA_RELEASE_DELAY`。

## 48 kHz 母带与 Steam Audio 正式导出

控制面保留两条职责不同的 CPU 路径。`POST /v1/audio/export` 是兼容接口，只接收一个已混合
双声道总线；WebUI 的 `standard` 档使用它完成 48 kHz 重采样、双遍 EBU R128 loudnorm 和编码，
不加入 Haas、aecho 或其他空间处理。

旧接口的实际默认 `profile` 仍为 `balanced`，并保留 `balanced`/`immersive` 的旧 Haas/aecho
滤镜，不能视为 Steam Audio 对象级渲染。标准母带调用必须显式传 `profile=standard`；
正式空间导出走下面的 `/v1/audio/spatial/render`，其 pre-master 后只做母带与编码。

```bash
curl -X POST http://127.0.0.1:8300/v1/audio/export \
  -F 'audio=@timeline-mix.wav;type=audio/wav' \
  -F 'profile=standard' \
  -F 'output_format=wav' \
  -o unitale-standard.wav
```

`POST /v1/audio/spatial/render` 是 `balanced`/`immersive` 的正式路径。WebUI 上传 Render
Manifest v1 和尚未预混的对白、SFX、环境声、BGM Blob；BGM 使用 `preserve_stereo`，其他对象
按有限语义 DSL 映射为位置、距离、移动和 HRTF spatial blend。renderer 对每个对象执行 Steam
Audio Direct Effect（距离衰减、空气吸收）和 Binaural Effect，再在磁盘上流式混合成 48 kHz
双声道 pre-master。随后只执行 loudnorm 和最终编码，绝不叠加旧 Haas/aecho 链。该任务不获取
GPU 锁，失败、超时和响应完成后都会清理暂存文件。

先将下列最小示例保存为 `render-manifest.json`；示例假设两份上传音频均至少有 1 秒：

```json
{
  "version": "1.0",
  "sample_rate": 48000,
  "timeline_duration_ms": 2000,
  "scene": {"room": "dry_studio", "acoustic_quality": "balanced"},
  "sources": [
    {
      "id": "narrator-1",
      "kind": "narrator",
      "asset_id": "narrator",
      "asset_filename": "asset_narrator.wav",
      "start_ms": 0,
      "duration_ms": 1000,
      "spatial": {"mode": "dry_center"}
    },
    {
      "id": "door-1",
      "kind": "sfx",
      "asset_id": "door",
      "asset_filename": "asset_door.wav",
      "start_ms": 1000,
      "duration_ms": 1000,
      "spatial": {"mode": "point", "location": "front_left", "distance": "near"}
    }
  ]
}
```

```bash
curl -X POST http://127.0.0.1:8300/v1/audio/spatial/render \
  -F 'manifest=<render-manifest.json' \
  -F 'assets=@narrator.wav;filename=asset_narrator.wav' \
  -F 'assets=@door.wav;filename=asset_door.wav' \
  -F 'profile=balanced' \
  -F 'output_format=wav' \
  -F 'job_id=job-example-12345678' \
  -o unitale-steam-balanced.wav
```

WebUI 会为每次正式导出生成唯一 `job_id`，并在 POST 执行期间轮询
`GET /v1/audio/spatial/render/progress/{job_id}`。状态包含 `state`、`stage`、`progress`（0–100）
和中文 `message`；成功或失败终态默认保留 1 小时，可由
`STEAM_AUDIO_PROGRESS_RETENTION_SECONDS` 调整。控制面终端使用同一 job ID 输出资产暂存、逐对象
标准化、Steam Audio 逐对象渲染、母带和完成/失败阶段；标准化完成首个对象后，消息还会按实际
平均耗时给出预计剩余时间。任务进度仅保存在当前控制面进程内，服务重启后不保留。
成功响应还会返回
`X-Spatial-Job-ID`；未传 `job_id` 的旧客户端仍可同步调用，但只能从响应头和终端获取服务端生成的 ID。

每个 `asset_filename` 必须是安全 basename，并与重复 `assets` 字段的上传文件名一一对应；缺失、
重复或未引用资产都会在暂存前拒绝。Manifest 固定 `version=1.0`、`sample_rate=48000`，最多
500 个对象、最长 2 小时。首版支持点声源、居中干声、BGM 立体声保留、距离/空气吸收和简单
移动；`diffuse`、非 `none` 遮挡、几何反射尚未实现，接口会显式拒绝而不是静默降级。输出 WAV
为 24-bit PCM，MP3 为 192 kbps。

### 构建正式 renderer

Steam Audio SDK 不提交到仓库。下载并解压 Valve 官方 SDK 后，仅在构建时提供路径；启动脚本
不会下载或编译第三方代码。SDK、动态库和 HRTF 的使用与分发须遵守 SDK 随附许可证，本仓库的
Apache-2.0 许可证不替代第三方许可：

```bash
STEAM_AUDIO_SDK_DIR=/opt/steam-audio \
  bash scripts/build_steam_audio_renderer.sh
```

Windows PowerShell 先设置 `$env:STEAM_AUDIO_SDK_DIR = "C:\\steam-audio"`，再运行
`.\scripts\build_steam_audio_renderer.ps1`。默认可执行文件为
`steam_audio_renderer/build/steam-audio-render`（Windows 为
`steam_audio_renderer/build/Release/steam-audio-render.exe`）。可选
`STEAM_AUDIO_HRTF_PATH` 指向 SOFA 文件；不设置时使用 SDK 内置 HRTF。`start.sh` 若发现可执行
文件缺失会告警但仍启动既有服务，正式路由返回 `503`；`GET /v1/control` 的
`spatial_renderer` 会报告 executable、SDK header/library、HRTF 和 GPU-lock 状态。

Stable Audio 3 默认允许上游的 flex-attention/SDPA 回退；只有需要严格检查
FlashAttention 时才设置 `STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN=1`。VoxCPM2、
LongCat、dots.tts-soar 和 Step-Audio-EditX 的默认项目路径不要求安装 `flash_attn`。
不要把本机某个 FlashAttention 源码 checkout 当作已安装的 Python 扩展。

## 参考音频克隆

Qwen3-TTS、VoxCPM2、LongCat、dots.tts-soar 和 FireRedTTS3 使用相同的三步 WebUI 流程：

1. `POST /v1/upload_audio`，表单字段为 `audio`、`full_path`，可选 `prompt_text`。
2. `GET /v1/check/audio?file_name=...` 检查服务自己的存储状态。
3. 调用当前模型的克隆路由，请求中的 `audio_path` 使用上传时完全相同的 `full_path` 标识。

`full_path` 是 1–1024 字符、带受支持音频扩展名的客户端逻辑标识，可以包含目录前缀，
服务端通过摘要映射解析，不能用它直接读取任意本地文件。共享音频策略支持
WAV/MP3/OGG/FLAC/M4A/AAC/WebM，检查扩展名与 Content-Type；普通上传默认按 1 MiB
分块暂存，总量最多 64 MiB。成功 JSON 返回 `filename`（逻辑标识）、`has_prompt_text`、
`sha256`、`size_bytes`、`storage`（`clone` 或 `timbre_reference`）。
检查接口应读取响应的 `exists`/`code`，不能只用 HTTP 状态判断文件存在。

Confucius4-TTS、Seed-VC、SoulX、Ditto 复用相同音频上传契约。
8300 和 Step-Audio-EditX 的上传接口只接收 `audio` 与 `full_path`，不保存参考转写；
编辑文本在 Step 的 JSON 请求中提供。MOSS-Audio、Qwen3-ASR 和 TIGER-DnR 则直接在业务
请求中发送 multipart 音频，不提供独立的上传/检查接口。

FireRedTTS3 Base 使用同样的上传与检查接口，最终克隆路由为
`POST http://127.0.0.1:8325/v1/FireRedTTS3/clone`。它要求参考音频对应的准确
`prompt_text`；默认语言为 `Chinese`，也可以在 JSON 请求中传入官方语言或方言标签。

上传示例（以 Qwen3-TTS 8321 为例）：

```bash
curl -X POST http://127.0.0.1:8321/v1/upload_audio \
  -F 'audio=@reference.wav' \
  -F 'full_path=reference.wav' \
  -F 'prompt_text=这是一句参考音频转写。'

curl 'http://127.0.0.1:8321/v1/check/audio?file_name=reference.wav'

curl -X POST http://127.0.0.1:8321/v1/qwen/clone \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，欢迎使用。","audio_path":"reference.wav","prompt_text":"这是一句参考音频转写。"}' \
  -o qwen-clone.wav
```

各模型的 `prompt_text` 语义不同：

| 服务 | 行为 |
| --- | --- |
| Qwen3-TTS Base | 有准确参考文本时映射为官方 `ref_text`；无文本或显式 `x_vector_only=true` 时使用仅音色向量克隆。 |
| VoxCPM2 | `clone_mode=ultimate` 使用参考文本；`clone_mode=controllable` 改用 `control_instruction`，二者互斥。`nonverbal_tags` 最多一个，且只能用于可控模式。 |
| LongCat-AudioDiT | 必须提供与参考音频逐字一致的文本，可来自请求或 sidecar；缺失时 worker 会失败。参考音频重采样为 24 kHz 单声道。 |
| dots.tts-soar | 有参考文本时使用 continuation cloning；省略时保留官方 x-vector-only cloning。输出为 48 kHz 单声道。 |
| FireRedTTS3 Base | 请求中的非空 `prompt_text` 优先于 sidecar；二者都缺失时返回 `400`。 |

五个克隆服务均要求 `text` 和 `audio_path`，可选参数按模型区分，不能跨模型混用：

| 服务 | 模型专用 JSON 字段 |
| --- | --- |
| Qwen3-TTS | `language`、`x_vector_only`、`device_map`、`dtype`、`attn_implementation`、`max_new_tokens`、`top_p`、`temperature`、`trim_leading_silence` 与 `trim_leading_silence_*` |
| VoxCPM2 | `clone_mode`、`control_instruction`、`nonverbal_tags`、`cfg_value`、`inference_timesteps`、`normalize`、`denoise`、`retry_badcase`、`load_denoiser`、`optimize`、`device`、`seed` |
| LongCat | `nfe`、`guidance_strength`、`guidance_method`（`cfg`/`apg`）、`seed`、`duration_scale`、`vae_dtype` |
| dots.tts-soar | `language`、`template_name`、`precision`、`seed`、`ode_method`、`num_steps`、`guidance_scale`、`speaker_scale`、`max_generate_length`、`normalize_text`、`profile_inference` |
| FireRedTTS3 | `language`、`n_timesteps`、`inference_cfg`、`stop_threshold`、`seed` |

前四个服务还支持 `max_chars_per_chunk` 和 `pause_ms`，默认分块长度依次为
120、0（不分块）、180、120 字符，段间停顿默认 250 ms。
LongCat 默认 `nfe=16`、`guidance_strength=4.0`、`guidance_method=apg`；dots 默认
`num_steps=10`、`guidance_scale=1.2`、`speaker_scale=1.5`。
VoxCPM2 默认 `cfg_value=2.0`、`inference_timesteps=10`。

VoxCPM2 可控克隆示例（先在 8322 上传参考音频）：

```bash
curl -fsS http://127.0.0.1:8322/v1/voxcpm2/clone \
  -H 'Content-Type: application/json' \
  -d '{"text":"你终于来了。","audio_path":"reference.wav","clone_mode":"controllable","control_instruction":"用轻松、带笑意的语气说话","nonverbal_tags":["laughing"]}' \
  -o voxcpm2-controllable.wav
```

`nonverbal_tags` 是白名单标签名，最多一个；不能把 `[laughing]` 等标记塞进 `text` 或
`prompt_text`。可控模式忽略已上传的参考文本 sidecar，且不允许在 JSON 中同时传参考文本。

所有这些克隆请求都拒绝 `style_prompt`；声音风格应通过音色设计或
VoxCPM2 的 `control_instruction` 表达。`text` 中的 Markdown 标题和列表标记会按服务
兼容逻辑清理，不能依赖它们传递控制指令。

## 音色设计

所有音色设计路由都接收 `voice_description` 和可选的 `text`，成功返回 `audio/wav`，
并把结果写入 `storage/timbre/`：

```bash
curl -X POST http://127.0.0.1:8301/v1/qwen/timbre \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o qwen-voice.wav

curl -X POST http://127.0.0.1:8302/v1/moss/timbre \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，温柔、清晰，语速中等。","text":"你好。"}' \
  -o moss-voice.wav

curl -X POST http://127.0.0.1:8303/v1/mimo/timbre \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o mimo-voice.wav

curl -X POST http://127.0.0.1:8304/v1/FireRedTTS3/timbre \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o fireredtts3-voice.wav
```

MiMo 的 8303 服务只做云端请求编排、重试、分段和本地音色缓存；8300 控制面保留兼容
代理。后端无法连接 MiMo API 时，独立服务和代理会返回 `503`，请检查
`MIMO_API_KEY`、`MIMO_BASE_URL`、DNS、HTTPS 出网和 `HTTPS_PROXY`。

FireRedTTS3 Instruct 音色设计监听 `8304`，基础字段为 `voice_description` 和可选 `text`，并将
生成 WAV 保存在 `storage/timbre/`。最终路由为
`POST http://127.0.0.1:8304/v1/FireRedTTS3/timbre`。

模型额外字段如下；完整限制可在各进程的 `/docs` 查看：

| 音色设计服务 | 可选 JSON 字段 |
| --- | --- |
| Qwen VoiceDesign | `language`、`max_chars_per_chunk`、`pause_ms`、`max_new_tokens`、`top_p`、`temperature`、`dtype`、`attn_implementation`、`device_map` |
| MOSS VoiceGenerator | `max_chars_per_chunk`、`pause_ms`、`max_new_tokens`、`audio_temperature`、`audio_top_p`、`audio_top_k`、`audio_repetition_penalty`、`dtype`、`attn_implementation` |
| MiMo | `model`、`timeout`、`max_chars_per_chunk`、`pause_ms`、`optimize_text_preview`、`min_request_interval_seconds`、`max_retries`、`retry_base_seconds`、`retry_max_seconds` |
| FireRedTTS3 Instruct | `language`、`n_timesteps`、`inference_cfg`、`seed`、`stop_threshold` |

FireRedTTS3 两个模式均默认 `n_timesteps=10`、`stop_threshold=0.5`，音色设计默认
`inference_cfg=1.2`、`seed=2`，克隆默认 `inference_cfg=2.0`、`seed=1234`。

MOSS VoiceGenerator 必须使用 **MOSS-Audio-Tokenizer v1**（24 kHz、单声道）。
不要把 48 kHz 双声道的 v2 codec 作为该服务的 tokenizer；8302 的健康检查中
`available.moss_audio_tokenizer` 应为 `true`。

## 音频理解

MOSS-Audio-4B-Thinking 监听 `8341`，MOSS-Audio-4B-Instruct 监听 `8342`，都是音频理解模型，不会合成 WAV。它们以 multipart
表单接收 `audio`，可通过 `prompt` 指定转写、描述或问答任务；请求会流式暂存、完成后删除，
响应返回最终文本和上传摘要。两个服务都需要 `$HOME/tts-depency/MOSS-Audio` 官方源码，并默认以离线模式分别从
`$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-4B-Thinking` 和
`$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-4B-Instruct` 加载；可分别通过对应的
`MOSS_AUDIO_4B_THINKING_*` 与 `MOSS_AUDIO_4B_INSTRUCT_*` 环境变量覆盖。

不先调用上传接口，也不传 `audio_path`。成功 JSON 包含 `code`、`text`、
`elapsed_seconds` 和 `audio`（SHA-256、大小、扩展名）；`strip_thinking=true` 可移除思考段。

```bash
curl -X POST http://127.0.0.1:8341/v1/mossAudioThinking/understand \
  -F 'audio=@reference.mp3;type=audio/mpeg' \
  -F 'prompt=请准确转写这段音频，仅输出转写文本。' \
  -F 'strip_thinking=true'
```

Instruct 服务使用相同路由和字段：

```bash
curl -X POST http://127.0.0.1:8342/v1/mossAudioThinking/understand \
  -F 'audio=@reference.mp3;type=audio/mpeg' \
  -F 'prompt=Describe this audio.'
```

可选字段包括 `max_new_tokens`（1–4096，默认 1024）、`do_sample`、`temperature`、`top_p`、
`top_k`、`enable_time_marker`、`strip_thinking`、`device` 与 `dtype`（`auto`、`bfloat16` 或
`float16`）。Thinking 服务默认关闭采样，Instruct 服务遵循上游 `hf_inference.py` 默认开启采样；
仅当 `do_sample=true` 时才把采样参数传给模型。两个服务使用共享 `GPU_LOCK_FILE`，一项请求对应一个
worker，worker 退出后释放显存。

单独启动时务必设置 `MOSS_AUDIO_4B_VARIANT`；若省略且环境中出现任何
`MOSS_AUDIO_4B_INSTRUCT_*` 变量，代码会推断为 Instruct。统一启动脚本已为两个进程明确设置变体。

## TIGER-DnR 三 Stem 分离

TIGER-DnR 监听 `8351`，接收任意支持的音频格式，以 44.1 kHz 按上游模型分离，再恢复输入的
采样率、声道数和准确采样数。成功响应是 ZIP，固定包含 `dialog.wav`、`effects.wav` 和
`music.wav`；相同文件也会作为一个原子批次保留在 `storage/separation/`。请求通过共享
`GPU_LOCK_FILE` 串行化，且每次请求都会启动并退出自己的 worker。默认 `device=cuda`，可在无
GPU 的调试环境显式使用 `device=cpu`。

```bash
curl -X POST http://127.0.0.1:8351/v1/tigerDnr/separate \
  -F 'audio=@cinematic-mix.wav;type=audio/wav' \
  -F 'device=cuda' \
  -o tiger-dnr-stems.zip
```

## SoundEffect 生成

MOSS-SoundEffect v2 接受中英文非语言声效提示词，输出 48 kHz 单声道 WAV，`seconds`
范围为 `(0, 30]`，默认 10 秒。默认字段为 `num_inference_steps=100`、`cfg_scale=4.0`、
`sigma_shift=5.0`、`seed=0`：

```bash
curl -X POST http://127.0.0.1:8312/v1/moss/soundEffect \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"雨夜中木门被轻敲三下，近距离，无可辨认说话声","seconds":3}' \
  -o moss-sfx.wav
```

Stable Audio 3 Medium 接受英文提示词，可生成音乐或声效，输出为 44.1 kHz 立体声 WAV。
`seconds` 和官方别名 `duration` 必须一致，最大时长为 380 秒；默认
`seconds=7`、`steps=8`、`cfg_scale=1.0`、`seed=-1`：

```bash
curl -X POST http://127.0.0.1:8311/v1/stableAudio/soundEffect \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A wooden door is knocked three times in a quiet room. TrackType: SFX","seconds":3}' \
  -o stable-audio-sfx.wav
```

两个服务只使用本文列出的最终接口与字段。脚本制作场景的提示词规范、
`prompt_en` 约束和 GPU 示例见 [`soundEffect/README.md`](soundEffect/README.md) 与
[`soundEffect/声效提示词说明.md`](soundEffect/声效提示词说明.md)。

MOSS 可按请求传 `device`、`torch_dtype`；Stable Audio 的设备/精度字段只允许
`device=cuda`、`dtype=float16`，其他值返回 `422`。两者均不接收参考音频。

## ACE-Step BGM 生成

ACE-Step 独立监听 `8313`，只负责有声小说纯配乐；Stable Audio 继续承担 ambience /
texture，MOSS-SoundEffect 继续承担明确事件型短音效。服务默认使用 BF16、model CPU
offload、VAE tiling 和一次性 worker，生成结果保存到 `storage/bgm/`，前端可直接把返回
的标准 48 kHz 双声道 WAV 写入已有 `bgmLibrary`。

```bash
curl -X POST http://127.0.0.1:8313/v1/aceStep/bgm \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Dark cinematic ambient underscore, sparse felt piano, low cello drone, restrained dynamics, designed underneath spoken narration, no vocals","seconds":10,"steps":8,"bpm":58,"keyscale":"D minor","timesignature":"4","seed":42}' \
  -o ace-step-bgm.wav
```

`prompt` 长度为 1–2000，`seconds` 为 10–600（默认 60），`steps` 为 1–20（默认 8），`bpm` 为 30–240；
`keyscale`、`timesignature` 可省略，`seed=-1` 表示随机。成功响应包含
`X-ACE-Step-Seed`、`X-ACE-Step-Sample-Rate` 和 `X-ACE-Step-Model` 响应头。先手动准备
依赖再启动：

```bash
uv sync --project ace_step_1_5 --locked
```

本仓库的 `start.sh` 对 ACE-Step 使用 `uv run --no-sync`，不会在启动时安装依赖。

## Step-Audio-EditX 编辑

8331 自己负责 prompt 音频的上传、检查和编辑，不经过 8300 代理。先上传，再调用编辑：

```bash
curl -X POST http://127.0.0.1:8331/v1/upload_audio \
  -F 'audio=@line-1.wav' \
  -F 'full_path=step-audio-editx/line-1.wav'

curl -X POST http://127.0.0.1:8331/v1/stepAudioEditx/edit \
  -H 'Content-Type: application/json' \
  -d '{"prompt_audio":"step-audio-editx/line-1.wav","prompt_text":"这是一条台词。","generated_text":"这是一条台词。","edit_type":"emotion","edit_info":"coldness"}' \
  -o edited.wav
```

`edit_type` 可为 `emotion`、`style`、`paralinguistic`、`denoise`、`vad` 或 `speed`。
`emotion`、`style` 和 `speed` 需要非空 `edit_info`；`denoise` 与 `vad` 不要求文本；
其他编辑类型需要与 prompt 音频匹配的 `prompt_text`。请求字段映射到上游命令的同名
编辑语义，输出保存到 `STEP_AUDIO_EDITX_OUTPUT_DIR`（默认 `storage/clone/`）。

## 文本润色训练实验

`scripts/training/` 与 `training/configs/` 提供有声书“原始小说片段 → 纯净可朗读文本”的
LoRA 流程实验，不属于 HTTP 服务，也不会随 `start.sh` 启动。模型为
`Qwen/Qwen3-4B-Instruct-2507`，模板为 `qwen3_nothink`；这部分处理文本，不训练语音模型。

| 文件 | 用途 |
| --- | --- |
| `scripts/training/build_audio_polisher_poc.py` | 抽取 Markdown、调用本地 Ollama 教师、按来源划分训练/验证/测试集，并生成 Alpaca 数据和导入清单 |
| `scripts/training/import_audio_polisher_to_easy_dataset.py` | 将来源和待确认候选导入独立运行的 Easy Dataset |
| `scripts/training/run_audio_polisher_inference.py` | 用 Transformers/PEFT 对基础模型与可选 adapter 执行确定性 GPU 推理 |
| `scripts/training/compare_audio_polisher_predictions.py` | 比较格式检查、相对教师标签的字符相似度与长度比例 |
| `training/configs/audio_polisher_qwen3_4b_lora_poc.yaml` | LLaMA-Factory LoRA 训练模板，rank 8、BF16、3 epochs |
| `training/configs/audio_polisher_qwen3_4b_predict_poc.yaml` | 隔离测试集预测模板 |

生成数据集前自行启动本地 Ollama 并准备教师模型（默认 `qwen3.5:9b`）。从仓库根目录执行：

```bash
uv run --project qa --locked python -m scripts.training.build_audio_polisher_poc \
  --source-dir /path/to/story-markdown \
  --output-dir storage/training/audio-polisher-poc
```

默认抽取 10 个来源、每个来源 3 条样本（不足则中止），来源按 8/1/1 分组；支持续跑缓存、固定 seed
和源文件 SHA-256。导入 Easy Dataset 的脚本会写入其项目，默认地址 `http://127.0.0.1:1717`，
需要操作者主动执行。候选标记为 `research-only`、`automated-poc`、`confirmed=false`，
须完成人工审核后再考虑生产训练。

训练与真实预测需要仓库外单独准备 LLaMA-Factory 或 Transformers/PEFT/CUDA 环境；
本仓库没有训练 uv 项目，不能在 `qa` 环境运行模型训练。配置中的 `model_name_or_path`
默认是模型 ID，离线运行时应替换为已准备的本地目录。
训练 CLI 和预测脚本不获取服务的共享 GPU 锁，应由操作者安排独占 GPU 时间。
adapter、数据集和预测都写入 `storage/training/`，不提交到 Git。

## 健康检查与错误语义

- 健康检查不会加载模型；`available`、`paths`、`runtime` 和 `last_errors` 用于区分依赖、
  权重、CUDA、worker 和配置问题。
- 请求校验或媒体类型失败通常返回 `422`，上传超过大小上限返回 `413`；
  找不到参考音频通常返回 `404`，但 Confucius4-TTS 的现有文件预检统一返回 `503`。
- GPU 排队超时返回 `503`。本地依赖缺失按服务返回 `503` 或 `500`；模型执行失败通常为
  `500`。SoulX/Ditto 的 worker 超时返回 `504`，其他服务通常转换为 `500`。
  Steam Audio 的非法 Manifest 返回 `422`、重复 `job_id` 返回 `409`、未知任务返回 `404`。
- 模型 worker 失败或超时会清理临时文件和进程组，再返回服务错误；共享 GPU 锁在
  `finally` 中释放。
- 音频生成接口响应体是 WAV，同时会写入语义对应的输出目录；JSON、ZIP、MP4 和临时导出
  的保存行为见服务总览。保留策略默认关闭：先通过
  `GET /v1/control` 的 `storage` 字段查看所在文件系统的容量与可用空间；需要治理历史
  生成结果时，再由运维显式配置并执行维护脚本，绝不自动删除用户上传或音色引用：

  ```bash
  export STORAGE_RETENTION_HOURS=168
  # 先预览，再显式确认删除；脚本只匹配已知的生成文件前缀。
  uv run --project qwen3_tts python main/storage_maintenance.py
  uv run --project qwen3_tts python main/storage_maintenance.py --apply
  ```

维护脚本目前仅处理四个默认音频目录中指定前缀的 WAV：Qwen/MOSS/MiMo 设计音色，
Qwen3-TTS/VoxCPM2/LongCat/dots/Step 的克隆或编辑结果，MOSS/Stable 声效及 ACE-Step BGM。
它保护普通上传和仍被 `.references` 引用的音色，但**尚不清理 FireRedTTS3、Confucius4-TTS、
Seed-VC 的生成结果、TIGER 分离批次、MP4、图片或训练产物**，也不自动遍历每个自定义
`*_OUTPUT_DIR`。`STORAGE_RETENTION_MAX_BYTES` 针对这些匹配的生成 WAV 总量，不是整个
`storage/` 容量限制。修改输出目录时需同步考虑维护脚本的扫描范围。

## 测试与开发

根目录回归测试不下载权重、不调用外部服务、不需要 CUDA：

```bash
bash -n start.sh
bash scripts/quality_gate.sh
```

统一质量门禁先执行 shell 语法和三个无 SDK 的 C++ 核心测试，再检查 19 个服务与 QA 的
锁文件、Ruff 和格式，最后执行根目录、ACE-Step、Stable Audio 的无模型测试。
GitHub Actions 使用同一脚本；需要 `uv` 和 `g++`，不需要 CUDA 或 Steam Audio SDK。
锁文件检查仍可能访问 Git/wheel 元数据，初次运行需具备这些依赖来源的网络或本地缓存。
已有完整缓存时可运行 `UV_OFFLINE=1 bash scripts/quality_gate.sh`。

仅运行无模型测试时使用轻量 QA 环境。两个服务内的测试必须从各自目录运行，以免同名
`runtime` 模块解析错误：

```bash
uv sync --project qa --locked
uv run --project qa --locked python -m unittest discover -s tests -v
(cd ace_step_1_5 && uv run --project ../qa --locked python -m unittest discover -s tests -v)
(cd stable_audio_3_medium && uv run --project ../qa --locked python -m unittest discover -s tests -v)
```

FastAPI TestClient 回归使用 QA 的 `httpx2` 兼容依赖。不得为无模型测试安装整套模型环境或
恢复旧客户端组合；worker、subprocess、网络与 CUDA 边界由测试替身替换。
`scripts/training/` 的纯文本逻辑也有根目录回归测试，但当前质量门禁的 Ruff 路径不包含训练脚本。

MOSS 的真实 CUDA/权重 smoke test 是独立流程：

```bash
bash soundEffect/run_moss_soundeffect_v2.sh
```

手动模型演示还包括 `tests/testConfucius4TTS/run_confucius4_tts.py`、
[`tests/testSoulX_FlashHead/testFlashHead.py`](tests/testSoulX_FlashHead/README.md) 和
[`tests/testDitto/testDitto.py`](tests/testDitto/README.md)。这些脚本需自行准备音频/图片并启动
对应模型服务，不属于常规无模型测试，不应把其历史样例结果视为当前机器已通过验证。

所有模型服务的 uv、Ruff、日志和无模型测试规范见
[`docs/python-engineering.md`](docs/python-engineering.md)。

修改请求契约、路由、存储解析或 worker 生命周期时，应同步添加/更新对应的 no-model
测试，并在 README 中更新兼容字段。不要提交模型权重、上传音频、生成 WAV、缓存、虚拟
环境、密钥或机器专用绝对路径。

模型专用说明见 [MiMo](mimo_tts/README.md)、[Step-Audio-EditX](Step_Audio_EditX/README.md)、
[LongCat](LongCat_AudioDiT_3.5B_bf16/README.md)、[dots](dots_tts_soar/README.md)、
[MOSS-SoundEffect](moss_soundEffect/README.md)、[Stable Audio](stable_audio_3_medium/README.md)、
[ACE-Step](ace_step_1_5/README.md)、[SoulX](SoulX-FlashHead-1_3B/README.md)、
[Ditto](ditto/README.md) 和 [Steam Audio renderer](steam_audio_renderer/README.md)。
子目录中的迁移记录和本机实验描述仅作背景，部署命令、服务清单与测试环境以本 README
和实际代码为准。贡献规则见 [AGENTS.md](AGENTS.md)，中文文档要求见
[CONSTITUTION.md](CONSTITUTION.md)。

## 许可证

本仓库代码采用 [Apache-2.0](LICENSE)。模型权重、上游推理源码、Steam Audio SDK 和云端
服务各自的许可与使用条款独立适用。
