# LLM Local Backend

面向本地部署的 LLM 音视频后端，提供参考音频语音克隆、文本音色设计、BGM 生成、语音转写、音乐转谱、人像与图生视频、
三路音源分离，以及 CPU 音频母带和 Steam Audio 对象级空间导出。
模型推理均使用本地权重。

仓库包含 **18 个模型服务 uv 项目，统一启动 19 个 HTTP 进程**，其中一个是轻量控制面。
HTTP 进程不加载重型模型，模型推理由对应目录
中的一次性 worker 完成；共享运行时和无模型 QA 环境分别位于 `llm_local_runtime/`、`qa/`。
接口、路径和默认值以当前 `main.py`、`worker.py`、`pyproject.toml` 与 `start.sh` 为准。

导航：[服务总览](#服务总览) · [接口兼容约束](#接口兼容约束) ·
[安装与启动](#安装与启动) · [模型与配置](#模型路径与主要配置) ·
[空间导出](#48-khz-母带与-steam-audio-正式导出) · [语音克隆](#参考音频克隆) ·
[训练实验](#文本润色训练实验) · [测试与开发](#测试与开发)。

LocalAI/audio.cpp 裸机升级与镜像清理记录见 [升级记录](docs/localai-audiocpp-upgrade.md)。
LocalAI v4.11.0 与 CUDA audio-cpp backend 已完成本机构建和无模型启停核验；当前保留 Qwen3-TTS
Base/VoiceDesign 的本地模型候选及五个 one-shot worker bridge。原生 provider 仍是默认值；未下载
模型或安装 Python 项目依赖。缺失模型和外部依赖见[手动准备状态](reports/localai/manual-model-readiness.md)。

## 服务总览

能力与原生输出依据各行链接的官方模型卡或推理文档；端口与 HTTP 路由依据本仓库
`main.py` 和 `start.sh`。模型原生能力列描述对应模型或官方完整模型包，接口列列出当前
服务入口；原生能力是否已接入某个接口，还需查看该接口实际字段和 worker 调用。

| 模型/服务 | 默认端口 | 主业务接口 | 模型原生能力/职责 | 原生输出 |
| --- | --- | --- | --- | --- |
| [控制面 `main/`](main/) | 8300 | `GET /v1/control`；`POST /v1/audio/export`；`POST /v1/audio/spatial/render` | 控制、共享上传/检查、CPU 母带与 Steam Audio 空间导出（[源码](main/main.py)） | 非模型服务，响应见下节 |
| [Qwen3-TTS 1.7B VoiceDesign `model-audio/design_qwen3-tts-1.7b-voicedesign/`](model-audio/design_qwen3-tts-1.7b-voicedesign/) | 8301 | `POST /v1/qwen/timbre` | 根据自然语言音色描述和待朗读文本生成语音，支持音色、情绪和韵律描述（[官方](https://github.com/QwenLM/Qwen3-TTS#voice-design)） | 语音波形 |
| [MOSS-Music-8B-Instruct `model-audio/analyse_moss-music-8b-Instruct/`](model-audio/analyse_moss-music-8b-Instruct/) | 8310 | `POST /v1/openMoss/moss-music` | 上传音乐并按文字指令生成音乐描述、问答或歌词文本；使用本地完整 GGUF 与 HOT-Step-CPP | JSON 分析文本 |
| [MOSS-SoundEffect v2.0 `model-audio/soundEffect-v2.0/`](model-audio/soundEffect-v2.0/) | 8362 | `POST /v1/moss/soundEffect-v2.0` | 根据中英文提示词生成最长 30 秒的环境音效（[模型卡](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)） | 48 kHz 单声道 WAV |
| [ACE-Step 1.5 XL Turbo `model-audio/music_ace-step-1.5/`](model-audio/music_ace-step-1.5/) | 8313 | `POST /v1/aceStep/bgm` | 按风格描述和可选音乐元数据生成 BGM；当前接口不接收歌词（[官方](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers)） | 音乐波形 |
| [YuE2-3B `model-audio/music_yue2-3b/`](model-audio/music_yue2-3b/) | 8314 | `POST /v1/m-a-p/YuE2` | 根据风格、歌词和可选 ABC 乐谱生成带歌声与伴奏的歌曲（[官方](https://huggingface.co/m-a-p/YuE2-3B)） | 48 kHz 立体声音乐波形 |
| [SheetSage2 `model-audio/score_sheetsage-2/`](model-audio/score_sheetsage-2/) | 8315 | `POST /v1/m-a-p/SheetSage2` | 将音乐转为可编辑乐谱、MIDI、节拍/和弦/调性/结构与时间标注（[官方](https://huggingface.co/m-a-p/SheetSage2)） | ABC、MIDI 与 JSON/LAB 标注 ZIP |
| [YuE2-Vae `model-audio/codec_yue2-vae/`](model-audio/codec_yue2-vae/) | 8316 | `POST /v1/m-a-p/YuE2-Vae` | 将 48 kHz 音频编码为 acoustic latent，或将 latent 解码为立体声音乐（[官方](https://huggingface.co/m-a-p/YuE2-Vae)） | `[T,64]` NPY latent 或 48 kHz WAV |
| [MERT-v2-FullSong `model-audio/embedding_mert-v2-fullsong/`](model-audio/embedding_mert-v2-fullsong/) | 8317 | `POST /v1/m-a-p/MERT-v2-FullSong` | 为 30–360 秒完整歌曲提取 MERT-v2 逐帧特征和整曲表征（[官方](https://huggingface.co/m-a-p/MERT-v2-FullSong)） | 1024 维 JSON embedding 或 25 Hz 特征 NPZ |
| [Qwen3-TTS 1.7B Base `model-audio/clone_qwen3-tts-1.7b-base/`](model-audio/clone_qwen3-tts-1.7b-base/) | 8321 | `POST /v1/qwen/clone` | 参考音频条件下的零样本音色克隆与多语言 TTS（[官方](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)） | 语音波形 |
| [VoxCPM2 `model-audio/clone_voxcpm-2/`](model-audio/clone_voxcpm-2/) | 8322 | `POST /v1/voxcpm2/clone` | 30 语种 TTS、文本描述音色设计、参考音频克隆，以及带风格指令的可控克隆（[官方](https://huggingface.co/openbmb/VoxCPM2)） | 语音波形 |
| [IndexTTS-2.5 `model-audio/indexTTS-2.5/`](model-audio/indexTTS-2.5/) | 8323 | `POST /v1/index-team/indexTTS-2.5` | 参考音频零样本克隆，支持五种语言、时长和情感向量控制（[官方](https://github.com/index-tts/index-tts/releases/tag/v2.5.0)） | 22.05 kHz WAV |
| [TIGER-DnR `model-audio/separate_tiger-dnr/`](model-audio/separate_tiger-dnr/) | 8351 | `POST /v1/tigerDnr/separate` | 将混音分离为对白、音效、音乐三路 Stem（[官方推理](https://github.com/JusperLee/TIGER/blob/main/inference_dnr.py)） | 三路音频波形 |
| [Confucius4-TTS `model-audio/clone_confucius-4/`](model-audio/clone_confucius-4/) | 8361 | `POST /v1/confucius4TTS/generate` | 无需参考转写的零样本克隆与跨语言 TTS，支持 14 语种及参考音频的情感迁移（[官方](https://huggingface.co/netease-youdao/Confucius4-TTS)） | 语音波形 |
| [Qwen3-ASR-1.7B `model-audio/transcribe_qwen3-asr-1.7b/`](model-audio/transcribe_qwen3-asr-1.7b/) | 8371 | `POST /v1/qwen3/asr` | 语种识别与语音转写，支持 30 种语言和 22 种中文方言（[官方](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)） | 转写文本与语种 |
| [MSST Mega 53 `model-audio/separate_msst-mega53/`](model-audio/separate_msst-mega53/) | 8380 | `POST /v1/msst/separate` | 按调用方选择音乐声部分离，保留 residual（[官方](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21)） | 44.1 kHz 双声道 FLOAT WAV 与元数据 ZIP |
| [SoulX-FlashHead 1.3B `model-video/avatar_soulx-flashhead-1.3b/`](model-video/avatar_soulx-flashhead-1.3b/) | 8391 | `POST /v1/soulX/flashHead` | 参考人像图片和驱动音频生成带声音的 MP4；支持 Lite/Pro 单 GPU | MP4 视频 |
| [Kandinsky 6 Lite 蒸馏版 `model-video/Kandinsky-6.0-Lite-Distill-BF16/`](model-video/Kandinsky-6.0-Lite-Distill-BF16/) | 8369 | `POST /v1/kandinskylab/Kandinsky-6.0-Lite-Distill-BF16` | 使用 BF16 蒸馏权重从参考图片生成 5 秒视频（[官方模型卡](https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers)） | 864×480、24 FPS MP4 |
| [MiniMax H3 16G HiFi `model-video/MiniMax-H3-16G-HiFi/`](model-video/MiniMax-H3-16G-HiFi/) | 8392 | `POST /v1/minimax/H3-16G-HiFi` | FL2VA 首尾帧或 Ref2VA 参考图生成视频，可选 H3 原生音轨（[模型卡](https://modelscope.cn/models/l1ngzi/MiniMax-H3-16G-HiFi)） | 默认无音轨；请求 `include_audio=true` 时输出带音轨 MP4 |

Demo 调用对应端口并保存服务原始输出；
都支持 `--check-only`，仅检查输入和原生服务健康状态。条件缺失时明确跳过，不写生成产物。
模型准备状态和未满足的外部依赖见[手动准备状态](reports/localai/manual-model-readiness.md)。

表中列出主业务入口。各 HTTP 进程还提供 `GET /v1/health`；FastAPI 的 `/docs` 和
`/openapi.json` 可查看实际注册的字段与路由。部分服务另有音频/图片上传和检查接口。
健康接口成功仅表示 HTTP 服务可访问，还须检查响应中的 `available` 和模型文件状态。

## HTTP 响应与存储

以下记录当前 HTTP 实现的响应与存储。

| HTTP 服务 | 成功响应 | 服务端保存行为 |
| --- | --- | --- |
| TTS 克隆、音色设计、BGM 生成 | `audio/wav` | 按用途保存 WAV |
| SheetSage2（8315） | `application/zip`，含可用时的 ABC、MIDI、事件与窗口输出 | 原子保存到 `storage/sheetsage2/`；上传音频和 worker 临时文件请求后清理 |
| YuE2-Vae（8316） | `audio/wav`（decode）或 `application/octet-stream`（encode NPY） | 原子保存到 `storage/yue2_vae/`；上传和 worker 临时文件请求后清理 |
| MSST Mega 53（8380） | `application/zip`，选定声部、residual 与 manifest.json | 原子保存到 `storage/msst/`；输入与 worker 临时文件请求后清理 |
| MERT-v2-FullSong（8317） | JSON mean embedding 或 `application/octet-stream` NPZ 逐帧特征 | 原子保存到 `storage/mert_v2_fullsong/`；请求音频和 worker 临时文件请求后清理 |
| Qwen3-ASR | JSON 文本及上传摘要 | 请求结束后删除上传暂存和 worker 结果，不保留输入音频 |
| TIGER-DnR | `application/zip`，含 `dialog.wav`、`effects.wav`、`music.wav` | 原子保存三路 Stem 与 ZIP 批次 |
| 控制面音频导出 | WAV 或 MP3 | 响应完成后删除临时成品 |

## 接口兼容约束

Qwen/VoxCPM2 克隆接口的 `style_prompt` 已禁用；
VoxCPM2 可控克隆使用 `control_instruction`。
`/v1/audio/export` 仍兼容旧总线滤镜，但正式空间导出使用
`/v1/audio/spatial/render`，详见音频导出章节。

## 目录与运行数据

模型目录按 `功能词_模型名-版本` 命名，例如 `music_ace-step-1.5`。功能词速查见
[音频目录说明](model-audio/README.md)和[视频目录说明](model-video/README.md)。

```text
main/                                            8300 控制面，不包含模型推理
model-audio/                                     当前维护的音频模型服务、worker 与 renderer
model-audio/analyse_moss-music-8b-Instruct/        MOSS 音乐理解 API 与 GGUF worker
model-audio/soundEffect-v2.0/                   MOSS-SoundEffect 文本音效生成 API 与 worker
model-audio/spatial_steam-audio/                 独立 C++17 Steam Audio CPU renderer
model-audio/clone_qwen3-tts-1.7b-base/           Qwen3-TTS Base 的 HTTP 服务和 worker
model-audio/clone_voxcpm-2/                      VoxCPM2 的 HTTP 服务和克隆 worker
model-audio/indexTTS-2.5/                        IndexTTS-2.5 HTTP 服务和一次性 worker
model-audio/music_ace-step-1.5/                  ACE-Step 1.5 BGM 服务和 worker
model-audio/music_yue2-3b/                       YuE2-3B 歌曲生成服务和一次性 worker
model-audio/score_sheetsage-2/                   SheetSage2 音乐转谱服务和一次性 worker
model-audio/codec_yue2-vae/                      YuE2 acoustic latent 编码/解码服务和一次性 worker
model-audio/embedding_mert-v2-fullsong/          MERT-v2-FullSong 音乐表征服务和一次性 worker
model-audio/design_qwen3-tts-1.7b-voicedesign/   Qwen VoiceDesign 服务和 worker
model-audio/clone_confucius-4/                   Confucius4-TTS 服务和一次性 worker
model-audio/separate_msst-mega53/                MSST 重点音乐声部分离与一次性 worker
model-audio/separate_tiger-dnr/                  TIGER-DnR 三 Stem 分离服务和 worker
model-audio/transcribe_qwen3-asr-1.7b/           Qwen3-ASR-1.7B 语音识别服务和 worker
model-text/                                      有声书纯文本润色训练实验
model-image/                                     预留图像模型目录
model-video/                                     SoulX-FlashHead、Kandinsky 6 与 MiniMax H3 视频生成服务
tests/                                           根目录无模型回归测试
qa/                                              锁定的轻量测试与 Ruff 环境，不安装模型依赖
llm_local_runtime/                               共享流式上传、引用存储、GPU 队列与容量工具
scripts/                                         质量门禁与 renderer 构建工具
model-text/scripts/training/                     有声书纯文本润色训练辅助工具
model-text/training/configs/                     有声书纯文本润色 LoRA 的实验配置
docs/                                            Python 工程规范
storage/                                         上传素材、生成音视频、sidecar、训练产物、缓存和 GPU 锁
```

根目录的 `流行*模型功能简介.md` 是调研资料，`空间音频讨论/` 是方案与验收记录，
`task*.md` 是任务资料；其中提到的模型或计划不表示已注册服务。实际服务清单以本 README
和 `start.sh` 为准。

默认运行数据目录为：

| 目录 | 内容 | 覆盖变量 |
| --- | --- | --- |
| `storage/timbre/` | Qwen VoiceDesign 生成的音色参考音频 | `TIMBRE_STORAGE_DIR` |
| `storage/bgm/` | ACE-Step 与 YuE2-3B 生成的 BGM、歌曲和 OST | `BGM_STORAGE_DIR`、`ACESTEP_OUTPUT_DIR`、`YUE2_3B_OUTPUT_DIR` |
| `storage/yue2_vae/` | YuE2-Vae 编码的 latent NPY 与解码 WAV | `YUE2_VAE_OUTPUT_DIR` |
| `storage/sheetsage2/` | SheetSage2 转写生成的 ABC、MIDI 与标注 ZIP | `SHEETSAGE2_OUTPUT_DIR` |
| `storage/msst/` | MSST 声部分离 ZIP；不受 WAV 自动保留策略清理 | `MSST_OUTPUT_DIR` |
| `storage/mert_v2_fullsong/` | MERT-v2-FullSong 的整曲 JSON embedding 与逐帧 NPZ 特征 | `MERT_V2_FULLSONG_OUTPUT_DIR` |
| `storage/clone/` | 普通参考音频与 TTS 克隆结果 | `CLONE_STORAGE_DIR`、各服务的 `*_OUTPUT_DIR` |
| `storage/separation/` | TIGER-DnR 每次分离的 dialog、effects、music Stem 与 ZIP | `TIGER_DNR_OUTPUT_DIR` |
| `storage/video/` | SoulX-FlashHead、Kandinsky 6 与 MiniMax H3 生成的 MP4 | `SOULX_FLASHHEAD_OUTPUT_DIR`、`KANDINSKY6_OUTPUT_DIR`、`MINIMAX_H3_OUTPUT_DIR` |
| `storage/training/` | 文本润色实验的数据集、adapter 和预测结果 | 训练脚本的 `--output-dir` 与训练配置 |
| `storage/.cache/runtime/` | worker 临时文件、母带/Steam Audio 任务缓存、库缓存和共享 GPU 锁 | `RUNTIME_CACHE_DIR`、`SPATIAL_EXPORT_CACHE_DIR`、`STEAM_AUDIO_RENDER_CACHE_DIR`、`GPU_LOCK_FILE` |

如果上传音频的内容与 `storage/timbre/` 中已有的设计音色一致，Qwen3-TTS、VoxCPM2 与 Confucius4-TTS 会在
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
`空间音频讨论/` 中的音频文件属于手动研究素材，不是服务或自动测试依赖；文档保留
实测记录，复现示例时请自备素材并修改输入路径。

统一启动和 GPU 锁使用 Linux 的 `setsid`、进程组信号与 `fcntl.flock`，部署目标为 Linux。
运行要求：Python `3.12.13`、`uv`、FFmpeg/ffprobe、可用的 CUDA/NVIDIA 驱动（本地模型服务），
以及下方列出的模型权重和外部源码目录。正式 Steam Audio renderer
需要 CMake 3.17+、C++17 编译器和本地 SDK。
不同模型锁定不同 Torch/CUDA 组合，必须使用各自环境。权重与第三方源码不放进本仓库。

先为需要的服务同步锁定依赖；部署全部服务时可以执行：

```bash
for project in model-video/avatar_soulx-flashhead-1.3b model-video/Kandinsky-6.0-Lite-Distill-BF16 model-video/MiniMax-H3-16G-HiFi model-audio/analyse_moss-music-8b-Instruct model-audio/clone_qwen3-tts-1.7b-base model-audio/clone_voxcpm-2 model-audio/indexTTS-2.5 model-audio/music_ace-step-1.5 model-audio/music_yue2-3b \
  model-audio/score_sheetsage-2 model-audio/codec_yue2-vae model-audio/embedding_mert-v2-fullsong \
  model-audio/design_qwen3-tts-1.7b-voicedesign model-audio/clone_confucius-4 model-audio/transcribe_qwen3-asr-1.7b model-audio/separate_tiger-dnr model-audio/separate_msst-mega53; do
  uv sync --project "$project" --locked
done
```

ACE-Step 的 Diffusers 使用锁定的 Git revision；首次安装和锁文件校验可能需要访问该依赖来源。
YuE2-3B 使用官方 `yue2_infer` 0.1.5 wheel 和 PyTorch 2.10；MOT 与 `YuE2-Vae`
权重需分别准备在下表默认目录，模型目录缺失时健康接口会报告 `ready=false`。
`LOCAL_FILES_ONLY` 只控制模型加载，不代替依赖安装。启动前应完成 `uv sync --locked`；
`start.sh` 不执行依赖同步，也不下载权重或外部源码。

YuE2-Vae 独立项目使用官方 `AutoModel` remote code，依赖 PyTorch 2.10、Transformers 4.57.6、
NumPy 2.2.6 与 SoundFile。准备 `m-a-p/YuE2-Vae` 本地快照后执行
`uv sync --project model-audio/codec_yue2-vae --locked`。`POST /v1/m-a-p/YuE2-Vae` 的
`operation=decode` 接收 `latent=@latent.npy` 并返回 48 kHz WAV；
`operation=encode` 接收 `audio=@song.wav` 并返回 `[T,64]` NPY。音频由 FFmpeg
转换为 48 kHz 双声道；VAE 使用 FP32 与官方 `decode_tiled` 分块解码。

SheetSage2 使用 `transformers` 官方 `AutoModel` 调用及一次性 worker。请按模型卡准备
SheetSage2 与其配置固定 revision 的 MERT-v2-FullSong 两个本地目录，再执行
`uv sync --project model-audio/score_sheetsage-2 --locked`。服务默认使用 BF16 推理和 SheetSage2 `default`
preset，FFmpeg 解码上传音频；转写结果 ZIP 原子保存到 `storage/sheetsage2/`。

MERT-v2-FullSong 有独立服务项目，使用模型卡测试过的 PyTorch/Torchaudio 2.6.0 与
Transformers 4.53.2。模型代码、processor 和权重来自 `$HF_MIRROR_DIR/m-a-p/MERT-v2-FullSong`；
执行 `uv sync --project model-audio/embedding_mert-v2-fullsong --locked` 后由本服务自己的 worker 加载，和 SheetSage2
使用不同的 uv 环境。

MOSS-Music 使用 `$HF_MIRROR_DIR/OpenMOSS/moss-music` 的 Q8 语言模型及 F16 音频编码器，
需要预先构建 `ace-caption`。它不加载 Torch，不在请求时编译或下载；
[服务说明](model-audio/analyse_moss-music-8b-Instruct/README.md)提供固定源码 revision、构建命令、字段与调用示例。

默认情况下 `LOCAL_FILES_ONLY=1`，本地 worker 不会从 Hugging Face 下载权重。确认模型、
Tokenizer 和上游源码就绪后启动全部服务：

```bash
bash start.sh
```

`start.sh` 会启动 8300、8301、8310、8313、8314、8315、8316、8317、8321、8322、8323、8351、8361、8362、8369、8371、8380、8391 和 8392，
共 19 个进程；8300 使用 `model-audio/clone_qwen3-tts-1.7b-base` uv 项目中的轻量 HTTP 依赖，其余服务使用
各自的 uv 项目。启动命令统一使用 `uv run --no-sync`，不会在运行阶段联网解析依赖；
本地 GPU 服务通过 `GPU_LOCK_FILE` 串行访问 GPU。默认最多排队 900 秒，超过时返回
`503`；用正数 `GPU_LOCK_WAIT_TIMEOUT` 调整。共享锁实现仍兼容非正值无限等待，
但 整套部署应保持正数。持锁期间用 `nvidia-smi` 采样，
`peak_vram_mib` 是所有可见 GPU 的已用显存总和峰值，并非某个 worker 的独立分配量。
任一子进程退出时脚本会终止其余进程组并清理 worker。
启动前会检查全部服务的监听地址；端口已被占用或配置相互冲突时，列出服务名和地址后退出。
更新代码后重新启动，应先在旧 `start.sh` 终端按 `Ctrl+C`，等待旧进程退出，再执行启动命令。
可用 `ss -lntp` 查看占用端口的进程。

#### LocalAI 裸机试接入

LocalAI 试接入只使用宿主机 Linux 二进制和宿主机子进程，不部署 Docker 容器。当前固定
LocalAI `v4.11.0` 源码 commit 与其内嵌 audio.cpp commit 见
[`localai/manifests/versions.json`](localai/manifests/versions.json)。在仓库外准备兼容的
LocalAI Linux amd64 二进制，以及固定版本的 CUDA 12 `audio-cpp` backend 包；将二进制路径设置到
`LLM_LOCALAI_BINARY`，backend 放在 `LLM_LOCALAI_BACKENDS_DIR/audio-cpp/run.sh`。
后端构建应使用 LocalAI v4.11.0 仓库内的 `backend/cpp/audio-cpp/Makefile` 和固定
`AUDIO_CPP_VERSION`，先执行 `grpc-server`、再执行 `package` 目标；具体编译产物与部署路径须按该
固定版本的 `package.sh` 核验。不要使用滚动版本的 Gallery 包或将滚动 backend 包作为运行时。

管理器只校验本机二进制报告 `v4.11.0`、backend 启动入口及整棵 backend 文件树 hash，不下载或编译第三方源码：

```bash
export LLM_LOCALAI_ENABLED=true
export LLM_LOCALAI_BINARY="$HOME/.local/opt/localai/bin/local-ai"
export LLM_LOCALAI_BACKENDS_DIR="${STORAGE_DIR:-storage}/.cache/localai/backends"
python3 scripts/localai/manager.py verify-install
export LLM_LOCALAI_MANAGED=true
# Qwen3-TTS Base 克隆可单独启用；启用前需手动准备模型目录并注册候选 YAML。
# export LLM_LOCAL_QWEN3_TTS_PROVIDER=localai
# Qwen3 VoiceDesign 使用独立 task:vdes 模型候选。
# export LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER=localai
export LLM_LOCALAI_BASE_URL=http://127.0.0.1:8400
bash start.sh
```

本机数据目录默认为 `${STORAGE_DIR}/.cache/localai`；模型配置/权重目录默认为
`${LLM_LOCALAI_DATA_DIR}/models`。`start.sh` 启动前检查端口，随后以宿主机进程组启动 LocalAI、
等待 `/readyz`，再验证显式选择的 Qwen3-TTS Base/VoiceDesign 候选权重、配置和模型注册。运行进程只绑定 loopback、最多加载一个
backend、关闭自动 Gallery 加载及 WebUI。管理器将 boot ID、PID、进程启动 tick 和独立运行标识写入状态文件；
停止时发出 TERM，超时后升级为 KILL，并在确认进程组退出且端口关闭后才删除状态。由 `start.sh`
启动时还会启动 watcher，在父 shell 异常退出后回收所属进程组。关闭 LLM Local Backend 启动组时，先停止业务
进程，再停止受管 LocalAI 进程组。

Qwen3-TTS Base 与 VoiceDesign 注册命令检查各自官方模型目录所需文件，
并在 LocalAI models 目录内建立 `<model-id>.hf` 符号链接；YAML 使用该相对名称，
verify 同时检查链接仍指向配置的完整本地快照，避免 LocalAI 拒绝绝对 `parameters.model`。
Base 的会话选项使用 `session.qwen3_tts.voice_prompt_cache_slots:0`。
VoiceDesign 使用 `QWEN_VOICEDESIGN_MODEL_DIR` 和
`python3 scripts/localai/manager.py register-qwen3-voicedesign-candidate`，固定 YAML 使用 `family:qwen3_tts`、
`task:vdes`，不与 Base 克隆的 `task:tts` 混用。这些命令只准备候选，不代表模型可运行或质量达标。LocalAI
固定版本 LocalAI 二进制和 audio.cpp CUDA gRPC backend 已在本机编译、打包、安装并通过无模型启停检查；
P7 worker bridge 已接入五个服务项目（五个固定 model ID）。2026-10-22 两个 Qwen 直接候选各完成一次
24 kHz 短句真实推理与 shutdown，尚未完成听感、相似度和长文本验收，见
[升级记录](docs/localai-audiocpp-upgrade.md#7-2026-10-22-复核修复与当前验证)。原生实现仍是默认。

#### LocalAI 一次性 worker bridge（P7）

LocalAI v4.11.0 通过 `/v1/completions` 接收版本化的 `llm-local.worker` v1 envelope，再转发到
`llm-local-worker` gRPC backend。backend 只允许 `localai/manifests/worker-bridge.json` 中登记的模型 ID、
项目目录和 `worker.py` 入口；每个请求使用对应 uv 项目启动一次性 worker，返回 JSON 或白名单文件类型。
LocalAI 与 gRPC bridge 默认只监听 `127.0.0.1`。受管启动会将 LocalAI 与 bridge 放进同一托管宿主机
进程组，停止时核对进程组、端口并回收带请求标记的遗留 worker。

当前接入的文件输出合同为：SheetSage2 ZIP；MERT embedding JSON / frame features NPZ；YuE2-Vae
encode NPY / decode WAV；YuE2-3B WAV；TIGER-DnR 三 Stem ZIP。五个 worker 的文件成品均通过随机
`file_ref` ID 从共享目录流式取回，单文件上限 4 GiB；服务与 bridge 须使用相同的
`LLM_LOCALAI_ARTIFACT_DIR`（默认 `$RUNTIME_CACHE_DIR/localai_artifacts`）。
取回后立即删除共享产物；未消费文件在下次发布时清理超过 24 小时的项目。
大文件不进入 gRPC Reply，避免固定 LocalAI 50 MiB 接收上限。
以下命令只登记和核验
不含权重路径的 bridge YAML，不下载模型或权重：

bridge 会在成功、失败、取消和超时后确认整个 worker 进程组退出；Free 复核未回收的组。
无法确认时保留 GPU 隔离标记，即便 LocalAI shutdown 返回 HTTP 200 也不会清除。
恢复这类标记须启用受管模式，由恢复脚本停止宿主机并复核 worker、进程组和监听端口。

```bash
python3 scripts/localai/manager.py register-worker-bridge-candidates
python3 scripts/localai/manager.py verify-worker-bridge-candidates
```

已有固定版本的本机 LocalAI 安装后，可独立运行无模型大文件检查：

```bash
uv run --project qa --locked python scripts/localai/smoke_worker_bridge.py
```

该命令使用临时配置和 QA 假 worker 生成 41 MiB、129 MiB 静音 WAV，经真实 LocalAI HTTP/gRPC
链路验证取回、内容、产物清理和停止。它不属于无需 LocalAI 安装的默认质量门禁。

在 `start.sh` 中受管运行时启用 bridge，并显式选择要切换的服务；各服务默认仍为 `native`：

```bash
export LLM_LOCALAI_ENABLED=true
export LLM_LOCALAI_MANAGED=true
export LLM_LOCALAI_WORKER_BRIDGE_ENABLED=true
export LLM_LOCAL_SHEETSAGE2_PROVIDER=localai
export LLM_LOCAL_MERT_V2_FULLSONG_PROVIDER=localai
export LLM_LOCAL_YUE2_VAE_PROVIDER=localai
export LLM_LOCAL_YUE2_3B_PROVIDER=localai
export LLM_LOCAL_TIGER_DNR_PROVIDER=localai
```

对应模型 ID 默认值见 `start.sh` 和 `localai/manifests/worker-bridge.json`。若 LocalAI 由其他进程托管，
需单独以 `uv run --project localai_worker_bridge --locked python -m localai_worker_bridge.server`
启动 bridge，并在 LocalAI 中配置 `LOCALAI_EXTERNAL_GRPC_BACKENDS=llm-local-worker:127.0.0.1:50052`。

这些接入已有 mock 路由与协议测试，状态为 `implemented_unverified`；尚未使用真实权重执行推理，
因此不能据此判断音频/视频质量、模型资源峰值或 E2/E3。`LLM_LOCALAI_API_KEY` 可从环境提供内部鉴权。
LocalAI 地址必须是本机 HTTP 回环地址。健康响应的 `selected_provider_ready` 会同时检查 LocalAI 健康状态
和所选候选的模型注册状态。

LocalAI 请求持有共享 `GPU_LOCK_FILE`，完成后请求 `/backend/shutdown`。backend 停止无法确认时，网关在锁文件旁
留下 `.quarantine` 标记，所有 GPU 服务都会拒绝新的推理准入。修复或停止后端后，可用
`LLM_LOCALAI_ENABLED=true` 运行 `scripts/localai/recover_quarantine.py`；该脚本持有 GPU 锁，先要求 LocalAI
确认停止记录中的模型。若 API 不可用且 `LLM_LOCALAI_MANAGED=true`，脚本停止受管宿主机进程组，并确认进程组退出、
状态文件消失且 loopback 端口关闭，再清除隔离标记。不要在模型仍运行时手动删除隔离标记。

控制面诊断与 HTTP 健康检查：

```bash
curl -fsS http://127.0.0.1:8300/v1/control
for port in 8300 8301 8313 8314 8315 8316 8317 8321 8322 8323 8351 8361 8371; do
  curl -fsS "http://127.0.0.1:${port}/v1/health" >/dev/null && echo "${port}: ok"
done
```

单独调试服务时，从仓库根目录执行，例如：

```bash
HOST=127.0.0.1 PORT=8321 \
  uv run --no-sync --project model-audio/clone_qwen3-tts-1.7b-base python model-audio/clone_qwen3-tts-1.7b-base/main.py
```

统一脚本没有按模型启停的开关；只部署部分模型时，准备并直接启动相应目录。
## 模型路径与主要配置

`start.sh` 默认使用 `HF_MIRROR_DIR`（默认为 `$HOME/hf-mirror`）和 `$HOME/LLM-depency`；
模型和运行数据路径可在启动前用环境变量覆盖。

| 服务 | 默认权重 | 权重覆盖变量与额外配置 |
| --- | --- | --- |
| Qwen3-TTS Base | `$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `QWEN3_TTS_MODEL_DIR` |
| IndexTTS-2.5 | `$HF_MIRROR_DIR/IndexTeam/IndexTTS-2.5` | `INDEXTTS25_MODEL_DIR`、`INDEXTTS25_W2V_BERT_MODEL_DIR`、`INDEXTTS25_CAMPPLUS_PATH`、`INDEXTTS25_BIGVGAN_MODEL_DIR` |
| Qwen VoiceDesign | `$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `QWEN_VOICEDESIGN_MODEL_DIR` |
| Confucius4-TTS | `$HF_MIRROR_DIR/netease-youdao/Confucius4-TTS` | `CONFUCIUS4_TTS_MODEL_DIR`、`CONFUCIUS4_TTS_CODE_PATH`、`CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR`、`CONFUCIUS4_TTS_VOCODER_MODEL_DIR`、`CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT` |
| Qwen3-ASR-1.7B | `$HF_MIRROR_DIR/Qwen/Qwen3-ASR-1.7B` | `QWEN3_ASR_MODEL_DIR` |
| ACE-Step 1.5 XL Turbo | `$HF_MIRROR_DIR/ACE-Step/acestep-v15-xl-turbo-diffusers` | `ACESTEP_MODEL_DIR`、`ACESTEP_OFFLOAD`、`ACESTEP_VAE_TILING` |
| YuE2-3B | `$HF_MIRROR_DIR/m-a-p/YuE2-3B` 与 `$HF_MIRROR_DIR/m-a-p/YuE2-Vae` | `YUE2_3B_MODEL_DIR`、`YUE2_VAE_MODEL_DIR`、`YUE2_3B_MEMORY_BUDGET_GIB`、`YUE2_3B_REQUEST_TIMEOUT` |
| YuE2-Vae | `$HF_MIRROR_DIR/m-a-p/YuE2-Vae` | `YUE2_VAE_MODEL_DIR`、`YUE2_VAE_OUTPUT_DIR`、`YUE2_VAE_DEVICE`、`YUE2_VAE_REQUEST_TIMEOUT`、`YUE2_VAE_MAX_AUDIO_SECONDS` |
| SheetSage2 | `$HF_MIRROR_DIR/m-a-p/SheetSage2` 与 `$HF_MIRROR_DIR/m-a-p/MERT-v2-FullSong` | `SHEETSAGE2_MODEL_DIR`、`SHEETSAGE2_BASE_MODEL_DIR`、`SHEETSAGE2_OUTPUT_DIR`、`SHEETSAGE2_DEVICE`、`SHEETSAGE2_DTYPE`、`SHEETSAGE2_PRESET`、`SHEETSAGE2_REQUEST_TIMEOUT` |
| MERT-v2-FullSong | `$HF_MIRROR_DIR/m-a-p/MERT-v2-FullSong` | `MERT_V2_FULLSONG_MODEL_DIR`、`MERT_V2_FULLSONG_OUTPUT_DIR`、`MERT_V2_FULLSONG_DEVICE`、`MERT_V2_FULLSONG_REQUEST_TIMEOUT` |
| MOSS-SoundEffect v2.0 | `$HF_MIRROR_DIR/OpenMOSS/MOSS-SoundEffect-v2.0` | `MOSS_SOUNDEFFECT_V2_MODEL_DIR`、`MOSS_SOUNDEFFECT_V2_CODE_PATH`、`MOSS_SOUNDEFFECT_V2_OUTPUT_DIR` |
| VoxCPM2 | `$HF_MIRROR_DIR/openbmb/VoxCPM2` | `VOXCPM2_MODEL_DIR`、仓库内 `model-audio/clone_voxcpm-2/voxcpm2_helpers.py` |
| TIGER-DnR | `$HF_MIRROR_DIR/JusperLee/TIGER-DnR` | `TIGER_DNR_MODEL_DIR`、`TIGER_DNR_SOURCE_DIR` |
| Kandinsky 6 Lite 蒸馏版 | `$HF_MIRROR_DIR/kandinskylab` | `KANDINSKY6_MODEL_DIR` |
| MiniMax H3 16G HiFi | `$HF_MIRROR_DIR/minimax/MiniMax-H3-16G-HiFi` | `MINIMAX_H3_MODEL_DIR` |

仓库外推理源码的默认位置如下，须在请求前准备；`start.sh` 不负责拉取源码：

| 服务 | 外部源码默认路径 | 覆盖变量 |
| --- | --- | --- |
| Confucius4-TTS | `$HOME/LLM-depency/Confucius4-TTS` | `CONFUCIUS4_TTS_CODE_PATH` |
| IndexTTS-2.5 | `$HOME/LLM-depency/index-tts-2.5.0`（tag `v2.5.0`） | `INDEXTTS25_CODE_PATH` |
| TIGER-DnR | `$HOME/.local/share/tiger-dnr/TIGER` | `TIGER_DNR_SOURCE_DIR` |
| MOSS-SoundEffect v2.0 | `$HOME/LLM-depency/MOSS-TTS/moss_soundeffect_v2`（固定 revision `934d6826b084c46a0d033402174d5f8ac4ed2519`） | `MOSS_SOUNDEFFECT_V2_CODE_PATH` |
| MiniMax H3 ComfyUI | `$HOME/LLM-depency/MiniMax-H3-ComfyUI`（ComfyUI `v0.37.0`） | `MINIMAX_H3_COMFYUI_CODE_PATH`、`MINIMAX_H3_COMFYUI_PYTHON` |

通用配置包括 `HOST`、`PORT`、`STORAGE_DIR`、`PROMPTS_DIR`、`RUNTIME_CACHE_DIR`、
`GPU_LOCK_FILE`、`LOCAL_FILES_ONLY` 和 `CUDA_RELEASE_DELAY`。48 kHz 导出可用
`SPATIAL_EXPORT_CACHE_DIR`、`SPATIAL_EXPORT_MAX_BYTES`、`SPATIAL_EXPORT_TIMEOUT` 和
`SPATIAL_EXPORT_FFMPEG_BIN` 覆盖旧总线母带的缓存目录、上传上限、处理超时和 FFmpeg 命令。
正式对象级导出使用 `STEAM_AUDIO_RENDERER_BIN`、`STEAM_AUDIO_SDK_DIR`、
`STEAM_AUDIO_HRTF_PATH`、`STEAM_AUDIO_RENDER_CACHE_DIR`、`STEAM_AUDIO_RENDER_TIMEOUT`、
`STEAM_AUDIO_RENDER_MAX_ASSETS`、`STEAM_AUDIO_RENDER_MAX_MANIFEST_BYTES`、
`STEAM_AUDIO_RENDER_MAX_BYTES` 和
`STEAM_AUDIO_RENDER_THREADS`。服务专用配置使用对应前缀，例如 `QWEN3_TTS_*`、
`QWEN3_ASR_*`、`VOXCPM2_*`、`ACESTEP_*`、`QWEN_VOICEDESIGN_*` 和
`CONFUCIUS4_TTS_*`。
每个服务的 `/v1/health` 会报告生效的路径、运行时和可用性。

重要共享配置如下；运行数据路径默认位于 `STORAGE_DIR`（仓库内 `storage/`）下：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HF_MIRROR_DIR` | `$HOME/hf-mirror` | 本地权重根目录 |
| `HOST`、`PORT` | `0.0.0.0`、`8300` | 统一启动时的控制面地址；其他服务有独立端口 |
| `PROMPTS_DIR` | `$CLONE_STORAGE_DIR` | 普通参考音频和 sidecar；多服务须共享该目录才能复用逻辑标识 |
| `UPLOAD_MAX_BYTES` | `67108864` | 参考音频、转写与分离输入的单文件上限 |
| `GPU_LOCK_FILE` | `$RUNTIME_CACHE_DIR/gpu-runtime.lock` | 本地模型共用排他锁 |
| `GPU_LOCK_WAIT_TIMEOUT` | `900` 秒 | GPU 排队时限；整套启动须为正数 |
| `LLM_LOCALAI_ENABLED` | `false` | 显式允许内部 LocalAI provider |
| `LLM_LOCALAI_MANAGED` | `false` | `start.sh` 是否启动并负责停止受管 LocalAI 宿主机进程组 |
| `LLM_LOCALAI_BASE_URL` | `http://127.0.0.1:8400` | 必须绑定本机回环地址 |
| `LLM_LOCALAI_DATA_DIR` | `$STORAGE_DIR/.cache/localai` | 本机模型、状态、日志与安装指纹目录 |
| `LLM_LOCALAI_BINARY` | `$LLM_LOCALAI_DATA_DIR/bin/local-ai` | 固定版本的宿主机 LocalAI 可执行文件 |
| `LLM_LOCALAI_BACKENDS_DIR` | `$LLM_LOCALAI_DATA_DIR/backends` | 宿主机 audio-cpp backend 包目录 |
| `LLM_LOCALAI_MODELS_DIR` | `$LLM_LOCALAI_DATA_DIR/models` | LocalAI 的模型配置与权重目录 |
| `LLM_LOCALAI_WORKER_BRIDGE_ENABLED` | `false` | 受管 LocalAI 启动时一并启动 loopback `llm-local-worker` gRPC backend |
| `LLM_LOCALAI_BRIDGE_ADDRESS` | `127.0.0.1:50052` | 外部 worker bridge 的 gRPC 回环监听地址 |
| `LLM_LOCAL_INFERENCE_PROVIDER` | `native` | 服务 provider 缺省值；原生实现仍为默认 |
| `LLM_LOCAL_QWEN3_TTS_PROVIDER` | `native` | Qwen3-TTS Base 克隆可单独选 `native` 或 `localai` |
| `LLM_LOCAL_QWEN3_TTS_MODEL` | `llm-local.qwen3-tts.1p7b-base` | LocalAI Qwen3-TTS Base 模型 ID |
| `LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER` | `native` | Qwen3 VoiceDesign 可单独选 `native` 或 `localai` |
| `LLM_LOCAL_QWEN3_VOICEDESIGN_MODEL` | `llm-local.qwen3-tts.1p7b-voicedesign` | LocalAI Qwen3-TTS VoiceDesign 模型 ID；YAML 固定为 `task:vdes` |
| `LLM_LOCAL_SHEETSAGE2_PROVIDER` / `LLM_LOCAL_SHEETSAGE2_MODEL` | `native` / `llm-local.sheetsage2.worker.v1` | SheetSage2 one-shot worker bridge；native 默认 |
| `LLM_LOCAL_MERT_V2_FULLSONG_PROVIDER` / `LLM_LOCAL_MERT_V2_FULLSONG_MODEL` | `native` / `llm-local.mert-v2-fullsong.worker.v1` | MERT 整曲/逐帧 worker bridge；native 默认 |
| `LLM_LOCAL_YUE2_VAE_PROVIDER` / `LLM_LOCAL_YUE2_VAE_MODEL` | `native` / `llm-local.yue2-vae.worker.v1` | YuE2-Vae encode/decode worker bridge；native 默认 |
| `LLM_LOCAL_YUE2_3B_PROVIDER` / `LLM_LOCAL_YUE2_3B_MODEL` | `native` / `llm-local.yue2-3b.worker.v1` | YuE2-3B WAV worker bridge；native 默认 |
| `LLM_LOCAL_TIGER_DNR_PROVIDER` / `LLM_LOCAL_TIGER_DNR_MODEL` | `native` / `llm-local.tiger-dnr.worker.v1` | TIGER-DnR three-stem ZIP file-reference bridge；native 默认 |
| `LLM_LOCALAI_CONNECT_TIMEOUT` | `5` 秒 | LocalAI 连接超时 |
| `LLM_LOCALAI_CLEANUP_TIMEOUT` | `30` 秒 | backend 停止请求的有界超时 |
| `LLM_LOCALAI_API_KEY` | 无 | 可选内部 API key，只从环境读取 |
| `GPU_METRICS_SAMPLE_INTERVAL` | `0.5` 秒 | 显存采样间隔，实际不低于 0.1 秒 |
| `CUDA_RELEASE_DELAY` | `2.0` 秒 | worker 退出后持锁等待显存回收 |
| `LOCAL_FILES_ONLY` | `1` | 本地模型离线加载 |
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
Qwen VoiceDesign 使用 `QWEN_VOICEDESIGN_HOST` / `QWEN_VOICEDESIGN_PORT`。
`*_PROJECT_DIR` 用于覆盖启动脚本中的 uv 项目路径；直接启动服务时使用该服务实际读取的
环境变量。省略 JSON 推理字段时，生效值由模块默认值和显式环境配置决定；
`start.sh` 当前也导出部分音效与 ACE-Step 默认参数，修改模块后需同时核对脚本。

TIGER-DnR 的官方推理代码不包含在本仓库。默认使用已准备的
`$HOME/.local/share/tiger-dnr/TIGER`；也可以通过 `TIGER_DNR_SOURCE_DIR` 指向作者的
`JusperLee/TIGER` 克隆。worker 只使用其中的 `look2hear` DnR 模型代码，并在
`LOCAL_FILES_ONLY=1` 下从本地 `config.json` 与 `model.safetensors` 加载权重。

## Confucius4-TTS 零样本克隆

Confucius4-TTS 的官方推理代码不包含在本仓库。默认使用
`$HOME/LLM-depency/Confucius4-TTS`，模型目录默认读取
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

## IndexTTS-2.5 零样本克隆

IndexTTS-2.5 在 `8323` 提供 `POST /v1/index-team/indexTTS-2.5`。服务使用官方 `v2.5.0`
推理源码（仓库外）以及本地 IndexTTS、Wav2Vec2-BERT、CAMPPlus 和 BigVGAN 权重；路径和
覆盖变量见[模型路径表](#模型路径与主要配置)与[服务 README](model-audio/indexTTS-2.5/README.md)。
服务固定使用独立 Python `3.12.13` / CUDA 12.8 uv 环境。上游打包元数据限定 Python `<3.12`，
因此本项目从已检出的源码导入推理入口，不安装上游包；API 会在离线模式下运行一次性 worker。
尚未准备的源码或资源会在健康检查中显示为 unavailable，请求预检返回 `503`。

先通过共享接口上传参考音频，再用返回的逻辑 `audio_path` 调用生成路由。成功返回 `audio/wav`，
并原子保存到 `storage/clone/`；`INDEXTTS25_OUTPUT_DIR` 可覆盖输出目录。

```bash
curl -fsS http://127.0.0.1:8323/v1/upload_audio \
  -F 'audio=@reference.wav;type=audio/wav' -F 'full_path=reference.wav'
curl -fsS http://127.0.0.1:8323/v1/index-team/indexTTS-2.5 \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，欢迎使用。","audio_path":"reference.wav","lang":"ZH"}' \
  -o index-tts-25.wav
```

`lang` 支持 `ZH`、`EN`、`JA`、`ES`、`AR`；`duration_factor` 范围为 0.5–2.0。可选
`emo_vector` 是 8 个 `[0, 1.2]` 数值，顺序为 happy、angry、sad、afraid、disgusted、
melancholic、surprised、calm；`emo_alpha` 范围为 0–1。情感文本模型未启用。

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
  -o llm-local-standard.wav
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
  -o llm-local-steam-balanced.wav
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

Steam Audio SDK 不提交到仓库。下载并解压 Valve 官方 SDK 后，启动和 Linux 构建脚本默认从
`$HOME/.local/share/steam-audio/4.8.1` 读取 SDK，不需要每次启动时设置环境变量；其他安装位置
可通过 `STEAM_AUDIO_SDK_DIR` 覆盖。启动脚本不会下载或编译第三方代码。
SDK、动态库和 HRTF 的使用与分发须遵守 SDK 随附许可证，本仓库的 Apache-2.0 许可证不替代第三方许可：

```bash
bash scripts/build_steam_audio_renderer.sh
```

SDK 安装在其他位置时，可仅为构建命令覆盖路径：

```bash
STEAM_AUDIO_SDK_DIR=/opt/steam-audio bash scripts/build_steam_audio_renderer.sh
```

Windows PowerShell 先设置 `$env:STEAM_AUDIO_SDK_DIR = "C:\\steam-audio"`，再运行
`.\scripts\build_steam_audio_renderer.ps1`。默认可执行文件为
`model-audio/spatial_steam-audio/build/steam-audio-render`（Windows 为
`model-audio/spatial_steam-audio/build/Release/steam-audio-render.exe`）。可选
`STEAM_AUDIO_HRTF_PATH` 指向 SOFA 文件；不设置时使用 SDK 内置 HRTF。`start.sh` 若发现可执行
文件缺失会告警但仍启动既有服务，正式路由返回 `503`；`GET /v1/control` 的
`spatial_renderer` 会报告 executable、SDK header/library、HRTF 和 GPU-lock 状态。

## 参考音频克隆

Qwen3-TTS、VoxCPM2 和 Confucius4-TTS 使用共享的上传、检查与逻辑引用流程：

1. `POST /v1/upload_audio`，表单字段为 `audio`、`full_path`，可选 `prompt_text`。
2. `GET /v1/check/audio?file_name=...` 检查已上传状态。
3. 调用克隆接口，请求中的 `audio_path` 使用上传时完全相同的 `full_path` 标识。

`full_path` 是客户端逻辑标识，不能用于直接读取服务器任意路径。共享音频策略支持
WAV/MP3/OGG/FLAC/M4A/AAC/WebM，分块暂存，默认上限为 64 MiB。成功 JSON 返回逻辑文件名、
SHA-256、大小和存储位置。检查接口应读取 `exists`/`code`，不能只用 HTTP 状态判断文件存在。

Qwen3-TTS 示例：

```bash
curl -X POST http://127.0.0.1:8321/v1/upload_audio \
  -F 'audio=@reference.wav' -F 'full_path=reference.wav' \
  -F 'prompt_text=这是一句参考音频转写。'
curl 'http://127.0.0.1:8321/v1/check/audio?file_name=reference.wav'
curl -X POST http://127.0.0.1:8321/v1/qwen/clone \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，欢迎使用。","audio_path":"reference.wav","prompt_text":"这是一句参考音频转写。"}' \
  -o qwen-clone.wav
```

Qwen3-TTS 可用准确参考文本或 `x_vector_only=true` 进行克隆。VoxCPM2 的
`clone_mode=controllable` 使用 `control_instruction`，且与参考转写互斥；`nonverbal_tags`
最多一个白名单标签。Confucius4-TTS 默认 `lang=zh`，不将上传时保存的参考文本 sidecar 送入推理。
三个接口都拒绝 `style_prompt`。具体字段与状态码以各服务 `/docs` 和 README 为准。

## 音色设计

Qwen VoiceDesign 在 `8301` 提供 `POST /v1/qwen/timbre`。请求包含 `voice_description` 和可选
`text`，成功返回 WAV 并将音色参考写入 `storage/timbre/`：

```bash
curl -X POST http://127.0.0.1:8301/v1/qwen/timbre \
  -H 'Content-Type: application/json' \
  -d '{"voice_description":"成年女性，声音清晰自然，语速中等。","text":"你好。"}' \
  -o qwen-voice.wav
```

`language`、`max_chars_per_chunk`、`pause_ms`、`max_new_tokens`、`top_p`、`temperature`、
`dtype`、`attn_implementation` 和 `device_map` 可按接口约束设置。其他模型的音色设计入口已退役；
VoxCPM2 保留的 `control_instruction` 属于参考音频可控克隆字段。



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

## MOSS-SoundEffect v2.0 音效生成

MOSS-SoundEffect v2.0 默认监听 `8362`，路由为 `POST /v1/moss/soundEffect-v2.0`。
服务接收 JSON 提示词与采样参数，按本地模型快照生成 48 kHz 单声道 WAV，并原子保存到
`storage/sound_effect_v2/`。默认时长为 10 秒，最长 30 秒；每个请求通过共享 GPU 锁启动并退出独立 worker。

官方源码不复制进仓库。先在仓库外准备固定 revision，再同步独立 uv 环境：

```bash
mkdir -p "$HOME/LLM-depency/MOSS-TTS"
git -C "$HOME/LLM-depency/MOSS-TTS" init
git -C "$HOME/LLM-depency/MOSS-TTS" remote add origin https://github.com/OpenMOSS/MOSS-TTS.git
git -C "$HOME/LLM-depency/MOSS-TTS" fetch --filter=blob:none --depth=1 origin 934d6826b084c46a0d033402174d5f8ac4ed2519
git -C "$HOME/LLM-depency/MOSS-TTS" sparse-checkout init --cone
git -C "$HOME/LLM-depency/MOSS-TTS" sparse-checkout set moss_soundeffect_v2
git -C "$HOME/LLM-depency/MOSS-TTS" checkout --detach FETCH_HEAD
uv sync --project model-audio/soundEffect-v2.0 --locked
```

模型默认目录为 `$HF_MIRROR_DIR/OpenMOSS/MOSS-SoundEffect-v2.0`。健康检查只核对四个权重文件的完整大小、
配置文件、依赖、官方源码和 GPU；不会加载模型。请求示例：

```bash
curl -fsS http://127.0.0.1:8362/v1/moss/soundEffect-v2.0 \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Rain falling steadily on a metal roof","seconds":8,"seed":42}' \
  -o sound-effect.wav
```

字段边界和配置见 [MOSS-SoundEffect 服务说明](model-audio/soundEffect-v2.0/README.md)。

## MOSS-Music 音乐理解

MOSS-Music-8B-Instruct 默认监听 `8310`，业务接口为 `POST /v1/openMoss/moss-music`。
`bash start.sh` 会启动该服务，并在“模型接口与端口映射”中显示它；设置 `MOSS_MUSIC_PORT`
可以覆盖默认端口，启动预检、服务监听和映射表会使用同一个配置值。

请求使用 `multipart/form-data`，上传音乐并按 `prompt` 提问：

```bash
curl -fsS http://127.0.0.1:8310/v1/openMoss/moss-music \
  -F 'audio=@song.wav' \
  -F 'prompt=请分析这段音乐的乐器、人声与情绪。' \
  -F 'max_new_tokens=1024' \
  -F 'temperature=0'
```

`audio` 必填，默认最大 64 MiB、时长不超过 420 秒；`prompt` 可省略，默认进行中文综合分析。
成功返回 JSON，包含 `model`、`runtime`、`text`、`duration_seconds` 和 `audio_sha256`。
音频及推理中间文件仅暂存，请求结束后清理。健康检查为 `GET http://127.0.0.1:8310/v1/health`。
完整构建与配置说明见 [MOSS-Music 服务文档](model-audio/analyse_moss-music-8b-Instruct/README.md)。

## ACE-Step BGM 生成

ACE-Step 独立监听 `8313`，是当前保留的音乐生成服务。
服务默认使用 BF16、model CPU
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
uv sync --project model-audio/music_ace-step-1.5 --locked
```

本仓库的 `start.sh` 对 ACE-Step 使用 `uv run --no-sync`，不会在启动时安装依赖。

## YuE2-3B 歌曲生成

YuE2-3B 在 `8314` 提供官方文本到音乐管线，输入风格描述和歌词，可选使用 `cot` 选择
完整和弦规划、旋律规划或关闭规划，也可提供 ABC 乐谱和原生采样参数。服务默认使用
`m-a-p/YuE2-3B` 与配套 `m-a-p/YuE2-Vae` 本地目录，生成 48 kHz 立体声 WAV，保存到
`storage/bgm/`，并在 HTTP 响应中返回同一 WAV。

```bash
curl -X POST http://127.0.0.1:8314/v1/m-a-p/YuE2 \
  -H 'Content-Type: application/json' \
  -d '{"style":"Mandarin funk, nu-disco, warm female lead vocal, bright bass and tight drums","lyrics":"[Verse 1]\n夜色落在窗台\n城市还没有睡\n\n[Chorus]\n今晚不眠 一起追着星光飞","cot":"full","seed":831001}' \
  -o yue2-song.wav
```

`style` 或其原生别名 `tags` 至少提供一项，`lyrics` 必填。`cot` 可取 `off`、`melody`、
`full`；`seed` 默认 `831001`；可选 `abc`、`cfg_scale`、`id`、`abc_sampling` 和
`semantic_sampling` 都映射到官方 pipeline 参数。采样对象只需传要覆盖的字段，未提供的
值继续使用 YuE2 官方默认配置。成功响应包含 `X-YuE2-3B-Seed`、
`X-YuE2-3B-Sample-Rate` 和 `X-YuE2-3B-Model`。健康接口不会加载模型；当依赖、MOT、VAE
或 CUDA 缺失时，`ready` 为 false，生成请求返回 `503`。

先同步该项目锁定环境，再准备模型文件：

```bash
uv sync --project model-audio/music_yue2-3b --locked
```

运行依赖固定 PyTorch `2.10.0` 和 Transformers `4.57.6`。官方 `yue2_infer` `0.1.5` wheel
从主模型目录加载，不需复制到仓库。主模型默认放在
`$HF_MIRROR_DIR/m-a-p/YuE2-3B`，VAE 默认放在 `$HF_MIRROR_DIR/m-a-p/YuE2-Vae`；
服务默认离线加载，请求不会下载权重。官方资源说明建议使用 24 GB GPU 和 24 GB 可用主机
内存，单次只生成一首歌。YuE2 使用共享 GPU 文件锁和一次性 worker；模型在每个请求中重新
加载，默认请求超时为 1800 秒，GPU 排队超时沿用正数 `GPU_LOCK_WAIT_TIMEOUT`。
本机显存较小时可通过 `YUE2_3B_MEMORY_BUDGET_GIB` 限制 worker 显存预算，但实际可生成的
歌曲长度仍取决于空闲显存。

## SheetSage2 音乐转谱

SheetSage2 在 `8315` 接收 multipart 音频，转写完整歌曲的旋律、和弦、节拍、调性和结构，
返回 `application/zip`，其中包含可用时的 ABC、MIDI、`events.json`、LAB、`result.json` 及模型实际
生成的其他原生文件。若 ABC 无法构建，`result.json` 会写入 `abc_error`，MIDI 和标注仍会返回。
可用 `dtype=bf16|fp32`、`preset=default|paper`、`max_seconds`、
`overlap_seconds`、`lookahead_seconds`、重复 `prompts`、`melody_only` 与特征导出开关控制
转写。`paper` preset 固定窗口参数为 `overlap_seconds=100`、`lookahead_seconds=0`。

```bash
curl -X POST http://127.0.0.1:8315/v1/m-a-p/SheetSage2 \
  -F 'audio=@song.mp3' \
  -F 'dtype=bf16' \
  -F 'preset=default' \
  -o sheetsage2-result.zip
```

模型代码和权重由 Hugging Face SheetSage2 本地快照提供，不复制到本仓库。SheetSage2
adapter 还要求配置指定 commit 的 MERT-v2-FullSong 权重和代码，加载时会按官方实现校验
父模型文件摘要。默认目录分别为 `$HF_MIRROR_DIR/m-a-p/SheetSage2` 与
`$HF_MIRROR_DIR/m-a-p/MERT-v2-FullSong`；`SHEETSAGE2_MODEL_DIR` 和
`SHEETSAGE2_BASE_MODEL_DIR` 可覆盖路径。上传经共享暂存器按块校验，单次 worker 退出后释放
GPU；请求音频临时文件会删除，生成 ZIP 保留在 `SHEETSAGE2_OUTPUT_DIR`，当前保留策略工具
不会自动清理这些 ZIP。

SheetSage2 固定 `pretty-midi==0.2.10`，其运行时依赖 `pkg_resources`，因此同时固定
`setuptools==80.10.2`。更新环境时使用 `uv sync --project model-audio/score_sheetsage-2 --locked`，
避免仅安装新版 setuptools 导致转谱阶段出现 `No module named 'pkg_resources'`。
健康检查会报告该传递依赖是否可用；更改服务代码后需重启对应 HTTP 进程使检查生效。

## YuE2-Vae latent 编码与解码

YuE2-Vae 在 `8316` 提供官方 VAE 编码和解码能力。解码接收 YuE2 acoustic latent 的 NPY 文件，
形状为 `[T,64]`，输出 48 kHz 立体声 PCM_24 WAV；编码接收音频，FFmpeg 先转为 48 kHz 双声道，
再输出 `[T,64]` float32 NPY。编码默认使用 posterior mean，可选 `sample=true` 和 `seed`
进行 posterior sampling。输入文件按块暂存，模型在一次性 worker 中加载，成品原子保存至
`storage/yue2_vae/`；该目录当前不由存储保留策略自动清理。

解码已有 YuE2 `song.save_artifacts(...)` 生成的 `latent.npy`：

```bash
curl -X POST http://127.0.0.1:8316/v1/m-a-p/YuE2-Vae \
  -F 'operation=decode' \
  -F 'latent=@outputs/song/latent.npy' \
  -o decoded.wav
```

编码输入音频：

```bash
curl -X POST http://127.0.0.1:8316/v1/m-a-p/YuE2-Vae \
  -F 'operation=encode' \
  -F 'audio=@song.wav' \
  -o song-latent.npy
```

`GET /v1/health` 检查本地模型、Python 依赖、FFmpeg/ffprobe 和 GPU，不加载权重。官方 VAE
要求 FP32；`decode_tiled` 按窗口解码以限制显存使用。请求默认最大音频时长为 600 秒，
最大 latent 帧数为 15000，可通过 `YUE2_VAE_MAX_AUDIO_SECONDS` 和
`YUE2_VAE_MAX_LATENT_FRAMES` 调整。

## MERT-v2-FullSong 音乐表征

MERT-v2-FullSong 在 `8317` 接收 30–360 秒完整歌曲，并按模型卡流程将音频混为单声道、转换到
24 kHz，再调用官方 `AutoFeatureExtractor` 与 `AutoModel`。模型输出 25 Hz、1024 维逐帧特征；
默认返回掩码平均后的整曲 JSON embedding，`output=frames` 则返回含 `last_hidden_state`、
`embedding` 和元数据的 NPZ。模型保持 FP32，使用 PyTorch SDPA attention。

```bash
curl -X POST http://127.0.0.1:8317/v1/m-a-p/MERT-v2-FullSong \
  -F 'audio=@song.flac' \
  -F 'output=embedding'

curl -X POST http://127.0.0.1:8317/v1/m-a-p/MERT-v2-FullSong \
  -F 'audio=@song.flac' \
  -F 'output=frames' \
  -o mert-v2-features.npz
```

权重、Transformers remote code 与 `preprocessor_config.json` 必须存在于本地快照，服务默认离线，
不会在请求中下载。默认目录为 `$HF_MIRROR_DIR/m-a-p/MERT-v2-FullSong`；模型状态、Python 依赖、
FFmpeg/ffprobe 和 CUDA 由 `GET /v1/health` 检查。成品原子保存到 `storage/mert_v2_fullsong/`，
当前存储维护工具不会自动清理该目录。



## SoulX-FlashHead 人像视频生成

服务在 `8391` 提供 `POST /v1/soulX/flashHead`：先上传人像和驱动音频，再用上传时的
`full_path` 逻辑标识生成视频。默认 Lite，支持 Pro；HTTP 进程不加载模型，推理使用独立 uv
环境的一次性 worker，成品原子保存到 `storage/video/` 并返回 `video/mp4`。

```bash
curl -fsS http://127.0.0.1:8391/v1/upload_image \
  -F 'image=@portrait.png;type=image/png' -F 'full_path=portrait.png'
curl -fsS http://127.0.0.1:8391/v1/upload_audio \
  -F 'audio=@speech.wav;type=audio/wav' -F 'full_path=speech.wav'
curl -fsS http://127.0.0.1:8391/v1/soulX/flashHead \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"portrait.png","audio_path":"speech.wav","model_type":"lite","seed":42}' \
  -o talking-head.mp4
```

主模型默认位于 `$HF_MIRROR_DIR/Soul-AILab/SoulX-FlashHead-1_3B`，音频编码器默认位于
`$HF_MIRROR_DIR/Soul-AILab/wav2vec2-base-960h`，官方源码默认位于 `$HOME/LLM-depency/SoulX-FlashHead`。
运行前执行 `uv sync --project model-video/avatar_soulx-flashhead-1.3b --locked`。
默认上传上限 64 MiB、驱动音频时长上限 300 秒；完整字段、预检和单 GPU 限制见
[SoulX-FlashHead 服务说明](model-video/avatar_soulx-flashhead-1.3b/README.md)。

## Kandinsky 6 Lite 蒸馏版图生视频

服务在 `8369` 提供 `POST /v1/kandinskylab/Kandinsky-6.0-Lite-Distill-BF16`。先用
`/v1/upload_image` 上传参考图片，再传入上传时的图片逻辑路径和提示词，成功响应为 `video/mp4`。
服务固定以 BF16、10 步、`guidance_scale=1.0`、864×480、121 帧生成无音轨视频；HTTP 进程不加载模型，
每次请求启动一次性 Diffusers worker 并通过共享 GPU 锁排队。

```bash
curl -fsS http://127.0.0.1:8369/v1/upload_image \
  -F 'image=@portrait.png;type=image/png' -F 'full_path=portrait.png'
curl -fsS http://127.0.0.1:8369/v1/kandinskylab/Kandinsky-6.0-Lite-Distill-BF16 \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"portrait.png","prompt":"A red dragon slowly unfolds its wings in a sunlit cave","seed":42}' \
  -o kandinsky6-video.mp4
```

权重默认位于 `$HF_MIRROR_DIR/kandinskylab`，与当前本地 Diffusers 快照结构一致。先执行
`uv sync --project model-video/Kandinsky-6.0-Lite-Distill-BF16 --locked`；模型完整性、运行依赖和
CUDA 状态可通过 `GET /v1/health` 查看。完整接口约定见
[Kandinsky 6 服务说明](model-video/Kandinsky-6.0-Lite-Distill-BF16/README.md)。

## MiniMax H3 16G HiFi 图生视频

服务在 `8392` 提供 `POST /v1/minimax/H3-16G-HiFi`，支持 FL2VA 首尾帧生成与 Ref2VA 多参考图生成。`include_audio` 默认 `false`，输出无音轨 MP4；设为 `true` 才封装 H3 原生音轨。默认 960×544、约 5.17 秒、24 FPS、20 步；分辨率、时长和空闲显存设有请求门槛。模型权重默认位于 `$HF_MIRROR_DIR/minimax/MiniMax-H3-16G-HiFi`。

推理使用仓库外的 ComfyUI `v0.37.0` 与模型仓库提供的 `ComfyUI-sol-attn-main.zip`，其 Python/CUDA 依赖按服务说明单独安装。先执行 `uv sync --project model-video/MiniMax-H3-16G-HiFi --locked`，再按
[MiniMax H3 服务说明](model-video/MiniMax-H3-16G-HiFi/README.md)准备 ComfyUI 和调用接口。`GET /v1/health` 检查权重、节点、依赖及 GPU 状态；完整准备不等于已完成真实生成验证。

## 文本润色训练实验

`model-text/scripts/training/` 与 `model-text/training/configs/` 提供有声书“原始小说片段 → 纯净可朗读文本”的
LoRA 流程实验，不属于 HTTP 服务，也不会随 `start.sh` 启动。模型为
`Qwen/Qwen3-4B-Instruct-2507`，模板为 `qwen3_nothink`；这部分处理文本，不训练语音模型。

| 文件 | 用途 |
| --- | --- |
| `model-text/scripts/training/build_audio_polisher_poc.py` | 抽取 Markdown、调用本地 Ollama 教师、按来源划分训练/验证/测试集，并生成 Alpaca 数据和导入清单 |
| `model-text/scripts/training/import_audio_polisher_to_easy_dataset.py` | 将来源和待确认候选导入独立运行的 Easy Dataset |
| `model-text/scripts/training/run_audio_polisher_inference.py` | 用 Transformers/PEFT 对基础模型与可选 adapter 执行确定性 GPU 推理 |
| `model-text/scripts/training/compare_audio_polisher_predictions.py` | 比较格式检查、相对教师标签的字符相似度与长度比例 |
| `model-text/training/configs/audio_polisher_qwen3_4b_lora_poc.yaml` | LLaMA-Factory LoRA 训练模板，rank 8、BF16、3 epochs |
| `model-text/training/configs/audio_polisher_qwen3_4b_predict_poc.yaml` | 隔离测试集预测模板 |

生成数据集前自行启动本地 Ollama 并准备教师模型（默认 `qwen3.5:9b`）。从仓库根目录执行：

```bash
PYTHONPATH=model-text uv run --project qa --locked python -m scripts.training.build_audio_polisher_poc \
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
  `500`。  模型目录或运行时未就绪时返回 `503`。
  Steam Audio 的非法 Manifest 返回 `422`、重复 `job_id` 返回 `409`、未知任务返回 `404`。
- 模型 worker 失败或超时会清理临时文件和进程组，再返回服务错误；共享 GPU 锁在
  `finally` 中释放。
- 音频生成接口响应体是 WAV，同时会写入语义对应的输出目录；JSON、ZIP 和临时导出
  的保存行为见服务总览。保留策略默认关闭：先通过
  `GET /v1/control` 的 `storage` 字段查看所在文件系统的容量与可用空间；需要治理历史
  生成结果时，再由运维显式配置并执行维护脚本，绝不自动删除用户上传或音色引用：

  ```bash
  export STORAGE_RETENTION_HOURS=168
  # 先预览，再显式确认删除；脚本只匹配已知的生成文件前缀。
  uv run --project model-audio/clone_qwen3-tts-1.7b-base python main/storage_maintenance.py
  uv run --project model-audio/clone_qwen3-tts-1.7b-base python main/storage_maintenance.py --apply
  ```

维护脚本处理默认音色、克隆和 BGM 目录中列出的生成 WAV 前缀：Qwen VoiceDesign、Qwen3-TTS、VoxCPM2、IndexTTS-2.5、ACE-Step 与 YuE2-3B。它保护普通上传和仍被 .references 引用的音色，但尚不清理 Confucius4-TTS、SheetSage2 ZIP、YuE2-Vae latent/WAV、MERT-v2-FullSong JSON/NPZ、TIGER 分离批次、MSST ZIP 或训练产物，也不自动遍历每个自定义输出目录。STORAGE_RETENTION_MAX_BYTES 只约束这些匹配的生成 WAV，不限制整个 storage/。修改输出目录时需同步考虑维护脚本的扫描范围。

## 测试与开发

根目录回归测试不下载权重、不调用外部服务、不需要 CUDA：

```bash
bash -n start.sh
bash scripts/quality_gate.sh
```

统一质量门禁先执行 shell 语法和三个无 SDK 的 C++ 核心测试，再检查工作区中存在的服务、
`localai_worker_bridge` 与 QA 锁文件、Ruff 和格式，最后执行根目录、Kandinsky 基准和 ACE-Step 的无模型测试。
所有当前维护的 uv 项目都必须存在并通过锁检查；缺失项目会使门禁失败。
GitHub Actions 使用同一脚本；需要 `uv` 和 `g++`，不需要 CUDA 或 Steam Audio SDK。
Ruff 检查仓库服务、共享运行时、根目录测试和 Kandinsky 基准脚本。
锁文件检查仍可能访问 Git/wheel 元数据，初次运行需具备这些依赖来源的网络或本地缓存。
已有完整缓存时可运行 `UV_OFFLINE=1 bash scripts/quality_gate.sh`。

仅运行无模型测试时使用轻量 QA 环境。两个服务内的测试必须从各自目录运行，以免同名
`runtime` 模块解析错误：

```bash
uv sync --project qa --locked
uv run --project qa --locked python -m unittest discover -s tests -v
(cd model-audio/music_ace-step-1.5 && uv run --project ../../qa --locked python -m unittest discover -s tests -v)
```

FastAPI TestClient 回归使用 QA 的 `httpx2` 兼容依赖。不得为无模型测试安装整套模型环境或
恢复旧客户端组合；worker、subprocess、网络与 CUDA 边界由测试替身替换。
`model-text/scripts/training/` 的纯文本逻辑也有根目录回归测试，但当前质量门禁的 Ruff 路径不包含训练脚本。

所有模型服务的 uv、Ruff、日志和无模型测试规范见
[`docs/python-engineering.md`](docs/python-engineering.md)。

修改请求契约、路由、存储解析或 worker 生命周期时，应同步添加/更新对应的 no-model
测试，并在 README 中更新兼容字段。不要提交模型权重、上传音频、生成 WAV、缓存、虚拟
环境、密钥或机器专用绝对路径。

模型专用说明见 [ACE-Step](model-audio/music_ace-step-1.5/README.md) 与 [Steam Audio renderer](model-audio/spatial_steam-audio/README.md)。
子目录中的迁移记录和本机实验描述仅作背景，部署命令、服务清单与测试环境以本 README
和实际代码为准。贡献规则见 [AGENTS.md](AGENTS.md)，中文文档要求见
[CONSTITUTION.md](CONSTITUTION.md)。

## 许可证

本仓库代码采用 [Apache-2.0](LICENSE)。模型权重、上游推理源码、Steam Audio SDK 和云端
服务各自的许可与使用条款独立适用。

## MSST Mega 53 音乐分轨（8380）

`model-audio/separate_msst-mega53` 提供 `POST /v1/msst/separate`。调用方上传 `audio`，
并通过必填 `stems` 选择本地模型支持的任意非空声部子集；`GET /v1/health` 返回完整支持清单。
模型层不固定业务推荐乐器，不默认分离全部 53 类。`chunk_seconds` 可选 5/10/20，默认 10 秒。

```bash
# 新增独立服务环境需要手动同步；外部 MSST .venv 不代替该步骤。
uv sync --project model-audio/separate_msst-mega53 --locked
curl --fail-with-body http://127.0.0.1:8380/v1/msst/separate \
  -F 'audio=@song.wav' -F 'stems=guitar,strings,synth' -F 'chunk_seconds=10' -o stems.zip
```

ZIP 包含请求声部、`residual.wav` 和分轨 `manifest.json`，WAV 均为 44.1 kHz 双声道 FLOAT。
外部源码默认位于 `~/LLM-depency/Music-Source-Separation-Training`（`MSST_SOURCE_DIR`），
完整模型位于 `$HF_MIRROR_DIR/oulianov/mvsep_mega_53`（`MSST_MODEL_DIR`）；
结果原子保存至 `storage/msst/`（`MSST_OUTPUT_DIR`），不由 WAV 保留策略清理。
服务拥有独立 uv 项目与一次性 worker，沿用共享 GPU 锁；不加入 LocalAI 五个固定 bridge。

业务侧按需挑选目标：吉他适合侧向定位，弦乐与合成器适合保留宽广立体声，管乐与打击乐可按配器增加；
这是业务建议而非模型限制。模型可能产生重叠或串音，原曲不能再次叠加，丢弃提取轨须回并 residual。
分轨元数据不能直接提交空间接口；业务方另行生成 Render Manifest v1，并将对象资产提交给
8300 `/v1/audio/spatial/render` 完成混合与最终导出。详见 [服务配置、合同与限制](model-audio/separate_msst-mega53/README.md)。

## 通用空间轨迹

支持静态、直线、靠近、远离、圆周与关键帧六类轨迹。HTTP 可选 `spatial.trajectory`，音乐可使用单位增益原生 CLI；合同、坐标和示例见[轨迹文档](docs/spatial-trajectories.md)。
