# Unitale AI Local Backend

Unitale 前端使用的本地语音后端，提供参考音频克隆、音色设计、语音编辑和
SoundEffect 生成。仓库采用“一个服务一个 uv 项目”的边界：HTTP 控制面不加载
重型模型，模型推理由对应目录中的一次性 worker 完成。

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
| dots.tts-soar | 8324 | 参考音频语音克隆 | `/v2/dotsTTS/clone` |
| FireRedTTS3 Base | 8325 | 参考音频语音克隆 | `/v1/FireRedTTS3/clone` |
| Step-Audio-EditX | 8331 | 语音编辑 | `/v1/stepAudioEditx/edit` |
| MOSS-Audio-4B-Thinking | 8341 | 音频转写、描述与问答 | `/v1/mossAudioThinking/understand` |
| MOSS-Audio-4B-Instruct | 8342 | 音频转写、描述与问答 | `/v1/mossAudioThinking/understand` |
| Confucius4-TTS | 8361 | 参考音频零样本、多语言语音克隆 | `/v1/confucius4TTS/generate` |
| TIGER-DnR | 8351 | 电影混音的对白、音效、音乐三 Stem 分离 | `/v1/tigerDnr/separate` |
| Qwen3-ASR-1.7B | 8371 | 本地多语言音频识别 | `/v1/qwen3/asr` |
| Seed-VC | 8381 | 原始语音到参考音色的转换，可选 F0 歌声转换 | `/v1/seedVc/voiceConversion` |

每个服务都提供 `GET /v1/health`。后端只注册表中列出的最终接口；模型生成接口返回
`audio/wav`，并在服务端保存一份 WAV。MOSS-Audio-4B Thinking/Instruct 与 Qwen3-ASR 接收音频并返回 JSON 文本，
不会生成或保留 WAV。TIGER-DnR 返回含 `dialog.wav`、`effects.wav`、`music.wav` 的 ZIP，并将
三路 Stem 原子保存。空间音频导出返回 WAV 或 MP3，响应完成后删除临时成品。

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
tests/                        根目录无模型回归测试
soundEffect/                  MOSS GPU 示例和提示词说明
storage/                      上传音频、生成音频、sidecar、缓存和 GPU 锁
```

默认运行数据目录为：

| 目录 | 内容 | 覆盖变量 |
| --- | --- | --- |
| `storage/timbre/` | Qwen、MOSS、MiMo、FireRedTTS3 生成的音色参考音频 | `TIMBRE_STORAGE_DIR` |
| `storage/soundEffect/` | MOSS 和 Stable Audio 生成的声效 | `SOUNDEFFECT_STORAGE_DIR`、`STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR` |
| `storage/bgm/` | ACE-Step 有声小说 BGM 和 OST | `BGM_STORAGE_DIR`、`ACESTEP_OUTPUT_DIR` |
| `storage/clone/` | 参考音频、克隆结果、Step 编辑结果和 Seed-VC 转换结果 | `CLONE_STORAGE_DIR`、各服务的 `*_OUTPUT_DIR` |
| `storage/separation/` | TIGER-DnR 每次分离的 dialog、effects、music Stem 与 ZIP | `TIGER_DNR_OUTPUT_DIR` |
| `storage/.cache/runtime/` | worker 临时文件、母带/Steam Audio 任务缓存、库缓存和共享 GPU 锁 | `RUNTIME_CACHE_DIR`、`SPATIAL_EXPORT_CACHE_DIR`、`STEAM_AUDIO_RENDER_CACHE_DIR`、`GPU_LOCK_FILE` |

如果上传音频的内容与 `storage/timbre/` 中已有的设计音色一致，Qwen3-TTS、VoxCPM2、
LongCat、dots.tts-soar 和 FireRedTTS3 会在 `storage/timbre/.references/` 保存带 SHA-256 和相对路径的
小型 JSON 引用映射，不再把同一 WAV 复制到 `storage/clone/`；普通用户上传的参考音频仍保存到
`storage/clone/`。上传按块暂存并通过原子替换提交，默认上限为 64 MiB，可用
`UPLOAD_MAX_BYTES` 覆盖。这些目录是运行数据，不要提交到 Git。

## 安装与启动

运行要求：Python `3.12.13`、`uv`、FFmpeg、可用的 CUDA/NVIDIA 驱动（本地模型服务），以及
下方列出的模型权重和外部源码目录。权重与第三方源码不放进本仓库。

先为需要的服务同步锁定依赖；部署全部服务时可以执行：

```bash
for project in qwen3_tts mimo_tts voxcpm2 LongCat_AudioDiT_3.5B_bf16 \
  dots_tts_soar moss_soundEffect stable_audio_3_medium ace_step_1_5 \
  qwen3_voiceDesign moss_voiceGenerator moss_audio_4b_thinking Confucius4_TTS Qwen3_ASR_1.7B seed-vc Step_Audio_EditX firered_tts3 TIGER-DnR; do
  uv sync --project "$project" --locked
done
```

ACE-Step 的 Diffusers 依赖当前使用官方 Git 版本；Step-Audio-EditX、MOSS-SoundEffect 和 Stable Audio 3 依赖外部源码或系统命令；
先准备对应路径，再执行 `uv sync`。启动前应完成依赖同步，不要把 `start.sh` 当作依赖
安装流程；使用 `--no-sync` 的服务尤其要求对应环境已经准备好。

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
8331、8341、8342、8351、8361、8371 和 8381 共 20 个进程；8300 使用 `qwen3_tts` uv 项目中的轻量 HTTP 依赖，其余服务使用
各自的 uv 项目。启动命令统一使用 `uv run --no-sync`，不会在运行阶段联网解析依赖；
本地 GPU 服务通过 `GPU_LOCK_FILE` 串行访问 GPU。默认最多排队 900 秒，超过时返回
`503`；用 `GPU_LOCK_WAIT_TIMEOUT` 调整（设为非正值可关闭时限）。健康检查可结合
`nvidia-smi` 观察显存，持锁期间会记录本次请求的峰值显存。
任一子进程退出时脚本会终止其余进程组并清理 worker。

健康检查：

```bash
for port in 8300 8301 8302 8303 8304 8311 8312 8313 8321 8322 8323 8324 8325 8331 8341 8342 8351 8361 8371 8381; do
  curl -fsS "http://127.0.0.1:${port}/v1/health" >/dev/null && echo "${port}: ok"
done
```

单独调试服务时，从仓库根目录执行，例如：

```bash
HOST=127.0.0.1 PORT=8321 \
  uv run --project qwen3_tts python qwen3_tts/main.py
```

## 模型路径与主要配置

`start.sh` 默认使用 `HF_MIRROR_DIR`（默认为 `$HOME/hf-mirror`）和 `$HOME/tts-depency`；
所有路径都可在启动前用环境变量覆盖。

| 服务 | 默认权重 | 其他必需路径 |
| --- | --- | --- |
| Qwen3-TTS Base | `$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `QWEN3_TTS_MODEL_DIR` |
| Qwen VoiceDesign | `$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `QWEN_VOICEDESIGN_MODEL_DIR` |
| MOSS VoiceGenerator | `$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-VoiceGenerator` | `MOSS_VOICEGENERATOR_MODEL_DIR`、`MOSS_AUDIO_TOKENIZER_PATH` |
| MOSS-Audio-4B-Thinking | `$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-4B-Thinking` | `MOSS_AUDIO_4B_THINKING_MODEL_DIR`、`MOSS_AUDIO_4B_THINKING_DEPENDENCY_PATH` |
| MOSS-Audio-4B-Instruct | `$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-4B-Instruct` | `MOSS_AUDIO_4B_INSTRUCT_MODEL_DIR`、`MOSS_AUDIO_4B_INSTRUCT_DEPENDENCY_PATH` |
| Confucius4-TTS | `$HF_MIRROR_DIR/netease-youdao/Confucius4-TTS` | `CONFUCIUS4_TTS_MODEL_DIR`、`CONFUCIUS4_TTS_CODE_PATH`、`CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR`、`CONFUCIUS4_TTS_VOCODER_MODEL_DIR`、`CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT` |
| Qwen3-ASR-1.7B | `$HF_MIRROR_DIR/Qwen/Qwen3-ASR-1.7B` | `QWEN3_ASR_MODEL_DIR` |
| Seed-VC | `$HF_MIRROR_DIR/Plachta/Seed-VC` | `SEED_VC_CODE_PATH`、`SEED_VC_WHISPER_MODEL_DIR`、`SEED_VC_VOCODER_MODEL_DIR`、`SEED_VC_STYLE_ENCODER_CHECKPOINT`；F0 模式另需 `SEED_VC_F0_VOCODER_MODEL_DIR`、`SEED_VC_RMVPE_CHECKPOINT` |
| MOSS-SoundEffect | `$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-SoundEffect-v2.0` | `MOSS_SOUNDEFFECT_CODE_PATH`、`MOSS_SOUNDEFFECT_MODEL_DIR` |
| Stable Audio 3 Medium | `$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium` | `STABLE_AUDIO_3_REPO_PATH`、`STABLE_AUDIO_3_MEDIUM_MODEL_DIR` |
| ACE-Step 1.5 XL Turbo | `$HF_MIRROR_DIR/ACE-Step/acestep-v15-xl-turbo-diffusers` | `ACESTEP_MODEL_DIR`、`ACESTEP_OFFLOAD`、`ACESTEP_VAE_TILING` |
| VoxCPM2 | `$HF_MIRROR_DIR/openbmb/VoxCPM2` | `VOXCPM2_MODEL_DIR`、仓库内 `voxcpm2/voxcpm2_helpers.py` |
| LongCat-AudioDiT | `$HF_MIRROR_DIR/drbaph/LongCat-AudioDiT-3.5B-bf16` | `LONGCAT_AUDIODIT_REPO_PATH`、`LONGCAT_AUDIODIT_TOKENIZER_PATH` |
| dots.tts-soar | `$HF_MIRROR_DIR/rednote-hilab/dots.tts-soar` | `DOTS_TTS_SOAR_MODEL_DIR` |
| Step-Audio-EditX | `$HF_MIRROR_DIR/stepfun-ai/Step-Audio-EditX` | `STEP_AUDIO_TOKENIZER_PATH`、`STEP_AUDIO_EDITX_CODE_PATH` |
| FireRedTTS3 Base/Instruct | `$HF_MIRROR_DIR/drbaph/FireRedTTS3-bf16` | `FIRERED_TTS3_MODEL_DIR`、`FIRERED_TTS3_CODE_PATH` |
| TIGER-DnR | `$HF_MIRROR_DIR/JusperLee/TIGER-DnR` | `TIGER_DNR_SOURCE_DIR`（默认 `$HOME/.local/share/tiger-dnr/TIGER`） |

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
`FIRERED_TTS3_*`、`TIGER_DNR_*`、`SEED_VC_*`。每个服务的 `/v1/health` 会报告
生效的路径、运行时和可用性。
FireRedTTS3 的官方源码默认位于 `$HOME/tts-depency/FireRedTTS3`，通过
`FIRERED_TTS3_CODE_PATH` 覆盖；8304 以 `timbre` 模式加载 Instruct，8325 以 `clone` 模式加载
Base，两者不会同时在 worker 中常驻显存。

TIGER-DnR 的官方推理代码不包含在本仓库。默认使用已准备的
`$HOME/.local/share/tiger-dnr/TIGER`；也可以通过 `TIGER_DNR_SOURCE_DIR` 指向作者的
`JusperLee/TIGER` 克隆。worker 只使用其中的 `look2hear` DnR 模型代码，并在
`LOCAL_FILES_ONLY=1` 下从本地 `config.json` 与 `model.safetensors` 加载权重。

Confucius4-TTS 的官方推理代码也不包含在本仓库。默认使用
`$HOME/tts-depency/Confucius4-TTS`，模型目录默认读取题目指定的
`$HF_MIRROR_DIR/netease-youdao/Confucius4-TTS`。除 Confucius4-TTS 自身权重外，还需准备
Wav2Vec2-BERT、BigVGAN 和 CAMPPlus 权重；当前默认目录分别是
`$HF_MIRROR_DIR/netease-youdao/facebook/w2v-bert-2.0`、
`$HF_MIRROR_DIR/netease-youdao/nv-community/bigvgan_v2_22khz_80band_256x` 和
`$HF_MIRROR_DIR/netease-youdao/funasr/campplus/campplus_cn_common.bin`，也可通过
`CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR`、`CONFUCIUS4_TTS_VOCODER_MODEL_DIR` 和
`CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT` 覆盖；`LOCAL_FILES_ONLY=1` 时 worker 不会隐式下载。
Confucius4-TTS 项目固定使用 TorchAudio 2.11，需先按锁文件同步其中声明的 `torchcodec` 依赖。

Qwen3-ASR 使用官方 `qwen-asr` Transformers 后端和题目指定的本地模型目录。项目锁定了 `qwen-asr`
及其 Transformers 依赖；`LOCAL_FILES_ONLY=1` 时不会隐式下载模型。服务通过一次性 worker 执行识别，
并使用与其他本地 GPU 服务共享的排他锁。

调用 `POST http://127.0.0.1:8371/v1/qwen3/asr` 时以 multipart/form-data 发送 `audio` 文件；
可选传入 `language` 强制指定识别语言、`context` 提供识别上下文、`max_new_tokens` 设置生成上限。
省略 `language` 时自动识别。成功响应为 JSON，包含 `text`、识别出的 `language`、耗时和上传文件摘要：

```bash
curl -X POST http://127.0.0.1:8371/v1/qwen3/asr \
  -F 'audio=@speech.wav;type=audio/wav' \
  -F 'language=Chinese' \
  -F 'context=人名：张三' \
  -F 'max_new_tokens=512'
```

Confucius4-TTS 使用与其他克隆服务相同的上传和检查流程：先调用 `POST /v1/upload_audio`，
再以保存后的 `audio_path` 调用 `POST http://127.0.0.1:8361/v1/confucius4TTS/generate`。
请求 JSON 至少包含 `text`、`lang` 和 `audio_path`，成功返回 `audio/wav`，并将生成结果保存到
`storage/clone/`。官方默认生成参数可通过 `CONFUCIUS4_TTS_*` 环境变量覆盖。

对单个极短非语言感叹音，可选传入 `vocalization_duration_seconds`（有限正数，最多 2 秒）。
worker 使用官方 S2A 的 `target_feat_len` 指定 Mel 帧数，完整保留 T2S 语义序列；不截断
最终 WAV，不用于压缩普通词汇台词。多合成块请求会明确失败。省略该字段时保持官方默认
生成长度，仍须对输出执行语义和时长验收。`GET /v1/health` 的
`runtime.native_vocalization_duration=true` 表示已加载支持此字段的 HTTP 版本；更新后需要
重新运行 `bash start.sh`，仅修改 worker 文件不能更新已启动的接口校验。

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
不加入 Haas、aecho 或其他空间处理：

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

```bash
manifest="$(tr -d '\n' < render-manifest.json)"
curl -X POST http://127.0.0.1:8300/v1/audio/spatial/render \
  -F "manifest=$manifest" \
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
平均耗时给出预计剩余时间。1× 倍速对象会跳过无意义的 `atempo=1`，避免 FFmpeg 6.1.1 在其后
衔接 SoXR 时偶发无法结束滤镜链。成功响应还会返回
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

1. `POST /v1/upload_audio`，表单字段为 `audio`、`full_path`；Qwen、VoxCPM2、LongCat、
   dots.tts-soar 和 FireRedTTS3 还接受可选的 `prompt_text`。
2. `GET /v1/check/audio?file_name=...` 检查服务自己的存储状态。
3. 调用当前模型的克隆路由，请求中的 `audio_path` 使用上传时的 `full_path` 文件名。

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
| Qwen3-TTS Base | 有准确参考文本时映射为官方 `ref_text`；也可使用仅音色向量克隆。 |
| VoxCPM2 | `clone_mode=ultimate` 使用参考文本；`clone_mode=controllable` 改用 `control_instruction`，二者互斥。`nonverbal_tags` 最多一个，且只能用于可控模式。 |
| LongCat-AudioDiT | 推荐提供与参考音频逐字一致的文本；参考音频会按官方流程重采样为 24 kHz 单声道。 |
| dots.tts-soar | 有参考文本时使用 continuation cloning；省略时保留官方 x-vector-only cloning。输出为 48 kHz 单声道。 |

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

FireRedTTS3 Instruct 音色设计监听 `8304`，只接收 `voice_description` 和 `text`，并将
生成 WAV 保存在 `storage/timbre/`。最终路由为
`POST http://127.0.0.1:8304/v1/FireRedTTS3/timbre`。

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
范围为 `(0, 30]`。默认字段为 `num_inference_steps=100`、`cfg_scale=4.0`、
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

`prompt` 长度为 1–2000，`seconds` 为 10–600，`steps` 为 1–20，`bpm` 为 30–240；
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

## 健康检查与错误语义

- 健康检查不会加载模型；`available`、`paths`、`runtime` 和 `last_errors` 用于区分依赖、
  权重、CUDA、worker 和配置问题。
- 请求校验失败通常返回 `422`；上传后找不到参考音频通常返回 `404`。
- 模型 worker 失败或超时会清理临时文件和进程组，再返回服务错误；共享 GPU 锁在
  `finally` 中释放。
- 生成接口响应体是 WAV，同时会写入语义对应的输出目录。保留策略默认关闭：先通过
  `GET /v1/control` 的 `storage` 字段查看所在文件系统的容量与可用空间；需要治理历史
  生成结果时，再由运维显式配置并执行维护脚本，绝不自动删除用户上传或音色引用：

  ```bash
  export STORAGE_RETENTION_HOURS=168
  # 先预览，再显式确认删除；脚本只匹配已知的生成文件前缀。
  uv run --project qwen3_tts python main/storage_maintenance.py
  uv run --project qwen3_tts python main/storage_maintenance.py --apply
  ```

## 测试与开发

根目录回归测试不下载权重、不调用外部服务、不需要 CUDA：

```bash
bash -n start.sh
bash scripts/quality_gate.sh
```

Stable Audio 的服务内测试需要从它自己的目录运行，否则 `test_migration.py` 无法解析
同目录的 `runtime.py`：

```bash
(cd stable_audio_3_medium && uv run --project . python -m unittest discover -s tests -v)
```

MOSS 的真实 CUDA/权重 smoke test 是独立流程：

```bash
bash soundEffect/run_moss_soundeffect_v2.sh
```

所有模型服务的 uv、Ruff、日志和无模型测试规范见
[`docs/python-engineering.md`](docs/python-engineering.md)。

修改请求契约、路由、存储解析或 worker 生命周期时，应同步添加/更新对应的 no-model
测试，并在 README 中更新兼容字段。不要提交模型权重、上传音频、生成 WAV、缓存、虚拟
环境、密钥或机器专用绝对路径。

各服务的依赖、单服务启动和模型专用配置可继续参考目录内 README：
`mimo_tts/README.md`、`Step_Audio_EditX/README.md`、`LongCat_AudioDiT_3.5B_bf16/README.md`、
`dots_tts_soar/README.md`、`moss_soundEffect/README.md` 和 `stable_audio_3_medium/README.md`。
