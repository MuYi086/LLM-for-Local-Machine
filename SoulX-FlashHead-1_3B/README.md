# SoulX-FlashHead-1_3B

独立 uv/FastAPI 服务，用一张人像图片和驱动音频生成带声音的 MP4。
默认端口 `8391`，唯一生成路由为 `POST /v1/soulX/flashHead`。
沿用 Confucius4-TTS 的轻量 HTTP 入口、文件预检、共享 GPU 队列和一次性 worker 结构。
HTTP 进程不导入 Torch 或模型；worker 使用当前服务的 Python 解释器。

## 准备与启动

需要 Python `3.12.13`、uv、CUDA GPU、FFmpeg（含 libx264/AAC）和 ffprobe。
参考[官方项目](https://github.com/Soul-AILab/SoulX-FlashHead)准备仓库外的源码与
[模型权重](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B)。
源码默认在 `$HOME/tts-depency/SoulX-FlashHead`；部署时固定自己验证过的源码版本。

```bash
mkdir -p "$HOME/tts-depency"
git clone https://github.com/Soul-AILab/SoulX-FlashHead.git "$HOME/tts-depency/SoulX-FlashHead"
export HF_MIRROR_DIR="${HF_MIRROR_DIR:-$HOME/hf-mirror}"
```

本地模型目录 `$HF_MIRROR_DIR/Soul-AILab/SoulX-FlashHead-1_3B` 中需要：

| 模式 | 必需文件 |
| --- | --- |
| Lite | `Model_Lite/config.json`、`Model_Lite/diffusion_pytorch_model.safetensors`、`VAE_LTX/config.json`、`VAE_LTX/diffusion_pytorch_model.safetensors` |
| Pro | `Model_Pro/config.json`、`Model_Pro/diffusion_pytorch_model.safetensors`、`VAE_Wan/Wan2.1_VAE.pth` |

两种模式都需要额外的 [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
音频编码器，放在 `$HF_MIRROR_DIR/facebook/wav2vec2-base-960h`。至少保留 `config.json`、
`preprocessor_config.json` 与 `model.safetensors` 或 `pytorch_model.bin`。已有 Hugging Face CLI 的主机可执行：

```bash
hf download facebook/wav2vec2-base-960h \
  --local-dir "$HF_MIRROR_DIR/facebook/wav2vec2-base-960h"
uv sync --project SoulX-FlashHead-1_3B --locked
uv run --no-sync --project SoulX-FlashHead-1_3B python SoulX-FlashHead-1_3B/main.py
```

从仓库根目录运行。部署全套服务时使用 `bash start.sh`，启动阶段使用 `--no-sync`。
项目锁定 Torch `2.7.1` / torchvision `0.22.1` / CUDA `12.8` 和 Transformers `4.57.3`。
单 GPU 使用官方 attention 的 PyTorch SDPA 兜底，基础环境无需编译 FlashAttention、
xformers 或 SageAttention；安装加速扩展时必须与本服务的 Torch/CUDA ABI 匹配。
Python 3.12 使用保留 `mediapipe.solutions` 的 MediaPipe `0.10.21`，配对 NumPy 1.x 和 OpenCV 4.11。
服务只运行单 GPU；Pro 多 GPU与实时 HTTP/WebSocket 视频流不属于这个文件生成接口。

## 接口

先上传图片和音频，以相同的 `full_path` 标识调用生成接口。`full_path` 是 WebUI 逻辑标识，
无需对应服务器真实路径。服务通过摘要映射解析输入，不能直接读取任意本地文件。

```bash
curl -fsS http://127.0.0.1:8391/v1/upload_image \
  -F 'image=@portrait.png;type=image/png' -F 'full_path=portrait.png'
curl -fsS http://127.0.0.1:8391/v1/upload_audio \
  -F 'audio=@speech.wav;type=audio/wav' -F 'full_path=speech.wav'
curl -fsS http://127.0.0.1:8391/v1/soulX/flashHead \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"portrait.png","audio_path":"speech.wav","model_type":"lite","seed":42,"use_face_crop":false}' \
  -o talking-head.mp4
```

| 接口 | 参数与返回值 |
| --- | --- |
| `POST /v1/upload_image` | multipart `image` + `full_path`；返回 `code`、`filename`、`sha256`、`size_bytes` |
| `POST /v1/upload_audio` | multipart `audio` + `full_path` + 可选 `prompt_text`；共享音频存储响应 |
| `GET /v1/check/image` | query `file_name`；返回 `code`、`exists` |
| `GET /v1/check/audio` | query `file_name`；返回 `code`、`exists`、`has_prompt_text` |
| `GET /v1/health` | 源码/权重/FFmpeg/显卡/存储可用性、worker 设置、最近错误与 GPU 指标 |
| `POST /v1/soulX/flashHead` | JSON 请求；返回 `video/mp4`，同时在服务端原子保存成品 |

生成请求字段如下，未知字段返回 `422`。默认值集中在 `main.py` 顶部。

| 字段 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `image_path` | 是 | — | 已上传图片的 `full_path`，1–1024 字符 |
| `audio_path` | 是 | — | 已上传音频的 `full_path`，1–1024 字符 |
| `model_type` | 否 | `lite` | `lite` 或 `pro` |
| `seed` | 否 | `42` | 0–4294967295 |
| `use_face_crop` | 否 | `false` | 官方人脸裁剪；关闭时按官方逻辑缩放/中心裁剪 |

图片支持 PNG/JPEG/WebP，边长不超过 8192；音频支持 WAV/MP3/OGG/FLAC/M4A/AAC/WebM。
上传默认最大 64 MiB，通过 `UPLOAD_MAX_BYTES` 覆盖。生成前使用 ffprobe 检查媒体内容、
音频流与时长；默认最多 300 秒。worker 将音频转换为 16 kHz 单声道，使用官方逐块编码和
pipeline 生成视频，逐帧编码 MP4，补齐末块并裁剪尾帧以保留短音频和尾音。
16 kHz 音频只用于模型特征；成品使用原始驱动音频编码 AAC 音轨。
输出尺寸/FPS 使用官方 `infer_params.yaml`。

缺少上传返回 `404`，参数或媒体无效返回 `422`，上传超限返回 `413`，
源码/权重/FFmpeg 不齐或 GPU 排队超时返回 `503`，worker 超时返回 `504`，推理/编码失败返回 `500`。
成功、失败和超时都会回收 worker 进程组与临时 JSON/WAV/MP4；释放显存的等待发生在持锁期间。

## 环境变量

| 变量 | 默认值 / 用途 |
| --- | --- |
| `SOULX_FLASHHEAD_PROJECT_DIR` | `start.sh` 项目目录，默认仓库内 `SoulX-FlashHead-1_3B` |
| `SOULX_FLASHHEAD_HOST` / `SOULX_FLASHHEAD_PORT` | `0.0.0.0` / `8391`；回退到 `HOST` / `PORT` |
| `SOULX_FLASHHEAD_MODEL_DIR` | `$HF_MIRROR_DIR/Soul-AILab/SoulX-FlashHead-1_3B` |
| `SOULX_FLASHHEAD_CODE_PATH` | `$HOME/tts-depency/SoulX-FlashHead` |
| `SOULX_FLASHHEAD_WAV2VEC_DIR` | `$HF_MIRROR_DIR/facebook/wav2vec2-base-960h` |
| `SOULX_FLASHHEAD_IMAGE_DIR` | `$STORAGE_DIR/flashhead/images`，上传图片 |
| `SOULX_FLASHHEAD_OUTPUT_DIR` | `$STORAGE_DIR/video`，生成 MP4 |
| `SOULX_FLASHHEAD_WORKER_TMP_DIR` | `$RUNTIME_CACHE_DIR/soulx_flashhead_worker` |
| `SOULX_FLASHHEAD_FFMPEG_BIN` / `SOULX_FLASHHEAD_FFPROBE_BIN` | `ffmpeg` / `ffprobe` |
| `SOULX_FLASHHEAD_REQUEST_TIMEOUT` | `1800` 秒 |
| `SOULX_FLASHHEAD_MAX_AUDIO_DURATION` | `300` 秒 |
| `SOULX_FLASHHEAD_CUDA_VISIBLE_DEVICES` | `0`；单个 GPU 索引或 UUID |
| `SOULX_FLASHHEAD_MODEL_TYPE` / `SOULX_FLASHHEAD_SEED` / `SOULX_FLASHHEAD_USE_FACE_CROP` | `lite` / `42` / `0`，请求默认值 |

复用 `STORAGE_DIR`、`CLONE_STORAGE_DIR`、`PROMPTS_DIR`、`TIMBRE_STORAGE_DIR`、`RUNTIME_CACHE_DIR`、
`HF_MIRROR_DIR`、`LOCAL_FILES_ONLY`（默认开启）、`UPLOAD_MAX_BYTES`、`GPU_LOCK_FILE`、
`GPU_LOCK_WAIT_TIMEOUT`、`GPU_METRICS_SAMPLE_INTERVAL` 和 `CUDA_RELEASE_DELAY`。
上传图片、音频、视频与编译缓存均为运行数据，不提交到仓库。

## 无模型验证

```bash
uv run --project qa --locked python -m unittest tests.test_soulx_flashhead_migration -v
bash scripts/quality_gate.sh
```

测试替换模型、worker、编码器和 CUDA 边界，不下载权重或执行真实 GPU 推理。
