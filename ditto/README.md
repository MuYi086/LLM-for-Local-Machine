# Ditto 人像视频服务

沿用 Confucius4-TTS 的独立 uv 项目、轻量 HTTP 入口与一次性 worker 结构，
以 `8392` 提供 `POST /v1/ditto/talkingHead`。上传人像图片和驱动音频后，
返回带声音的 `video/mp4`，成品原子保存在 `storage/video/ditto_*.mp4`。

## 安装与启动

需要 Python `3.12.13`、uv、FFmpeg/ffprobe、GCC 和可用的 NVIDIA CUDA 驱动。
GCC 用于官方源码中 blend 的 Cython 编译；编译缓存位于单次 worker 临时目录。
准备 [antgroup/ditto-talkinghead 官方源码](https://github.com/antgroup/ditto-talkinghead)：

```bash
git clone https://github.com/antgroup/ditto-talkinghead "$HOME/tts-depency/ditto-talkinghead"
uv sync --project ditto --locked
```

默认权重根目录是 `$HF_MIRROR_DIR/thewintersun/ditto-talkinghead`，
`HF_MIRROR_DIR` 默认 `$HOME/hf-mirror`。使用已有的本地文件：

```text
ditto_cfg/v0.4_hubert_cfg_trt.pkl
ditto_onnx/appearance_extractor.onnx
ditto_onnx/blaze_face.onnx
ditto_onnx/decoder.onnx
ditto_onnx/face_mesh.onnx
ditto_onnx/hubert.onnx
ditto_onnx/insightface_det.onnx
ditto_onnx/landmark106.onnx
ditto_onnx/landmark203.onnx
ditto_onnx/lmdm_v0.4_hubert.onnx
ditto_onnx/motion_extractor.onnx
ditto_onnx/stitch_network.onnx
ditto_onnx/warp_network_ori.onnx
```

服务使用 ONNX Runtime CUDA 后端。官方 TensorRT 8.6.1 无 Python 3.12 wheel，
因此 worker 将本地 TRT 配置中的路径映射到 ONNX 模型，并将原始 warp 图中的三维
GridSample 通过官方 ONNX 转换器升级为 opset 20，保持其他算子的语义。转换文件只写入请求临时目录；
模型目录内的 pickle、ONNX 和 TensorRT 权重保持原样。该服务要求
`warp_network_ori.onnx`，普通 `warp_network.onnx` 使用 TensorRT 专用 GridSample3D 插件。
`onnxruntime-gpu` 限定为 `>=1.21,<1.27`，与 Torch cu128 使用同一组 CUDA 12/cuDNN 9 动态库；
ORT 1.27 起 PyPI 默认构建切换到 CUDA 13，见
[官方 CUDA 兼容表](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)。

从仓库根目录单独启动，或使用统一启动脚本：

```bash
uv run --no-sync --project ditto python ditto/main.py
# 启动全部服务
bash start.sh
curl -fsS http://127.0.0.1:8392/v1/health
```

## 上传与生成

`full_path` 是客户端逻辑标识；生成请求使用完全相同的标识，不能直接读取服务器任意文件。
上传按块暂存并计算 SHA-256，默认上限 `64 MiB`，支持 `UPLOAD_MAX_BYTES` 覆盖。

```bash
curl -fsS http://127.0.0.1:8392/v1/upload_image \
  -F 'image=@portrait.png;type=image/png' -F 'full_path=portrait.png'
curl -fsS http://127.0.0.1:8392/v1/upload_audio \
  -F 'audio=@speech.wav;type=audio/wav' -F 'full_path=speech.wav'
curl --fail-with-body http://127.0.0.1:8392/v1/ditto/talkingHead \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"portrait.png","audio_path":"speech.wav","seed":42,"sampling_timesteps":50,"max_size":1920}' \
  -o talking-head.mp4
```

图片支持 PNG/JPG/JPEG/WebP；音频支持共享上传策略中的 WAV/MP3/FLAC/OGG/M4A/AAC 等格式。
图片可解码且边长不超过 8192；驱动音频须包含音频流，时长大于 0，默认不超过 300 秒。
worker 将驱动音频转为单声道 16 kHz，按官方离线 SDK 的 25 FPS 流程生成视频，
并使用原始驱动音频合并 AAC 声轨。

| 路由 | 方法 | 字段/响应 |
| --- | --- | --- |
| `/v1/health` | GET | 文件、依赖、CUDA、存储状态和最近一次 GPU 指标；不加载模型 |
| `/v1/upload_image` | POST | multipart `image`、`full_path`；返回 SHA-256、字节数 |
| `/v1/upload_audio` | POST | multipart `audio`、`full_path`、可选 `prompt_text` |
| `/v1/check/image` | GET | query `file_name`；JSON `exists` |
| `/v1/check/audio` | GET | query `file_name`；JSON `exists`、`has_prompt_text` |
| `/v1/ditto/talkingHead` | POST | 下表中的 JSON；返回 MP4 文件 |

| 生成字段 | 默认值 | 约束 |
| --- | --- | --- |
| `image_path` | 必填 | 上传图片的 `full_path`，长度 1–1024 |
| `audio_path` | 必填 | 上传音频的 `full_path`，长度 1–1024 |
| `seed` | 42 | 0–4294967295 |
| `sampling_timesteps` | 50 | 1–100，官方 motion diffusion 采样步数 |
| `max_size` | 1920 | 256–4096，官方图片处理最大尺寸 |

请求拒绝额外字段。缺少上传输入返回 `404`，媒体或参数不合法返回 `422`，
上传超限返回 `413`，本地源码/权重/FFmpeg 缺失或 GPU 排队超时返回 `503`，
worker 超时返回 `504`，推理、编码或成品校验失败返回 `500`。

## 配置与资源管理

| 环境变量 | 默认值 |
| --- | --- |
| `DITTO_PROJECT_DIR` | 仓库 `ditto/`，仅启动脚本使用 |
| `DITTO_HOST`、`DITTO_PORT` | `0.0.0.0`、`8392`；直接启动可回退到 `HOST`、`PORT` |
| `DITTO_MODEL_DIR` | `$HF_MIRROR_DIR/thewintersun/ditto-talkinghead` |
| `DITTO_CODE_PATH` | `$HOME/tts-depency/ditto-talkinghead` |
| `DITTO_DATA_ROOT` | `$DITTO_MODEL_DIR/ditto_onnx` |
| `DITTO_CONFIG_PATH` | `$DITTO_MODEL_DIR/ditto_cfg/v0.4_hubert_cfg_trt.pkl` |
| `DITTO_OUTPUT_DIR` | `$STORAGE_DIR/video` |
| `DITTO_IMAGE_DIR` | `$STORAGE_DIR/ditto/images` |
| `DITTO_WORKER_TMP_DIR` | `$RUNTIME_CACHE_DIR/ditto_worker` |
| `DITTO_REQUEST_TIMEOUT` | 1800 秒 |
| `DITTO_MAX_AUDIO_DURATION` | 300 秒 |
| `DITTO_CUDA_VISIBLE_DEVICES` | `0`，只允许一个可见设备 |
| `DITTO_FFMPEG_BIN`、`DITTO_FFPROBE_BIN` | `ffmpeg`、`ffprobe` |
| `DITTO_SEED`、`DITTO_SAMPLING_TIMESTEPS`、`DITTO_MAX_SIZE` | 42、50、1920；默认值集中在 `main.py` |

共享配置沿用 `STORAGE_DIR`、`RUNTIME_CACHE_DIR`、`PROMPTS_DIR`、`CLONE_STORAGE_DIR`、
`TIMBRE_STORAGE_DIR`、`GPU_LOCK_FILE`、`GPU_LOCK_WAIT_TIMEOUT`、`GPU_METRICS_SAMPLE_INTERVAL`、
`CUDA_RELEASE_DELAY`、`LOCAL_FILES_ONLY`。常规上传音频保存在 `storage/clone/`，
设计音色引用使用共享 `.references` 机制。

输入、文件依赖和媒体格式在获取 GPU 锁之前检查。每次请求用本服务 uv 解释器启动一个
worker；成功、失败或超时都会终止其进程组并清理中间文件，再在持锁期间等待 CUDA 释放。
每个 ONNX session 检查实际启用的 CUDA provider，加载失败不会静默退回 CPU。
健康接口提供排队时间、worker 时间和峰值显存。默认离线运行，权重和第三方源码不纳入 Git。

## 无模型验证

```bash
uv run --project qa --locked python -m unittest discover -s tests -p 'test_ditto_service.py' -v
bash scripts/quality_gate.sh
```

测试 mock worker、模型包、网络和子进程边界，不下载权重，不执行真实模型。
