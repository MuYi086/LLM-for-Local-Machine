# Kandinsky 6 Lite 蒸馏版图生视频

服务在 `8369` 提供 `POST /v1/kandinskylab/Kandinsky-6.0-Lite-Distill-BF16`，固定使用
BF16、10 步、`guidance_scale=1.0`、864×480、121 帧和 24 FPS 生成约 5 秒的无音轨 MP4。
这组采样参数对应 Lite 蒸馏 checkpoint；HTTP 进程不导入 Torch 或 Diffusers，每个请求使用
独立 worker 与共享 GPU 文件锁。

## 本地模型目录

默认目录为 `$HF_MIRROR_DIR/kandinskylab/`（默认展开为 `~/hf-mirror/kandinskylab/`）。
此服务按官方 Diffusers 快照结构直接加载，不移动或改名权重。`GET /v1/health` 会检查以下文件：

```text
model_index.json
scheduler/scheduler_config.json
transformer/config.json
transformer/diffusion_pytorch_model.safetensors
vae/config.json
vae/diffusion_pytorch_model.safetensors
text_encoder/config.json
text_encoder/model.safetensors.index.json
text_encoder/model-00001-of-00005.safetensors
text_encoder/model-00002-of-00005.safetensors
text_encoder/model-00003-of-00005.safetensors
text_encoder/model-00004-of-00005.safetensors
text_encoder/model-00005-of-00005.safetensors
text_encoder/preprocessor_config.json
text_encoder/tokenizer.json
text_encoder/tokenizer_config.json
text_encoder/merges.txt
text_encoder/vocab.json
tokenizer/processor_config.json
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
text_encoder_2/config.json
text_encoder_2/model.safetensors
tokenizer_2/tokenizer.json
tokenizer_2/tokenizer_config.json
audio_vae/config.json
audio_vae/diffusion_pytorch_model.safetensors
vocoder/config.json
vocoder/diffusion_pytorch_model.safetensors
```

完整 pipeline 快照保留音频 VAE 与 vocoder 文件；调用时通过 `sample_audio=False` 关闭音频生成。
服务只加载本地文件，不会在请求期间从 Hugging Face 下载内容。

## 安装和启动

```bash
uv sync --project model-video/Kandinsky-6.0-Lite-Distill-BF16 --locked
bash start.sh
```

项目锁定 Python 3.12.13、PyTorch 2.7.1/CUDA 12.8 与 Diffusers 0.42.0。推理使用
`enable_sequential_cpu_offload()` 控制显存峰值，并启用 VAE tiled decoding。该方式会增加模块搬运，
因此推理速度低于显存充足时的常驻 GPU 模式。

## API

先上传 PNG、JPEG 或 WebP 图片；`full_path` 是客户端逻辑标识，只用于随后请求引用：

```bash
curl -fsS http://127.0.0.1:8369/v1/upload_image \
  -F 'image=@portrait.png;type=image/png' -F 'full_path=portrait.png'
curl -fsS http://127.0.0.1:8369/v1/kandinskylab/Kandinsky-6.0-Lite-Distill-BF16 \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"portrait.png","prompt":"A red dragon slowly unfolds its wings in a sunlit cave","seed":42}' \
  -o kandinsky6-video.mp4
```

| 字段 | 类型/范围 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `image_path` | 非空字符串，最多 1024 字符 | 必填 | 上传时提供的逻辑图片标识，不能传服务器文件路径 |
| `prompt` | 1–12000 字符 | 必填 | 描述参考图之后的主体运动和镜头变化 |
| `seed` | `0` 到 `2^63-1` | `42` | 随机种子 |

响应是 `video/mp4`。图片先完成扩展名、内容类型、哈希和解码检查，再等待 GPU 锁；上传上限由
`UPLOAD_MAX_BYTES` 控制。输入放在 `storage/kandinsky/images/`，输出原子保存到 `storage/video/`。

## 配置

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `KANDINSKY6_HOST` / `KANDINSKY6_PORT` | `0.0.0.0` / `8369` | HTTP 监听地址 |
| `KANDINSKY6_MODEL_DIR` | `$HF_MIRROR_DIR/kandinskylab` | 本地 Diffusers 模型目录 |
| `KANDINSKY6_IMAGE_DIR` | `storage/kandinsky/images/` | 逻辑上传图片 |
| `KANDINSKY6_OUTPUT_DIR` | `storage/video/` | 成品 MP4 |
| `KANDINSKY6_WORKER_TMP_DIR` | `storage/.cache/runtime/kandinsky6_worker/` | worker 请求和临时视频 |
| `KANDINSKY6_REQUEST_TIMEOUT` | `2400` 秒 | 一次推理超时 |
| `KANDINSKY6_CUDA_VISIBLE_DEVICES` | `0` | 单个可见 GPU 编号 |
| `KANDINSKY6_FFPROBE_BIN` | `ffprobe` | MP4 校验程序 |
| `GPU_LOCK_FILE` | `storage/.cache/runtime/gpu-runtime.lock` | 所有本地 GPU 服务共享锁 |
| `GPU_LOCK_WAIT_TIMEOUT` | `900` 秒 | GPU 排队时限，必须为正数 |
| `LOCAL_FILES_ONLY` | `1` | 离线加载本地模型 |

健康接口不会加载模型；`available.selected_provider_ready` 还要求完整模型、Python 依赖、ffprobe 和 CUDA
全部可用。
