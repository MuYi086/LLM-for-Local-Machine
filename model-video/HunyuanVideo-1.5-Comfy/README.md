# HunyuanVideo-1.5 Comfy 图生视频

独立 FastAPI 服务在 `8392` 提供 `POST /v1/Tencent-Hunyuan/HunyuanVideo-1.5-Comfy`。
当前只接入本地 480p step-distilled FP8 图生视频模型，固定输出 848×480、24 FPS MP4。
步数支持 8 或 12，帧数支持 49、73、97 或 121。视频生成需使用图生视频输入，不接入纯文本生视频。

HTTP 进程不加载 ComfyUI、Torch 或模型。每个请求启动一个一次性 worker，由 worker 在 loopback
随机端口启动 ComfyUI API、提交 workflow 并在完成后停止 ComfyUI 进程；worker 退出后释放 GPU。
所有本地模型请求使用共享 `GPU_LOCK_FILE`，等待时限由正数 `GPU_LOCK_WAIT_TIMEOUT` 控制。

## 模型和 ComfyUI 源码

默认模型目录 `$HF_MIRROR_DIR/HunyuanVideo-1.5-Comfy/` 必须包含以下五个文件：

```text
split_files/diffusion_models/hunyuanvideo1.5_480p_i2v_step_distilled_fp8_scaled.safetensors
split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors
split_files/clip_vision/sigclip_vision_patch14_384.safetensors
split_files/vae/hunyuanvideo15_vae_fp16.safetensors
```

ComfyUI 源码默认路径为 `$HOME/LLM-depency/ComfyUI`，可用 `HUNYUAN_VIDEO_COMFY_CODE_PATH`
覆盖。源码需包含 ComfyUI 核心 `HunyuanVideo15ImageToVideo`、`SaveVideo` 节点。示例准备命令：

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git "$HOME/LLM-depency/ComfyUI"
uv sync --project model-video/HunyuanVideo-1.5-Comfy --locked
```

服务通过每次 worker 独有的 `extra_model_paths.yaml` 映射本地模型目录，不复制权重、不加载自定义节点，
也不会在请求期间联网下载模型。服务依赖和 ComfyUI Python 运行依赖在本项目锁文件中安装；源码仓库留在项目之外。

## API

先上传 PNG、JPEG 或 WebP 参考图片，`full_path` 只用作逻辑标识：

```bash
curl -fsS http://127.0.0.1:8392/v1/upload_image \
  -F 'image=@portrait.png;type=image/png' -F 'full_path=portrait.png'
```

生成请求使用上传返回时的 `full_path` 作为 `image_path`。成功响应为 `video/mp4`：

```bash
curl -fsS http://127.0.0.1:8392/v1/Tencent-Hunyuan/HunyuanVideo-1.5-Comfy \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"portrait.png","prompt":"A person walks through a quiet garden at dawn, slow cinematic camera movement","seed":42,"width":480,"height":832,"steps":12,"num_frames":81}' \
  -o hunyuan-video.mp4
```

| 字段 | 类型/范围 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `image_path` | 非空字符串，最多 1024 字符 | 必填 | 上传时的逻辑图片路径，不能传服务器任意路径 |
| `prompt` | 1–12000 字符 | 必填 | 场景和运动描述 |
| `negative_prompt` | 最多 4000 字符 | 空字符串 | 可选负向提示词 |
| `seed` | `0` 到 `2^64-1` | `42` | 随机种子 |
| `width` / `height` | 256–1280，16 的倍数；像素总量不超过 848×480 | `848` / `480` | 输出分辨率；480×832 可用于竖屏评测 |
| `steps` | `8` 或 `12` | `8` | step-distilled 模型采样步数 |
| `num_frames` | `49`、`81`、`97` 或 `121` | `121` | 帧数，对应约 2–5 秒 |

输入图片默认最多 64 MiB，由 `UPLOAD_MAX_BYTES` 覆盖。图片在 GPU 排队前完成格式、解码和尺寸检查；
边长不得超过 8192 像素。上传图片存于 `storage/hunyuan/images/`，成品原子保存至
`storage/video/`，由接口以 `FileResponse` 返回。运行数据与模型素材不纳入 Git。

## 配置

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `HUNYUAN_VIDEO_COMFY_HOST` / `HUNYUAN_VIDEO_COMFY_PORT` | `0.0.0.0` / `8392` | HTTP 监听地址 |
| `HUNYUAN_VIDEO_COMFY_MODEL_DIR` | `$HF_MIRROR_DIR/HunyuanVideo-1.5-Comfy` | 本地 ComfyUI 格式模型目录 |
| `HUNYUAN_VIDEO_COMFY_CODE_PATH` | `$HOME/LLM-depency/ComfyUI` | ComfyUI 核心源码目录 |
| `HUNYUAN_VIDEO_COMFY_IMAGE_DIR` | `storage/hunyuan/images/` | 逻辑上传图片 |
| `HUNYUAN_VIDEO_COMFY_OUTPUT_DIR` | `storage/video/` | 成品 MP4 |
| `HUNYUAN_VIDEO_COMFY_WORKER_TMP_DIR` | `storage/.cache/runtime/hunyuan_video_comfy_worker/` | 请求 JSON、ComfyUI 输入/输出及临时文件 |
| `HUNYUAN_VIDEO_COMFY_REQUEST_TIMEOUT` | `2400` 秒 | worker 总超时 |
| `HUNYUAN_VIDEO_COMFY_CUDA_VISIBLE_DEVICES` | `0` | 单个可见 GPU 编号 |
| `HUNYUAN_VIDEO_COMFY_STEPS` | `8` | 默认步数，只能设为 8 或 12 |
| `HUNYUAN_VIDEO_COMFY_FRAMES` | `121` | 默认帧数，只能设为 49、73、97 或 121 |
| `HUNYUAN_VIDEO_COMFY_SEED` | `42` | 默认随机种子 |
| `HUNYUAN_VIDEO_COMFY_FFPROBE_BIN` | `ffprobe` | MP4 输出校验程序 |
| `GPU_LOCK_FILE` | `storage/.cache/runtime/gpu-runtime.lock` | 所有本地模型服务共享的 GPU 锁 |
| `GPU_LOCK_WAIT_TIMEOUT` | `900` 秒 | GPU 排队时限 |
| `LOCAL_FILES_ONLY` | `1` | 离线加载模型文件 |

`GET /v1/health` 不加载模型，报告每个模型文件、ComfyUI 核心目录、Python 依赖、ffprobe 和 CUDA 状态。
需要先检查 `available.selected_provider_ready` 为 `true`，再发送生成请求。缺少 ComfyUI 源码、模型、
依赖或 CUDA 时，接口会在 GPU 排队前返回 `503`。
