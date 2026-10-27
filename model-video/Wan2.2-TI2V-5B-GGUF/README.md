# Wan2.2 TI2V 5B GGUF

独立 FastAPI 服务在 `8392` 提供 `POST /v1/QuantStack/Wan2.2-TI2V-5B-GGUF`，支持文本生视频和参考图生视频，
成功时返回 `video/mp4`。HTTP 进程不加载 ComfyUI、Torch 或模型；每个请求启动一个一次性 worker，
worker 在 loopback 随机端口启动 ComfyUI，并仅加载白名单中的 ComfyUI-GGUF 自定义节点。
全部视频模型任务共用 `GPU_LOCK_FILE`，由正数 `GPU_LOCK_WAIT_TIMEOUT` 控制排队时间。

## 本地模型和源码

默认模型目录 `$HF_MIRROR_DIR/QuantStack/Wan2.2-TI2V-5B-GGUF/` 必须包含以下文件：

```text
Wan2.2-TI2V-5B-Q8_0.gguf
umt5_xxl_fp8_e4m3fn_scaled.safetensors
wan2.2_vae.safetensors
```

ComfyUI 核心源码默认位于 `$HOME/LLM-depency/ComfyUI/`，GGUF 自定义节点源码位于
`$HOME/LLM-depency/ComfyUI-GGUF/`。worker 在每个请求的临时目录中创建节点软链接并通过
`--disable-all-custom-nodes --whitelist-custom-nodes ComfyUI-GGUF` 加载它；不会修改、复制或安装源码。
模型权重从外部模型目录映射，不下载、不复制。

为 16 GB 显存设置默认输出尺寸为 832×480、49 帧，文本编码放在 CPU，并使用 VAE 分块解码。
宽高必须是 32 的倍数且至少 256，像素总量不能超过 1280×704；较大分辨率或更长帧数会显著增加显存、
运行时和输出文件大小。帧数只允许 49、81 或 121。当前固定采样器为 `uni_pc`，调度器为 `simple`，
shift 为 8、CFG 为 5；步数可设为 1–50，默认 20。

## 安装与启动

```bash
uv sync --project model-video/Wan2.2-TI2V-5B-GGUF --locked
bash start.sh
```

`start.sh` 默认从 `$HOME/hf-mirror/QuantStack/Wan2.2-TI2V-5B-GGUF` 加载模型，并使用上述 ComfyUI
与节点源码路径。请求期间启用本地离线环境；缺少权重、源码、依赖、ffprobe 或 CUDA 时，健康状态为
`selected_provider_ready=false`，生成接口在获取 GPU 锁前返回 `503`。

## API

### 文本生视频

```bash
curl -fsS http://127.0.0.1:8392/v1/QuantStack/Wan2.2-TI2V-5B-GGUF \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A small boat glides across a misty lake at sunrise, slow cinematic camera movement","width":832,"height":480,"num_frames":49,"steps":20,"seed":42}' \
  -o wan22-video.mp4
```

### 参考图生视频

上传图片后，生成请求使用 `full_path` 作为 `image_path`。该值只作为逻辑标识，服务不会按其内容访问服务器路径：

```bash
curl -fsS http://127.0.0.1:8392/v1/upload_image \
  -F 'image=@landscape.png;type=image/png' -F 'full_path=landscape.png'
curl -fsS http://127.0.0.1:8392/v1/QuantStack/Wan2.2-TI2V-5B-GGUF \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"landscape.png","prompt":"A sailboat crosses the lake as morning mist lifts","width":832,"height":480,"num_frames":49,"steps":20,"seed":42}' \
  -o wan22-i2v.mp4
```

| 字段 | 类型/范围 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `prompt` | 1–12000 字符 | 必填 | 场景与运动描述 |
| `negative_prompt` | 最多 4000 字符 | 空字符串 | 可选负向提示词 |
| `image_path` | 可选字符串，最多 1024 字符 | `null` | 已上传图片的逻辑标识；省略时使用文生视频 |
| `width` / `height` | 256–1280，32 的倍数，面积不超过 1280×704 | 832 / 480 | 输出分辨率 |
| `num_frames` | `49`、`81` 或 `121` | `49` | 输出帧数，须符合 Wan latent 的时间压缩步长 |
| `steps` | 1–50 | `20` | 采样步数 |
| `seed` | `0` 到 `2^64-1` | `42` | 随机种子 |
| `fps` | 1–60 | `24` | 输出帧率 |

图片格式支持 PNG、JPEG 和 WebP，上传上限由共享 `UPLOAD_MAX_BYTES` 设置，默认 64 MiB。服务会先验证
图片可解码，再原子保存到 `storage/wan22/images/`；视频验证通过后原子保存到 `storage/video/`。

## 配置与健康检查

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `WAN22_TI2V_HOST` / `WAN22_TI2V_PORT` | `0.0.0.0` / `8392` | HTTP 监听地址 |
| `WAN22_TI2V_MODEL_DIR` | `$HF_MIRROR_DIR/QuantStack/Wan2.2-TI2V-5B-GGUF` | 本地三件套模型文件 |
| `WAN22_TI2V_COMFYUI_CODE_PATH` | `$HOME/LLM-depency/ComfyUI` | ComfyUI 核心源码 |
| `WAN22_TI2V_GGUF_NODE_PATH` | `$HOME/LLM-depency/ComfyUI-GGUF` | GGUF 自定义节点源码 |
| `WAN22_TI2V_IMAGE_DIR` | `storage/wan22/images/` | 上传图片 |
| `WAN22_TI2V_OUTPUT_DIR` | `storage/video/` | 生成 MP4 |
| `WAN22_TI2V_WORKER_TMP_DIR` | `storage/.cache/runtime/wan22_ti2v_worker/` | worker 请求、ComfyUI 输入/输出与日志 |
| `WAN22_TI2V_REQUEST_TIMEOUT` | `3600` 秒 | worker 总超时 |
| `WAN22_TI2V_CUDA_VISIBLE_DEVICES` | `0` | 单个可见 GPU 编号 |
| `WAN22_TI2V_WIDTH` / `WAN22_TI2V_HEIGHT` | `832` / `480` | 默认尺寸 |
| `WAN22_TI2V_LENGTH` | `49` | 默认帧数 |
| `WAN22_TI2V_STEPS` | `20` | 默认采样步数 |
| `WAN22_TI2V_SEED` | `42` | 默认随机种子 |
| `WAN22_TI2V_FPS` | `24` | 默认帧率 |
| `WAN22_TI2V_FFPROBE_BIN` | `ffprobe` | 输出 MP4 校验程序 |
| `GPU_LOCK_FILE` | `storage/.cache/runtime/gpu-runtime.lock` | 本地模型服务共享 GPU 锁 |
| `GPU_LOCK_WAIT_TIMEOUT` | `900` 秒 | GPU 排队时限 |
| `CUDA_RELEASE_DELAY` | `2.0` 秒 | worker 退出后释放 GPU 锁前的等待 |
| `LOCAL_FILES_ONLY` | `1` | 禁止请求期从模型仓库下载 |

`GET /v1/health` 不加载模型，逐项报告权重文件、ComfyUI/GGUF 源码、Python 依赖、ffprobe 和 CUDA 状态。
开始生成前检查 `available.selected_provider_ready` 为 `true`。
