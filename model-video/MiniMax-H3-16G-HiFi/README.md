# MiniMax H3 16G HiFi 本地服务

服务在 `8392` 提供 `POST /v1/minimax/H3-16G-HiFi`，支持 FL2VA 首尾帧生成和 Ref2VA 多参考图生成。请求字段 `include_audio` 默认 `false`，默认输出无音轨 MP4；传 `true` 才会解码并封装 H3 原生音轨。HTTP 进程不加载 Torch 或 ComfyUI；请求在共享 GPU 锁内启动一次性 worker 和 loopback ComfyUI 子进程，结束后回收进程组。

默认使用 HiFi FL2VA/Ref2VA 权重、20 步 `res_multistep/simple` sampler、960×544、5 秒和 24 FPS。H3 按 `17k+5` 帧网格对齐，5 秒请求输出 124 帧（约 5.17 秒）。请求最多 960×544、5 秒；进入队列前及拿到 GPU 锁后均要求至少 13,000 MiB 空闲显存，并让 ComfyUI 预留 2.5 GiB。该门槛不能通过配置调低；显存不足返回 `503`。

## 准备 ComfyUI

H3 原生节点在 ComfyUI 核心中；HiFi 工作流使用模型仓库附带的 Sol/Fused/Chunk 节点包。ComfyUI 源码和环境放在仓库外：

```bash
COMFYUI_DIR="$HOME/LLM-depency/MiniMax-H3-ComfyUI"
uv sync --project "$COMFYUI_DIR" --locked
unzip -o "$HOME/hf-mirror/minimax/MiniMax-H3-16G-HiFi/ComfyUI-sol-attn-main.zip" -d "$COMFYUI_DIR/custom_nodes"
```

ComfyUI 0.37.0 源码需放在 `COMFYUI_DIR`；项目配置锁定 Python 3.12.13、CUDA 13.0 PyTorch 2.10.0、Triton 3.6.0 和 ComfyUI `requirements.txt` 中的核心依赖（包括启动必需的 Pydantic 与 Pydantic Settings）。确认扩展入口为 `$COMFYUI_DIR/custom_nodes/ComfyUI-sol-attn-main/__init__.py`。服务的 `/v1/health` 会分别报告 ComfyUI H3 核心节点、Python、扩展、依赖和本地权重状态。模型默认位于 `$HOME/hf-mirror/minimax/MiniMax-H3-16G-HiFi`。

同步本服务轻量 API 依赖：

```bash
uv sync --project model-video/MiniMax-H3-16G-HiFi --locked
```

`start.sh` 不下载 ComfyUI、Python 包或模型权重。HTTP 环境通过 `MINIMAX_H3_COMFYUI_CODE_PATH`、`MINIMAX_H3_COMFYUI_PYTHON` 和 `MINIMAX_H3_MODEL_DIR` 覆盖默认路径。

## 调用

FL2VA 可只传提示词（T2VA），或传首帧、尾帧或两者：

```bash
curl -fsS http://127.0.0.1:8392/v1/upload_image \
  -F 'image=@first.png;type=image/png' -F 'full_path=first.png'
curl -fsS http://127.0.0.1:8392/v1/upload_image \
  -F 'image=@last.png;type=image/png' -F 'full_path=last.png'
curl -fsS http://127.0.0.1:8392/v1/minimax/H3-16G-HiFi \
  -H 'Content-Type: application/json' \
  -d '{"mode":"fl2va","prompt":"A red dragon slowly unfolds its wings in a sunlit cave","first_frame_path":"first.png","last_frame_path":"last.png","seed":42,"include_audio":false}' \
  -o minimax-h3.mp4
```

需要 H3 原生音轨时，将请求中的 `include_audio` 设为 `true`。静音请求不需要音频 VAE 权重，也不执行音频 VAE 解码或音轨封装；H3 的 DiT 仍联合采样音视频 latent，因此此选项不会跳过全部音频计算。

```json
{"mode":"fl2va","prompt":"A red dragon slowly unfolds its wings in a sunlit cave","first_frame_path":"first.png","include_audio":true}
```

Ref2VA 的图片也先通过 `/v1/upload_image` 上传，再按顺序传 `reference_image_paths`。提示词可用 `<Picture 1>`、`<Picture 2>` 等标记对应图片：

```json
{
  "mode": "ref2va",
  "prompt": "Use <Picture 1> as the identity reference and create a cinematic scene.",
  "reference_image_paths": ["person.png", "style.png"],
  "seed": 42
}
```

图片路径是上传时的逻辑标识，接口不会读取任意服务器路径。请求字段见 `GET /openapi.json`；健康、模型完整性、ComfyUI 和当前显存见 `GET /v1/health`。`available.selected_provider_ready` 报告静音视频就绪状态，`available.native_audio_ready` 单独报告原生音频所需权重是否可用。
