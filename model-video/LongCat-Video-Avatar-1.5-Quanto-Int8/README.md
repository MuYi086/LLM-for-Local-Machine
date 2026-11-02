# LongCat-Video-Avatar-1.5 Quanto-Int8

独立 FastAPI 服务监听 `8390`，通过 `POST /v1/meituan/LongCat-Video-Avatar-1.5-Quanto-Int8`
提供单人音频驱动视频生成。8390 只处理上传、输入预检、共享 GPU 排队和 MP4 保存；每个请求由一次性 worker
调用 WanGP 官方 Python API。输入权重保持只读，结果写入 `storage/video/longcat_avatar/`。

## WanGP 与 MMGP

推理使用仓库外的 [WanGP](https://github.com/deepbeepmeep/WanGP) 项目及其推荐的
`shared.api.init()` / `session.submit_task()` 调用方式，模型类型为 `longcat_avatar_v1_5`。
WanGP 仓库内已经包含 `mmgp/`，默认 Profile 0 会按工作负载动态预加载显存，并避免长期固定主模型权重到预留主存；无需另行
clone `mmgp` 或在 8390 的 uv 环境安装它。服务将推理依赖放在 WanGP 自己的 Python 环境中，8390 环境只装
FastAPI、Pillow 和共享运行包。

可通过 `LONGCAT_AVATAR_MEMORY_PROFILE=4` 回退到 Profile 4；Profile 4 会将主 Transformer 长期保留在预留主存并逐块送入显存。Profile 0/4 均通过同一环境变量配置，其他档位会在启动时拒绝。

WanGP 建议使用 Python 3.11 与匹配显卡的 CUDA PyTorch。当前主机使用 WanGP `.venv`、PyTorch 2.10.0
CUDA 13.0、Triton 3.6；从 WanGP 目录可按项目要求准备环境：

```bash
cd "$HOME/LLM-depency/WanGP"
uv venv --seed --python 3.11.15 .venv
uv pip install --python .venv/bin/python \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python 'triton>=3.6,<3.7'
```

如果使用 WanGP 安装器建立在其他目录的环境，通过 `LONGCAT_AVATAR_WANGP_PYTHON` 指向该环境的 Python。
8390 默认读取 `$HOME/LLM-depency/WanGP/.venv/bin/python`。

## 本地模型文件

| 环境变量 | 默认目录 | 需要的文件 |
| --- | --- | --- |
| `LONGCAT_AVATAR_MODEL_DIR` | `$HF_MIRROR_DIR/meituan/LongCat-Video-Avatar-1.5` | Quanto DiT、Quanto T5、`umt5-xxl/` tokenizer、`lora/dmd_lora.safetensors`、`whisper-large-v3/` 的模型及配置 |
| `LONGCAT_AVATAR_BASE_MODEL_DIR` | `$HF_MIRROR_DIR/meituan/LongCat-Video` | `vae/diffusion_pytorch_model.safetensors` |
| `LONGCAT_AVATAR_WANGP_DIR` | `$HOME/LLM-depency/WanGP` | WanGP 源码、自带 MMGP 和独立 Python 环境 |

WanGP 从自己的源码目录读取 Avatar scheduler、DiT/ VAE 配置和推理代码，不需要另装 LongCat 官方源码。
`configure_wangp.py` 在 `WanGP/ckpts/` 与 `WanGP/loras/` 建立指向上表文件的软链接，并更新
`wgp_config.json`，将检查点目录设为 `ckpts`、视频档设为当前 MMGP Profile（默认 0）、attention 设为 SDPA。
不会复制或覆盖模型文件；遇到同名的非目标文件时会停止并报出冲突。

部署时先安装轻量 HTTP 环境，再启动服务：

```bash
uv sync --project model-video/LongCat-Video-Avatar-1.5-Quanto-Int8 --locked
bash start.sh
```

`start.sh` 在启动 HTTP 进程前会执行 `configure_wangp.py`，配置缺失时由 WanGP API 生成默认配置后再应用
本服务需要的模型目录与低显存档。单独启动 8390 时，先用 WanGP Python 执行该配置脚本：

```bash
"$HOME/LLM-depency/WanGP/.venv/bin/python" \
  model-video/LongCat-Video-Avatar-1.5-Quanto-Int8/configure_wangp.py
uv run --project model-video/LongCat-Video-Avatar-1.5-Quanto-Int8 --locked \
  python model-video/LongCat-Video-Avatar-1.5-Quanto-Int8/main.py
```

默认只暴露 GPU 0，要求至少有 `10,000 MiB` 空闲显存；可通过
`LONGCAT_AVATAR_CUDA_VISIBLE_DEVICES` 和 `LONGCAT_AVATAR_MIN_FREE_VRAM_MIB` 调整。MMGP Profile 0 在分块运行时会按实测工作负载使用空闲显存，并让预留主存供正在运行的模型部分共享；Profile 4 则长期保留主 Transformer 权重在预留主存。两者都能运行大于显存的模型。共享 GPU 锁默认最多排队 900 秒。
输入音频最多约 16.52 秒，WanGP 按 93 帧滑窗、13 帧重叠延续生成，服务最多接收 5 段覆盖范围内的音频。

排查阶段耗时时可设置 `LONGCAT_AVATAR_TRACE_DIR` 为可写目录。每个 worker 会生成一个 JSONL，记录 worker 阶段计时和 WanGP 的进度/状态回调；不设置时不会写 trace，也不会改变推理参数。诊断文件写入失败不会中断推理。

`GET /v1/health` 会检查权重、WanGP 源码/依赖、配置、软链接、FFmpeg 和显存；健康检查不会加载模型或执行
生成。模型真实运行情况以完整推理请求为准。

## API

健康状态：

```bash
curl -fsS http://127.0.0.1:8390/v1/health
```

上传音频与参考图片。WanGP 的 LongCat Avatar 运行配置会强制启用参考图模式，因此图片必填；`full_path` 是安全逻辑文件名，不能包含目录；文件按块暂存、计算 SHA-256，再原子提交。

```bash
curl -fsS http://127.0.0.1:8390/v1/upload_audio \
  -F 'full_path=speech.wav' -F 'audio=@speech.wav'
curl -fsS http://127.0.0.1:8390/v1/upload_image \
  -F 'full_path=portrait.png' -F 'image=@portrait.png'
```

生成请求：

```bash
curl -f http://127.0.0.1:8390/v1/meituan/LongCat-Video-Avatar-1.5-Quanto-Int8 \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"一位年轻女性在明亮的咖啡馆里自然说话","audio_path":"speech.wav","image_path":"portrait.png","resolution":"480p"}' \
  --output avatar.mp4
```

JSON 字段：

- `prompt`：必填，1–12,000 字符。
- `audio_path`：必填，先上传得到的音频逻辑文件名。
- `image_path`：必填，先上传得到的参考图片逻辑文件名。
- `resolution`：可选，`480p` 或 `720p`，默认 `480p`。
- `num_segments`：可选，1–5；显式值必须覆盖完整音频。

成功输出经 ffprobe 确认同时包含视频和音频流，再原子保存到 `LONGCAT_AVATAR_OUTPUT_DIR`。请求工作区、WanGP
临时输出和 worker 进程组会在成功、失败或超时后清理。
