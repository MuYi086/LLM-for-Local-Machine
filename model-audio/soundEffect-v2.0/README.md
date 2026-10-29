# MOSS-SoundEffect v2.0 文本音效服务

服务默认监听 `8362`，生成入口为 `POST /v1/moss/soundEffect-v2.0`。它使用 OpenMOSS 的
MOSS-SoundEffect v2.0，根据中文或英文提示词生成 48 kHz 单声道 WAV，时长最多 30 秒。
HTTP 进程只做参数和本地依赖检查、共享 GPU 排队及一次性 worker 编排；模型只在 worker 内加载。

## 本地准备

模型快照默认位于 `$HF_MIRROR_DIR/OpenMOSS/MOSS-SoundEffect-v2.0`，`HF_MIRROR_DIR` 默认
`$HOME/hf-mirror`。服务检查 4 个权重的完整字节数和必需配置文件，不会在推理时下载模型。

官方推理源码不提交到本仓库，固定使用 MOSS-TTS revision
`934d6826b084c46a0d033402174d5f8ac4ed2519`。在仓库外准备该源码并同步隔离环境：

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

服务环境依据官方推理依赖固定 Diffusers、Transformers、PyTorch 2.9.0 和 CUDA 12.8 wheel。
启动使用 `uv run --no-sync`，不会在请求或服务启动时安装依赖。

## 请求

请求体为 JSON：

| 字段 | 默认值 | 边界 | 说明 |
| --- | --- | --- | --- |
| `prompt` | 必填 | 1–8000 字符，不能只含空白 | 中英文声音描述 |
| `seconds` | `10` | 大于 0 且不超过 30 | 目标音频时长 |
| `num_inference_steps` | `100` | 1–200 | Flow Matching 推理步数 |
| `cfg_scale` | `4.0` | 0–20 | Classifier-free guidance 强度 |
| `sigma_shift` | `5.0` | 0–20 | Flow Matching scheduler shift |
| `seed` | `0` | 0–4294967295 | 随机种子 |

示例：

```bash
curl -fsS http://127.0.0.1:8362/v1/moss/soundEffect-v2.0 \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"雨水均匀地落在金属屋顶上","seconds":8,"seed":42}' \
  -o sound-effect.wav
```

成功返回 `audio/wav`，响应头包含 `X-Audio-Sample-Rate: 48000` 和实际时长。生成成品原子保存到
`storage/sound_effect_v2/`（可用 `MOSS_SOUNDEFFECT_V2_OUTPUT_DIR` 覆盖）。

## 配置与健康检查

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `MOSS_SOUNDEFFECT_V2_HOST` / `MOSS_SOUNDEFFECT_V2_PORT` | `0.0.0.0` / `8362` | 服务监听地址 |
| `MOSS_SOUNDEFFECT_V2_MODEL_DIR` | `$HF_MIRROR_DIR/OpenMOSS/MOSS-SoundEffect-v2.0` | 本地模型快照 |
| `MOSS_SOUNDEFFECT_V2_CODE_PATH` | `$HOME/LLM-depency/MOSS-TTS/moss_soundeffect_v2` | 官方推理源码目录 |
| `MOSS_SOUNDEFFECT_V2_OUTPUT_DIR` | `$STORAGE_DIR/sound_effect_v2` | 成品 WAV 目录 |
| `MOSS_SOUNDEFFECT_V2_REQUEST_TIMEOUT` | `900` 秒 | 单次 worker 时限 |
| `MOSS_SOUNDEFFECT_V2_DEVICE` | `cuda` | worker 推理设备 |
| `GPU_LOCK_FILE` | `$RUNTIME_CACHE_DIR/gpu-runtime.lock` | 跨模型共享 GPU 锁 |
| `GPU_LOCK_WAIT_TIMEOUT` | `900` 秒 | GPU 队列最大等待时间，必须为正数 |
| `CUDA_RELEASE_DELAY` | `2` 秒 | worker 退出后释放 CUDA 资源的等待时间 |

`GET /v1/health` 检查权重、配置、上游源码、Python 依赖和 GPU 可见性，不加载模型。
HTTP 200 仅表示健康接口可访问；`available.ready` 才表示本地推理依赖已准备完整。
服务不会导入音频上传、参考克隆、母带或分离功能。

## 无模型验证

```bash
uv run --project qa --locked python -m unittest tests.test_moss_soundeffect_service -v
uv run --project model-audio/soundEffect-v2.0 --locked ruff check model-audio/soundEffect-v2.0
```

测试 mock worker，不下载权重、不使用 CUDA，也不执行真实推理。
