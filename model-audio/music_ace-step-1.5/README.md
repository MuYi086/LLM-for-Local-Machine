# ACE-Step 1.5 XL Turbo BGM 服务

这是 LLM Local Backend 独立的 ACE-Step 1.5 XL Turbo Diffusers 服务，默认监听 `8313`。
它只负责生成有声小说背景音乐；模型不会在 HTTP 进程常驻，而是每个请求启动一个
worker，完成推理后退出并释放 CUDA。

## 依赖准备

本目录是独立 uv project。当前 Diffusers 官方文档要求从 Git 版本使用
`AceStepPipeline`，因此 `pyproject.toml` 不依赖整个 ACE-Step 原仓库，也不会在
服务启动时联网安装依赖。

部署时由操作者手动执行：

```bash
uv sync --project model-audio/music_ace-step-1.5 --locked
```

以上命令从仓库根目录执行。

## 模型与运行配置

默认权重目录：

```text
~/hf-mirror/ACE-Step/acestep-v15-xl-turbo-diffusers
```

健康检查只验证 `model_index.json`、组件目录和 `transformer/*.safetensors`，不会
假设单一权重文件名。目标机器默认使用 BF16、model CPU offload、VAE tiling 和
共享 GPU 锁；长音频发生 OOM 时可将 `ACESTEP_OFFLOAD=sequential`。

主要环境变量：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `ACESTEP_MODEL_DIR` | `~/hf-mirror/ACE-Step/acestep-v15-xl-turbo-diffusers` | 本地权重 |
| `BGM_STORAGE_DIR` | `storage/bgm` | 生成 WAV 保存目录 |
| `ACESTEP_OFFLOAD` | `model` | `model` / `sequential` / `none` |
| `ACESTEP_VAE_TILING` | `1` | 启用 VAE tiling |
| `ACESTEP_DEFAULT_SECONDS` | `60` | 默认时长 |
| `ACESTEP_DEFAULT_STEPS` | `8` | 默认推理步数 |
| `ACESTEP_REQUEST_TIMEOUT` | `1800` | worker 超时秒数 |
| `ACESTEP_PORT` | `8313` | 启动脚本使用的端口 |

## HTTP 接口

### `GET /v1/health`

只做本地权重、依赖可见性和 `nvidia-smi` 检查，不加载模型。

### `POST /v1/aceStep/bgm`

请求 JSON：

```json
{
  "prompt": "Dark cinematic ambient underscore, sparse felt piano, low cello drone, designed underneath spoken narration, no vocals",
  "seconds": 60,
  "steps": 8,
  "bpm": 58,
  "keyscale": "D minor",
  "timesignature": "4",
  "seed": -1
}
```

成功时返回 `audio/wav`，原生 48kHz 双声道，并附带 `X-ACE-Step-Seed`、
`X-ACE-Step-Sample-Rate` 和 `X-ACE-Step-Model` 响应头。

## 无模型测试

测试不会下载权重、导入真实 Diffusers Pipeline、调用 CUDA 或执行网络请求：

```bash
(cd model-audio/music_ace-step-1.5 && uv run --project ../../qa --locked python -m unittest discover -s tests -v)
```
