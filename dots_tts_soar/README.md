# dots.tts-soar uv 服务

这是 `dots.tts-soar` 的独立 Python 3.12.13 uv 服务，默认监听 `8324`。HTTP
控制面在 `main.py`，模型推理在 `worker.py`；每个合成请求都会启动一个同一 uv
环境中的 worker，完成后清理 CUDA 并退出。

## 安装和启动

```bash
uv sync --project dots_tts_soar --locked
uv run --project dots_tts_soar python dots_tts_soar/main.py
```

模型默认读取 `$HF_MIRROR_DIR/rednote-hilab/dots.tts-soar`，普通参考音频和克隆输出
目录使用根项目的 `storage/clone/`；音色设计音频只使用 `storage/timbre/`，同步到本服务
时仅在 `storage/timbre/.references/` 保存引用映射。运行缓存使用 `storage/.cache/runtime/`。
可以通过 `DOTS_TTS_SOAR_MODEL_DIR`、`PROMPTS_DIR`、`RUNTIME_CACHE_DIR`、
`GPU_LOCK_FILE` 和现有 `DOTS_TTS_SOAR_*` 参数覆盖。

## 接口

保持 WebUI 兼容的接口为：

- `GET /v1/health`
- `POST /v1/upload_audio`
- `GET /v1/check/audio?file_name=...`
- `POST /v1/dotsTTS/clone`

`/v1/dotsTTS/clone` 成功返回 `audio/wav`，默认输出 48 kHz 单声道。`prompt_text`
可以直接传入，也可以使用上传时保存的 sidecar；省略参考文本时保留官方
x-vector-only cloning 行为。

## FlashAttention 判断

本服务不需要 `flash_attn`。官方 `dots.tts` runtime 使用 PyTorch 原生
`torch.nn.attention.flex_attention`，项目依赖和 uv lock 均不包含 `flash-attn`。
`/v1/health` 会报告 `available.flash_attn` 和
`runtime.flash_attention_policy`。当前 `/home/muyi086/tts-depency/flash-attention`
源码仓库既不是本服务的运行依赖，也不能替代已编译、与当前 Torch/CUDA/Python
匹配的扩展。该 checkout 的 Hopper FlashAttention 3 路径面向 H100/H800，而
当前机器是 RTX 4070 Ti SUPER；即使未来需要性能优化，也应先针对实际 Torch、
CUDA 和 GPU 架构单独编译并做 canary，不要在默认迁移路径中编译它。

## 迁移状态

dots.tts-soar 已完成迁移，`start.sh` 固定使用本目录的 uv 项目启动 8324
服务。旧 API、worker 和对应 Conda 环境均已移除。
