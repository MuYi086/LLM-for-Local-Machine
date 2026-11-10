# 图像模型

当前接入 MOSS-Image-SFT-Q8_0，服务目录为
[`MOSS-Image-SFT-Q8_0/`](MOSS-Image-SFT-Q8_0/)。它复用 `inference_gateway.sdcpp` 的结构化参数、
共享 GPU 锁、进程组回收和 quarantine 恢复。

`start.sh` 默认在 8501 端口启动服务，路由为
`POST /v1/openmoss/MOSS-Image-SFT-Q8_0`。模型目录、Qwen3.5 文本编码器、FLUX.2 VAE 和官方 tokenizer
均从 `$HF_MIRROR_DIR/stable-diffusion-cpp-gguf/MOSS-Image-SFT-GGUF/` 只读加载；模型文件和 sd.cpp CUDA
二进制不放入仓库。默认参数和资源限制见 [`sdcpp/manifests/models.json`](../sdcpp/manifests/models.json)。

设置 sd.cpp 二进制及其 SHA-256 后启动完整服务：

```bash
export SDCPP_BIN="$HOME/.local/src/stable-diffusion.cpp/build/bin/sd-cli"
export SDCPP_BIN_SHA256="$(sha256sum "$SDCPP_BIN" | cut -d ' ' -f 1)"
bash start.sh
```

服务启动前会校验二进制摘要和四个模型组件。健康检查为
`GET http://127.0.0.1:8501/v1/health`，不加载模型。

服务运行期间可生成真实示例：

```bash
uv run --project qa --locked python tests/MOSS-Image-SFT-Q8_0/generate_example.py
```

示例输出写入 `storage/llm_tests/MOSS-Image-SFT-Q8_0/`，每次使用独立文件名且不覆盖已有产物；API
推理中间产物保留在 `$RUNTIME_CACHE_DIR/sdcpp/<request-id>/`。这些运行结果不提交到仓库。
