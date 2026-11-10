# 图像模型

当前接入两种 stable-diffusion.cpp 图片服务。两个服务复用 `inference_gateway.sdcpp` 的结构化参数、共享 GPU 锁、
进程组回收和 quarantine 恢复，模型文件与 CUDA `sd-cli` 保持在仓库外。

## MOSS-Image-SFT-Q8_0

服务目录为 [`MOSS-Image-SFT-Q8_0/`](MOSS-Image-SFT-Q8_0/)，默认端口 8501，路由为
`POST /v1/openmoss/MOSS-Image-SFT-Q8_0`。四个只读组件从
`$HF_MIRROR_DIR/stable-diffusion-cpp-gguf/MOSS-Image-SFT-GGUF/` 加载。

## FLUX-3-Image-Klein-GGUF-Q8

服务目录为 [`FLUX-3-Image-Klein-GGUF-Q8/`](FLUX-3-Image-Klein-GGUF-Q8/)，默认端口 8502。模型清单读取：

```text
$HF_MIRROR_DIR/stable-diffusion-cpp-gguf/FLUX-3-Image-Klein-GGUF/
├── flux3-image-klein-Q8_0.gguf
├── Qwen3VL-4B-Instruct-Q8_0.gguf
├── mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
└── ae.safetensors
```

文生图路由为 `POST /v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8`；图片编辑路由为其 `/edit` 子路径，
通过 `image` multipart 字段上传 PNG/JPEG 参考图。上传默认限制为 64 MiB，使用 `$RUNTIME_CACHE_DIR/sdcpp_uploads`
下的唯一临时文件，在推理结束后删除。默认生成尺寸为 1024×1024，16 步、CFG 1.0、Euler，并启用 CPU offload。
编辑调用由模型清单映射到 stable-diffusion.cpp 的 `--ref-image`。

启动前设置 sd.cpp 二进制及 SHA-256，并准备两种模型的全部外部组件：

```bash
export SDCPP_BIN="$HOME/.local/src/stable-diffusion.cpp/build/bin/sd-cli"
export SDCPP_BIN_SHA256="$(sha256sum "$SDCPP_BIN" | cut -d ' ' -f 1)"
bash start.sh
```

健康检查为 `GET http://127.0.0.1:8502/v1/health`，不加载权重。真实生成和编辑 demo：

```bash
uv run --project qa --locked python tests/FLUX-3-Image-Klein-GGUF-Q8/generate_example.py
```

Demo 会先生成图片，再把生成结果作为参考图提交到 `/edit`，并把两个 PNG 以独立文件名写入
`storage/llm_tests/FLUX-3-Image-Klein-GGUF-Q8/`。真实 FLUX.3 Klein 推理尚未由无模型质量门禁验证；运行该 demo
需要本机模型文件、CUDA sd.cpp 二进制和已启动服务。

MOSS demo 见 [`tests/MOSS-Image-SFT-Q8_0/`](../tests/MOSS-Image-SFT-Q8_0/)。完整模型清单和参数限制见
[`sdcpp/manifests/models.json`](../sdcpp/manifests/models.json)，provider 及隔离恢复流程见
[`docs/sdcpp-runtime.md`](../docs/sdcpp-runtime.md)。
