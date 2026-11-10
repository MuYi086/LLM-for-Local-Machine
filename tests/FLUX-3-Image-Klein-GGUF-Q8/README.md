# FLUX-3-Image-Klein-GGUF-Q8 HTTP 示例

先根据 [`model-image/README.md`](../../model-image/README.md) 准备仓库外的 FLUX.3 Klein Q8 权重、Qwen3-VL 4B
文本模型、mmproj、VAE 和 CUDA `sd-cli`，然后运行 `bash start.sh`。图片 API 默认监听 8502。

运行生成与编辑 demo：

```bash
uv run --project qa --locked python tests/FLUX-3-Image-Klein-GGUF-Q8/generate_example.py
```

脚本先向 `POST /v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8` 发送 1024×1024、16 步、CFG 1.0 的文生图请求；
然后将该 PNG 作为 `image` multipart 文件发给 `/edit`，并要求模型修改自行车颜色。两次请求都校验 PNG 签名，输出原子写入
`storage/llm_tests/FLUX-3-Image-Klein-GGUF-Q8/`，使用独立文件名且不覆盖已有文件。该目录属于运行数据，不应提交。

可通过环境变量覆盖服务地址：

```bash
FLUX3_IMAGE_BASE_URL=http://127.0.0.1:8502 \
  uv run --project qa --locked python tests/FLUX-3-Image-Klein-GGUF-Q8/generate_example.py
```

图片编辑路由只接受 PNG/JPEG，单张参考图默认上限 64 MiB。无模型测试不加载 FLUX 权重，也不验证本机实际生成效果。
