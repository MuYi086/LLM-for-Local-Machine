# MOSS-Image-SFT-Q8_0 HTTP 示例

先按 [`model-image/README.md`](../../model-image/README.md) 配置仓库外的 stable-diffusion.cpp CUDA `sd-cli`、
SHA-256 和四个 MOSS 模型组件，并通过 `bash start.sh` 启动全部服务。图片 API 默认监听 8501。

执行 HTTP demo：

```bash
uv run --project qa --locked python tests/MOSS-Image-SFT-Q8_0/generate_example.py
```

脚本向 `POST /v1/openmoss/MOSS-Image-SFT-Q8_0` 发送固定提示词与 1024×1024、20 步、CFG 3、seed 13579，
校验返回的 PNG 签名，并将图片原子写入 `storage/llm_tests/MOSS-Image-SFT-Q8_0/`。每次生成使用唯一文件名，
不会覆盖已有图片。该目录属于运行数据，不应提交。

脚本支持通过 `MOSS_IMAGE_BASE_URL` 覆盖 API 地址，例如：

```bash
MOSS_IMAGE_BASE_URL=http://127.0.0.1:8501 \
  uv run --project qa --locked python tests/MOSS-Image-SFT-Q8_0/generate_example.py
```
