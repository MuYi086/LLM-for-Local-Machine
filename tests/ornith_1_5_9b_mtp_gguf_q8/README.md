# Ornith-1.5-9B-MTP-GGUF-Q8 调用示例

服务由仓库根目录 `start.sh` 启动。新 API 监听 `8504`，路由为
`/v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8`；llama-server backend 只绑定 loopback
`127.0.0.1:8506`。模型与 vision projector 从 `$HF_MIRROR_DIR/ornith-ai/Ornith-1.5-9B-MTP-GGUF/`
只读加载。请求通过与仓库其他 GPU provider 共用的 GPU 锁。

启动服务后生成一条文本样例：

```bash
uv run --project qa --locked python tests/ornith_1_5_9b_mtp_gguf_q8/generate_example.py
```

为模型提供本地 PNG 或 JPEG 图片：

```bash
uv run --project qa --locked python tests/ornith_1_5_9b_mtp_gguf_q8/generate_example.py \
  --image /path/to/image.png \
  --prompt '请描述图片中的主要内容。'
```

脚本只保存 assistant 文本，默认输出至
`storage/llm_tests/ornith_1_5_9b_mtp_gguf_q8/`，使用独占方式创建文件，不会覆盖已有样例。
可用 `ORNITH_MTP_9B_TEST_BASE_URL` 和 `ORNITH_MTP_9B_TEST_OUTPUT_DIR` 修改 API 地址和输出目录。

无模型合同测试不访问真实 API：

```bash
uv run --project qa --locked python -m unittest discover \
  -s tests/ornith_1_5_9b_mtp_gguf_q8 -t . -v
```

真实调用 demo 与无模型质量门禁分开运行；它不下载权重，也不会保存输入图片。
