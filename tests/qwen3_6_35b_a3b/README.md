# Qwen3.6-35B-A3B-MTP 测试

`test_service.py` 和 `test_generate_example.py` 只使用 mock HTTP transport，不加载模型，也不需要 CUDA。

服务启动后，生成真实文本样例：

```bash
uv run --project qa --locked python tests/qwen3_6_35b_a3b/generate_example.py
```

默认请求 `http://127.0.0.1:8402/v1/qwen/Qwen3.6-35B-A3B-MTP`，并将非空结果写入
`storage/llm_tests/qwen3_6_35b_a3b/`。可通过 `QWEN36_TEST_OUTPUT_DIR` 覆盖输出目录。
