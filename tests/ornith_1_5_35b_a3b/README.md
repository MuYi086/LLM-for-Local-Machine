# Ornith-1.5-35B-A3B 样例测试

该目录提供不依赖模型的 API 合同测试，以及一条显式执行的真实模型样例生成脚本。无模型测试只使用 mock HTTP transport；真实推理不会进入仓库质量门禁。

## 生成真实样例

先在另一个终端启动服务：

```bash
bash start.sh
```

再生成一条中文样例：

```bash
uv run --project qa --locked python tests/ornith_1_5_35b_a3b/generate_example.py
```

默认输出到 `storage/llm_tests/ornith_1_5_35b_a3b/`。可以通过 `ORNITH_TEST_OUTPUT_DIR` 或 `--output-dir` 改目录，通过 `--prompt` 改提示，通过 `--timeout` 调整最长等待时间。脚本只保存 assistant 文本，不覆盖同名文件。

也可直接检查流式 API：

```bash
curl -N http://127.0.0.1:8401/v1/AtomicChat/Ornith-1.5-35B-A3B-AD-Q5 \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"请用中文介绍一下你自己。"}],"max_tokens":128,"stream":true}'
```

主 GGUF、F16 mmproj 和独立 `mtp-Ornith-1.5-35B-A3B-Q8_0.gguf` 从 `$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/` 读取。llama-server 必须是 CUDA 构建且报告 `CUDA0`，默认用 CUDA 承载模型层、将前 23 层 MoE 专家放到 CPU，并在空闲后卸载模型。每个请求使用公共 GPU 锁；如果 backend 未确认休眠，服务会返回 503 并保留 quarantine 标记。

`start.sh` 默认使用线程 6/8、主模型 Q8 KV、F16 草稿 KV、MTP 草稿长度 4 和 load-mode none；复测步骤与完整结论见 [网页任务对比](../model_web_comparison/README.md)。
