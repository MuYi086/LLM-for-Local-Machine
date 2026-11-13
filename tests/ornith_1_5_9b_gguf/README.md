# Ornith-1.5-9B-GGUF 调用示例

96k OpenCode 性能分析、MTP/batch 对照与回退方法见 [performance.md](performance.md)。

摘要预算保护的真实验证：`uv run --project qa --locked python tests/ornith_1_5_9b_gguf/benchmark_context_budget.py`。它通过实际适配器与共享锁核对模板分词/usage，并在生成前拒绝挤占输出空间的请求；使用已运行的 loopback backend，不启动或重启服务。它不属于无模型门禁，机制与生效步骤见 [压缩预算修复](../../docs/opencode-compaction-budget.md)。

跨休眠保存主模型、MTP 草稿及混合注意力状态的优化见 [部署与实测记录](../../docs/llama-cpp-sleep-cache.md)。真实多轮追加、工具结果、分叉、前部改写、换题和容量回退由 `benchmark_sleep_cache.py` 显式运行验证；它不属于无模型质量门禁。

显式运行后端参数对照实验（须先配置下列权重与可执行文件环境变量）：

```bash
uv run --project main --no-sync python tests/ornith_1_5_9b_gguf/tune_backend.py \
  --server-bin "$LLAMA_SERVER_BIN" --model-file "$ORNITH_9B_MODEL_FILE" \
  --profiles baseline mtp4-b1024 --repeats 2 --context-copies 11
```

实验脚本在独立 loopback 端口 28504 使用既有 startup/shutdown 助手管理临时 backend；启动、推理和退出均取得共享 GPU 锁，推理前核验既有 8504 backend 休眠。异常卸载无法确认时保留 quarantine。若部署覆盖了 GPU 锁或既有 backend 地址，需传入相同的 `--lock-file`、`--existing-backend`（也读取 `GPU_LOCK_FILE`、`ORNITH_9B_LLAMA_BASE_URL`）。请先让 OpenCode 停止发请求；脚本不重启既有服务。原生 timings、首 token、RSS、guest 内存与整卡显存及输出样本保存于运行数据目录；不属于无模型测试。

此目录包含不加载模型的 API 合同测试和一条显式执行的真实模型调用 demo。服务由仓库根目录的 `start.sh` 启动；9B API 监听 `8503`，llama-server backend 只绑定 `127.0.0.1:8504`。模型从 `$HF_MIRROR_DIR/ornith-ai/Ornith-1.5-9B-GGUF/Ornith-1.5-9B-Q8_0.gguf` 只读加载，每次推理使用与其他 GPU provider 共用的 GPU 锁。

要验证实际模型调用并记录性能，服务启动后运行：

```bash
uv run --project qa --locked python tests/ornith_1_5_9b_gguf/benchmark_live.py
```

实测脚本先检查 API 路由、backend 的 sleep-idle 状态和 GPU quarantine 标记，再通过 8503 的 API 流式请求一条短中文提示，不直连 backend。它以默认 0.25 秒间隔记录 GPU 0 全局显存、llama-server RSS、WSL/Linux guest 已用及可用内存，并计算端到端首 token 延迟和正文生成阶段平均 token/s。输出文本、JSON 与 Markdown 报告独占写入 `storage/llm_tests/ornith_1_5_9b_gguf/`；显存统计包含同一 GPU 上其他进程的占用。

可调整生成长度、采样间隔和报告目录：

```bash
uv run --project qa --locked python tests/ornith_1_5_9b_gguf/benchmark_live.py \
  --max-tokens 256 \
  --sample-interval 0.25 \
  --output-dir storage/llm_tests/ornith_1_5_9b_gguf
```

启动后生成一条中文样例：

```bash
uv run --project qa --locked python tests/ornith_1_5_9b_gguf/generate_example.py
```

脚本请求 `POST /v1/ornith-ai/Ornith-1.5-9B-GGUF`，只保存 assistant 文本，默认输出到 `storage/llm_tests/ornith_1_5_9b_gguf/`。文件以独占方式创建，不会覆盖已有样例。

可通过参数自定义请求和输出位置：

```bash
uv run --project qa --locked python tests/ornith_1_5_9b_gguf/generate_example.py \
  --base-url http://127.0.0.1:8503 \
  --prompt '用中文解释什么是混合量化。' \
  --max-tokens 256 \
  --timeout 1800 \
  --output-dir storage/llm_tests/ornith_1_5_9b_gguf
```

也可使用 `ORNITH_9B_TEST_BASE_URL`、`ORNITH_9B_TEST_OUTPUT_DIR` 设置默认地址和目录。服务端口、上下文、线程和 GPU 层配置见仓库根目录 [README](../../README.md) 的环境变量表。

无模型测试命令：

```bash
uv run --project qa --locked python -m unittest discover -s tests/ornith_1_5_9b_gguf -t . -v
```

测试使用 mock HTTP transport，不下载权重、不调用外部模型服务，也不执行真实推理。
实测指标工具的无模型单元测试可单独运行：

```bash
uv run --project qa --locked python -m unittest tests.ornith_1_5_9b_gguf.test_benchmark_live -v
```
