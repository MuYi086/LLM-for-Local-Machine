# 文本模型网页任务记录

当前唯一可运行的文本模型是 Ornith。`run_comparison.py` 和 `run_tuning_probe.py` 只向现存的 Ornith API 发请求。

此目录中的 Qwen3.6 对比报告、JSON、网页检查记录和原始 HTML 都是 2026-11-09 的历史测试结果，保留用于追溯，不代表当前服务仍然存在。历史对比使用同一份 [task_prompt.txt](task_prompt.txt)，并按 Ornith、Qwen 的顺序串行请求；当时的两模型启动参数和结果在归档报告中。Qwen3.6 服务、代码及权重现已移除。

## 当前 Ornith 复测

先正常重启服务，再确认 Ornith backend 的 `/props.is_sleeping` 为 true，且没有遗留 quarantine。若筛选参数，可运行：

```bash
uv run --project qa --locked python tests/model_web_comparison/run_tuning_probe.py --label ornith-profile --models ornith --repeats 2
```

探测数据默认写入 `storage/llm_tests/web_comparison_probes/`，拒绝覆盖同名文件。完整网页任务通过下面命令执行：

```bash
uv run --project qa --locked python tests/model_web_comparison/run_comparison.py
```

脚本保留原提示词、请求用量和资源数据，只生成 Ornith 的报告与网页。运行前先退出已启动的 OpenCode/模型客户端，避免并发使用共享 GPU。

Ornith 的已选参数为 CUDA0、23 层 CPU MoE、线程 6/8、主 KV Q8、MTP 草稿长度 4、草稿 KV F16、上下文 32768、batch 512、load-mode none。参数筛选与真实模型请求需由使用者手动安排；本仓库维护者不在质量门禁中运行模型。

## 历史结果

旧版双模型报告曾记录 Ornith 与 Qwen3.6 的速度、资源和网页质量。它们保存在 `baseline_report.json`、`optimized_report.json`、`optimized_v2_report.json`、`optimized_v3_*.json`、对应 Markdown/HTML，以及 `tests/qwen3_6_35b_a3b/outputs/`。这些数据保持原样；后续任务不再重跑或更新 Qwen 项。
