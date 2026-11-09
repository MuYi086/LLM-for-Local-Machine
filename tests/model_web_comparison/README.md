# Ornith / Qwen 网页任务复测

两款模型使用同一份 [task_prompt.txt](task_prompt.txt)，生成可离线打开的「周末城市微旅行规划器」。测试脚本通过公开 API 串行调用，不绕过共享 GPU 锁；每次请求结束后等待 backend 卸载。

## 当前默认配置

直接运行仓库根目录的 `bash start.sh` 即使用三次完整实测选定的参数，不需要额外的优化启动脚本。环境变量仍可覆盖默认值。

| 项目 | Ornith | Qwen3.6 |
| --- | --- | --- |
| CPU MoE 专家层数 | 23 | 23 |
| 生成 / 批量线程 | 6 / 8 | 8 / 8 |
| 主模型 K/V 缓存 | Q8 / Q8 | Q8 / Q8 |
| MTP | 独立 Q8_0 文件 | 主 GGUF 内置 |
| 最大草稿长度 | 4 | 4 |
| 草稿 K/V 缓存 | F16 / F16 | F16 / F16 |
| 加载方式 | none | none |
| 上下文 / batch | 32768 / 512 | 32768 / 512 |

保持原 Q5 权重、Ornith F16 mmproj、CUDA0、单并发、共享 GPU 锁、请求后 sleep-idle 卸载及 CUDA 释放等待。`ORNITH_SPEC_TYPE=none` 可显式关闭 Ornith MTP。

## 保留的测试步骤

1. 正常停止已有服务，再运行 `bash start.sh`。核对实际后端参数，并确认两款 backend 的 `/props.is_sleeping` 都为 true，且没有遗留 quarantine。
2. 若要比较参数，固定权重、提示词、上下文、batch 和采样方式；每次只改变一个变量，通过环境变量设置并正常重启。
3. 使用 1024 token、每项两次的短测筛选候选：

   ```bash
   uv run --project qa --locked python tests/model_web_comparison/run_tuning_probe.py --label my_profile --models ornith qwen --repeats 2
   ```

   默认写入 `storage/llm_tests/web_comparison_probes/`，拒绝覆盖同名记录。请求级草稿长度在已测 llama.cpp 版本中未生效，因此草稿参数通过启动配置调整。
4. 对选定配置执行完整网页任务：

   ```bash
   uv run --project qa --locked python tests/model_web_comparison/run_comparison.py
   ```

   使用原提示词、max_tokens=8192、temperature=0、top_p=1、关闭 thinking、SSE 用量统计和 0.25 秒资源采样；每轮 Ornith → Qwen 串行。完整测试重复三次，每轮完成后按报告的 `run_id` 单独归档 `latest_report.json`、`latest_report.md` 和 `latest_comparison.html`，比较中位数与范围。原始 HTML 自带运行时间文件名。
5. 测速全部结束后检查原始 HTML 的内联 JavaScript 语法，并在离线浏览器验证预算、目的地、行程增删及勾选、行李清单；保留模型原始输出及问题，不用修复后的网页替代质量结果。
6. 确认 backend 休眠、无遗留 quarantine；结束实验时正常停止服务并确认进程组退出。无模型门禁使用 `bash scripts/quality_gate.sh`。

## 结论与保留记录

- Ornith 最终三次为 76.71 / 76.41 / 77.19 token/s，中位数 76.71，相比上轮 58.60 提高 30.9%。输出从 7930 减至 6898 token，耗时下降也包含长度变化。
- Qwen 最终中位数 72.76 token/s，上轮为 73.56；补测未发现值得采用的额外收益，因此保留第二版参数。三次 HTML 与第二版逐字一致。
- 批量线程 4 较慢；6 与 8 差异不足以确认收益。Ornith 草稿长度 4 优于 3/5/6；Qwen 长度 6 仅有约 1.3% 短测差异，未采用。
- 原始网页仍有功能问题：Ornith 删除行程无效；Qwen 缺少目的地输入，行李原生勾选状态未保留。推理速度与网页功能质量分别评估。

[完整对比结论](optimization_v3_comparison.md) 保留历史各轮指标、三次最终结果、资源数据和参数筛选表；[固定网页预览](optimized_v3_comparison.html) 保留最终第三次输出，[latest_comparison.html](latest_comparison.html) 会在下次复测更新。

历史完整指标保存在 `baseline_report.json`、`optimized_report.json`、`optimized_v2_report.json`；最终三次为 `optimized_v3_run{1,2,3}_report.json`，汇总为 `optimized_v3_summary.json`，浏览器及语法检查为 `quality_checks_v3.json`。所有这些报告引用的原始网页均保留在对应模型的 `outputs/`。

运行数据目录 `storage/llm_tests/web_comparison_optimization_v{1,2,3}/` 保留关键启动参数与最终日志；v2/v3 的短测请求、计时和实际参数已归并为各目录的 `probe_summary.json`。v3 另保留 `selection.json`、`environment.json`、`browser_checks.json`、`browser_smoke.mjs`、`cleanup_verification.json` 和最终质量门禁日志；默认参数提升的验证日志为 `defaults_quality_gate.log`，核对记录为 `defaults_verification.json`。旧启动包装脚本、过期计划、重复预览、临时浏览器目录及探测 SSE 已清理。

## 指标口径与适用范围

任务耗时包含请求排队、模型加载、生成和卸载等待；首 token 延迟从发送请求计时。生成速度使用 API 返回的真实 completion token 数除以首末文本到达间隔，缺少用量时不以字符数替代。

RSS 为对应 llama.cpp 进程采样峰值，系统已用内存增量与 RSS 是不同口径；显存为 GPU 0 总占用，包含桌面等后台负载，报告同时保留基线差值。已测配置适用于本机 RTX 4070 Ti SUPER 16 GiB、i5-13600KF；仍保留约 0.7–0.9 GiB 显存余量，未为短测数字进一步迁移专家或缩小上下文。32K 为配置上限，本网页任务不是满 32K 压力测试，也不能保证所有提示词达到相同吞吐。
