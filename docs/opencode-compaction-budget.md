# OpenCode 归档简报与压缩预算保护

## 已确认的失败链

2026-11-13 15:02 的 `4ee035ba` 轮次已实际复用 sleep cache，读取和报告生成加速。但执行 end 后，模型把 Python 交给 JavaScript Code Mode，再尝试不可用的 require('fs')，随后直接 read 汇总 JSON：返回 1699 行、60502 字符且 truncated=true，随后发生摘要失败。

原始记录确认摘要申请 max_completion_tokens=4096、temperature=0；实际输入 97441，输出仅 862，合计 98303，接近后端容量 98304；日志 truncated=1，finish_reason=length。不是 4096 仍不够，也不是 reasoning 消耗（该请求 reasoning=0），而是输入挤占了输出空间。OpenCode 将 length 统一报为 summary reached output token limit。

原报告的窗口在故障前已结束，“舒适”自评没有覆盖最终失败。旧报告和会话数据库保持原样作为历史证据，不据此声称没有压缩故障。

## 归档与工具输出

- `end/report --brief` 输出固定指标、覆盖率、warnings 摘录/省略数和 attached 状态，整个 stdout 不超过 4096 UTF-8 字节。详细 JSON/Markdown 仍完整保存；简报不携带 requests、backend_unassigned 或逐进程列表。
- 固定任务 v6 与插件要求一次 `end --attach-result --brief` 完成收尾。attached=true 且 test_state=ended 即完成标记验证，不再 read JSON 或整份报告，直接给出路径和缺项的最终回复。
- 基准会话直接 read 遥测 JSON/JSONL 会被拒绝，包括目录和文件的符号链接别名。结果 Markdown 大于 16 KiB 时拒绝 read，小型初始结果最多 40 行；五个源码仍严格按 150 行分页。
- 归档使用 shell 调用既有 CLI。Code Mode execute 接受 JavaScript program body，文件和命令操作通过提供的 tools，不支持 Python/require/import/fs。插件自身运行在 Node 环境，其 fs 权限与 Code Mode 沙箱不同。

例子（由插件提供本轮实际参数）：

```bash
python3 scripts/model_telemetry.py --directory '<directory>' end '<test_id>' \
  --settle-seconds 30 --attach-result '<report>' --brief
```

合并失败后用同一目录/ID 的 `report --test-id '<test_id>' --attach-result '<report>' --brief` 恢复，不重复 end。

## 实际 token 预算

仅 local-ornith 的 compaction HTTP body 添加 `llm_context_guard=true`；普通请求不增加模板或分词调用。OpenCode 的 MTP 默认路由经 8500→8504→8506；普通 9B 仍可经 8500→8503→8505 选择。网关不转发 Key 给 backend，预算检查使用被选模型 backend 的实际上下文。

适配器在取得既有 GPU 租约、确认初始 sleep 后，剥离本地字段，调用同一 backend：

1. `/props` 获取实际 n_ctx。
2. `/apply-template` 使用与聊天相同的参数构造文本，不生成 token。
3. `/tokenize` 使用 add_special=true、parse_special=true，与原聊天的分词路径一致。
4. 要求 `actual_input + requested_output + 256 <= n_ctx`，保留原消息和申请输出，不静默缩短摘要。

超限返回 HTTP 400、根级 OpenAI error、code=context_length_exceeded，未提交 `/v1/chat/completions` 生成。OpenCode 2.2.7 识别该错误后可安全缩减摘要历史重试。计数不可用返回 503，不跳过保护；错误或取消仍由 finally/shield 等待卸载、释放锁，不能确认则保留 quarantine。遥测只记录预算数字与 context_guard_start/end，不记录模板正文。

如果不可省略的最新用户输入或检查点本身已经过大，OpenCode 仍可能明确报告无法继续缩减；预算保护不保证任意大输入都能自动摘要成功。隔离测试用 32k 临时模型配置，让首次正常请求有充足空间，实际读取大工具结果后才触发压缩；标题辅助请求不伪造过大的 usage，避免在尚无可缩减内容时提前触发。

MTP 与普通 9B 的 context 均为 98304；OpenCode output=4096、buffer=16384、keep.tokens=10240 保持原值。buffer 是触发余量，无法独自阻止单次大工具结果跨过余量，因此同时限制工具输出和核验实际预算。

## 验证

- CLI 测试包括 5000 条逐进程资源、大量 warnings、控制字符、巨型计数、缺指标和真实 end/attach；简报可解析且不泄露详细记录。
- Node 测试覆盖基准作用域、JSON/大报告、符号链接别名、源码分页和仅压缩请求携带标记。
- 隔离 OpenCode 集成使用临时 XDG/假模型；首次压缩返回上下文过长，确认后续摘要请求更小、压缩成功并继续任务，不接触真实会话库。
- API 无模型测试复现 97441/4096、精确边界、计数不可用、普通请求，以及 tokenization 期间取消后的隔离标记与锁释放。
- 显式真实测试 `tests/ornith_1_5_9b_gguf/benchmark_context_budget.py` 通过真实适配器与同一 GPU 锁复用当前 loopback backend：小请求预检 16 token，与生成返回 usage=16 一致；97012 输入/4096 输出被拒绝，遥测无 backend_send，无生成。证据目录：`storage/llm_tests/ornith_1_5_9b_gguf/context-budget-20261113T073904598805Z/`。

真实测试不属于无模型质量门禁；未执行完整长会话固定任务，因此不能把以上验证当成新一轮基准已完成。

## 生效

无需重编译 llama.cpp。正常 Ctrl+C 退出当前服务栈并重新 `bash start.sh`；退出重启 OpenCode，新会话运行 v6 固定任务。插件/适配器的旧进程不会自动加载文件改动。本次不修改活动服务或旧失败会话，不自动后台重启服务。
