# OpenCode 本地模型压缩失败排查与修复

## 本次已确认的故障

本机为 OpenCode 2.2.7，本地模型声明 context=32768、output=4096，compaction.auto=true、buffer=8192、keep.tokens=4096。

失败会话 `ses_e324e4744ffeF0aAlf4pZ7vGhL` 的持久化消息显示：模型虽然声称每次读 150 行，却在同一次回复里发了四个只有 path、没有 offset/limit 的 read。工具实际返回 start.sh 1–816、网关 1–605、适配器 1–492、runtime 1–643，总计 2556 行。原生 read 默认最多 2000 行、50 KiB，因此不会自动执行提示词里的 150 行限制。

随后自动压缩失败，记录为输入 21824、输出 671 tokens，错误是 `Compaction summary did not match the required template`。这表示 OpenCode 未在返回文本中找到合法的检查点标题，并非该错误已经证明显存 OOM 或 4096 输出耗尽。压缩会提醒模型重写一次，仍不合法则终止当前任务。未保存的失败摘要正文无法还原，不能断言它具体写了什么。

另外，用户只提交了简短执行指令，没有给本轮性能目录和测试 ID，所以该轮无法绑定遥测；这是统计缺失的原因，不是压缩中断的原因。

## 仓库修复

`.opencode/plugins/local-model-context/` 是 OpenCode V2 自动发现的本地插件，无 npm 依赖。只针对 local-ornith provider，在本仓库生效：

- 识别执行 CONTEXT_BENCHMARK_PROMPT.md 的会话，在 read 执行前为四个源码补齐/限制 limit≤150、offset≥1。同次模型回复的第二个源码 read 返回明确工具错误，避免一轮灌入四个文件；其他文件/会话不改。
- 压缩请求单独使用精简摘要 system，明确将历史任务格式当作待总结数据，要求原生英文检查点标题。只替换压缩请求的 system 和工具，不更改持久化历史或主任务 system。
- 仅压缩请求设置 temperature=0、输出上限最多 4096（保留更低的既有预算），并在 HTTP 请求构造后设置 chat_template_kwargs.enable_thinking=false；正文、既有其他模板参数和鉴权头保留。仍只通过原有网关和共享 GPU 锁，不新增模型请求或直连后端。
- 不关闭自动压缩，不修改 32768 上下文、主任务输出上限或全局 OpenCode 配置。buffer 8192/keep 4096 保持原值。

固定提示词 v4 缩短了操作说明，加入实际 read JSON 参数、一次一个分页调用、检查点格式优先和压缩后续读规则。v4 与旧版工具行为不同，性能比较需要重新建立各档基线，不把旧失败样本当成功结果。

## 使用与边界

1. 退出旧 OpenCode，再在仓库根目录运行 `opencode --standalone`，开启全新会话。插件在新实例加载；无需因此重启模型服务。
2. 模型服务仍需按性能记录流程启动。每轮先运行 begin，再将性能目录和测试 ID 带入固定任务指令，例子见 CONTEXT_BENCHMARK_PROMPT.md。
3. 按返回实际行号连续读页，压缩后不重头读取。压缩失败的旧会话不自动篡改或删除；保留为失败样本。
4. 本插件是上下文管理修复，不保证模型永远遵守指令，也不将不完整读取冒充完成。read 限制不覆盖 shell 中自行 cat 文件等绕过行为，固定任务明确禁止绕过。
5. 若需停用插件，在 OpenCode 项目配置的 plugins 中加入 `-llm-local-model-context` 并重新启动。不要直接修改打包的 OpenCode 可执行文件。

## 验证

```bash
node --test tests/opencode/context-guard.test.mjs
uv run --project qa --locked python -m unittest tests.test_opencode_context_guard -v
```

Node 单元测试覆盖压缩模板隔离、非本地模型不变、150 行限制、同次批量读取拒绝，以及仅压缩请求关闭 thinking 并保留鉴权。安装了 OpenCode 时，Python 集成测试用临时 XDG 配置/数据目录、临时源码和本机假 OpenAI SSE 服务，验证插件自动发现、分页执行、压缩成功后继续。没有 OpenCode/Node 的环境相应测试明确跳过，不下载依赖、不调用真实模型、CUDA 或已有服务。

该验证证明本机 2.2.7 的插件接口和执行链可用；真实 Qwen 对摘要模板的遵循仍需下一轮真实基准确认，不能把假模型成功写成真实模型已修复。

## 上游依据

- [OpenCode 2.2.7 压缩实现](https://github.com/anomalyco/opencode/blob/v2.2.7/packages/core/src/session/compaction.ts)：摘要模板、合法标题校验、一次提醒和终止错误。
- [原生 read 工具](https://opencode.ai/docs/tools/#read)：offset/limit 与默认页大小。
- [V2 插件与钩子](https://opencode.ai/docs/build/plugins/server)：session context/compaction、HTTP 请求与工具 execute.before。
- [压缩配置](https://opencode.ai/docs/compaction/)：阈值、摘要与保留尾部的区别；实际触发还受估计和单次工具输出增长影响，压缩请求的输入 token 不能直接当触发点。

## 2026-11-12：分页和压缩成功后出现后端 400

会话 `ses_e3235e786ffet7X7ly5t1Id8y5` 实际读完 start.sh，并成功压缩、续读；最终在读取共享适配器之后收到
`Field 'reasoning_budget_start_tag': vector::_M_range_check`。因此截图里的第 451—600 行并不是最终停止位置，上一轮的压缩修复在本轮确实成功了一次。

后端日志在失败请求中记录：模型 load 完成后约 1.08 秒进入 sleep，尚无 launch_slot 日志；随后 API 返回 400。
本机 llama.cpp `7d7f0287530bc8981690f361d8c1627dbae1c99a` 的 HTTP response generator 仅等待唤醒，未阻止请求预处理期间再次休眠。
主循环仅按已入队任务计时，模板渲染/分词尚未入队时可能卸载词表。这与报错位置和负索引一致；无模型真实任务队列测试也复现了该生命周期竞态。

修复见 [后端补丁部署说明](llama-cpp-sleep-guard.md)：为需要模型的 HTTP 请求持有唤醒引用，直到 response generator 析构；支持嵌套引用、异常退出，最后一个引用释放后恢复 idle 计时。保留 1 秒自动卸载、共享 GPU 锁和 quarantine 流程，不通过增大上下文、关闭 thinking 或重复提交失败请求掩盖错误。

此补丁改变后端库，**必须正常 Ctrl+C 退出旧 start.sh，等待后端退出后再启动**；仅重启 OpenCode 不会加载新库。该修复经过无模型竞态回归与 CUDA 后端编译；当时真实 Qwen 固定任务仍需重启后重新测试。当前项目只保留 Ornith 文本模型，历史会话中的 Qwen 名称仅作为故障记录。


## 当前模型配置

后续已移除项目中的 Qwen3.6 服务与权重，仓库插件及 OpenCode 用户配置改用 `local-ornith/Ornith-1.5-35B-A3B-AD-Q5`。本文前面的会话 ID、错误和日志是当时 Qwen 运行记录，保留用于故障追溯。

## Ornith 摘要输出耗尽与中文交互

用户报告 `Compaction summary reached the output token limit`。检查发现插件此前将摘要预算压到 2048 tokens；现改为保留既有预算、最多 4096 tokens，并要求摘要正文精简为最多约 600 个汉字。固定英文标题保留，正文用中文。正常会话通过 context hook 要求进度和最终回复使用简体中文。

固定任务取消恰好 8 条证据的硬约束，缺失行号可标为待复核，避免报告阶段为凑齐证据循环回读。成功压缩并正确续读不会自动降低舒适度评价。

退出并重新启动 OpenCode 后插件才生效；本次改动无需重启模型服务。单元检查不代表真实模型长任务已通过，仍需新会话确认。终端的 Thought/Explored 等界面标签由 OpenCode 自身决定，中文提示不能更改这些标签。
