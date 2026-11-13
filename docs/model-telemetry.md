# 模型性能记录

`bash start.sh` 默认启用性能记录。采集器不加载模型、不调用推理 API、不获取 GPU 锁；Ornith 仍使用原有共享 GPU 租约及 quarantine 收尾。

跨 sleep RAM cache 启用后，manifest 记录 `ORNITH_9B_SLEEP_CACHE`、`ORNITH_9B_CACHE_RAM_MIB`。`usage.prompt_tokens` 仍为完整输入，`timings.cache_n` 为复用部分，`timings.prompt_n` 才是本轮实际 prefill 的 token 数。命中后应比较 prefill 毫秒、首事件及持锁时间，不以小批次 tok/s 判断是否加速。CPU RSS 包括有界缓存，sleep 后 RSS 较高不代表 CUDA 未释放；保存开销计入请求收尾。部署及实测见 [缓存补丁说明](llama-cpp-sleep-cache.md)。

## 开关与目录

模型执行收尾时使用 `end/report --brief`：简报只含固定服务指标、资源峰值、warnings 摘录和 attached 状态，stdout 最多 4096 UTF-8 字节；完整数据仍落盘。attached=true 且 test_state=ended 即确认统计块，不再读完整 JSON/Markdown。当前工作流与预算保护见 [说明](opencode-compaction-budget.md)。

修改仓库中的 `config/model-telemetry.env`：

```bash
MODEL_TELEMETRY_ENABLED=true
MODEL_TELEMETRY_INTERVAL_SECONDS=1
```

测试完成后将 `true` 改成 `false`，正常退出旧服务并重新运行 `bash start.sh` 生效。也可以临时执行 `MODEL_TELEMETRY_ENABLED=false bash start.sh`；显式环境变量优先于配置文件。关闭性能记录不会关闭必要的控制台和故障日志。

每次启动创建独立目录，终端输出 `[性能记录] 已开启，目录：...`，默认位于 `storage/.cache/runtime/telemetry/run-.../`。可用 `MODEL_TELEMETRY_ROOT` 更换根目录；不会清理历史记录。启动预检失败可能不产生采样。单独启动 API 时使用 `MODEL_TELEMETRY_DIR` 或默认 `telemetry/standalone/`。

| 文件 | 内容 |
| --- | --- |
| `manifest.jsonl` | 启动时间、采样频率、WSL/Linux 范围、启动器传入的上下文/线程/KV/MTP 配置；配置不是后端运行值的独立证明 |
| `starts-服务-PID.jsonl` | 请求起点、请求 ID 和本层 trace ID，发现没有结束记录的调用 |
| `requests-服务-PID.jsonl` | 单调时钟耗时、状态、用量、原生 timing、finish reason、首个正文/工具/reasoning 事件及适配器阶段 |
| `resources.jsonl` | Linux/WSL 内存、swap、进程 RSS/间歇 PSS/CPU 累计秒数、整卡和可用的进程显存、利用率、温度、功耗 |
| `backend.jsonl` | 现有混合启动日志中白名单提取的 llama.cpp timing/MTP 数字；未可靠关联请求 |
| `tests/` | 每轮测试独立的开始/结束标记 |
| `report-*.json`、`report-*.md` | 离线生成的详细记录和可阅读汇总，文件名唯一，不覆盖历史报告 |

记录器只保存固定指标，不保存提示词、回答、工具参数、鉴权头或进程命令行。每个 JSONL 文件只有一个进程写入，进程内用锁串行追加。采集缺失、响应事件过大或磁盘写入失败不应影响推理；磁盘失败会告警。不要将运行数据提交到 Git。

## 每轮测试怎么用

服务启动并就绪后，在仓库根目录的另一个终端执行：

```bash
python3 scripts/model_telemetry.py begin --label 96k-run1
```

保存输出的“性能目录”和“测试 ID”，然后在新的 OpenCode 会话提交原有固定任务。记录不需要 agent 读取日志，也不改变固定任务。

等 OpenCode 本轮回复结束后执行（替换 ID）：

```bash
python3 scripts/model_telemetry.py end <测试ID>
```

`end` 写入结束标记，并最多等待 10 秒让最后一次请求完成卸载和写盘，随后输出 JSON 和 Markdown 报告路径。等待不改变结束标记的时间；跨越窗口边界的请求保留完整数据并注明。若仍有未结束请求，报告会列出 `unfinished`，不要当作完整成功样本。可增加 `--settle-seconds 30`，或稍后重新汇总：

```bash
python3 scripts/model_telemetry.py report --test-id <测试ID>
```

默认选择最近启动批次。如果测试期间重启过服务、使用自定义根目录，或有 standalone 目录，请在子命令前明确指定当轮目录，防止混入别的批次：

```bash
python3 scripts/model_telemetry.py --directory <性能目录> end <测试ID>
```

没有提前标记时，也可以按带时区的时间范围汇总；但标记前后的人为等待会影响任务窗口时长：

```bash
python3 scripts/model_telemetry.py --directory <性能目录> report \
  --start '2026-11-12T10:00:00+08:00' \
  --end '2026-11-12T10:15:00+08:00'
```

自动采集不要求每轮都运行 begin/end；这两个命令用于准确区分多次测试。原始数据始终持续写入。建议把性能报告路径补到原有 `测试结果/*.md`，不要让 agent 为补齐统计额外发起模型请求。32K、48K、64K 等档位分别测试至少三轮，并分别比较中位数与质量。

## 将统计写入 OpenCode 的同一份结果报告

当前固定任务 v6 由插件自动绑定本轮目录、ID 和已有结果文件，具体流程见 `CONTEXT_BENCHMARK_PROMPT.md`，无需额外 begin/end。下列标记和手动命令用于说明格式及人工恢复；不要让 agent 寻找“最新日志”或另开分析循环。服务启动批次和测试 ID 在整轮中固定，不得复用旧 ID。时间窗口筛选不等于会话识别，本轮应独占模型调用；如有其他客户端请求，必须注明归属不确定。

OpenCode 在自己的结果 Markdown 中创建且只创建一个统计块，示意如下（实际 ID 替换占位符，标记本身不放在代码围栏中）：

```markdown
<!-- MODEL_TELEMETRY_BEGIN test_id=<32位测试ID> -->
待汇总。
<!-- MODEL_TELEMETRY_END -->
```

它先执行离线 report，将当前阶段快照组装进该统计块：

```bash
python3 scripts/model_telemetry.py --directory '<本轮性能目录>' \
  report --test-id '<本轮测试ID>' \
  --attach-result '测试结果/<本轮实际文件名>.md'
```

只有匹配同一个测试 ID 的唯一统计块才能更新。目标必须是已有的普通 Markdown 文件，符号链接、缺少标记、重复标记和错误 ID 均拒绝；使用文件锁及原子替换，不修改块外的配置、审查正文或手动补录的客户端界面统计。并发修改冲突会报错，不强行覆盖。

模型无法在自己的最终回复完成前读取该次完整用量，因此此时明确标记“阶段快照”，不能当整轮最终统计。回复结束后，测试者执行 OpenCode 给出的已替换好实际参数的命令：

```bash
python3 scripts/model_telemetry.py --directory '<本轮性能目录>' \
  end '<本轮测试ID>' --settle-seconds 30 \
  --attach-result '测试结果/<本轮实际文件名>.md'
```

该命令不再调用模型，将结束窗口统计更新到同一结果文件，包含最终回复在内的已记录请求。如果超时仍有 unfinished，报告保留该警告，不能当完整成功样本；稍后用相同目录和 ID 的 report --attach-result 重新汇总。若 end 已写结束标记但合并失败，也使用 report 重试，不重复 end。

合并模式必须显式提供 --directory 和测试 ID，不支持“自动最新批次”或任意 --start/--end 的结果写回。独立的时间范围报告仍可用来排查历史数据，但不能冒充本轮身份确定的统计。没有本轮目录/ID 时，OpenCode 只能标注缺失，不能从旧记录补数。

## 指标口径与缺失

- `bridge` 是网关观测位置；`ornith` 是适配器观测位置。共同 `request_id` 用于关联同一调用，各层独立 `trace_id` 区分记录。不同层的输入/输出 token 和耗时不能直接相加。
- `first_content_ms` 是本层请求进入至首次观测到非空 `delta.content` 的延迟，排除 role、心跳、空事件、工具参数和 reasoning。它不是 OpenCode 实际发出请求到界面显示的 TTFT。非流式响应仅能观测整份 JSON 到达时间，不能作为流式 TTFT 比较。
- 仅工具调用而无正文时，正文延迟为 null，查看 `first_tool_ms`；reasoning 另记。一个 SSE 事件可能包含多个 token，不宣称单 token 精度。
- 适配器记录 `queue_enter`、`lease_acquired`、`backend_send`、`backend_headers`、`response_body_end`、`cleanup_start`、`cleanup_end`、`lease_released`。`duration_ms` 包括本层请求生命周期及适配器清理；网关已发完正文时，适配器可能仍持锁等待 sleep-idle，下一次请求会排队。
- 启用记录时，适配器为缺少该选项的流式请求补 `stream_options.include_usage=true`，保留调用方显式 false。用量以实际收到的 usage 为准，不按字符数估算。llama.cpp 返回的 `timings` 保存 prefill/decode 时间、速度、cache_n、draft_n 和 draft_n_accepted；版本不支持或中途取消则可能缺失。
- `prompt_n` 是实际处理的 prompt token，缓存 token 单独记录，不能直接替代完整输入 usage；decode 原生速度不等于界面 tok/s。MTP 接受率可由原生 accepted/draft 数字计算，不将某一请求的速度当作整轮速度。
- 当前原生 timing 不包含独立模型加载耗时，不能用首正文时间减 prefill 反推。模型加载阶段仍需参照原始启动日志；汇总不会编造加载时间。混合日志的后端 task 未可靠绑定模型/请求，`backend_unassigned` 只作辅助；其 observed_at 是采集时刻，存在采样延迟。
- 资源默认约每 1 秒采样，PSS 每五次读取一次。实际采样时间、耗时保存在原始文件；nvidia-smi 超时可能降低采样频率。峰值仅为采样观测峰值。基线取测试开始前最近一次采样，缺少前置采样时基线与增量为 null；峰值只计算窗口内采样，不包含窗口前基线；增量可以为负值。结束值是窗口内最后一次可见采样，缺失覆盖会告警。
- RAM 是 `/proc/meminfo` 的 MemTotal−MemAvailable；仅代表 Linux/WSL guest，不包括 Windows 宿主的完整内存。RSS 可能重复统计共享页，不求和冒充任务物理内存。PSS 权限不足时为 null。后端除启动树外还按 PID 文件定位并核对可执行文件，避免 helper 退出后遗漏；OpenCode 进程及其后代按 PID/starttime 记录，其他 OpenCode 会话也可能被采集，不自动归属当前任务。
- GPU 用 UUID 标识，记录整卡占用及可用的进程显存；WSL 不支持进程指标时为 null，不填零。整机/整卡增量不是任务独占用量。
- OpenCode 界面总耗时、界面 tok/s、自动压缩、工具耗时、最终报告显示时刻仍需客户端导出或测试者补录。报告成功不代表没有发生 OOM 或重试；没有直接证据时保持未知。进程被强杀时可能只留下起点，最终统计不会伪装成完成。


## 固定上下文测试准备

运行 `python3 scripts/prepare_context_benchmark.py --directory '<本轮采集目录>' --context 98304`，将输出的完整指令交给新 OpenCode 会话。准备器校验采样新鲜度和启动器上下文，提前创建如 `96k-日期-时间-ID.md` 的报告及 begin 标记。OpenCode 必须在同一文件追加正文；回复结束后执行准备器打印的 end 命令，才会计入最终回复。缺少标记的旧报告不能凭时间相近或最新目录补成精确样本。

汇总包含 RAM/VRAM 基线、观测峰值与增量；首事件平均延迟包括正文、reasoning 和工具调用，另保留首正文延迟。加权 decode 速度为有效请求原生 predicted_n 总和 / predicted_ms 总和，显示覆盖率。它不是 agent 总耗时吞吐率，也不是客户端 UI 的 tok/s。
