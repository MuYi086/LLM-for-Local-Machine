# Ornith 跨 sleep 的 RAM prompt cache

## 作用与生命周期

原流程每轮请求后释放模型与 KV，下一轮 `cache_n=0`，重新处理完整历史。补丁在 sleep 前调用现有 `slot.prompt_save()`，把主模型、MTP 草稿状态和混合注意力检查点保存在有界 CPU RAM cache；随后清除 slot 的 CPU 引用，执行原有 `destroy()` 释放模型与 CUDA。唤醒后不重建 RAM cache，继续由 llama.cpp 原生 token 前缀匹配、检查点回退与 `prompt_load()` 恢复。

请求、保存、唤醒与恢复仍在既有 GPU 租约内。队列持有 mutex 执行保存和 destroy，`/props.is_sleeping` 也取得同一 mutex，确认 sleeping 时 CUDA 卸载已经结束。前端仍等待 CUDA_RELEASE_DELAY，再释放锁；没有延长 sleep-idle 或保持 GPU 常驻。

缓存只在同一后端进程内存在，不跨进程重启、不写磁盘。容量包括原生计算的主状态、草稿与检查点；不是整个进程的 RSS 上限。超过容量的条目不保存，内存分配/快照异常丢弃缓存并继续卸载。前缀变化由原生实现选择可用检查点或回退冷算；不按客户端文件路径或会话标题猜测状态。

## 部署补丁

本机基于 llama.cpp `7d7f0287530bc8981690f361d8c1627dbae1c99a` 的外部构建，并保留既有 sleep-request-guard。仅将局部补丁存入仓库，不纳入权重、第三方源码或二进制。

先核对外部源码的已有改动，再应用：

```bash
git -C "$LLAMA_SOURCE_DIR" apply --check "$PWD/scripts/llama_cpp/sleep-prompt-cache.patch"
git -C "$LLAMA_SOURCE_DIR" apply "$PWD/scripts/llama_cpp/sleep-prompt-cache.patch"
python scripts/llama_cpp/build_sleep_cache.py \
  --build-dir "$LLAMA_BUILD_DIR" \
  --output-dir "$LLAMA_SOURCE_DIR/build-sleep-cache/bin"
```

已应用时用 `git apply --reverse --check` 判断，不重复应用。构建助手读取既有 `compile_commands.json`，直接编译唯一的 `server-context.cpp` 到私有目录，复制 archive 并替换对应对象，再独立链接实现库与入口。它不执行 CMake 依赖重建；输出加锁并原子替换，避免原地改写活动的库。新二进制仍依赖原 CUDA 构建中的基础库，不能单独搬走新目录使用。

本机新构建已放在 `$HOME/.local/src/llama.cpp/build-sleep-cache/bin/llama-server`。`start.sh` 在未设置显式路径、且 PATH 未发现服务时，优先发现此目录；否则可显式设置 `LLAMA_SERVER_BIN`。

在原终端 Ctrl+C，等旧服务退出后重新执行 `bash start.sh`。正在运行的旧进程不会因文件更新自动启用补丁；本次没有后台重启用户服务栈。

## 配置与核验

本机 32k OpenCode 对比配置（2026-11-14）：MTP Q8、draft=3、纯文本、显式 `--fit off`。
适配器支持 `<backend_env_prefix>_CUDA_RELEASE_DELAY` 覆盖全局缓冲；本轮
`ORNITH_MTP_9B_CUDA_RELEASE_DELAY=0.5`，全局 `CUDA_RELEASE_DELAY` 仍为原值。
本机补丁在 `destroy()` 完成后才报告休眠，适配器仍在持锁期间等待缓冲并二次确认休眠，
确认失败仍保留 quarantine。每次减少 1.5 秒固定等待，实际端到端收益与 GPU 释放表现待真实复测。
这轮继续逐请求卸载；连续驻留需要单独设计租约交接。回退时删除 MTP 专属覆盖项或设为 `2`。

| 环境变量 | 默认值 | 作用 |
| --- | ---: | --- |
| `ORNITH_MTP_9B_SLEEP_CACHE` | `1` | 启用跨 sleep 缓存；设 `0` 回退原生命周期 |
| `ORNITH_MTP_9B_CACHE_RAM_MIB` | `4096` | 正数的原生 RAM 缓存预算，超限跳过该条目 |
| `LLAMA_SERVER_SLEEP_PROMPT_CACHE` | 由上一开关导出 | 本地补丁识别的后端开关，精确值 `1` 才启用 |

96k、Q8 主权重、MTP 4、batch/ubatch 1024、F16 KV、线程 6/8、1 秒 sleep-idle 均保持原配置。启动器在初次休眠后核验 `/props` 的 `sleep_prompt_cache_supported=true` 和 `sleep_prompt_cache_enabled=true`。旧二进制缺少能力或预算导致实际关闭时，停止本次 backend，不静默忽略；停止不能确认则沿用 quarantine 恢复协议。

回退命令：

```bash
ORNITH_MTP_9B_SLEEP_CACHE=0 bash start.sh
```

OpenCode 仍使用 8500 网关、同一模型 ID 和 Key；无需修改客户端配置。服务只保留计算状态，健康检查不加载权重。

## 真实多轮验证（2026-11-13）

使用独立 loopback 端口、相同 GPU 锁与 98304 上下文，固定同一参考源码，关闭 thinking，temperature=0、seed=42。工具轮次先生成 `report_tag` 调用，再提交真实 assistant/tool/user 消息。所有请求均实际 sleep 卸载后才运行下一轮；比较相同工具流在缓存关闭/开启时的表现。

| 约 6.2 万输入的追加请求 | 缓存关闭 | 最终补丁开启 |
| --- | ---: | ---: |
| 总 prompt tokens | 61935 | 61935 |
| 复用 `cache_n` | 0 | 61893 |
| 实际 prefill `prompt_n` | 61935 | 42 |
| 原生 prefill | 39.220 秒 | 0.479 秒 |
| 首文本事件 | 42.577 秒 | 4.269 秒 |
| 持锁时间（含采样、保存、卸载与 2 秒释放等待） | 46.052 秒 | 9.516 秒 |

此前开启缓存的同场景也测到 prefill 0.242 秒、首事件 4.534 秒、持锁 9.719 秒。最终补丁的分叉请求复用 61916 token，只重算 23 token；修改历史最前部的目标标签后 `cache_n=0`，回答采用新标签；换题 `cache_n=0`。无 MTP 与 MTP 的约 5.6k 输入验证均通过。1 MiB 预算测试用于确认超限条目安全冷算，具体结果见运行记录。

缓存不是免费的：最终长输入测试保存约 2380–2532 MiB 状态，保存耗时约 1.28–1.91 秒，已计入持锁时间。观测的整个 lease 显存峰值约 14538 MiB，sleep 后整卡回到约 1803–1808 MiB；backend RSS 峰值约 4984 MiB、sleep 后约 3042–3243 MiB，guest RAM 峰值约 9412 MiB。关闭缓存的对照 sleep RSS 约 805–977 MiB，说明约多占用 2.2 GiB CPU RAM，而非 GPU 常驻。整卡与 guest 指标包含其他进程，不能当任务独占资源。

API `usage.prompt_tokens` 仍是完整输入量；`timings.prompt_n` 是实际本轮计算量。命中后仅计算几十 token，prefill tok/s 可能下降，但耗时显著减少，应比较实际 token 数、毫秒和首事件，不能按 tok/s 判断缓存失效。

冷请求、前部改写、压缩后前缀不匹配和超限仍会完整 prefill。模型权重每次仍需加载，所以这里不是零首 token 延迟。该测试验证状态复用和目标标签/工具功能，不能代替完整 OpenCode 固定任务的运行时间与质量评估；96k 是分配上限，实际验证最大约 61.9k 输入。

## 测试与运行证据

```bash
# 无模型：缓存容量、主/草稿/checkpoint 计数、前缀去重与淘汰
bash scripts/llama_cpp/test_sleep_prompt_cache.sh "$LLAMA_SOURCE_DIR" "$LLAMA_BUILD_DIR"

# 真实模型：显式执行；先让其他模型客户端停止发请求
uv run --project main --no-sync python tests/ornith_1_5_9b_mtp_benchmarks/benchmark_sleep_cache.py \
  --server-bin "$LLAMA_SERVER_BIN" --model-file "$ORNITH_MTP_9B_MODEL_FILE" \
  --mtp draft-mtp --context-copies 11 --tool-turn --cache 1 --expect-cache
```

缓存关闭对照用 `--cache 0`，超限测试用 `--cache 1 --cache-mib 1`（不加 expect-cache）。可通过 `--source-file` 固定跨实验相同输入；脚本保存实际 prompt、请求 hash、JSON timings、SSE 输出、整个 lease 的资源汇总、sleep 快照及 backend 日志，且不会覆盖已有运行目录。GPU 锁和既有 backend URL 可用 CLI 覆盖，默认读取相同部署环境。

证据目录位于 `storage/llm_tests/ornith_1_5_9b_mtp_benchmarks/`：

- `sleep-cache-20261113T063651785634Z`：原二进制、缓存关闭，无 MTP。
- `sleep-cache-20261113T063808158036Z`：无 MTP，跨 sleep 追加与分叉。
- `sleep-cache-20261113T063939953532Z`：MTP，跨 sleep 追加与分叉。
- `sleep-cache-20261113T064231910951Z`：长输入工具轮次，缓存关闭。
- `sleep-cache-20261113T064456799654Z`：长输入工具轮次，缓存开启。
- `sleep-cache-20261113T065124080571Z`：最终补丁、整个 lease 采样及前部历史变更失效验证。
- `sleep-cache-20261113T065329324416Z`：最终补丁，1 MiB 预算无法容纳条目；追加、分叉、前部改写和换题全部安全冷算，答案检查通过。

原报告窗口遗漏会话尾段的问题仍然存在于手动标记流程；后续 OpenCode 复测须在最终回答完成后再结束窗口，并单独记录客户端总耗时。

## 32k 第二阶段等待优化（修正）

当前 CLI 拒绝 sleep-idle=0，内部队列允许该值不能证明 CLI 支持。已恢复
`ORNITH_MTP_9B_SLEEP_IDLE_SECONDS=1` 并在启动器拒绝 0，避免启动第二个 backend 后才失败。
MTP 专属释放缓冲保持 0.5 秒。

已按用户要求恢复 145f5307 对应的 32k 运行行为：采样线程退出后再释放 GPU 文件锁，撤回采样回收顺序优化。
当前参数为 context=32768、draft=3、vision=off、fit=off、sleep-idle=1、MTP CUDA 缓冲=0.5 秒；
OpenCode output=4096、buffer=8192、keep.tokens=4096。实际运行进程须重启才能加载恢复后的代码。

当前本机复测档位已调整为 96k：context=98304、OpenCode output=4096、buffer=16384、keep.tokens=10240；继续沿用 145f5307 对应的 draft=3、vision=off、fit=off、sleep-idle=1、MTP CUDA 缓冲=0.5 秒与采样线程退出后释放锁的行为。
