# Ornith 9B 96k 性能分析与实测

后续已实现跨 sleep 的有界 CPU RAM 状态复用，长输入追加请求 prefill 从约 39.2 秒降至约 0.48 秒，仍在每轮释放 GPU。部署、资源代价、回退与验证边界见 [缓存优化记录](../../docs/llama-cpp-sleep-cache.md)。本文以下保留改动前的分析和 MTP/batch 对照，未将其当作缓存优化后的 OpenCode 总耗时。

分析日期：2026-11-13。原报告为仓库 `测试结果/96k-20261113-11时23分29秒-f71d4bb2.md`。

## 原报告反映的问题

- 27 条已列出的原生记录共处理 972,947 输入 token，prefill 503.42 秒；输出 13,806 token，decode 310.92 秒，按输出 token / decode 时间约为 44.4 token/s。这些记录有窗口与关联限制，不能直接相加当作 OpenCode 会话耗时，也不能把 bridge、adapter 同一请求重复计数。
- 后端原始启动日志 `start-20261113T032252Z-hE3Y0f.log` 明确记录 `n_ctx_slot=98304`。原报告末尾模型自述的“未核实上下文”不代表实际后端没有设置成功。
- 原配置没有启用 speculative decoding。每次加载都忽略 `blk.32.nextn.*` 等内置 MTP 张量。全 GPU offload、Flash Attention 已启用，未发现把主模型错误放在 CPU 的证据。
- 每次请求必须在共享 GPU 锁内加载，结束后卸载并确认休眠；KV 与 prompt cache 随之释放。长工具会话反复计算完整历史，6 万输入时 prefill 约 28–33 秒，工具调用之间还会等待前一请求休眠与 CUDA 释放。单纯提高 decode 无法消除这部分成本。
- 遥测“首正文”可能晚于推理或工具参数开始生成，不能直接当成客户端首生成 token，也不能据此认定 SSE 被缓冲。

## 为什么旧 35B A3B 可以更快

官方将 [9B 标为稠密模型](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF)，[配置含一个 MTP 层](https://huggingface.co/ornith-ai/Ornith-1.5-9B/blob/main/config.json)。35B A3B 为稀疏模型，单 token 激活规模和稠密 9B 不同；总参数量不能直接推算速度。历史 `tests/model_web_comparison/optimized_v3_report.md` 记录的是启用 MTP 的 35B 网页任务，77.19 token/s，按客户端首末正文时间计算，提示与已用上下文也不同。

9B Q8 的逐 token 计算和权重读取成本较高。13 GiB 以下的显存占用只说明容量余量，不能证明计算或带宽尚有同等比例的余量；本轮没有硬件计数器证据，不能把带宽瓶颈当成已确诊的唯一原因。

## 同机串行对照

RTX 4070 Ti SUPER 16 GiB、i5-13600KF、WSL2。所有实验保留同一 Q8_0 权重、98304 上下文、CUDA0 全 offload、线程 6/8、sleep-idle=1、load-mode=none。通过独立 loopback backend 与共享 GPU 锁运行，每次请求冷加载，结束后确认休眠，最后通过既有 shutdown 助手确认退出。原服务保持运行并处于休眠。

固定中文 Python LRUCache 任务，temperature=0、seed=42、关闭 thinking、输出预算 512 token；长提示在任务前重复参考源码，实际输入数见下表。速度为服务返回的原生 `timings.predicted_per_second`，不是字符估算或客户端正文吞吐。首 token 为探测客户端首个非空文本事件；请求耗时截至 SSE 结束，不含后续休眠与释放等待。

| 配置 | 46-token 输入 decode tok/s | 5655-token 输入 decode tok/s | 观测整卡显存峰值 MiB |
| --- | ---: | ---: | ---: |
| 原配置：无 MTP，batch 512/512，F16 KV | 46.48–55.56（2 次） | 46.12 | 13237 |
| MTP 2，batch 512/512，F16 KV | 76.34–83.46（2 次） | 82.27 | 14089 |
| MTP 4，batch 512/512，F16 KV | 93.47–94.94（2 次） | 78.92 | 14181 |
| MTP 4，batch 512/512，Q8 主 KV | 52.14–74.39（2 次） | 54.30 | 13070 |
| 无 MTP，batch 1024/1024，F16 KV | 48.13 | — | 13431 |
| **MTP 4，batch 1024/1024，F16 KV** | **97.93** | — | **14593（含长输入复测）** |

| 61615-token 输入配置 | 原生 prefill 秒 | 首 token 秒 | decode tok/s | 请求秒 |
| --- | ---: | ---: | ---: | ---: |
| 原配置，无 MTP，batch 512/512 | 32.03 | 34.34 | 40.08 | 47.09 |
| MTP 4，batch 512/512 | 40.79 | 43.76 | 49.63 | 54.06 |
| 无 MTP，batch 1024/1024 | 34.12 | 36.49 | 42.49 | 48.51 |
| **MTP 4，batch 1024/1024，第一次** | **38.10** | **41.34** | **74.19** | **48.23** |
| **MTP 4，batch 1024/1024，复测** | **39.78** | **42.84** | **70.58** | **50.08** |

所选组合在约 6 万输入时，decode 较原配置提升约 76%–85%；短输入测到 97.93 token/s。MTP 草稿长度 4 的接受率约 55%–61%。整卡显存峰值 14593 MiB（14.25 GiB），backend RSS 峰值约 2.3 GiB；JSON 还记录 guest 内存基线与峰值。GPU 显存与 guest 内存均包含环境影响，不代表本任务独占资源。

MTP 增加冷加载及草稿 prefill 成本。这次长输入、短输出样本总耗时比原配置增加约 1–3 秒；输出更长时 decode 节省才更明显。**不能据此声称整个 OpenCode 工具任务已变快。** Q8 KV 在本轮明显降低吞吐，因此不采用；batch 1024 单独也没有改善首 token。

## 已采用的默认值与回退

`start.sh` 默认设置：

```text
ORNITH_9B_CONTEXT_SIZE=98304
ORNITH_9B_BATCH_SIZE=1024
ORNITH_9B_UBATCH_SIZE=1024
ORNITH_9B_SPEC_TYPE=draft-mtp
ORNITH_9B_DRAFT_MAX=4
```

使用主 GGUF 内置 MTP，不增加独立模型文件；主/草稿 KV 保持 llama.cpp 默认 F16。启动预检检查所需 CLI 参数，遥测记录新增参数。OpenCode 的模型 ID、96k 上限、buffer 与 keep.tokens 无须调整。

当前已运行的服务不会自动采用新值。在原启动终端 Ctrl+C，等全部进程退出后执行 `bash start.sh`。本次调优没有后台重启服务栈。

若主要执行长历史、短输出的工具任务，可完整回退到原 profile 作会话对照：

```bash
ORNITH_9B_SPEC_TYPE=none ORNITH_9B_BATCH_SIZE=512 ORNITH_9B_UBATCH_SIZE=512 bash start.sh
```

该轮 MTP/batch 调优时，固定的每请求卸载策略使长历史 prefill 成为剩余瓶颈。当前共享 GPU 协议要求卸载后才释放锁，不能只延长 sleep-idle 来保留显存和缓存，否则会拖住后续 GPU 请求。后续的缓存补丁通过 CPU RAM 保存/恢复状态解决重复 prefill，见本文顶部链接。

## 复现与证据

实验脚本为 [tune_backend.py](tune_backend.py)，运行方式见 [README](README.md)。本机原始 JSON、输出样本、内存与显存采样汇总、backend 参数和日志位于运行数据目录 `storage/llm_tests/ornith_1_5_9b_gguf/` 下：

- `tuning-20261113T051558170512Z/`：四组 MTP/KV 对照。
- `tuning-20261113T051953231158Z/`：原配置与 MTP 4 的 61615-token 对照。
- `tuning-20261113T052227148682Z/`：batch 1024 对照。
- `tuning-20261113T052532599314Z/`：所选配置长输入复测。

样本数量有限，后台负载、GPU 时钟与提示内容会造成波动；未填满全部 96k，仅验证 96k 分配与最大约 61.6k 的实际输入。512-token 输出预算是吞吐探测，不能当作完整代码正确性评测。不同 profile 输出并非字节一致，未完成质量等价评估。新默认 profile 尚需用户在 OpenCode 中完成原固定任务的端到端复测。
