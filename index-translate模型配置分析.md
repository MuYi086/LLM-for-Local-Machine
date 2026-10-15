# Index-Translate 模型配置分析

分析日期：2026-10-15。

本报告根据 [ModelScope 官方集合](https://www.modelscope.cn/collections/IndexTeam/Index-Translate)、
IndexTeam 官方模型卡、官方推理文档和本机硬件检查，评估 Echo S2ST、Homura、
Nailong / NativeLong、Index-Translate 四个模型系列的部署版本。

**推荐的生产候选组合：Echo 2B 完整原生包、Homura 9B Q6_K、Nailong 9B Q5_K_M、
Index-Translate 9B Q5_K_M 加官方 mmproj（视觉编码与投影组件）。**

这是配置选型分析。下文建议尚未应用到 `start.sh`，也尚未执行每个候选的最大输入压力测试。
估算显存不能视为经过实测的硬上限；最终生产配置需要固定模型、运行时和输入边界后验收。

## 一、本机配置与显存核算范围

| 项目 | 检查结果 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| 显存 | 16,376 MiB，约 16 GiB |
| CPU | Intel Core i5-13600KF |
| 系统环境 | Linux / WSL |
| WSL 可见内存 | 约 35 GiB |
| 工作流 | 从 `start.sh` 启动，模型请求通过共享 GPU 锁串行执行 |

四个模型的显存分别计算，不累计四份模型权重。单个模型内部的必要组件必须一起计入：

```text
单任务显存 = 模型权重 + 原生音频/视觉组件 + 缓存 + CUDA 和推理工作区
```

多套 Python / Torch 环境安装在磁盘上，不会仅因安装而同时占用模型显存；实际加载到 GPU
的组件、上下文和工作区才计入单任务预算。GPU 锁负责避免请求重叠，不负责限制单请求显存。

建议给桌面、系统和波动预留约 3 GiB，把模型单任务预算控制在约 13 GiB。
检查期间已有约 10–12 GiB 显存占用，这只是当时的动态快照；下文选型以前一任务退出、
显存释放后的单任务环境为前提。未遵守同一把 GPU 锁的其他进程需要另行管理。

文件大小沿用官方十进制 GB，显存预算使用 GiB，两者不是同一单位。

## 二、推荐部署版本

| 模型 | 推荐版本 | 官方权重文件大小 | 包含运行时的显存评估 | 建议部署边界与原生输入输出 |
| --- | --- | --- | --- | --- |
| Echo S2ST | [Index-Echo-S2ST-2B 完整原生包](https://huggingface.co/IndexTeam/Index-Echo-S2ST-2B)，当前 v4.1 | 包含翻译主干、音频编码器、连接器、Hidden2CV、CosyVoice3，不能只按 2B 解码器计算 | 官方要求单卡 ≥12 GB；本机满足门槛，完整峰值仍需本机验收 | 单次一段、约 ≤30 秒；中/日/英语音输入，输出目标语种 24 kHz WAV |
| Homura | [Index-Homura-9B-GGUF：Q6_K](https://huggingface.co/IndexTeam/Index-Homura-9B-GGUF) | 7.36 GB | 在下述限制下，估算约 9–12 GiB | 上下文先设 8,192 token；文本＋目标音节数＋可选术语表 → 译文文本 |
| Nailong / NativeLong | [Index-Nailong-9B-GGUF：Q5_K_M](https://huggingface.co/IndexTeam/Index-Nailong-9B-GGUF) | 6.47 GB | 64K 窗口下，估算约 10–13 GiB | 上下文先设 65,536 token；整篇原文 → 整篇译文，源文、提示词和译文共同占用窗口 |
| Index-Translate | [Index-Translate-9B-GGUF：Q5_K_M＋mmproj-Q8_0](https://huggingface.co/IndexTeam/Index-Translate-9B-GGUF) | 6.64＋0.62＝7.26 GB | 限制图片与上下文后，估算约 9–13 GiB | 上下文先设 16,384 token；图片先限制为单张、约 ≤100 万像素；文本/图片＋翻译约束 → 译文文本 |

三个文本模型的估算按以下条件计算：

- 单 worker、`--parallel 1`。
- FP16 KV cache（键值缓存）。
- 启用 Flash Attention（节省注意力计算工作区的实现），使用较小的预填充批量。
- 图片预处理明确设置像素上限，不能只限制上传文件字节数。
- 上下文上限包含提示词、输入和输出，不代表允许同等数量的源文 token。

这些数值用于选择有余量的候选，尚未通过本机压力测试确认为峰值上限。
Echo 的“至少 12 GB”是官方硬件门槛，也不是对所有输入的显存上限承诺。

推荐文本权重文件为：

| 用途 | 文件 |
| --- | --- |
| Homura | `Index-Homura-9B.Q6_K.gguf` |
| Nailong / NativeLong | `Index-Nailong-9B.Q5_K_M.gguf` |
| Index-Translate 文本主干 | `Index-Translate-9B.Q5_K_M.gguf` |
| Index-Translate 原生图片组件 | `Index-Translate-9B.mmproj-Q8_0.gguf` |

Echo 必须下载完整 `IndexTeam/Index-Echo-S2ST-2B` 模型包并保持配套代码、组件和依赖版本一致。
它内置的 CosyVoice3 属于官方原生包，显存预算包含该组件。四个端口应分别保留原模型支持的
输入输出，不引入 Qwen3-ASR、Confucius4-TTS 等外部模型编排。

Echo 官方还记录了约 1–2% 短英语输入失败的情况；生产验收应检查空音频、异常短输出和推理失败。
详见 [Echo 2B 官方说明](https://huggingface.co/IndexTeam/Index-Echo-S2ST-2B)。

## 三、其他版本的筛选结果

| 版本范围 | 本机适用性 | 判断依据 |
| --- | --- | --- |
| 三个文本模型的 2B 原版、GGUF、FP8 | 可作为更省显存的备选 | 官方 BF16 部署参考约 8 GB，长上下文缓存另计。2B GGUF 的 Q8 权重约 2 GB；Nailong 2B 尤其值得作为更长文档的候选，但最大窗口仍需压测。[官方部署说明](https://github.com/bilibili/Index-Translate/blob/main/inference/llm/README.md)、[Nailong 2B 文件表](https://huggingface.co/IndexTeam/Index-Nailong-2B-GGUF) |
| 三个文本模型的 9B BF16 / F16 | 不推荐作为 16 GB 生产方案 | F16 GGUF 文件约 17.92–18.41 GB，缺少运行时余量；官方 BF16 部署参考约 24 GB，长上下文缓存另计。[官方部署说明](https://github.com/bilibili/Index-Translate/blob/main/inference/llm/README.md) |
| 9B GGUF Q4_K_M | 适合，更偏向显存余量 | Homura/Nailong 约 5.63 GB，Translate 约 5.78 GB，也是官方模型卡推荐的平衡档位。[Homura 文件表](https://huggingface.co/IndexTeam/Index-Homura-9B-GGUF)、[Translate 文件表](https://huggingface.co/IndexTeam/Index-Translate-9B-GGUF) |
| 9B GGUF Q8_0 | 短上下文可选 | 权重约 9.53–9.79 GB；Homura 等短文本任务可以评估保留。Nailong 最大窗口不适合直接沿用这一组合。[Nailong 文件表](https://huggingface.co/IndexTeam/Index-Nailong-9B-GGUF) |
| 三个文本模型的 9B FP8 | 有条件可选，暂不作为默认推荐 | Ada GPU 支持 FP8 W8A8，但仍有未量化组件，以及 vLLM 缓存、CUDA Graph 等开销，需要限制上下文并实测。[官方 FP8 模型卡](https://huggingface.co/IndexTeam/Index-Translate-9B-FP8)、[vLLM 硬件支持](https://docs.vllm.ai/en/latest/features/quantization/) |
| Echo 2B FP8 完整包 | 值得作为备选验证 | 包含完整语音流程，但只量化文本主干；还需验证对应版本的加载器兼容性和完整语音峰值。[官方 FP8 模型卡](https://huggingface.co/IndexTeam/Index-Echo-S2ST-2B-FP8) |
| Echo 9B 原版／FP8／FP4 | 不列入本机首选生产清单 | 原版官方要求 ≥24 GB；FP8/FP4 仅量化文本主干，其他语音组件仍保留，缺少完整流程在 16 GB 上的峰值证据。[原版说明](https://huggingface.co/IndexTeam/Index-Echo-S2ST-9B)、[FP8 说明](https://huggingface.co/IndexTeam/Index-Echo-S2ST-9B-FP8) |
| Echo 2B／9B GGUF | 跳过，无法满足原生 S2ST 功能 | 只有文本主干，没有音频塔、连接器和合成栈，不能用它验证完整语音翻译。[官方 GGUF 说明](https://huggingface.co/IndexTeam/Index-Echo-S2ST-2B-GGUF) |
| 各模型 FP4 | 不作为这张卡的默认方案 | 4070 Ti SUPER 没有原生 NVFP4 加速；官方说明旧架构可走权重反量化兼容路径，需要额外验证实际显存与性能。[官方 FP4 说明](https://huggingface.co/IndexTeam/Index-Translate-9B-FP4) |
| Translate 35B-A3B-preview | 不推荐本机全 GPU 生产部署 | Q4_K_M 已达 21.71 GB；Q2_K 虽降至 13.25 GB，仍有运行时余量和质量问题。A3B 是激活参数量，不能按 3B 权重计算。CPU 卸载可另行评估，但吞吐未经本机验证。[官方文件表](https://huggingface.co/IndexTeam/Index-Translate-35B-A3B-preview-GGUF) |

Q2 / Q3 等更低精度版本即使能装入显存，也需要单独验证翻译质量、术语约束和格式保留效果，
不能只凭文件小就列为生产推荐。

## 四、当前 Nailong 默认配置的显存风险

本次检查时，[start.sh](start.sh) 配置了：

```bash
INDEX_NAILONG_MODEL_PATH="$INDEX_NAILONG_MODEL_DIR/Index-Nailong-9B.Q8_0.gguf"
INDEX_NAILONG_CONTEXT_SIZE=229376
```

以上代码块是当前默认值的摘录；实际脚本允许环境变量覆盖。

读取本地 GGUF 元数据得到：Nailong 9B 有 32 个模型层，每 4 层一个完整注意力层，
共 8 个完整注意力层；KV head 数为 4，head dimension（注意力头维度）为 256。
其余为循环状态层，需要另计状态内存。

在单序列、FP16 KV 下，完整注意力层的缓存核算为：

```text
KV 字节数 = token 数 × 8 层 × 4 个 KV head × 256 维 × K/V 两份 × 每元素 2 字节
```

| 配置窗口 | 完整注意力层的 FP16 KV 缓存 |
| --- | ---: |
| 32,768 token | 约 1 GiB |
| 65,536 token | 约 2 GiB |
| 229,376 token | 约 7 GiB |

当前 Q8_0 文件大小为 9,527,501,472 字节，约 8.87 GiB。
按整份权重计入 GPU 的保守预算为：

```text
8.87 GiB 权重 + 7 GiB KV ≈ 15.87 GiB
```

该预算尚未计入循环状态、CUDA 上下文和计算工作区，无法留下生产余量。
实际 GPU 权重分配还受 llama.cpp 的张量放置策略影响，以上不是实测常驻显存。
GPU 锁可以避免不同模型重叠，但不能消除这项单任务风险。

缓存核算依据为 llama.cpp 的[混合缓存实现](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-memory-hybrid.cpp)
和 [KV 分配实现](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp)。
[服务参数文档](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
说明了上下文窗口、并发槽、批量大小和 KV 类型等控制项。

如果必须处理超过 64K 的整篇文档，应进一步验证 Nailong 2B Q8_0，或验证 9B 的量化 KV
缓存方案。量化 KV 的兼容性、翻译质量和实际峰值需要单独验收，不能直接把模型标称最大窗口
当成本机安全窗口，也不能把 64K 总窗口误当成 64K 源文输入额度。

## 五、当前配置与建议的差异

| 服务 | 当前 `start.sh` 默认值 | 本报告候选值 |
| --- | --- | --- |
| Echo | 9B 完整原生包 | 2B v4.1 完整原生包；切换时需同步核对服务的硬件预检、语言边界和依赖 |
| Homura | 9B Q8_0，32,768 token | 9B Q6_K，8,192 token |
| Nailong | 9B Q8_0，229,376 token | 9B Q5_K_M，65,536 token |
| Index-Translate | 9B Q8_0＋mmproj-Q8_0，32,768 token | 9B Q5_K_M＋mmproj-Q8_0，16,384 token，并限制单张图片像素 |

上述候选在保留四个模型原生功能的前提下，为单任务运行时留出显存空间。
最终生产验收需要覆盖模型加载、最大允许输入的预填充、完整输出生成和 worker 退出释放显存
这几个阶段；只有完成这些测量，才能把候选配置记录为本机已验证可用的配置。
