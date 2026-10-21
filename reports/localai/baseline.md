# LocalAI/audio.cpp P0 基线报告

> 状态：**部分完成，尚未满足 P0 版本与模型基线验收条件**。LocalAI 尚未安装或运行，候选权重未下载，原生默认值未改变。
> 采集日期：2026-10-20。详细机器/部署值见 `../../localai/manifests/versions.json`；接口静态盘点见 `../../localai/manifests/capabilities.json`。

## 仓库基线

- 起始 HEAD：`2895c3d399b0771359db4519d8538d55a6b75750`（`feat: 架构调整升级文档`）；方案文档的核对基线为 `26d03f28184598ba6fe906843347794d8fc1832f`。起始工作区干净，当前分支比 `origin/main` 超前 1 个提交。
- 初步盘点 27 个服务 uv 项目和 `main/` 控制面，共 28 个项目、87 条 HTTP 路由（含 `/v1/control` 与 `/v1/health` 同一健康处理器的两个路径）、20 个 Pydantic 请求 schema。
- [能力清单](../../localai/manifests/capabilities.json)登记各服务路由参数、类型/默认值/必填性、20 个请求 schema（含继承字段）、响应注解/媒体类型、源码行号、handler/module 状态码、worker、测试、存储目录、能力去向和已有约束。87 条路由、58 条带输入字段的路由及 294 个输入字段均有去向记录；Qwen3-ASR 的 4 个字段有标准 API E0 映射候选，其余字段按 audio-cpp 待核实或版本化原生桥接候选映射，控制面/MiMo 保留在边界。桥接字段与响应仍未实现或验证。
- [固定 audio.cpp family 筛选](audio-cpp-family-screen.md)以 LocalAI v4.11.0 内嵌源码 commit 对应的 README 为依据，覆盖全部 28 个服务入口：13 个候选、13 个无精确模型族、2 个范围外。候选只是 E1 family/task/variant 声明，不代表实际 RPC 或模型加载兼容。
- `scripts/localai/validate_capabilities.py` 已纳入质量门禁：AST 核验 87 条路由的参数名/类型/默认值/必填性、Pydantic 字段声明及映射一致性。核验中发现并修正 28 个 Pydantic 必填标记，避免将无默认值字段记为可选；该静态检查不替代接口运行或模型实测。
- 每项能力目前仍为 E0/native 默认。8300 控制面、MiMo 云端代理和 Steam Audio CPU 导出保留为边界；Index 文本与视频等不适用 audio.cpp 的能力仍需经 P7 原生桥接方案评估。

## 版本与制品

- 已固定源码候选：LocalAI `v4.11.0`，commit `58830f7ac508845a6f4efa32cfca06af422d4d82`；其官方 tag Makefile 的 `AUDIO_CPP_VERSION` 是 `9a02e61326aaaf9d462b584ca5e0daba22c0abfc`。来源分别为 [LocalAI v4.11.0 release](https://github.com/mudler/LocalAI/releases/tag/v4.11.0) 和 [该 tag 的 audio-cpp Makefile](https://raw.githubusercontent.com/mudler/LocalAI/v4.11.0/backend/cpp/audio-cpp/Makefile)。
- Qwen3-ASR 按方案 §5.2 增补了 E0 候选字段表：`audio → file`、`language → language`；`context` 暂不映射到 `prompt`，`max_new_tokens` 只有确认后端确实透传后才可使用。响应保留原请求 hash/大小/后缀、语言与计时语义。此表尚未通过固定宿主机 LocalAI/backend 验证。
- 部署候选改为宿主机二进制：LocalAI `v4.11.0` commit `58830f7ac508845a6f4efa32cfca06af422d4d82`，其音频后端 Makefile 锁定 audio.cpp `9a02e61326aaaf9d462b584ca5e0daba22c0abfc`。CUDA 12 backend 需从该固定源码构建；目前没有构建产物或进程行为证据。
- 官方 audio.cpp 文档说明后端需显式安装和选择。独立上游 release `v0.9.1` 的 commit `cf124a67cc55d8f65a9a15eec69edbff0fb212c8` 只作为参考；LocalAI v4.11.0 使用的实际上游源码以 `9a02…` 为准，不另行覆盖。
- Qwen3-ASR 1.7B 的 Q8 GGUF 已登记为 E0 候选：固定 package revision `09fe073ba154561f4474162e8bd4ab233a848eca`，文件大小 `2,473,010,048` 字节，LFS SHA-256 为 `da4fc2ac7f24dee784d1684eb1f35836cdbf559519452ae11777670734c0a4f8`。包仓库模型表标注该模型 Apache-2.0；仍需核对完整上游许可文件、运行所需组件，并验证它是否兼容 LocalAI v4.11.0 内嵌的 audio.cpp `9a02…`。见 [固定 package commit](https://huggingface.co/audio-cpp/audio.cpp-gguf/commit/09fe073ba154561f4474162e8bd4ab233a848eca)。本次未下载权重，候选不代表可运行。
- 目前仅有该候选模型包记录；尚无经兼容性验证的 LocalAI package。配置指向的 36 个 Hugging Face 缓存目录中有 34 个存在；Confucius4 TTS 与 Index Echo 的目录缺失。目录存在不代表权重可用。

## 本机与依赖环境

| 项目 | 采集结果 |
| --- | --- |
| OS / WSL | Ubuntu 24.04.4 LTS，WSL2，内核 `6.18.33.2-microsoft-standard-WSL2` |
| GPU / 驱动 | NVIDIA GeForce RTX 4070 Ti SUPER，driver 610.47，显存总量 16,376 MiB；采集时空闲 14,526 MiB |
| RAM | 35 GiB 总量、约 29 GiB 可用；低于方案规划假设的 48 GB |
| Linux 文件系统 | 1,007 GiB 总量，470 GiB 已用、487 GiB 可用 |
| `/mnt/c` | 953 GiB 总量，887 GiB 已用、67 GiB 可用 |
| 工具链 | Python 3.12.13、uv 0.12.3、FFmpeg/ffprobe 6.1.1、SoX 14.4.2、CMake 3.28.3、GCC 13.3.0 |
| LocalAI / llama-server / Go | LocalAI 与 llama-server 均未安装；Go toolchain 不可用，GitHub 源码连接超时 |
| 模型缓存 | `~/hf-mirror` 存在（225 GiB）；Qwen3-ASR 原生权重目录约 4.4 GiB 且服务预检完整；未读取或使用 `storage/` 中的用户生成音频 |

当前没有可用 LocalAI 二进制或 audio-cpp backend，也未尝试以非固定版本替代。GitHub 源码访问本次超时，故没有执行裸机安装或构建。没有已准备并授权的短、中、长 benchmark corpus，尚未运行完整 native 模型质量/延迟/资源基线。P0 不从 `storage/` 中默认选取已有音频作为样本。

## Native ASR 短样本基线

使用工作区中的 `tests/testIndexTranslate/speech.wav`（11.008 秒、48 kHz、双声道）通过本机
Qwen3-ASR 原生 `/v1/qwen3/asr` HTTP 路由一次。该 WAV 被 `.gitignore` 的 `*.wav` 规则忽略，
路径也是 Index Echo Demo 的默认输入，但文件来源和内容不受 Git 版本控制。相邻的
`subtitles.md` 是 Index 文本 Demo 的默认输入；没有证据证明它与这份本机 WAV 对齐，因此不作为
ASR 参考文本。实际请求返回 HTTP 200、`language=Chinese`、41 个转写字符；报告不复制转写正文，
没有计算 CER/WER，也没有质量通过结论。

| 指标 | 结果 | 口径 |
| --- | --- | --- |
| API `elapsed_seconds` | 3.863 秒 | worker 内模型计时；不含 API 上传和 GPU 排队 |
| Worker 进程时间 | 8.62 秒 | `communicate()`；含 Python 启动与权重加载 |
| 持锁时间 | 10.63 秒 | worker 退出后包含 2 秒 `CUDA_RELEASE_DELAY` |
| GPU 已用显存峰值 | 6,350 MiB | `nvidia-smi` 全卡汇总；启动前 1,640 MiB，结束后 1,611 MiB，不是模型独占量 |
| 返回音频大小 | 2,113,614 字节 | 上传 staging 的原始大小 |
| 语言 / 转写长度 | Chinese / 41 字符 | 只记长度，不写入音频或转写内容 |

本次只有一个短样本和一次调用，不构成性能统计或 E3。该 native 缓存没有固定 Hub revision；仅记录
用于复现此次调用的权重与配置 SHA-256，不能与未下载的 Q8 候选作精度对照：

| 文件 | SHA-256 |
| --- | --- |
| `model-00001-of-00002.safetensors` | `a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6` |
| `model-00002-of-00002.safetensors` | `6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc` |
| `config.json` | `2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f` |
| `generation_config.json` | `1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c` |

## 质量门禁

初始 P0 记录时的门禁包含根回归 292 项。纳入 LocalAI gateway、管理器与完整能力筛选检查后，最近一次
`UV_OFFLINE=1 bash scripts/quality_gate.sh` 退出码 0：`bash -n start.sh`、3 个无 SDK C++ renderer 核心测试、
28 个 uv 项目 lock check、Ruff 与格式检查通过（200 个文件已格式化）；根回归 326 项、ACE-Step 9 项、
Stable Audio 7 项均通过。测试中的 mock 失败日志属于预期场景。

另发现锁文件规则不一致：仓库说明称 `unitale_runtime/` 没有独立 `uv.lock`，但 checkout 中存在受跟踪的 `unitale_runtime/uv.lock`；质量门禁也未将该项目列入锁检查。需确认实际依赖管理意图后同步文档或门禁。

## P0 未闭合项与后续入口

1. LocalAI 与 audio.cpp 源码 commit 已固定；还需准备固定二进制/源码、检查 `package.sh` 产物与 wrapper，再运行 backend smoke。
2. 能力清单为 294 个源字段登记了候选 destination。Qwen3-ASR 的本机服务路由已有 LocalAI 候选实现与映射拒绝逻辑；其他本地模型字段仍是待实现的版本化桥接 envelope。Qwen3-ASR Q8 GGUF 仍为 E0 候选，尚未完成固定 commit 兼容验证。
3. Native ASR 只覆盖一个短样本；仍需有参考文本的短/中/长授权样本和多语种集合，再完成 native 重复基线与质量评价。
4. 当前没有 E2/E3 LocalAI 实测，全部服务仍以 native 为默认；P1/P2 mock 与代码证据见 [`P1-P2.md`](P1-P2.md)。

本报告记载初始基线；后续管理器现采用宿主机进程方案，LocalAI 默认仍关闭且所有服务仍使用 native。
