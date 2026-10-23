# LocalAI 与 audio.cpp 升级及镜像清理记录

> 状态：宿主机构建与当前维护范围已完成；2026-10-22 修复升级复核中的五项问题及真实 Qwen 调用发现的候选配置遗漏。完整无模型门禁、真实 LocalAI 大文件传输和两个 Qwen 短句推理通过。完整模型部署与音质验收仍需单独进行。本轮不使用 Docker，不下载模型，不安装 Python 项目依赖。更新日期：2026-10-22。

## 1. 目标与完成标准

LocalAI 以宿主机进程承载固定 audio.cpp CUDA backend；服务保留 HTTP 参数校验、逻辑文件引用、
GPU 锁、原子保存和原有响应格式。自定义 gRPC bridge 只负责将版本化 RPC 分发到各服务环境的一次性 worker。

升级交付包括固定版本、可复核的宿主机构建、进程管理、provider/bridge 路由、测试、能力清单和运维文档。
镜像清理依据是用户指定的 ~/hf-mirror：不存在的主模型或服务必需辅助模型视为退役，不留下等待下载的服务代码。

## 2. 架构与边界

LocalAI 与 worker bridge 使用 loopback；provider 显式启用，默认仍为 native。调用服务持有共享 GPU 锁，
直到 backend shutdown 或 worker 进程组退出及 CUDA 回收等待结束。shutdown 无法确认时保留 quarantine 标记，
阻止后续 GPU 请求。

bridge 只允许 localai/manifests/worker-bridge.json 登记的模型 ID、uv 项目、worker.py、输出类型、
超时与环境白名单。输入和大文件通过流式 staging、逻辑标识或随机 artifact ID 传递，不信任任意路径。

Steam Audio renderer 是独立 CPU 路径；其 SDK 不属于 LocalAI/audio.cpp backend，也不由 LocalAI 管理。

## 3. 软件构建与版本

固定 LocalAI v4.11.0，audio.cpp revision 为 9a02e61326aaaf9d462b584ca5e0daba22c0abfc。
使用用户提供的本地源码包和 Ubuntu 24.04 amd64 deb，在私有前缀构建 CUDA 12 sm_89 gRPC backend，
再按固定上游 package 脚本打包。没有使用 Docker，没有用 apt/dpkg 修改系统安装状态。
LocalAI binary、backend 包和安装指纹位于忽略的本地运行目录；版本和 hash 见 localai/manifests/versions.json。

manager.py verify-install 校验 LocalAI 版本、backend 入口和文件树 hash。manager start/stop 管理
loopback 宿主机进程组、owner watcher、readiness、TERM/KILL、进程身份和端口回收。
候选登记只创建或验证本地 YAML，不下载模型。

## 4. 镜像筛选与维护范围

当前维护 13 个模型 uv 项目与 1 个控制面，共 14 个 HTTP 进程。当前镜像中的 13 个服务主模型根目录均存在。
目录存在不代表所有文件、依赖、真实推理或输出质量已验收。

### 保留的 LocalAI 路径

- Qwen3-TTS Base 克隆与 Qwen3 VoiceDesign 使用本地官方 Hugging Face 目录，保留直接候选 YAML、结构预检和显式 provider。
- SheetSage2、MERT-v2-FullSong、YuE2-Vae、YuE2-3B、TIGER-DnR 使用五个固定 one-shot worker bridge ID。
- MOSS-Music-8B-Instruct 使用本地完整 GGUF 与 HOT-Step-CPP 原版 `ace-caption`；通过独立原生 worker 提供 8310 的音乐理解 API，没有 LocalAI provider，也不加入五个固定 worker bridge。
- 其余原生服务保持各自原生 worker；原生 provider 仍为默认值。

- SoulX-FlashHead 已在补齐 `$HF_MIRROR_DIR/Soul-AILab/wav2vec2-base-960h` 后恢复独立原生服务（8391）；不接入 LocalAI，不加入五个固定 worker bridge。

### 已清理的服务与候选

- Stable Audio 3 Medium 的主模型目录缺失，完整服务代码和全部启动/维护/测试引用已移除。
- Seed-VC 主目录虽存在，但必需 Whisper-small 不存在；其 44 kHz BigVGAN/RMVPE 也缺失。VC/SVC 服务和候选已移除。
- Qwen3-ASR、VoxCPM2、Confucius4-TTS、ACE-Step 原生模型根目录存在；相应直接 GGUF 不在 hf-mirror，因此仅移除 LocalAI provider、GGUF 候选、注册/预检和专属测试，原生 API 保留。
- Index 相关代码此前已按用户要求退役；task10 中列出的服务也不属于当前部署范围。

维护清单见 reports/localai/manual-model-readiness.md。固定 audio.cpp 上游 family 清单只作为 E1 研究证据，
不能推断模型文件存在或 LocalAI provider 已接入。

## 5. Worker bridge

当前固定五个 worker ID：

| 服务 | 固定 model ID | 输出合同 |
| --- | --- | --- |
| SheetSage2 | llm-local.sheetsage2.worker.v1 | ZIP |
| MERT-v2-FullSong | llm-local.mert-v2-fullsong.worker.v1 | embedding JSON / frames NPZ |
| YuE2-Vae | llm-local.yue2-vae.worker.v1 | encode NPY / decode WAV |
| YuE2-3B | llm-local.yue2-3b.worker.v1 | WAV |
| TIGER-DnR | llm-local.tiger-dnr.worker.v1 | 三 Stem ZIP 文件引用 |

注册和检查：

~~~
python3 scripts/localai/manager.py register-worker-bridge-candidates
python3 scripts/localai/manager.py verify-worker-bridge-candidates
~~~

无模型测试覆盖 RPC envelope、allowlist、服务路由、文件合同、大小限制和清理；
五个 bridge 的真实模型加载、资源和输出质量仍需模型运行时单独验证。
当前五个 worker 文件成品统一使用最大 4 GiB 的共享 `file_ref`，不将 WAV/NPZ/ZIP 放进 gRPC Reply；
服务与 bridge 必须使用相同的 `LLM_LOCALAI_ARTIFACT_DIR`。取回时流式复制并删除共享原文件。

## 6. 验证与当前限制

离线可运行的检查：

~~~
python3 scripts/localai/manager.py verify-install
python3 scripts/localai/manager.py verify-worker-bridge-candidates
UV_OFFLINE=1 bash scripts/quality_gate.sh
~~~

能力清单当前校验 14 个服务入口、13 个模型 uv 项目、43 条路由、6 个请求 schema 和 132 个请求字段。
quality gate 还包含 shell 语法、Steam Audio renderer 的无 SDK 核心测试、锁检查、Ruff、格式检查与两组无模型回归。
2026-10-21 的历史 `UV_OFFLINE=1 bash scripts/quality_gate.sh` 记录为通过：13 个 uv 项目锁检查、Ruff 和 104 个 Python 文件格式检查通过；
能力清单通过；根目录 198 项、ACE-Step 9 项无模型测试全部通过。`bash -n start.sh` 和 Steam Audio renderer
无 SDK 核心测试也包含在门禁中。

LocalAI `verify-install`、两个 Qwen3-TTS 候选和五个 worker bridge 候选核验通过。对 11 个服务 uv 项目、
`localai_worker_bridge/` 和 `qa/` 执行 `uv sync --locked --offline --dry-run` 均显示 `Would make no changes`；
该只读检查没有安装依赖。13 份维护文档的相对文件链接检查通过。

本机 Steam Audio SDK 4.8.1 已放置于 `$HOME/.local/share/steam-audio/4.8.1`，空间 renderer
可执行文件已存在于 `model-audio/spatial_steam-audio/build/`；`ldd` 可解析 `libphonon.so`。
Linux 启动与构建脚本均默认使用该 SDK 路径，无需每次额外设置环境变量。SDK 保持在仓库外，
不会提交。模型镜像中没有保留服务需要用户补充的缺失主模型；缺失 GGUF 和辅助模型对应的功能均已退役。
模型权重及真实测试样本仍由用户管理。下节记录当前目录布局修复后的新验证结果；
历史门禁结果不能代替当前提交验证。

## 7. 2026-10-22 复核修复与当前验证

复核基线为 `1bd58ae`（目录优化），详细问题和修复证据见 [升级后问题](../升级后问题.md)。

- manager 从已验证的 worker manifest 获取回收入口，覆盖五个迁移后的项目；
  子进程继承入口标记，父 worker 提前退出后仍可识别并回收遗留组。
- bridge 对成功、非零退出、取消和超时统一执行整个进程组的 TERM/KILL 收尾；
  Free 重试尚未确认的组，失败时保留隔离。固定 LocalAI 可能吞掉 Free 错误，
  gateway 因此另行检测被 bridge 更新的隔离标记，恢复脚本要求宿主机退出证据。
- 五个 bridge 文件产物均使用共享引用，泛用内联产物上限降为 32 MiB，
  为固定 LocalAI 的 50 MiB gRPC 接收限制预留 base64 和 envelope 空间。
- 修复服务路由测试、uv 配置测试、ACE-Step 启动断言及文本训练测试的目录迁移遗漏，
  修正训练推理脚本的包内/直接执行导入，完成 Ruff 安全修复并同步能力清单行号。
- 真实 Qwen 调用发现绝对 `parameters.model` 被 LocalAI 拒绝，以及 Base session 选项缺少
  `session.` 前缀。候选现通过 models 内 `<model-id>.hf` 符号链接引用本地快照，
  verify 命令同时检查链接目标；不复制权重。旧本机 YAML 已备份后重新登记。
- LocalAI 可变数据设置到 `$LLM_LOCALAI_DATA_DIR/data`，避免污染仓库根目录。
  本机残留的两份退役 worker YAML 已备份移出 models；备份不在 LocalAI 扫描目录。

当前验证结果：

| 验证 | 结果与边界 |
| --- | --- |
| 完整离线质量门禁 | 13 份锁、Ruff、107 个文件格式检查、能力清单、三个 C++ 核心测试、根目录 205 项及 ACE-Step 9 项测试通过 |
| 生命周期回归 | 五个 worker 入口、继承入口标记、四种退出结果的抗 TERM 子进程、Free 失败及隔离恢复均有无模型覆盖 |
| `smoke_worker_bridge.py` | 41 MiB、129 MiB 静音 WAV 经真实 HTTP → Go gRPC → bridge 取回，内容、消费清理及宿主机停止确认通过；不加载模型 |
| Qwen VoiceDesign | 一个真实短句，24 kHz、76,800 帧、153,644 字节 WAV；shutdown 确认成功 |
| Qwen Base | 使用上述合成参考做一次 x-vector-only 克隆，24 kHz、76,800 帧、153,644 字节 WAV；shutdown 确认成功 |
| Qwen 资源记录 | 本轮两次调用可见整卡峰值分别约 5,310 / 5,217 MiB，最终宿主机停止后采样为 1,394 MiB；不等于模型独占显存或完整资源基准 |
| Qwen 制品指纹 | 本地 metadata revision 与每个快照 13 个非缓存文件 SHA-256 已记入 `versions.json`；未联网核对上游快照 |

大文件 smoke 可独立复现：

```bash
uv run --project qa --locked python scripts/localai/smoke_worker_bridge.py
```

两个 Qwen 短句仅证明加载、调用、WAV 合同和本次 shutdown 路径可运行；
不代表听感、克隆相似度、长文本、多语种、错误注入及所有模型的 E2/E3 验收完成。
五个 bridge 的真实模型推理、YuE2-3B 在本机 16 GiB GPU 上的资源适配仍需另行验证。
