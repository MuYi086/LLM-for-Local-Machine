# 仓库开发指南

## Music Postprocessing Rules

### Skill routing

- audio-analyzer：仅负责音频分析，输出 JSON，不修改源音频。
- mix-engineer：仅负责分轨 EQ、动态处理和混音，不执行最终母带标准化。
- mastering-engineer：唯一负责最终母带、响度及峰值控制。
- ffmpeg-skill：仅负责音频裁剪、探测、格式转换和通用导出，不接管混音与母带任务。
- music-evaluate：仅负责质量评估和生成报告，未经明确授权不得重新处理音频。
- batch-processor：负责批量任务执行，不决定音乐处理策略。

### Execution rules

1. 输入源文件只读，每个阶段写入独立输出目录；前一阶段验证后才能执行下一阶段。
2. 同一任务不得由两个 Skill 重复执行响度归一化。
3. 禁止多个进程同时写入同一输出文件。
4. 默认不并行执行多个 GPU 音频模型；全部阶段完成后执行最终质量检查。
5. 不得把 AI 分离的估计分轨当作原生录音分轨。

## 项目范围和服务

仓库维护九个音频模型服务、Ornith 和 Qwen 两个文本模型 API、一个轻量控制面和 Steam Audio CPU renderer。
start.sh 管理十四个 HTTP 进程，包括只绑定 loopback 的两个 llama.cpp backend；Steam Audio renderer 由控制面按需调用，
不是 HTTP 服务。八个 audio.cpp 服务共用一个轻量 uv 项目，避免重复安装 Torch 和 Python 推理依赖。

| 服务 | 默认端口 | 主业务接口 | 职责 |
| --- | ---: | --- | --- |
| main/ 控制面 | 8300 | GET /v1/control；POST /v1/audio/export；POST /v1/audio/spatial/render | 诊断、共享音频处理和空间导出 |
| model-text/ornith_1_5_35b_a3b/ | 8401 | POST /v1/AtomicChat/Ornith-1.5-35B-A3B-AD-Q5 | llama.cpp 聊天补全适配器 |
| model-text/qwen3_6_35b_a3b/ | 8402 | POST /v1/qwen/Qwen3.6-35B-A3B-MTP | llama.cpp Qwen3.6 MTP 聊天补全 |
| Ornith llama.cpp backend | 8403（仅 127.0.0.1） | POST /v1/chat/completions | Ornith GGUF、MTP 与 mmproj 推理 |
| Qwen3.6 llama.cpp backend | 8400（仅 127.0.0.2） | POST /v1/chat/completions | Qwen3.6 GGUF 与 MTP 推理；与 127.0.0.1:8400 LocalAI 并存 |
| design_qwen3-tts-1.7b-voicedesign/ | 8301 | POST /v1/qwen/Qwen3-TTS-1.7B-VoiceDesign | 文本音色设计与 TTS |
| tts_confucius4-tts/ | 8331 | POST /v1/netease/Confucius4-TTS-GGUF | 参考音频克隆 |
| tts_indextts2.5/ | 8312 | POST /v1/indexTeam/IndexTTS2.5-GGUF | 参考音频克隆 |
| gen_midashenglm/ | 8341 | POST /v1/mi/MiDashengLM-Gen-GGUF | 文本提示音频生成 |
| asr_qwen3-1.7b/ | 8321 | POST /v1/qwen/Qwen3-ASR-1.7B-GGUF | 语音转写 |
| tts_qwen3-1.7b-base/ | 8311 | POST /v1/qwen/Qwen3-TTS-12Hz-1.7B-Base-GGUF | 参考音频克隆 |
| vc_seed-vc/ | 8342 | POST /v1/SeedVC/SeedVC-MLX-GGUF | 音色转换 |
| gen_stable-audio-3-small-sfx/ | 8343 | POST /v1/stabilityai/Stable-Audio-3-Small-SFX-GGUF | 音效生成 |
| tts_voxcpm2/ | 8313 | POST /v1/OpenBMB/VoxCPM2-GGUF | TTS、音色设计和可选参考克隆 |
| spatial_steam-audio/ | 无 HTTP 端口 | 由 8300 控制面调用 | C++17 CPU 空间 renderer |

文档按源码描述已接入的接口，不把上游 family 声明等同于模型能力。项目介绍、部署和 API 示例见
[README.md](README.md)。GGUF 文件名、环境变量、LocalAI model ID、family/task、路由与端口以
[audio-cpp-models.json](localai/manifests/audio-cpp-models.json) 为准。

## 目录职责和运行环境

- main/main.py 负责健康/控制、共享上传检查、CPU 导出和空间渲染编排，不加载模型。
- llm_local_runtime/ 是轻量共享包，负责流式上传、引用、原子提交、容量/保留策略和 GPU 锁；不得导入
  FastAPI、Torch 或模型包。由维护中的 uv 项目以 editable 依赖引用。
- inference_gateway/ 只依赖标准库与 llm_local_runtime，负责 LocalAI provider、GPU 锁执行、backend
  shutdown 和 quarantine。backend 停止无法确认时必须保留锁旁的 .quarantine 标记；仅用
  scripts/localai/recover_quarantine.py 在确认进程组和 loopback API 退出后解除隔离。
- inference_gateway.sdcpp 提供按请求启动的 sd.cpp CLI provider；sdcpp/manifests/ 记录外部版本与默认空模型清单。
  scripts/sdcpp/manager.py 负责本机状态、预检和生成。尚无图片/视频 HTTP 服务；不得把未验证模型计入当前能力。
  CLI 无 HTTP 端口，隔离恢复按 PID/starttime 与进程组核验，统一使用既有 recover_quarantine.py 入口。
- model-audio/audio_cpp_runtime/ 是八个新增模型 API 的共享轻量 uv 项目，不安装 Torch。各模型目录的
  main.py 只从清单选择对应的共享服务合同。
- scripts/localai/manager.py 管理裸机 LocalAI/audio.cpp 固定版本、九个 GGUF 候选和宿主生命周期；不使用 Docker。
- model-audio/spatial_steam-audio/ 使用仓库外 Steam Audio SDK 与动态库，是 CPU renderer，不获取 GPU 锁。
- qa/ 是无模型测试和 Ruff 项目；scripts/quality_gate.sh 与 CI 共用质量门禁。
- model-text/scripts/training/、model-text/training/configs/ 是纯文本润色 LoRA 实验，不是语音模型训练或 HTTP 服务。
- model-video/ 保留已退役视频模型源码和历史说明，不属于当前启动项或质量门禁。
- 历史报告、任务和方案只记录当时状态，不据此推断当前接口。

## 安装、启动与验证

部署使用 Linux、Python 3.12.13、uv、FFmpeg/ffprobe、CUDA/NVIDIA 驱动及固定版本 LocalAI/audio.cpp。
统一启动依赖 setsid、进程组信号和 fcntl.flock；renderer 构建需要 CMake 3.17+ 和 C++17。

准备权重、仓库外 LocalAI/backend 和 llama.cpp `llama-server` 后，执行 uv sync --project <dir> --locked。
Ornith GGUF、MTP 与 mmproj 默认位于 `$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/`，Qwen GGUF 默认位于
`$HF_MIRROR_DIR/unsloth/Qwen3.6-35B-A3B-MTP/`；可通过 `LLAMA_SERVER_BIN`、`ORNITH_MODEL_FILE`、
`ORNITH_MMPROJ_FILE`、`ORNITH_MTP_FILE` 和 `QWEN36_MODEL_FILE` 覆盖默认位置。维护中的 Python 环境为
main/、VoiceDesign 独立项目、model-audio/audio_cpp_runtime/ 和 qa/。依赖变化后在对应项目执行
uv lock，提交 pyproject 与锁文件，并用 uv lock --check 验证。

九个音频 GGUF 服务通过 LocalAI/audio.cpp 推理；Ornith 与 Qwen 文本模型由 CUDA 构建的 llama.cpp 推理。
Ornith 前 23 层、Qwen 前 23 层 MoE 专家默认留在 CPU，Ornith 的 `mmproj` 负责图片输入。模型、mmproj 和
llama.cpp 源码/二进制均在仓库外，不提交权重或第三方源码。
文本模型默认采用串行实测配置：Ornith 线程 6/8、Qwen 线程 8/8，主 KV 为 Q8、MTP 草稿长度为 4、
草稿 KV 为 F16、load-mode 为 none；详细步骤与结论见 tests/model_web_comparison/README.md。

bash start.sh 先检查全部十四个 HTTP 端口、两个文本模型权重和 CUDA llama-server，注册九个 LocalAI 候选并执行 readiness
预检；之后通过共享 GPU 锁依次预热 Ornith 与 Qwen，确认模型进入 sleep-idle 后再启动控制面和全部 API。
任一服务退出时清理全部服务进程组并分别确认两个 llama.cpp backend 停止。使用 Ctrl+C 等待正常退出；若 backend 停止状态不能确认，
保留 quarantine 标记并按恢复流程处理。

无模型质量门禁：

~~~bash
bash -n start.sh
bash scripts/quality_gate.sh
uv run --project qa --locked python -m unittest discover -s tests -v
~~~

测试不得下载权重、依赖 CUDA、调用外部模型服务或执行真实模型。真实模型串行生成单独执行，音频样例用
tests/audio_cpp/generate_examples.py，Ornith 和 Qwen 文本样例分别用
tests/ornith_1_5_35b_a3b/generate_example.py 和 tests/qwen3_6_35b_a3b/generate_example.py；
输出放入运行数据目录。

## 架构与接口约束

- HTTP 入口负责校验、响应、存储和 worker 生命周期。阻塞锁、subprocess 和网络任务使用同步路由或线程池，
  不能阻塞异步事件循环。
- 所有 GPU provider 必须使用同一个 GPU_LOCK_FILE，保证跨进程串行执行，避免显存 OOM。纯 CPU/文件预检
  在获取锁前完成；Ornith 和 Qwen 请求在持锁后先确认对应 backend `/props.is_sleeping`，再提交推理；结束后仍须确认模型卸载并等待
  CUDA_RELEASE_DELAY，最终由 finally 释放锁。队列默认等待 900 秒，部署值必须为正数，超时返回 503。
- `.quarantine` 标记存在时，任何 GPU provider 都不得推理。LocalAI 要确认 backend 已关闭；Ornith 与 Qwen 要确认 llama.cpp
  进程组和各自 loopback API 已退出；仅使用 `scripts/localai/recover_quarantine.py` 完成确认后移除标记。
- 模型 API 健康检查不加载权重。音频上传按块暂存，默认单文件上限 64 MiB；不得整文件 read()、
  直接覆盖 sidecar 或将临时输入长期保存。
- audio.cpp 服务默认值、候选和 API 合同集中在候选 manifest 与 audio_cpp_runtime；修改服务合同后同步
  localai/manifests/capabilities.json 并运行能力清单校验。
- Ornith API 将 OpenAI Chat Completions 请求转发到 loopback llama-server；主 GGUF 与 F16 mmproj 从镜像目录只读加载，
  不注册为 LocalAI 候选，也不提交模型文件。
- Qwen3.6 API 将 OpenAI Chat Completions 请求固定转发到 `127.0.0.2:8400`；GGUF 已包含 MTP tensors，默认启用
  `draft-mtp`，使用 CUDA0 并将前 23 层 MoE 专家留在 CPU。它与绑定在 `127.0.0.1:8400` 的 LocalAI 并存。
- Qwen3-TTS Base 通过 WAV 参考音频克隆，且必须提供与音频逐字匹配的 `reference_text`；输入 `zh`/`zh-CN`
  会映射为 audio.cpp 接受的 `Chinese`。Confucius4 和 IndexTTS2.5 也通过 WAV 参考音频克隆；SeedVC 需要源音频和目标 WAV；
  Qwen3-ASR 返回转写 JSON；MiDashengLM-Gen 和 Stable Audio SFX 返回音频；VoxCPM2 允许无参考音频运行。
- Steam Audio 正式 balanced/immersive 路径必须保留对象边界，使用 Manifest v1 和重复 assets 字段，
  不在浏览器预混且不取 GPU 锁。Steam Audio pre-master 后只允许 loudnorm/编码，不叠加 Haas/aecho。
- diffuse、非 none 遮挡和几何反射尚未实现；不得将保留的语义字段描述为可用能力。

## 存储、清理和提交

- 上传经过 llm_local_runtime 流式暂存、校验、SHA-256 和原子提交。客户端 full_path 是逻辑标识，不能直接
  读取服务器任意路径。
- 默认目录：VoiceDesign storage/timbre/；控制面与服务临时文件位于 $RUNTIME_CACHE_DIR。Qwen 与 Ornith 真实样例
  写入 storage/llm_tests/；新增 audio.cpp API 不长期保存上传或输出。
- 保留策略默认关闭；显式设置时先预览清理计划。不得删除用户上传、历史运行数据、生成结果或策略未覆盖目录。
- 训练代码不自动取得服务 GPU 锁；执行真实训练/预测需单独安排 GPU。
- 不提交权重、第三方源码、上传素材、生成音视频、训练数据、虚拟环境、缓存、密钥或机器专用绝对路径。
  Commit subject 使用简洁 Conventional Commit，例如 feat:、fix:、docs:。
- Python 注释和 Docstring 默认简体中文；标识符、类型名、库/API 名与通用技术术语保留英文。公共函数、公共类和
  重要业务入口提供 Google Style / PEP 257 Docstring。复杂并发、进程生命周期、GPU/CUDA 和兼容 workaround
  说明其原因；显然的赋值与简单条件不写逐行注释。
- 详细 Python 约定见 [docs/python-engineering.md](docs/python-engineering.md)。
