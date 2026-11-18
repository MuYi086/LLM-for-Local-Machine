# LocalAI/audio.cpp 模型服务运维

本文描述当前十五个 LocalAI/audio.cpp GGUF 候选、统一启动和故障恢复。模型文件名、环境变量、family/task、
model ID、端口与 API 路由以 [audio-cpp-models.json](../localai/manifests/audio-cpp-models.json) 为准；
所有模型制品都由部署者预先放入本地镜像，manager 不下载权重。

## 固定运行合同

- LocalAI 是宿主机 loopback 服务，默认地址为 http://127.0.0.1:8400，使用固定 LocalAI v4.11.0 和相应
  audio.cpp CUDA backend。版本与制品指纹见 [versions.json](../localai/manifests/versions.json)。
- 当前候选包括 Qwen3-TTS VoiceDesign、Confucius4-TTS、IndexTTS2.5、MiDashengLM-Gen、Qwen3-ASR、
  Qwen3-TTS Base、SeedVC、Stable Audio SFX、Stable Audio 3 Medium、MOSS-SoundEffect v2、VoxCPM2、
  BS-RoFormer、Qwen3 ForcedAligner、Mel-Band RoFormer 和 HTDemucs。
- GGUF 默认从 $HF_MIRROR_DIR/audio.cpp-gguf/ 下的候选目录读取，环境变量可覆盖路径；候选文件名必须固定。
- 8300 控制面、8301 VoiceDesign 与十四个 audio.cpp 模型 API 共十六个 HTTP 进程。十四个 API 共用
  model-audio/audio_cpp_runtime 的轻量 uv 环境，不安装 Torch。
- 所有 GPU API 必须共用 GPU_LOCK_FILE。Steam Audio renderer 是 CPU 路径，不获取 GPU 锁。
- 当前 LocalAI provider 固定启用。start.sh 管理 LocalAI 进程组，按模型请求加载任务，推理完成后请求 backend
  shutdown，并等待 CUDA_RELEASE_DELAY 后释放 GPU 锁。

## 安装与启动

先准备 LocalAI binary、audio.cpp 固定 revision `0382d37fe6e1a817b6b8fed5dcaab0e11fd1e8e6` 构建的 `audio-cpp` backend，以及 manifest 中列出的十五个 GGUF 文件，然后同步锁定依赖。旧 revision `9a02e61326aaaf9d462b584ca5e0daba22c0abfc` 不支持 MOSS-SoundEffect v2；`verify-install` 只核对并记录文件指纹，不下载或构建 backend。

LocalAI v4.11.0 的 backend adapter 尚未镜像该 audio.cpp revision 新增的 `TurnDetection`、`WakeWord` 和
`AudioClassification` task。构建前，从 LocalAI v4.11.0 源码根目录应用
[`audio-cpp-task-kinds.patch`](../scripts/localai/patches/audio-cpp-task-kinds.patch)，再按 `versions.json`
锁定的 CUDA 12 配置构建 backend。该补丁同步 adapter 的 task enum、名称和双向映射，保留 `-Werror=switch`
检查。部署新 backend 后再运行 `verify-install`；源码 revision 变更时，manager 会拒绝仅刷新旧 backend 的指纹。

~~~bash
uv sync --project main --locked
uv sync --project model-audio/design_qwen3-tts-1.7b-voicedesign --locked
uv sync --project model-audio/audio_cpp_runtime --locked
uv sync --project qa --locked
python3 scripts/localai/manager.py verify-install
~~~

启动整组服务：

~~~bash
bash start.sh
~~~

start.sh 会依次检查所有二十一个 HTTP 端口、注册十五个候选、启动受管 LocalAI、执行候选 readiness 检查，
再启动控制面、VoiceDesign 和十四个 audio.cpp API。任一 HTTP 服务退出时，脚本会清理其他服务进程组并停止
受管 LocalAI。启动期间不会加载 GGUF 模型；模型只在请求到达后加载。

候选注册命令会在 LocalAI models 目录建立指向本地 GGUF 的受控符号链接，并原子写入 YAML，不复制权重。
常用检查：

~~~bash
python3 scripts/localai/manager.py verify-audio-cpp-candidates
python3 scripts/localai/manager.py status
curl -fsS http://127.0.0.1:8300/v1/control
curl -fsS http://127.0.0.1:8301/v1/health
curl -fsS http://127.0.0.1:8311/v1/health
~~~

新增模型服务的健康检查验证 LocalAI readiness 和 model ID 注册，不触发推理或权重加载。

## 全局 GPU 队列与 quarantine

所有模型 API 在共享 llm_local_runtime 的 GPU_LOCK_FILE 上获取有限时长的跨进程 flock；默认排队上限为
900 秒。GPU 锁同时覆盖 LocalAI 推理、backend shutdown 和 CUDA_RELEASE_DELAY。排队超时或发现 quarantine
标记时，服务返回 503。该锁必须由以后加入的 GPU 模型继续共用，避免并发占用显存。

每次 LocalAI 推理持锁期间都会创建锁旁的 .quarantine 标记。若 backend shutdown 或进程回收无法确认，标记会保留，
阻止所有其他 GPU 模型开始推理。禁止手动删除该标记。

正常退出 start.sh 会终止全部二十一个 HTTP 服务进程组，再要求 manager 停止 LocalAI 受管进程组。独立检查或停止：

~~~bash
python3 scripts/localai/manager.py status
python3 scripts/localai/manager.py stop
~~~

完成 backend 修复后，使用恢复脚本。脚本在 GPU 锁内确认模型/受管进程退出和 loopback 端口关闭后才解除隔离：

~~~bash
LLM_LOCALAI_ENABLED=true python3 scripts/localai/recover_quarantine.py
~~~

具体的隔离文件合同和新增 provider 要求见[全局 GPU 锁说明](global-gpu-lock.md)。

## 诊断和能力清单

LocalAI/audio.cpp API 由当前活跃服务清单描述；修改请求字段、路由、端口或请求 schema 后，同步生成并校验能力清单：

~~~bash
uv run --project qa --locked python scripts/localai/validate_capabilities.py
bash scripts/quality_gate.sh
~~~

新增 audio.cpp 模型服务的能力清单只描述接口合同；真实推理与听感仍需运行对应的生成样例验证。
按顺序运行模型生成样例、指定 WAV 输入要求和输出位置见
[audio.cpp 冒烟生成说明](../tests/audio_cpp/README.md)。
