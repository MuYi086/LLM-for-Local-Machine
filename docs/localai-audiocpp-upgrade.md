# LocalAI/audio.cpp 模型服务运维

本文描述当前九个 LocalAI/audio.cpp GGUF 候选、统一启动和故障恢复。模型文件名、环境变量、family/task、
model ID、端口与 API 路由以 [audio-cpp-models.json](../localai/manifests/audio-cpp-models.json) 为准；
所有模型制品都由部署者预先放入本地镜像，manager 不下载权重。

## 固定运行合同

- LocalAI 是宿主机 loopback 服务，默认地址为 http://127.0.0.1:8400，使用固定 LocalAI v4.11.0 和相应
  audio.cpp CUDA backend。版本与制品指纹见 [versions.json](../localai/manifests/versions.json)。
- 当前候选包括 Qwen3-TTS VoiceDesign、Confucius4-TTS、IndexTTS2.5、MiDashengLM-Gen、Qwen3-ASR、
  Qwen3-TTS Base、SeedVC、Stable Audio SFX 和 VoxCPM2。
- GGUF 默认从 $HF_MIRROR_DIR/audio.cpp-gguf/ 下的候选目录读取，环境变量可覆盖路径；候选文件名必须固定。
- 8300 控制面、8301 VoiceDesign 与八个新增模型 API 共十个 HTTP 进程。八个新 API 共用
  model-audio/audio_cpp_runtime 的轻量 uv 环境，不安装 Torch。
- 所有 GPU API 必须共用 GPU_LOCK_FILE。Steam Audio renderer 是 CPU 路径，不获取 GPU 锁。
- 当前 LocalAI provider 固定启用。start.sh 管理 LocalAI 进程组，按模型请求加载任务，推理完成后请求 backend
  shutdown，并等待 CUDA_RELEASE_DELAY 后释放 GPU 锁。

## 安装与启动

先准备 LocalAI binary、audio-cpp backend 和 manifest 中列出的九个 GGUF 文件，然后同步锁定依赖：

~~~bash
uv sync --project main --locked
uv sync --project model-audio/design_qwen3-tts-1.7b-voicedesign --locked
uv sync --project model-audio/audio_cpp_runtime --locked
uv sync --project qa --locked
python3 scripts/localai/manager.py verify-install
~~~

verify-install 检查宿主 LocalAI 固定版本与 audio.cpp backend 文件指纹，记录安装状态，不下载文件。
启动整组服务：

~~~bash
bash start.sh
~~~

start.sh 会依次检查所有十个 HTTP 端口、注册九个候选、启动受管 LocalAI、执行候选 readiness 检查，
再启动控制面、VoiceDesign 和八个 audio.cpp API。任一 HTTP 服务退出时，脚本会清理其他服务进程组并停止
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

正常退出 start.sh 会终止全部十个 HTTP 服务进程组，再要求 manager 停止 LocalAI 受管进程组。独立检查或停止：

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

新增八个模型服务前，能力清单记录了 VoiceDesign 一次真实短句样例；这不能代替新增模型的真实推理与听感验证。
按顺序运行模型生成样例、指定 WAV 输入要求和输出位置见
[audio.cpp 冒烟生成说明](../tests/audio_cpp/README.md)。
