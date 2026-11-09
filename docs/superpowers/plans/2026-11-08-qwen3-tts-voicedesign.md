# Qwen3-TTS VoiceDesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用现有固定版本 LocalAI/audio.cpp 加载本机 VoiceDesign BF16 GGUF，在 8301 提供指定 TTS 路由，并提供无模型回归和真实生成示例。

**Architecture:** 新建轻量独立 uv/FastAPI 服务，复用 `inference_gateway` 的 LocalAI 客户端、GPU 排队、停机确认和隔离逻辑。LocalAI 候选引用单个 GGUF 文件；服务验证输入与 WAV，将生成音频原子保存至 `storage/timbre/` 并返回 WAV。

**Tech Stack:** Python 3.12.13、FastAPI、uv、LocalAI v4.11.0 audio-cpp、Qwen3-TTS GGUF、QA unittest/Ruff。

**Spec:** `task20.md` 和仓库根目录 `AGENTS.md`。

## Global Constraints

- 输入模型文件只读，模型权重和生成音频不提交。
- HTTP 服务只处理输入校验、存储和生命周期；推理由 LocalAI/audio.cpp 处理。
- GPU 推理复用共享 `GPU_LOCK_FILE`，队列超时必须为正数；停机无法确认时保留 quarantine 标记。
- 新依赖在独立 uv 项目锁定；回归测试不下载权重、不依赖 CUDA、不访问真实模型服务。
- 服务默认监听 8301，提供 `GET /v1/health` 和 `POST /v1/qwen/Qwen3-TTS-1.7B-VoiceDesign`。

## Review Focus

- 空白或超限文本/音色描述：在取得 GPU 锁前拒绝，覆盖服务请求测试。
- GGUF 缺失、空文件或引用目标错误：注册/预检明确失败，覆盖 manager 测试。
- LocalAI 返回无效 WAV 或请求失败：不留下半成品输出，覆盖服务测试。
- backend 停止未确认：沿用 gateway 隔离语义，不清除 quarantine 标记，复用 gateway 的既有测试约束。
- 真实模型推理：单独运行生成示例脚本，输出仅写入本机 storage，不进入 QA 门禁或 Git。

---

### Task 1: VoiceDesign 服务合同与本地推理调用

**Files:**
- Create: `model-audio/design_qwen3-tts-1.7b-voicedesign/main.py`
- Create: `model-audio/design_qwen3-tts-1.7b-voicedesign/pyproject.toml`
- Create: `model-audio/design_qwen3-tts-1.7b-voicedesign/.python-version`
- Create: `tests/qwen3_tts_voicedesign/__init__.py`
- Create: `tests/qwen3_tts_voicedesign/test_service.py`

- [x] 先写请求校验、正确路由、WAV 响应与原子存储行为测试，并确认失败原因是服务模块/行为缺失。
- [x] 实现同步 FastAPI 路由，通过 `execute_localai_operation` 调用 `LocalAIClient.design_voice`；不在 API 进程加载模型。
- [x] 写入单段或按请求分段合并后的 WAV，并登记到 `AudioReferenceStore`；失败时清理临时文件。
- [x] 运行服务合同测试，确认正确通过。
- [x] 生成本服务的锁文件并检查锁一致性。

### Task 2: GGUF 候选注册及预检

**Files:**
- Modify: `scripts/localai/manager.py`
- Modify: `localai/manifests/qwen3-tts-voicedesign-candidate.yaml`
- Modify: `localai/manifests/versions.json`
- Create: `tests/qwen3_tts_voicedesign/test_localai_candidate.py`

- [x] 先写测试，要求只接受任务指定文件名、存在且非空的 GGUF；验证候选引用为受控相对文件引用。
- [x] 实现可逆的单文件模型引用注册与核验，保持 `family:qwen3_tts`、`task:vdes` 和 CUDA 候选选项。
- [x] 更新本机模型文件环境变量与能力记录，不再要求 VoiceDesign safetensors。
- [x] 运行候选测试并确认版本清单 JSON 有效；源码能力清单校验放在 Task 3 同步路由后运行。

### Task 3: start.sh、配置、文档和质量门禁

**Files:**
- Modify: `start.sh`
- Modify: `scripts/localai/preflight.py`
- Modify: `scripts/quality_gate.sh`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `localai/manifests/capabilities.json`

- [x] 为 8301/指定路由和 GGUF 路径补配置测试，确认 `start.sh` 仍通过 `bash -n`。
- [x] 默认启动路径启用这个仅 audio.cpp 的 provider，确保 LocalAI 注册、managed 生命周期和预检顺序正确。
- [x] 在速查表、配置表、API 文档、LocalAI 报告和质量门禁中同步新项目与路由。
- [x] 运行配置回归、Ruff、锁校验和 VoiceDesign 子集能力清单校验；完整校验遇到其他服务源码缺失，见最终记录。

### Task 4: 真实模型生成示例

**Files:**
- Create: `tests/qwen3_tts_voicedesign/generate_example.py`
- Create: `tests/qwen3_tts_voicedesign/README.md`

- [x] 示例脚本请求本机 8301 的 VoiceDesign 路由，验证 RIFF/WAVE 和有效采样帧，并核对服务已保存到 `storage/timbre/` 的同一文件。
- [x] 使用本机 LocalAI/audio.cpp 和 GGUF 实际生成一个短中文示例；记录生成路径和验证结果，不提交音频文件。

### Final verification

- [x] `bash -n start.sh`
- [x] `uv run --project qa --locked python -m unittest discover -s tests -v`（11 项通过；使用 QA 提供的 `httpx2` TestClient 兼容依赖）
- [x] QA Ruff check/format 和 VoiceDesign uv lock 检查通过。
- [x] VoiceDesign 子集能力快照校验通过（2 routes / 1 schema / 8 fields）；完整能力校验被既有缺失的 `model-audio/clone_confucius-4/main.py` 阻断。
- [x] 完整质量门禁已运行；Steam Audio 核心检查通过，后续因既有缺失的 `model-audio/analyse_moss-music-8b-Instruct` 项目停止。VoiceDesign 的独立 QA、Ruff、锁和真实推理检查已通过。

**Final runtime sample:** `storage/timbre/qwen3-tts-voicedesign-56e80fd15f7e42f6abd865819d4ed83c.wav` — 24 kHz mono PCM WAV, 3.36 seconds, 161324 bytes; LocalAI backend shutdown and managed host stop confirmed. GPU peak sample: 5522 MiB. Subjective audio quality was not evaluated.
