# Four audio.cpp Separation and Alignment Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register BS-RoFormer, Qwen3-ForcedAligner, Mel-Band RoFormer, and HTDemucs as four LocalAI/audio.cpp HTTP services on ports 8344–8347, with per-model live API demos.

**Architecture:** Keep one shared `audio_cpp_runtime` project and LocalAI provider. Add source separation by fetching each named stem listed in the LocalAI `X-Audio-Stems` response header after one transform request; add forced alignment by passing the supplied transcript as LocalAI transcription `prompt` and preserving verbose JSON timing fields. Add fixed candidate metadata, start supervision, capability snapshots, and a live example runner for each model.

**Tech Stack:** Python 3.12.13, FastAPI, stdlib HTTP/multipart/WAV/ZIP, LocalAI v4.11.0 with pinned audio.cpp backend, Bash, unittest.

**Spec:** `docs/superpowers/specs/2026-11-17-audio-cpp-separation-alignment-design.md`

## Global Constraints

- 输入源文件只读，每个阶段写入独立输出目录；前一阶段验证后才能执行下一阶段。
- 禁止多个进程同时写入同一输出文件；默认不并行执行多个 GPU 音频模型。
- 音频上传按块暂存，默认单文件上限 64 MiB；不得整文件 `read()`、直接覆盖 sidecar 或长期保存临时输入。
- 所有 GPU provider 必须使用同一个 `GPU_LOCK_FILE`；队列默认等待 900 秒，超时返回 503。
- `.quarantine` 标记存在时，任何 GPU provider 都不得推理；LocalAI 要确认 backend 已关闭。
- audio.cpp 服务默认值、候选和 API 合同集中在候选 manifest 与 `audio_cpp_runtime`；修改服务合同后同步 `localai/manifests/capabilities.json` 并运行能力清单校验。
- 测试不得下载权重、依赖 CUDA、调用外部模型服务或执行真实模型。
- Python 注释和 Docstring 默认简体中文；公共函数、公共类和重要业务入口提供 Google Style / PEP 257 Docstring。

## Review Focus

- `X-Audio-Stems` 缺失、格式错误、重复名或非法 URL：拒绝不完整/越界响应，且不留下请求文件；测试解析及拒绝分支。
- 某 stem GET 失败或不是 PCM WAV：API 返回错误并清理此前已暂存的所有文件；测试中途失败清理。
- 对齐模型没有 `words` 时间戳：返回上游合同错误，不把普通 ASR 文本误报为对齐结果；测试无 words 的响应。
- 分轨上传不是 WAV、采样率不是 44.1 kHz 或声道不受支持：在推理前返回 415/422；测试格式、采样率和声道边界。
- 自定义端口或缺少默认权重：启动脚本应把四个 API 端口纳入冲突检查、把四个权重纳入预检；扩展启动合同测试覆盖。

---

### Task 1: Register four fixed LocalAI candidates and supervised services

**Files:**
- Modify: `localai/manifests/audio-cpp-models.json`
- Modify: `start.sh`
- Test: `tests/test_start_services.py`
- Test: `tests/test_audio_cpp_candidate_registration.py`

**Interfaces:**
- Consumes: four GGUF folders and filenames under `$HF_MIRROR_DIR/audio.cpp-gguf`.
- Produces: keys `bs_roformer`, `qwen3_forced_aligner`, `mel_band_roformer`, `htdemucs`; ports 8344–8347; family/task pairs `bs_roformer/sep`, `qwen3_forced_aligner/align`, `mel_band_roformer/sep`, `htdemucs/sep`; route `/v1/audio-cpp/<directory-name>`.

- [ ] **Step 1: Write failing candidate and start-contract tests.** Assert each candidate's model ID, source directory, exact required filename, env var, family/task, runtime, service directory, route and port. Assert `start.sh` checks all four model ports, launches all four entries once, and defaults their model-file env vars to the HF mirror paths.
- [ ] **Step 2: Run focused tests and verify they fail because candidates and services are absent.**
- [ ] **Step 3: Add four manifest entries and extend the parallel start arrays and environment defaults.** Add each service to `PORT_CHECK_ARGS`; preserve single-process-group cleanup and the shared uv project.
- [ ] **Step 4: Run focused tests and verify all four services are declared and supervised.**

### Task 2: Extend LocalAI client for forced alignment and named stems

**Files:**
- Modify: `inference_gateway/src/inference_gateway/localai_client.py`
- Test: `tests/test_audio_cpp_localai_client.py`

**Interfaces:**
- Consumes: existing `GatewayConfig`, bounded multipart sender, pinned LocalAI contracts.
- Produces: transcription with selectable verbose JSON response format; transform result containing response WAV and parsed stem paths; a bounded GET for LocalAI-generated artifact paths.

- [ ] **Step 1: Write failing tests for verbose transcription fields, `params[stem]`, parsing `X-Audio-Stems`, and artifact GET.** Assert all requests remain on the configured LocalAI origin and keep bounded response sizes.
- [ ] **Step 2: Run the client tests and verify the new methods/options are absent.**
- [ ] **Step 3: Implement typed result handling and validate every stem URL is a single `/generated-audio/<filename>` path before fetching it.** Reject malformed JSON, empty stem sets, duplicate names, path traversal, query strings and fragments.
- [ ] **Step 4: Run client tests; verify valid responses succeed and invalid/missing headers fail closed.**

### Task 3: Add source-separation and forced-alignment service contracts

**Files:**
- Modify: `model-audio/audio_cpp_runtime/src/audio_cpp_runtime/specs.py`
- Modify: `model-audio/audio_cpp_runtime/src/audio_cpp_runtime/service.py`
- Test: `tests/test_audio_cpp_services.py`

**Interfaces:**
- Consumes: `source_separation` and `forced_alignment` candidate kinds; new LocalAI client operations.
- Produces: separation route taking WAV `audio` and returning an `application/zip` of named WAV files; alignment route taking WAV `audio`, required `text`, and required `language`, returning verbose JSON with nonempty `words`.

- [ ] **Step 1: Write failing service tests for exact registry scope, 44.1 kHz mono/stereo validation, zip stem members, uploaded file cleanup, required alignment fields, LocalAI prompt forwarding, and missing timestamp errors.**
- [ ] **Step 2: Run focused service tests and verify unsupported kinds/contracts fail.**
- [ ] **Step 3: Register fourteen runtime services and implement the two kinds using `_run_locked`, `_stage_upload`, WAV validation, ZIP writing, and `finally` cleanup.** Keep every generated file in a unique request directory under the existing runtime cache.
- [ ] **Step 4: Run focused service tests and verify lock/error/cleanup behavior.**

### Task 4: Add four live demos and synchronize active-scope documentation

**Files:**
- Modify: `tests/audio_cpp/generate_examples.py`
- Modify: `tests/audio_cpp/README.md`
- Modify: `model-audio/README.md`
- Create: `tests/BS-RoFormer-ep368-GGUF/README.md`, `generate_example.py`
- Create: `tests/Qwen3-ForcedAligner-0.6B-GGUF/README.md`, `generate_example.py`
- Create: `tests/Mel-Band-RoFormer-GGUF/README.md`, `generate_example.py`
- Create: `tests/HTDemucs-GGUF/README.md`, `generate_example.py`
- Modify: `tests/test_audio_cpp_model_test_directories.py`
- Modify: `tests/test_localai_active_scope.py`
- Modify: `tests/test_start_services.py`
- Modify: `tests/test_audio_cpp_services.py`
- Modify: `localai/manifests/capabilities.json`
- Modify: `localai/manifests/versions.json`

**Interfaces:**
- Consumes: live APIs from Tasks 1–3; input WAV from CLI/environment; output root `storage/audio_cpp_tests/`.
- Produces: one API-only demo per model; ZIP validation and atomic save for separators; JSON validation for aligner; updated service/candidate counts and routes.

- [ ] **Step 1: Write failing tests for the four demo directories, runner request fields, zip/json output validation, and capability snapshot parity.**
- [ ] **Step 2: Run focused tests and verify the four demos and capability entries are missing.**
- [ ] **Step 3: Implement per-model runner selection and model-specific CLI arguments; update READMEs with routes, ports, 44.1 kHz requirements, transcript/language input and output formats.**
- [ ] **Step 4: Update capabilities and version manifests to 15 candidate models, 16 API services, 3 uv projects, 35 routes, and 15 family candidates.** Preserve the distinction between E1 family declaration and runtime acceptance.
- [ ] **Step 5: Run all focused tests and the repository's listed no-model quality gates.**
