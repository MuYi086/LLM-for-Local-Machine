# MOSS-SoundEffect v2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local text-to-sound-effects service at port 8362 and route `/v1/moss/soundEffect-v2.0`.

**Architecture:** A lightweight FastAPI process validates requests and local model files, then serializes GPU work through `llm_local_runtime` and starts one isolated worker per request. The worker loads the pinned official MOSS-SoundEffect pipeline and saves a 48 kHz WAV; the API atomically persists and returns it.

**Tech Stack:** Python 3.12.13, FastAPI, Pydantic v2, uv, PyTorch CUDA 12.8, official `moss-soundeffect-v2`, Hugging Face local snapshot.

**Spec:** `docs/superpowers/specs/2026-10-29-moss-soundeffect-v2-design.md`

## Global Constraints

- Model execution must use one request, one worker, and process exit to release CUDA context.
- Local inference must not download weights; model validation and request validation happen before GPU locking.
- GPU lock waiting uses a finite positive timeout and worker process groups and temporary files are cleaned on every exit.
- Source audio, model weights, external model source, caches, generated audio, and virtual environments are not committed.
- New Python projects use Python `3.12.13`, locked dependencies, Ruff rules `E`, `F`, `I`, `UP`, `B`, line length 100, and formatter settings matching other model services.
- Service changes require targeted model-free tests and synchronized README, `start.sh`, quality gate, and service quick-reference updates.

## Review Focus

- Missing or truncated model files must report unavailable before queueing; test exact required-file readiness without reading weight contents.
- Empty, whitespace-only, out-of-range, non-finite, or extra JSON fields must fail before starting a worker; test request validation.
- GPU queue timeout, worker nonzero exit, timeout, and malformed WAV must return the expected HTTP error and clean temporary files; test mocked worker lifecycle.
- Concurrent requests must share the configured GPU lock and hold the CUDA release delay through worker completion; test lock ordering and release.
- Port, route, process count, project lock, README inventory, and AGENTS inventory must all identify the new 8362 service; test startup and project registration.

---

### Task 1: Service contract and local runtime

**Files:**
- Create: `tests/test_moss_soundeffect_service.py`
- Create: `model-audio/soundEffect-v2.0/main.py`
- Create: `model-audio/soundEffect-v2.0/worker.py`
- Create: `model-audio/soundEffect-v2.0/soundeffect_runtime.py`
- Create: `model-audio/soundEffect-v2.0/.python-version`
- Create: `model-audio/soundEffect-v2.0/pyproject.toml`
- Create: `model-audio/soundEffect-v2.0/uv.lock`
- Create: `model-audio/soundEffect-v2.0/README.md`

**Interfaces:**
- Consumes: `llm_local_runtime.gpu_runtime_lock(lock_path, label, timeout_seconds=...)` and the official `MossSoundEffectPipeline.from_pretrained(path, torch_dtype=..., device=...)`.
- Produces: `MossSoundEffectRequest`, `POST /v1/moss/soundEffect-v2.0`, `GET /v1/health`, model readiness and a worker manager returning validated WAV bytes.

- [x] **Step 1: Write model-free tests first** for request bounds, complete local model files, health without loading Torch, successful mocked worker WAV response and persistence, and worker timeout/failure cleanup.
- [x] **Step 2: Run `uv run --project qa --locked python -m unittest tests.test_moss_soundeffect_service -v` and confirm expected failures because the service files do not exist.**
- [x] **Step 3: Implement the lightweight API, local file checks, finite shared GPU locking, process-group cleanup, one-shot worker, and atomic WAV persistence.** The API must not import Torch or the model package.
- [x] **Step 4: Add a Python 3.12.13 uv project pinned to the fixed external MOSS-TTS source revision `934d6826b084c46a0d033402174d5f8ac4ed2519`, CUDA 12.8 Torch wheels, shared runtime, and Ruff; generate the lock with `uv lock --project model-audio/soundEffect-v2.0`.**
- [x] **Step 5: Run the focused test and Ruff on the new service; confirm both pass without model weights or CUDA inference.**

### Task 2: Deployment registration and documentation

**Files:**
- Modify: `start.sh`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `model-audio/README.md`
- Modify: `tests/test_startup_ports.py`
- Modify: `tests/test_python_quality_configuration.py`
- Modify: `scripts/quality_gate.sh`

**Interfaces:**
- Consumes: Task 1 service path, environment prefix `MOSS_SOUNDEFFECT_V2_*`, port `8362`, route `/v1/moss/soundEffect-v2.0`.
- Produces: startup port preflight and child-process supervision; documented setup, configuration and request contract; project lock/lint/test registration.

- [x] **Step 1: Extend startup and project registration tests** to assert the 8362 port, route, uv project, supervised PID, 18 HTTP processes, and model project quality baseline.
- [x] **Step 2: Run the focused startup/configuration tests and confirm expected failures before changing deployment files.**
- [x] **Step 3: Register the new service in `start.sh`, `scripts/quality_gate.sh`, README inventories and AGENTS service table/counts.**
- [x] **Step 4: Document model location, locked environment setup, route fields, WAV output, limits, and model-free verification in Chinese.**
- [x] **Step 5: Run focused tests, `bash -n start.sh`, project lock check and `bash scripts/quality_gate.sh`; confirm expected pass output.**
