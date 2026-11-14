# Ornith MTP 9B Q8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the MTP Q8 Ornith API on external port 8504 while retaining the existing API on 8503.

**Architecture:** Reuse `llama_cpp_runtime` for a second API adapter and llama-server lifecycle. Move the existing loopback backend to 8505, run the MTP backend on 8506 with its vision projector, and serialize both through the existing shared GPU lock.

**Tech Stack:** Python 3.12, FastAPI, httpx, uvicorn, shell, llama.cpp, unittest.

**Spec:** `docs/superpowers/specs/2026-11-13-ornith-mtp-9b-q8-design.md`

## Global Constraints

- Model and projector files are read-only and remain outside the repository.
- The new API listens on `8504`; only its llama-server backend binds loopback `127.0.0.1:8506`.
- Preserve the existing API on `8503`; its llama-server backend binds loopback `127.0.0.1:8505`.
- Both APIs use the same `GPU_LOCK_FILE`, sleep-idle confirmation, quarantine lifecycle, and shutdown helper.
- Default context is `98304`, with MTP draft length `4`.
- The standard quality gate must not load weights, require CUDA, call a real model, or contact external model services.
- Update the managed HTTP process count from 15 to 17; llama-server backends remain loopback-only.

## Review Focus

- Missing or empty MTP GGUF or projector must fail before managed LocalAI starts; cover in Task 3.
- A port conflict among public API `8504` and loopback backends `8505`/`8506` must be caught before model startup; cover in Task 3.
- New API requests must use the exact route and fixed model alias while forwarding image content unchanged; cover in Tasks 1 and 2.
- One backend failing during startup or shutdown must leave process cleanup and quarantine behavior intact for both models; cover in Task 3 with the fake llama-server.
- Demo output collisions and malformed/empty model responses must fail without overwriting prior samples; cover in Task 2.

---

### Task 1: Add the MTP API adapter

**Files:**
- Create: `model-text/Ornith-1.5-9B-MTP-GGUF/app.py`
- Create: `model-text/Ornith-1.5-9B-MTP-GGUF/main.py`
- Test: `tests/ornith_1_5_9b_mtp_gguf_q8/test_service.py`

**Interfaces:**
- Consumes: `llama_cpp_runtime.app.create_app` and its shared lock/lifecycle contract.
- Produces: fixed route `/v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8`, model alias `Ornith-1.5-9B-MTP-GGUF-Q8`, ASGI `app`, and `main()` defaulting to port `8504`.

- [x] **Step 1: Write the failing adapter and entrypoint tests**

Assert the exact route, forced model alias, successful mocked chat forwarding, and `main()` default host/port `0.0.0.0:8504`.

- [x] **Step 2: Run the focused tests and confirm they fail because the service files do not exist**

Run: `uv run --project qa --locked python -m unittest tests.ornith_1_5_9b_mtp_gguf_q8.test_service -v`
Expected: import failure for the missing new adapter.

- [x] **Step 3: Implement the thin adapter and entrypoint**

Pass the MTP model alias, route, and `ORNITH_MTP_9B` environment prefix to the shared app factory; keep all request lifecycle behavior in `llama_cpp_runtime`.

- [x] **Step 4: Run the focused adapter tests**

Run: `uv run --project qa --locked python -m unittest tests.ornith_1_5_9b_mtp_gguf_q8.test_service -v`
Expected: all adapter and entrypoint tests pass without a model backend.

### Task 2: Add the text and optional image demo

**Files:**
- Create: `tests/ornith_1_5_9b_mtp_gguf_q8/generate_example.py`
- Create: `tests/ornith_1_5_9b_mtp_gguf_q8/test_generate_example.py`
- Create: `tests/ornith_1_5_9b_mtp_gguf_q8/README.md`

**Interfaces:**
- Consumes: Task 1 API route and model response contract.
- Produces: an explicit real-call demo that defaults to `http://127.0.0.1:8504`, accepts an optional local image path, and writes a uniquely named text result under `storage/llm_tests/ornith_1_5_9b_mtp_gguf_q8/`.

- [x] **Step 1: Write failing demo tests**

Assert the default route, chat-completion request shape, optional PNG/JPEG data URL content, nonempty assistant extraction, and exclusive output creation.

- [x] **Step 2: Run the focused tests and confirm they fail because the demo module does not exist**

Run: `uv run --project qa --locked python -m unittest tests.ornith_1_5_9b_mtp_gguf_q8.test_generate_example -v`
Expected: import failure for the missing demo module.

- [x] **Step 3: Implement the demo and usage documentation**

Read an optional local image only when supplied, encode it as an OpenAI-compatible data URL, and save only generated text with exclusive file creation.

- [x] **Step 4: Run focused demo contract tests**

Run: `uv run --project qa --locked python -m unittest tests.ornith_1_5_9b_mtp_gguf_q8.test_generate_example -v`
Expected: all tests pass without calling a live API.

### Task 3: Integrate both Ornith backends into `start.sh`

**Files:**
- Modify: `start.sh`
- Modify: `tests/test_start_services.py`
- Modify: `model-text/Ornith-1.5-9B-GGUF/app.py`
- Modify: `tests/ornith_1_5_9b_gguf/test_service.py`
- Modify: `tests/ornith_1_5_9b_gguf/benchmark_context_budget.py`
- Modify: `tests/ornith_1_5_9b_gguf/benchmark_live.py`
- Modify: `tests/ornith_1_5_9b_gguf/benchmark_sleep_cache.py`
- Modify: `tests/ornith_1_5_9b_gguf/tune_backend.py`
- Modify: `tests/model_web_comparison/run_comparison.py`
- Modify: `tests/ornith_1_5_9b_gguf/README.md`
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: Tasks 1 and 2 service paths and environment prefix.
- Produces: validated two-backend startup, monitoring, shutdown and port checks; MTP model/projector defaults; updated service inventory and run instructions.

- [x] **Step 1: Extend startup integration tests first**

Add fake MTP model/projector files and assert 8503/8504 public APIs, loopback backends 8505/8506, exact route logging, MTP `--model`, `--mmproj`, alias and MTP flags, both startup/shutdown calls, and early failure for a missing MTP asset.

- [x] **Step 2: Run the startup tests and confirm the new expectations fail**

Run: `uv run --project qa --locked python -m unittest tests.test_start_services -v`
Expected: failures for missing MTP variables, ports, backend launch, and API entrypoint.

- [x] **Step 3: Implement dual-backend configuration and lifecycle**

Move the existing backend default to 8505; set MTP API/backend defaults to 8504/8506; validate both assets and all three ports; start the backends sequentially through the shared runtime; include both APIs, watchers, PID files, and backends in monitoring and cleanup. Preserve the existing Easel upstream at 8503.

- [x] **Step 4: Update deployment and service documentation**

Document both APIs, both loopback backends, new model/projector environment variables, the 17-process count, and the text/image demo commands.

- [x] **Step 5: Run startup integration tests**

Run: `uv run --project qa --locked python -m unittest tests.test_start_services -v`
Expected: all fake-startup and preflight tests pass without loading weights or starting real model services.

### Task 4: Run repository verification

**Files:**
- Verify: all changed files and the task requirements.

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: verified shell syntax, formatting, quality gate, Ornith no-model tests, and a clean diff check.

- [x] **Step 1: Run `bash -n start.sh` and focused new Ornith tests**

Expected: shell parses and both new test modules pass.

- [x] **Step 2: Run the required repository checks**

Run: `bash scripts/quality_gate.sh && uv run --project qa --locked python -m unittest discover -s tests -v`
Expected: both commands pass; no weights, CUDA, or live inference are used.

- [x] **Step 3: Inspect final diff and verify model files remain external**

Run: `git diff --check` and review `git status --short` plus the changed-file diff.
Expected: no whitespace errors, no model assets staged or copied, and the existing `task10.md` edit is preserved.
