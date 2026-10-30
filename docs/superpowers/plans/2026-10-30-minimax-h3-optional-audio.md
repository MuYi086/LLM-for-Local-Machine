# MiniMax H3 Optional Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-request `include_audio` to MiniMax H3, defaulting to silent MP4 output.

**Architecture:** The API forwards the boolean into its one-shot worker. The ComfyUI graph includes the audio VAE decoder and CreateVideo audio input only when enabled. Output validation requires the requested audio-track state. The H3 benchmark opts in to native audio to preserve its prior comparison contract.

**Tech Stack:** Python 3.12, FastAPI/Pydantic, ComfyUI API graph, unittest.

**Spec:** `docs/superpowers/specs/2026-10-30-minimax-h3-optional-audio-design.md`

## Global Constraints

- HTTP layer remains free of Torch and ComfyUI imports.
- GPU work remains one request, one worker, one ComfyUI child, with the existing shared lock and VRAM gate.
- Silent output skips audio decoding and muxing; the H3 DiT still produces its joint audio/video latent.
- No tests may download weights, initialize CUDA, or call external model services.

## Review Focus

- Omitted `include_audio` must produce a silent-output worker contract; pin default `false` in API tests.
- `include_audio=true` must preserve audio VAE decoding and audio muxing; pin graph nodes and connection.
- Worker output must reject an MP4 whose audio-track state differs from the request; pin both mismatch directions.
- H3 benchmark must continue to request native audio explicitly; pin its request payload.
- `ref2va` must not reference an omitted audio VAE in silent mode; pin its graph inputs.

---

### Task 1: H3 request, worker graph, validation, benchmark, and docs

**Files:**
- Modify: `model-video/MiniMax-H3-16G-HiFi/main.py`
- Modify: `model-video/MiniMax-H3-16G-HiFi/worker.py`
- Modify: `model-video/MiniMax-H3-16G-HiFi/minimax_h3_runtime.py`
- Test: `tests/test_minimax_h3_service.py`
- Modify: `tests/i2v-benchmark/minimax_h3/benchmark.py`
- Modify: `tests/i2v-benchmark/minimax_h3/config.json`
- Modify: `tests/i2v-benchmark/extended5/minimax_h3.json`
- Modify: `tests/i2v-benchmark/tests/test_minimax_h3_benchmark.py`
- Modify: `model-video/MiniMax-H3-16G-HiFi/README.md`

**Interfaces:**
- Request: `include_audio: bool = False`.
- Worker graph: `build_api_prompt(...)` reads `payload["include_audio"]` and conditionally adds audio nodes/edges.
- Validation: `validate_video(path, ffprobe_bin, *, expect_audio: bool = False)` verifies exact audio-track presence.
- Benchmark request always carries `include_audio: true` because its expected audio stream count is one.

- [x] Add no-model API and worker graph tests for omitted/false, true, and Ref2VA cases.
- [ ] Run focused tests and confirm the new expectations fail before implementation.
- [x] Add the request field and forward it through worker validation.
- [x] Conditionally build audio nodes and mux edge; strictly validate audio-track presence.
- [x] Set both H3 benchmark configs to audio enabled and forward that setting in requests.
- [x] Document silent default, opt-in audio, and the joint-latent limitation.
- [x] Parse changed Python files and JSON configs; run `git diff --check`.
