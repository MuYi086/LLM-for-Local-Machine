# MOSS-Image-SFT-Q8_0 Integration Plan

> **For agentic workers:** Implement this plan inline, task by task. Each task uses test-first verification.

**Goal:** Expose the locally available MOSS-Image-SFT-Q8_0 model through the shared stable-diffusion.cpp runtime at port 8501.

**Architecture:** Declare the four read-only model components in the sd.cpp manifest and reuse its process-group lifecycle, shared GPU lock, and quarantine recovery. Add a synchronous FastAPI text-to-image endpoint, managed by `start.sh`, and a model-specific HTTP demo.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, `uv`, stable-diffusion.cpp CLI, GGUF and safetensors files outside the repository.

**Spec:** `task1.md`; runtime contracts in `docs/superpowers/specs/2026-11-10-sdcpp-runtime-design.md`.

## Global Constraints

- Model files and the sd.cpp binary stay outside the repository and are loaded read-only.
- All GPU providers use the exact same `GPU_LOCK_FILE`; quarantine blocks inference until the existing recovery script confirms worker exit.
- Health checks do not load weights or compute binary digests; startup validation does.
- The MOSS SFT route is text-to-image only; width and height must be multiples of 16.
- Use the upstream MOSS defaults: 1024×1024, 20 steps, CFG 3, Euler/discrete, flow shift 5, and CPU offload.
- Keep test suites independent of CUDA, model downloads, and external services. Run real inference only in the explicit demo step.
- New Python public entry points and comments/docstrings follow the repository's Simplified Chinese convention.

## Review Focus

- Missing external Qwen3.5 tokenizer or wrong MOSS component mapping — cover with manifest/argv tests and startup validation.
- Incorrect dimensions, task, or prompt mapping — cover with API 422 and captured request tests.
- Quarantine and queue-timeout behavior — cover with API 503 tests and no model invocation.
- Startup failure ordering and cleanup — cover with fake-command start tests for port, binary digest, and weight paths.
- Demo overwrites a prior generated image or accepts a non-PNG response — cover with isolated example-script tests.

---

### Task 1: Add MOSS component support to the generic sd.cpp contract

**Files:**
- Modify: `inference_gateway/src/inference_gateway/sdcpp/models.py`
- Test: `tests/test_sdcpp_config.py`
- Modify: `sdcpp/manifests/models.json`

**Interfaces:**
- `ModelSpec.arguments(request, output)` maps `tokenizer` to `--tokenizer` and `runtime.offload_to_cpu=true` to `--offload-to-cpu`.
- The manifest contains only model ID `MOSS-Image-SFT-Q8_0`, task `text-to-image`, and the diffusion model, LLM, VAE, and tokenizer components.

- [x] Write failing tests for tokenizer/offload argv mapping, MOSS defaults and component paths, and 16-pixel dimension alignment.
- [x] Run the focused tests and confirm they fail because the contract does not yet support the new component/runtime key.
- [x] Implement the minimum schema and argv changes, then run the focused tests to green.

### Task 2: Build and test the model-specific HTTP API

**Files:**
- Create: `model-image/MOSS-Image-SFT-Q8_0/__init__.py`
- Create: `model-image/MOSS-Image-SFT-Q8_0/main.py`
- Create: `model-image/MOSS-Image-SFT-Q8_0/service.py`
- Create: `tests/test_moss_image_sft_api.py`

**Interfaces:**
- `POST /v1/openmoss/MOSS-Image-SFT-Q8_0` accepts prompt, optional model ID, size, steps, CFG, seed, and `b64_json` response format.
- `GET /v1/health` checks the sd.cpp binary, all four model files, and quarantine without loading the model.
- The synchronous handler delegates to the shared provider and returns one PNG in an OpenAI Images-style response.

- [x] Write failing tests for successful argument/response mapping, limits and unsupported options, health, queue timeout, and quarantine.
- [x] Run the focused API tests and confirm the expected missing-module failures.
- [x] Implement the app factory, schemas, routes, and `run_service`; run focused tests to green.

### Task 3: Manage service startup and document runtime requirements

**Files:**
- Modify: `start.sh`
- Modify: `tests/test_start_services.py`
- Modify: `AGENTS.md`, `README.md`, `model-image/README.md`, `docs/sdcpp-runtime.md`
- Modify: `scripts/quality_gate.sh`

- [x] Extend fake-start tests for 8501, model preflight, pinned binary digest, service launch, and process cleanup.
- [x] Run the focused start tests and confirm they fail on the missing MOSS configuration and service.
- [x] Add MOSS environment defaults, manifest validation, startup line, managed PID cleanup, and the corresponding service table and usage docs.
- [x] Run focused startup tests, `bash -n start.sh`, Ruff, and manifest validation.

### Task 4: Add and run the model HTTP demo

**Files:**
- Create: `tests/MOSS-Image-SFT-Q8_0/generate_example.py`
- Create: `tests/MOSS-Image-SFT-Q8_0/README.md`
- Create runtime output under: `storage/llm_tests/MOSS-Image-SFT-Q8_0/`
- Modify: `sdcpp/manifests/versions.json`

- [x] Add a no-overwrite HTTP demo and tests for PNG decoding and output publication.
- [x] Run its focused unit tests.
- [x] Start the MOSS API with the configured local binary and shared GPU lock, request one real image, and inspect the PNG and service cleanup/quarantine state.
- [x] Record the inference-verified model state only after real generation succeeds.

### Task 5: Final verification

- [x] Run `bash -n start.sh`.
- [x] Run `bash scripts/quality_gate.sh` and record the full result.
- [x] Run `git diff --check` and inspect the final diff for stale SD 1.5 references, model paths, route names, and generated files staged for commit.
- [x] Perform a separate self-review of the final changes; do not commit or remove user-owned worktree changes.
