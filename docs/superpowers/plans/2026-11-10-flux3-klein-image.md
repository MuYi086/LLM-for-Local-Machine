# FLUX.3 Image Klein Q8 Image API Implementation Plan

> **For agentic workers:** Implement inline, task by task. Each task uses test-first verification.

**Goal:** Add FLUX-3-Image-Klein-GGUF-Q8 text-to-image and reference-image editing through the shared stable-diffusion.cpp provider, exposed on port 8502.

**Architecture:** Keep model files outside the repository and declare them in the sd.cpp manifest. Add a model-specific synchronous FastAPI service that streams a bounded temporary reference upload, uses the shared GPU lock and worker lifecycle, and returns OpenAI Images-style Base64 PNG responses. Manage it from `start.sh` beside MOSS.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, `uv`, stable-diffusion.cpp CLI, GGUF and safetensors files outside the repository.

**Spec:** `task10.md`; shared runtime contract in `docs/superpowers/specs/2026-11-10-sdcpp-runtime-design.md`.

## Global Constraints

- Use external FLUX.3 Klein components from `$HF_MIRROR_DIR/stable-diffusion-cpp-gguf/FLUX-3-Image-Klein-GGUF/`; do not add weights, upstream source, or CUDA runtime files to the repository.
- All GPU inference uses the existing `GPU_LOCK_FILE`, sd.cpp worker process-group cleanup, and quarantine recovery.
- Keep health checks weight-free and reject malformed, oversized, or unsupported edit uploads before GPU execution.
- Preserve generic sd.cpp provider behavior; model-specific CLI choices come from strict manifest fields.
- Use pinned FLUX.3 Klein settings: 1024×1024, 16 steps, CFG 1.0, Euler, CPU offload, and `--ref-image` for editing.
- No automated test may download weights, initialize CUDA, or call a real model.

## Review Focus

- Reference uploads larger than the configured cap — test a 413 response and no generator call.
- Mislabeled or unsupported image bytes — test rejection before generation.
- FLUX reference flag differs from generic `--init-img` — test configured `--ref-image` while retaining video `--init-img`.
- Missing one of the four external components — test startup preflight failure before service launch.
- Temporary references survive too long or overwrite — test cleanup on success and failure with unique staged paths.

---

### Task 1: Extend the generic sd.cpp model contract

**Files:**
- Modify: `inference_gateway/src/inference_gateway/sdcpp/models.py`
- Modify: `sdcpp/manifests/models.json`
- Test: `tests/test_sdcpp_config.py`

**Interfaces:**
- Add `llm_vision` to accepted component names, emitted as the upstream `--llm_vision` argument.
- Add runtime `image_to_image_flag`, restricted to `init-img` or `ref-image`, defaulting to `init-img`.
- For `image-to-image`, map `GenerationRequest.init_image` through that field; preserve `--init-img` for `image-to-video`.
- Register FLUX components `diffusion_model`, `llm`, `llm_vision`, and `vae`, text-to-image and image-to-image tasks, and the values in the Goal/Global Constraints above.

- [x] Write failing tests for manifest values, `llm_vision`, FLUX `--ref-image`, and unchanged video `--init-img` behavior.
- [x] Run `uv run --project qa --locked python -m unittest discover -s tests -p 'test_sdcpp_config.py' -v` and confirm the new assertions fail for missing support.
- [x] Implement the strict manifest and argv mapping.
- [x] Re-run the focused test file and confirm it passes.

### Task 2: Add FLUX generation and edit API

**Files:**
- Create: `model-image/FLUX-3-Image-Klein-GGUF-Q8/__init__.py`
- Create: `model-image/FLUX-3-Image-Klein-GGUF-Q8/main.py`
- Create: `model-image/FLUX-3-Image-Klein-GGUF-Q8/service.py`
- Create: `tests/test_flux3_klein_image_api.py`

**Interfaces:**
- `POST /v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8` accepts JSON prompt/size/steps/CFG/seed.
- `POST /v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8/edit` accepts multipart prompt, image, and the same optional generation controls.
- Both endpoints return one Base64 PNG. The edit endpoint streams PNG/JPEG input to a unique temporary file, enforces a configurable 64 MiB cap, and removes that file after inference on success or failure.
- Health checks binary, four model components, and quarantine without loading weights.

- [x] Write failing tests for generation defaults, reference edit mapping, limits, content validation, upload cap, and temp-file cleanup.
- [x] Run the focused API tests and confirm they fail because the FLUX service is absent.
- [x] Implement the app factory, bounded upload staging, routes, and `run_service` using the shared provider.
- [x] Re-run the focused API tests and confirm they pass.

### Task 3: Manage FLUX service from `start.sh`

**Files:**
- Modify: `start.sh`
- Modify: `tests/test_start_services.py`

**Interfaces:**
- Add FLUX port 8502 to collision checks, preflight its four external model files, validate the manifest model, launch the service in a managed process group, and include its PID in cleanup.
- Export host/port, upload directory, and maximum upload size with safe defaults.

- [x] Extend startup tests for the 8502 listener, FLUX preflight files, manifest validation, service launch, and cleanup accounting.
- [x] Run the focused startup tests and confirm failure for missing FLUX wiring.
- [x] Add the service environment, preflight, validation, banner, launch, and cleanup integration.
- [x] Re-run the focused startup tests and `bash -n start.sh`.

### Task 4: Add generate/edit demo and update operational docs

**Files:**
- Create: `tests/FLUX-3-Image-Klein-GGUF-Q8/generate_example.py`
- Create: `tests/FLUX-3-Image-Klein-GGUF-Q8/README.md`
- Create: `tests/test_flux3_klein_image_demo.py`
- Modify: `AGENTS.md`, `README.md`, `model-image/README.md`, `docs/sdcpp-runtime.md`

**Interfaces:**
- Demo creates one image, uses it as the edit reference, and saves both responses under `storage/llm_tests/FLUX-3-Image-Klein-GGUF-Q8/` with exclusive no-overwrite publication.
- Document port 8502, both routes, component filenames, defaults, limits, and that real FLUX inference remains unverified until the demo is run on the host with model files.

- [x] Write failing demo tests for both routes, PNG decoding, and no-overwrite output behavior.
- [x] Run the focused demo tests and confirm expected failure before the demo exists.
- [x] Implement the demo and update service counts/configuration/docs consistently.
- [x] Re-run the focused demo tests.

### Task 5: Verify the change

**Files:**
- No additional implementation files.

- [x] Run `bash -n start.sh`.
- [x] Run `bash scripts/quality_gate.sh`.
- [x] Run `uv run --project qa --locked python -m unittest discover -s tests -v` (also run by the quality gate).
- [x] Run `git diff --check`, inspect changed files, and confirm no model weights or generated images entered the repository.
