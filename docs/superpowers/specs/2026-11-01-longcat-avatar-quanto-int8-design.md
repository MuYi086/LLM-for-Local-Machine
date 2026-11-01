# LongCat Avatar 1.5 Quanto-Int8 Service Design

## Intent

Expose the local LongCat Avatar 1.5 Quanto model on port 8390 and run it on the host's single RTX 4070 Ti SUPER
with 16 GiB VRAM. The API accepts a prompt, uploaded audio and a required reference image, then returns an MP4
with the supplied audio track.

## Design

- The FastAPI process handles validation, safe uploads, readiness checks, the shared GPU lock, and worker lifetime.
  It does not import Torch or model code.
- Each generation launches a one-shot worker with the WanGP Python interpreter. The worker uses WanGP's supported
  `shared.api.init()` and `session.submit_task()` API for `longcat_avatar_v1_5`; WanGP owns model loading, Quanto
  weight handling, LoRA loading, audio conditioning, and sliding-window generation.
- WanGP's bundled MMGP 4.5.0 manages low-memory placement with Profile 4. SDPA attention avoids requiring an
  additional attention package. The request settings use the model's 8-step distillation defaults, 25 fps, 93-frame
  windows and 13-frame overlap.
- `configure_wangp.py` creates only symlinks in WanGP `ckpts/` and `loras/`, then updates the existing
  `wgp_config.json` atomically. Existing same-name files pointing elsewhere cause an error instead of being
  overwritten. Source model files remain read-only.
- The worker is run offline. Missing model files are reported before taking the GPU lock; model weights are never
  downloaded during a request.
- A generated MP4 must contain both video and audio streams before it is atomically saved. Success, error, timeout,
  and process-group cleanup remove the temporary request workspace.

## Runtime boundary

- The service keeps its own lightweight Python 3.12 uv project. WanGP has a separate Python 3.11 environment with
  the Torch/CUDA and model dependencies. The worker starts with the WanGP interpreter, and the HTTP process does not
  share its CUDA context.
- The default visible GPU is `0`; the default preflight requires 10,000 MiB of free VRAM. MMGP Profile 4 can move
  model blocks through host memory, trading inference speed for lower VRAM use.
- Local Quanto DiT/T5 and Whisper files live under
  `$HF_MIRROR_DIR/meituan/LongCat-Video-Avatar-1.5`. The VAE weight lives under
  `$HF_MIRROR_DIR/meituan/LongCat-Video`. WanGP source and MMGP live under `$HOME/LLM-depency/WanGP`.
- Audio is limited to about 16.52 seconds (five windows); segment count must cover the full input.
- WanGP fixes LongCat Avatar to `KI` reference-image mode, so `image_path` is required and missing images are
  rejected before GPU admission.
- `GET /v1/health` checks files, source, Python module specs, WanGP config, symlink targets, FFmpeg, and available
  GPU memory. It does not load model weights and cannot establish that generation itself succeeds.

## Acceptance

- Readiness reports all local weights, WanGP/MMGP, expected symlink targets, Profile 4, SDPA, and single-card
  capacity as available.
- The official Python API receives absolute audio/image paths, an 8-step distilled task, and duration in seconds;
  the service returns only an MP4 validated to contain both audio and video.
- Upload safety, bounded GPU locking, timeout process-group termination, atomic save, and workspace cleanup remain
  covered by no-model regression checks.
- End-to-end inference is claimed only after a real local generation completes through port 8390.

## Local inference check

One 3.5-second audio request with a generated portrait completed through port 8390. The endpoint returned a
3.56-second 624×624 MP4 at 25 fps with H.264 video and the supplied 16 kHz mono AAC audio track; peak VRAM was
about 5.6 GiB with MMGP Profile 4 on the RTX 4070 Ti SUPER.
