# LongCat Avatar 1.5 Quanto-Int8 Integration Plan

**Goal:** Run the local Quanto LongCat Avatar model through port 8390 on a single 16 GiB GPU.

**Architecture:** Lightweight FastAPI parent, shared bounded GPU lock, WanGP's recommended Python API,
WanGP-bundled MMGP Profile 4, read-only checkpoint symlinks, one-shot worker, ffprobe validation, atomic save.

**Specification:** `docs/superpowers/specs/2026-11-01-longcat-avatar-quanto-int8-design.md`

## Work completed

- [x] Replaced the official dual-GPU torchrun path with WanGP `shared.api.init()` and `submit_task()`.
- [x] Confirmed WanGP reports the local `longcat_avatar_v1_5` checkpoint set as available.
- [x] Prepared an isolated WanGP Python 3.11 / PyTorch 2.10 CUDA 13 environment with MMGP 4.5.0.
- [x] Added an idempotent WanGP config/link setup that only creates read-only symlinks to existing local weights.
- [x] Updated 8390 readiness, startup configuration, service dependencies and deployment documentation for one GPU.
- [x] Confirmed `/v1/health` reports selected-provider readiness on the RTX 4070 Ti SUPER 16 GiB host.
- [x] Generated a reference portrait with the repository imagegen skill and completed one real generation through port 8390; ffprobe confirmed H.264 video and the input AAC audio track.
- [x] Made the reference image mandatory at request validation because WanGP fixes LongCat Avatar to `KI` reference-image mode.

## Remaining verification

- [x] Review updated source/config/docs; the repository no-model suite, Ruff checks, lock check, shell syntax and LocalAI capability validation pass.
