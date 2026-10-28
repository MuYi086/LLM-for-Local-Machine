#!/usr/bin/env bash
# 只使用 qa 轻量环境执行无模型质量门禁，不解析任一 CUDA/模型项目依赖。
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"

bash -n start.sh
bash scripts/test_steam_audio_renderer_core.sh

service_projects=(
    model-video/HunyuanVideo-1.5-Comfy
    model-video/avatar_soulx-flashhead-1.3b
    model-audio/analyse_moss-music-8b-Instruct
    model-audio/clone_qwen3-tts-1.7b-base model-audio/clone_voxcpm-2 model-audio/music_ace-step-1.5 model-audio/music_yue2-3b
    model-audio/indexTTS-2.5
    model-audio/score_sheetsage-2 model-audio/codec_yue2-vae model-audio/embedding_mert-v2-fullsong
    model-audio/design_qwen3-tts-1.7b-voicedesign model-audio/clone_confucius-4 model-audio/transcribe_qwen3-asr-1.7b
    model-audio/separate_tiger-dnr model-audio/separate_msst-mega53
)

lock_projects=("${service_projects[@]}" localai_worker_bridge qa)
lint_paths=(main llm_local_runtime inference_gateway scripts/localai)
for project in "${lock_projects[@]}"; do
    if [[ -d "$project" ]]; then
        uv lock --project "$project" --check
        if [[ "$project" != qa ]]; then
            lint_paths+=("$project")
        fi
    else
        echo "缺少当前维护的 uv 项目：$project" >&2
        exit 1
    fi
done
lint_paths+=(tests)

uv run --project qa --locked ruff check "${lint_paths[@]}"
uv run --project qa --locked ruff format --check "${lint_paths[@]}"

uv run --project qa --locked python scripts/localai/validate_capabilities.py
uv run --project qa --locked python -m unittest discover -s tests -v
(cd model-audio/music_ace-step-1.5 && uv run --project ../../qa --locked python -m unittest discover -s tests -v)
