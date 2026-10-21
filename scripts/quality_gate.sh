#!/usr/bin/env bash
# 只使用 qa 轻量环境执行无模型质量门禁，不解析任一 CUDA/模型项目依赖。
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"

bash -n start.sh
bash scripts/test_steam_audio_renderer_core.sh

service_projects=(
    qwen3_tts mimo_tts voxcpm2 LongCat_AudioDiT_3.5B_bf16
    dots_tts_soar moss_soundEffect stable_audio_3_medium ace_step_1_5 YuE2-3B
    SheetSage2 YuE2-Vae MERT-v2-FullSong
    qwen3_voiceDesign moss_voiceGenerator moss_audio_4b_thinking Confucius4_TTS
    Qwen3_ASR_1.7B seed-vc SoulX-FlashHead-1_3B ditto Step_Audio_EditX
    firered_tts3 TIGER-DnR
)

lock_projects=("${service_projects[@]}" unitale_localai_bridge qa)
lint_paths=(main unitale_runtime inference_gateway scripts/localai)
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
(cd ace_step_1_5 && uv run --project ../qa --locked python -m unittest discover -s tests -v)
(cd stable_audio_3_medium && uv run --project ../qa --locked python -m unittest discover -s tests -v)
