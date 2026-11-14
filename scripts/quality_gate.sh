#!/usr/bin/env bash
# 只使用 qa 轻量环境执行无模型质量门禁，不解析模型项目依赖。
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"

bash -n start.sh
bash scripts/test_steam_audio_renderer_core.sh

lock_projects=(
    main
    model-audio/design_qwen3-tts-1.7b-voicedesign
    model-audio/audio_cpp_runtime
    qa
)
lint_paths=(
    main
    llm_local_runtime
    inference_gateway
    scripts/localai
    scripts/llama_cpp
    scripts/sdcpp
    scripts/model_telemetry.py
    model-text/llama_cpp_runtime
    model-text/Ornith-1.5-9B-GGUF
    model-text/Ornith-1.5-9B-MTP-GGUF
    model-image
    easel_bridge
)
for project in "${lock_projects[@]}"; do
    if [[ ! -d "$project" ]]; then
        echo "缺少当前维护的 uv 项目：$project" >&2
        exit 1
    fi
    uv lock --project "$project" --check
    if [[ "$project" != qa ]]; then
        lint_paths+=("$project")
    fi
done
lint_paths+=(tests)
lint_paths+=(
    model-audio/tts_confucius4-tts
    model-audio/tts_indextts2.5
    model-audio/gen_midashenglm
    model-audio/asr_qwen3-1.7b
    model-audio/tts_qwen3-1.7b-base
    model-audio/vc_seed-vc
    model-audio/gen_stable-audio-3-small-sfx
    model-audio/tts_voxcpm2
)

uv run --project qa --locked ruff check "${lint_paths[@]}"
uv run --project qa --locked ruff format --check "${lint_paths[@]}"

uv run --project qa --locked python scripts/localai/validate_capabilities.py
uv run --project qa --locked python -m unittest discover -s tests -v
# 此目录没有 __init__.py，默认 discovery 不会递归进入，需要显式执行。
