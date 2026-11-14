#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STORAGE_DIR="${STORAGE_DIR:-$PROJECT_DIR/storage}"
export RUNTIME_CACHE_DIR="${RUNTIME_CACHE_DIR:-$STORAGE_DIR/.cache/runtime}"
START_LOG_DIR="$RUNTIME_CACHE_DIR/logs"
mkdir -p "$START_LOG_DIR"
START_LOG_FILE="$(mktemp --suffix=.log "$START_LOG_DIR/start-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
# 终端信号交给主脚本收尾。日志进程若先退出，cleanup 的输出会触发 SIGPIPE，
# 使主脚本在停止独立 setsid 进程组前退出；终端断开时仍继续写日志文件。
exec > >(trap '' INT TERM HUP; exec tee -a --output-error=warn-nopipe "$START_LOG_FILE") 2>&1
START_LOG_PID=$!
printf '[日志] 本次完整控制台日志：%s\n' "$START_LOG_FILE"

MAIN_DIR="$PROJECT_DIR/main"
MAIN_PROJECT_DIR="$MAIN_DIR"
VOICEDESIGN_PROJECT_DIR="$PROJECT_DIR/model-audio/design_qwen3-tts-1.7b-voicedesign"
AUDIO_CPP_RUNTIME_PROJECT_DIR="$PROJECT_DIR/model-audio/audio_cpp_runtime"
AUDIO_CPP_RUNTIME_SRC="$AUDIO_CPP_RUNTIME_PROJECT_DIR/src"
MODEL_TEXT_DIR="$PROJECT_DIR/model-text"
ORNITH_9B_SERVICE_DIR="$MODEL_TEXT_DIR/Ornith-1.5-9B-GGUF"
ORNITH_MTP_9B_SERVICE_DIR="$MODEL_TEXT_DIR/Ornith-1.5-9B-MTP-GGUF"
MOSS_IMAGE_SERVICE_DIR="$PROJECT_DIR/model-image/MOSS-Image-SFT-Q8_0"
FLUX3_IMAGE_SERVICE_DIR="$PROJECT_DIR/model-image/FLUX-3-Image-Klein-GGUF-Q8"
EASEL_BRIDGE_ENV_FILE="${EASEL_BRIDGE_ENV_FILE:-$PROJECT_DIR/.env.easel-bridge}"
LLM_LOCAL_RUNTIME_SRC="$PROJECT_DIR/llm_local_runtime/src"
INFERENCE_GATEWAY_SRC="$PROJECT_DIR/inference_gateway/src"

if [[ ! -f "$LLM_LOCAL_RUNTIME_SRC/llm_local_runtime/__init__.py" ]]; then
  echo "缺少共享运行时源码：$LLM_LOCAL_RUNTIME_SRC/llm_local_runtime" >&2
  exit 1
fi
if [[ ! -f "$INFERENCE_GATEWAY_SRC/inference_gateway/__init__.py" ]]; then
  echo "缺少共享推理网关源码：$INFERENCE_GATEWAY_SRC/inference_gateway" >&2
  exit 1
fi
if [[ ! -f "$VOICEDESIGN_PROJECT_DIR/main.py" ]]; then
  echo "缺少 Qwen3-TTS VoiceDesign 服务：$VOICEDESIGN_PROJECT_DIR/main.py" >&2
  exit 1
fi
if [[ ! -f "$AUDIO_CPP_RUNTIME_SRC/audio_cpp_runtime/__init__.py" ]]; then
  echo "缺少共享 audio.cpp runtime 包：$AUDIO_CPP_RUNTIME_SRC/audio_cpp_runtime" >&2
  exit 1
fi
if [[ ! -f "$MOSS_IMAGE_SERVICE_DIR/main.py" ]]; then
  echo "缺少 MOSS-Image-SFT 图片服务：$MOSS_IMAGE_SERVICE_DIR/main.py" >&2
  exit 1
fi
if [[ ! -f "$FLUX3_IMAGE_SERVICE_DIR/main.py" ]]; then
  echo "缺少 FLUX.3 Klein 图片服务：$FLUX3_IMAGE_SERVICE_DIR/main.py" >&2
  exit 1
fi
if [[ ! -f "$ORNITH_9B_SERVICE_DIR/main.py" ]]; then
  echo "缺少 Ornith-1.5-9B-GGUF 服务：$ORNITH_9B_SERVICE_DIR/main.py" >&2
  exit 1
fi
if [[ ! -f "$ORNITH_MTP_9B_SERVICE_DIR/main.py" ]]; then
  echo "缺少 Ornith-1.5-9B-MTP-GGUF 服务：$ORNITH_MTP_9B_SERVICE_DIR/main.py" >&2
  exit 1
fi
if [[ ! -f "$PROJECT_DIR/easel_bridge/app.py" ]]; then
  echo "缺少 OpenAI-compatible 网关：easel_bridge/app.py" >&2
  exit 1
fi

# 与单独启动网关时使用同一份本地配置，不在日志中输出密钥。
if [[ -f "$EASEL_BRIDGE_ENV_FILE" ]]; then
  set -a
  source "$EASEL_BRIDGE_ENV_FILE"
  set +a
fi
# 显式环境变量优先于可编辑的默认配置，false 不影响必要的运行/故障日志。
telemetry_enabled_override="${MODEL_TELEMETRY_ENABLED-}"
telemetry_interval_override="${MODEL_TELEMETRY_INTERVAL_SECONDS-}"
MODEL_TELEMETRY_CONFIG_FILE="${MODEL_TELEMETRY_CONFIG_FILE:-$PROJECT_DIR/config/model-telemetry.env}"
if [[ -f "$MODEL_TELEMETRY_CONFIG_FILE" ]]; then
  source "$MODEL_TELEMETRY_CONFIG_FILE"
fi
export MODEL_TELEMETRY_ENABLED="${telemetry_enabled_override:-${MODEL_TELEMETRY_ENABLED:-true}}"
export MODEL_TELEMETRY_INTERVAL_SECONDS="${telemetry_interval_override:-${MODEL_TELEMETRY_INTERVAL_SECONDS:-1}}"
case "${MODEL_TELEMETRY_ENABLED,,}" in
  true|false|1|0|yes|no|on|off) ;;
  *) echo "MODEL_TELEMETRY_ENABLED 必须为 true 或 false。" >&2; exit 1 ;;
esac
if [[ ! "$MODEL_TELEMETRY_INTERVAL_SECONDS" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] ||
   [[ -z "${MODEL_TELEMETRY_INTERVAL_SECONDS//[.0]/}" ]]; then
  echo "MODEL_TELEMETRY_INTERVAL_SECONDS 必须为正数。" >&2
  exit 1
fi
export EASEL_BRIDGE_PORT="${EASEL_BRIDGE_PORT:-8500}"
export EASEL_BRIDGE_API_KEY="${EASEL_BRIDGE_API_KEY:-}"
if [[ ${#EASEL_BRIDGE_API_KEY} -lt 32 ]]; then
  echo "EASEL_BRIDGE_API_KEY 必须是至少 32 字节的随机值，请配置 .env.easel-bridge。" >&2
  exit 1
fi
if [[ "${EASEL_BRIDGE_API_KEY,,}" == replace-with-* || "$EASEL_BRIDGE_API_KEY" == *[[:space:]]* ]]; then
  echo "EASEL_BRIDGE_API_KEY 必须是随机值，不能使用模板占位符或空白。" >&2
  exit 1
fi
if [[ ! "$EASEL_BRIDGE_PORT" =~ ^[1-9][0-9]*$ ]] || (( EASEL_BRIDGE_PORT > 65535 )); then
  echo "EASEL_BRIDGE_PORT 必须在 1 到 65535 之间。" >&2
  exit 1
fi

export PYTHONPATH="$MODEL_TEXT_DIR:$AUDIO_CPP_RUNTIME_SRC:$LLM_LOCAL_RUNTIME_SRC:$INFERENCE_GATEWAY_SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_MIRROR_DIR="${HF_MIRROR_DIR:-$HOME/hf-mirror}"
export QWEN_VOICEDESIGN_MODEL_FILE="${QWEN_VOICEDESIGN_MODEL_FILE:-$HF_MIRROR_DIR/audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF/qwen3-tts-12hz-1.7b-voicedesign-bf16.gguf}"
export AUDIO_CPP_CONFUCIUS4_TTS_MODEL_FILE="${AUDIO_CPP_CONFUCIUS4_TTS_MODEL_FILE:-$HF_MIRROR_DIR/audio.cpp-gguf/Confucius4-TTS-GGUF/confucius4-tts-orig.gguf}"
export AUDIO_CPP_INDEX_TTS25_MODEL_FILE="${AUDIO_CPP_INDEX_TTS25_MODEL_FILE:-$HF_MIRROR_DIR/audio.cpp-gguf/IndexTTS2.5-GGUF/index-tts2_5-f16.gguf}"
export AUDIO_CPP_MIDASHENGLM_GEN_MODEL_FILE="${AUDIO_CPP_MIDASHENGLM_GEN_MODEL_FILE:-$HF_MIRROR_DIR/audio.cpp-gguf/MiDashengLM-Gen-GGUF/midashenglm-gen-q8_0.gguf}"
export AUDIO_CPP_QWEN3_ASR_MODEL_FILE="${AUDIO_CPP_QWEN3_ASR_MODEL_FILE:-$HF_MIRROR_DIR/audio.cpp-gguf/Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-f16.gguf}"
export AUDIO_CPP_QWEN3_TTS_BASE_MODEL_FILE="${AUDIO_CPP_QWEN3_TTS_BASE_MODEL_FILE:-$HF_MIRROR_DIR/audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf}"
export AUDIO_CPP_SEED_VC_MODEL_FILE="${AUDIO_CPP_SEED_VC_MODEL_FILE:-$HF_MIRROR_DIR/audio.cpp-gguf/SeedVC-MLX-GGUF/seed-vc-mlx-f16.gguf}"
export AUDIO_CPP_STABLE_AUDIO_SFX_MODEL_FILE="${AUDIO_CPP_STABLE_AUDIO_SFX_MODEL_FILE:-$HF_MIRROR_DIR/audio.cpp-gguf/Stable-Audio-3-Small-SFX-GGUF/stable-audio-3-small-sfx-f16.gguf}"
export AUDIO_CPP_VOXCPM2_MODEL_FILE="${AUDIO_CPP_VOXCPM2_MODEL_FILE:-$HF_MIRROR_DIR/audio.cpp-gguf/VoxCPM2-GGUF/voxcpm2-q8_0.gguf}"
export ORNITH_9B_MODEL_FILE="${ORNITH_9B_MODEL_FILE:-$HF_MIRROR_DIR/ornith-ai/Ornith-1.5-9B-GGUF/Ornith-1.5-9B-Q8_0.gguf}"
export ORNITH_MTP_9B_MODEL_FILE="${ORNITH_MTP_9B_MODEL_FILE:-$HF_MIRROR_DIR/ornith-ai/Ornith-1.5-9B-MTP-GGUF/Ornith-1.5-9B-MTP-Q8_0.gguf}"
export ORNITH_MTP_9B_MMPROJ_FILE="${ORNITH_MTP_9B_MMPROJ_FILE:-$HF_MIRROR_DIR/ornith-ai/Ornith-1.5-9B-MTP-GGUF/mmproj-Ornith-1.5-9B-BF16.gguf}"
export SDCPP_BIN="${SDCPP_BIN:-$HOME/.local/src/stable-diffusion.cpp/build/bin/sd-cli}"
export SDCPP_BIN_SHA256="${SDCPP_BIN_SHA256:-}"
export SDCPP_MODEL_MANIFEST="${SDCPP_MODEL_MANIFEST:-$PROJECT_DIR/sdcpp/manifests/models.json}"
export SDCPP_OUTPUT_DIR="${SDCPP_OUTPUT_DIR:-$RUNTIME_CACHE_DIR/sdcpp}"
export SDCPP_RUN_TIMEOUT="${SDCPP_RUN_TIMEOUT:-3600}"
export SDCPP_CLEANUP_TIMEOUT="${SDCPP_CLEANUP_TIMEOUT:-30}"
export FLUX3_IMAGE_UPLOAD_DIR="${FLUX3_IMAGE_UPLOAD_DIR:-$RUNTIME_CACHE_DIR/sdcpp_uploads}"
export FLUX3_IMAGE_MAX_UPLOAD_BYTES="${FLUX3_IMAGE_MAX_UPLOAD_BYTES:-67108864}"
export TIMBRE_STORAGE_DIR="${TIMBRE_STORAGE_DIR:-$STORAGE_DIR/timbre}"

if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
  LLAMA_SERVER_BIN="$(command -v llama-server || true)"
  if [[ -z "$LLAMA_SERVER_BIN" ]]; then
    local_cached_llama_server="$HOME/.local/src/llama.cpp/build-sleep-cache/bin/llama-server"
    local_cuda_llama_server="$HOME/.local/src/llama.cpp/build-cuda/bin/llama-server"
    if [[ -x "$local_cached_llama_server" ]]; then
      LLAMA_SERVER_BIN="$local_cached_llama_server"
    elif [[ -x "$local_cuda_llama_server" ]]; then
      LLAMA_SERVER_BIN="$local_cuda_llama_server"
    fi
  fi
elif [[ "$LLAMA_SERVER_BIN" != */* ]]; then
  LLAMA_SERVER_BIN="$(command -v "$LLAMA_SERVER_BIN" || true)"
fi
export LLAMA_SERVER_BIN

export LLM_LOCALAI_ENABLED="${LLM_LOCALAI_ENABLED:-true}"
export LLM_LOCALAI_MANAGED="${LLM_LOCALAI_MANAGED:-true}"
export LLM_LOCALAI_BASE_URL="${LLM_LOCALAI_BASE_URL:-http://127.0.0.1:8400}"
export LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER="${LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER:-localai}"
export LLM_LOCAL_QWEN3_VOICEDESIGN_MODEL="${LLM_LOCAL_QWEN3_VOICEDESIGN_MODEL:-llm-local.qwen3-tts.1p7b-voicedesign}"
export LLM_LOCAL_AUDIO_CPP_PROVIDER="${LLM_LOCAL_AUDIO_CPP_PROVIDER:-localai}"
export LLM_LOCALAI_ARTIFACT_DIR="${LLM_LOCALAI_ARTIFACT_DIR:-$RUNTIME_CACHE_DIR/localai_artifacts}"
export LLM_LOCALAI_CONNECT_TIMEOUT="${LLM_LOCALAI_CONNECT_TIMEOUT:-5}"
export LLM_LOCALAI_CLEANUP_TIMEOUT="${LLM_LOCALAI_CLEANUP_TIMEOUT:-30}"
export QWEN_VOICEDESIGN_REQUEST_TIMEOUT="${QWEN_VOICEDESIGN_REQUEST_TIMEOUT:-900}"
export AUDIO_CPP_REQUEST_TIMEOUT="${AUDIO_CPP_REQUEST_TIMEOUT:-900}"
export AUDIO_CPP_TEMP_DIR="${AUDIO_CPP_TEMP_DIR:-$RUNTIME_CACHE_DIR/audio_cpp_uploads}"

export SPATIAL_EXPORT_CACHE_DIR="${SPATIAL_EXPORT_CACHE_DIR:-$RUNTIME_CACHE_DIR/spatial_exports}"
export SPATIAL_EXPORT_MAX_BYTES="${SPATIAL_EXPORT_MAX_BYTES:-536870912}"
export SPATIAL_EXPORT_TIMEOUT="${SPATIAL_EXPORT_TIMEOUT:-600}"
export SPATIAL_EXPORT_FFMPEG_BIN="${SPATIAL_EXPORT_FFMPEG_BIN:-ffmpeg}"
export STEAM_AUDIO_RENDER_CACHE_DIR="${STEAM_AUDIO_RENDER_CACHE_DIR:-$RUNTIME_CACHE_DIR/steam_audio_renders}"
export STEAM_AUDIO_RENDERER_BIN="${STEAM_AUDIO_RENDERER_BIN:-$PROJECT_DIR/model-audio/spatial_steam-audio/build/steam-audio-render}"
export STEAM_AUDIO_SDK_DIR="${STEAM_AUDIO_SDK_DIR:-$HOME/.local/share/steam-audio/4.8.1}"
export STEAM_AUDIO_HRTF_PATH="${STEAM_AUDIO_HRTF_PATH:-}"
export STEAM_AUDIO_RENDER_TIMEOUT="${STEAM_AUDIO_RENDER_TIMEOUT:-900}"
export STEAM_AUDIO_RENDER_MAX_ASSETS="${STEAM_AUDIO_RENDER_MAX_ASSETS:-500}"
export STEAM_AUDIO_RENDER_MAX_MANIFEST_BYTES="${STEAM_AUDIO_RENDER_MAX_MANIFEST_BYTES:-8388608}"
export STEAM_AUDIO_RENDER_MAX_BYTES="${STEAM_AUDIO_RENDER_MAX_BYTES:-2147483648}"
export STEAM_AUDIO_RENDER_THREADS="${STEAM_AUDIO_RENDER_THREADS:-4}"

export GPU_LOCK_FILE="${GPU_LOCK_FILE:-$RUNTIME_CACHE_DIR/gpu-runtime.lock}"
export GPU_LOCK_WAIT_TIMEOUT="${GPU_LOCK_WAIT_TIMEOUT:-900}"
export GPU_METRICS_SAMPLE_INTERVAL="${GPU_METRICS_SAMPLE_INTERVAL:-0.5}"
export CUDA_RELEASE_DELAY="${CUDA_RELEASE_DELAY:-2.0}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8300}"
export MOSS_IMAGE_HOST="${MOSS_IMAGE_HOST:-127.0.0.1}"
export MOSS_IMAGE_PORT="${MOSS_IMAGE_PORT:-8501}"
export FLUX3_IMAGE_HOST="${FLUX3_IMAGE_HOST:-$HOST}"
export FLUX3_IMAGE_PORT="${FLUX3_IMAGE_PORT:-8502}"
export QWEN_VOICEDESIGN_HOST="${QWEN_VOICEDESIGN_HOST:-$HOST}"
export QWEN_VOICEDESIGN_PORT="${QWEN_VOICEDESIGN_PORT:-8301}"
export ORNITH_9B_HOST="${ORNITH_9B_HOST:-$HOST}"
export ORNITH_9B_PORT="${ORNITH_9B_PORT:-8503}"
export ORNITH_9B_BACKEND_HOST=127.0.0.1
export ORNITH_9B_LLAMA_CPP_PORT="${ORNITH_9B_LLAMA_CPP_PORT:-8505}"
export ORNITH_9B_LLAMA_BASE_URL="${ORNITH_9B_LLAMA_BASE_URL:-http://127.0.0.1:$ORNITH_9B_LLAMA_CPP_PORT}"
export ORNITH_9B_CONTEXT_SIZE="${ORNITH_9B_CONTEXT_SIZE:-98304}"
export ORNITH_9B_BATCH_SIZE="${ORNITH_9B_BATCH_SIZE:-1024}"
export ORNITH_9B_UBATCH_SIZE="${ORNITH_9B_UBATCH_SIZE:-1024}"
export ORNITH_9B_THREADS="${ORNITH_9B_THREADS:-6}"
export ORNITH_9B_THREADS_BATCH="${ORNITH_9B_THREADS_BATCH:-8}"
export ORNITH_9B_N_GPU_LAYERS="${ORNITH_9B_N_GPU_LAYERS:-99}"
export ORNITH_9B_SPEC_TYPE="${ORNITH_9B_SPEC_TYPE:-draft-mtp}"
export ORNITH_9B_DRAFT_MAX="${ORNITH_9B_DRAFT_MAX:-4}"
export ORNITH_9B_SLEEP_CACHE="${ORNITH_9B_SLEEP_CACHE:-1}"
export ORNITH_9B_CACHE_RAM_MIB="${ORNITH_9B_CACHE_RAM_MIB:-4096}"
export LLAMA_SERVER_SLEEP_PROMPT_CACHE="$ORNITH_9B_SLEEP_CACHE"
export ORNITH_9B_SLEEP_IDLE_SECONDS="${ORNITH_9B_SLEEP_IDLE_SECONDS:-1}"
export ORNITH_9B_STARTUP_TIMEOUT="${ORNITH_9B_STARTUP_TIMEOUT:-1800}"
export ORNITH_9B_SLEEP_TIMEOUT="${ORNITH_9B_SLEEP_TIMEOUT:-300}"
export ORNITH_9B_SLEEP_POLL_INTERVAL="${ORNITH_9B_SLEEP_POLL_INTERVAL:-0.25}"
export ORNITH_9B_BACKEND_PID_FILE="${ORNITH_9B_BACKEND_PID_FILE:-$RUNTIME_CACHE_DIR/ornith-9b-llama-server.pid}"
export ORNITH_MTP_9B_HOST="${ORNITH_MTP_9B_HOST:-$HOST}"
export ORNITH_MTP_9B_VISION_ENABLED="${ORNITH_MTP_9B_VISION_ENABLED:-1}"
export ORNITH_MTP_9B_SLEEP_IDLE_SECONDS="${ORNITH_MTP_9B_SLEEP_IDLE_SECONDS:-$ORNITH_9B_SLEEP_IDLE_SECONDS}"
export ORNITH_MTP_9B_PORT="${ORNITH_MTP_9B_PORT:-8504}"
export ORNITH_MTP_9B_BACKEND_HOST=127.0.0.1
export ORNITH_MTP_9B_LLAMA_CPP_PORT="${ORNITH_MTP_9B_LLAMA_CPP_PORT:-8506}"
export ORNITH_MTP_9B_LLAMA_BASE_URL="${ORNITH_MTP_9B_LLAMA_BASE_URL:-http://127.0.0.1:$ORNITH_MTP_9B_LLAMA_CPP_PORT}"
export ORNITH_MTP_9B_STARTUP_TIMEOUT="${ORNITH_MTP_9B_STARTUP_TIMEOUT:-$ORNITH_9B_STARTUP_TIMEOUT}"
export ORNITH_MTP_9B_SLEEP_TIMEOUT="${ORNITH_MTP_9B_SLEEP_TIMEOUT:-$ORNITH_9B_SLEEP_TIMEOUT}"
export ORNITH_MTP_9B_SLEEP_POLL_INTERVAL="${ORNITH_MTP_9B_SLEEP_POLL_INTERVAL:-$ORNITH_9B_SLEEP_POLL_INTERVAL}"
export ORNITH_MTP_9B_BACKEND_PID_FILE="${ORNITH_MTP_9B_BACKEND_PID_FILE:-$RUNTIME_CACHE_DIR/ornith-mtp-9b-llama-server.pid}"
export EASEL_BRIDGE_CHAT_URL="${EASEL_BRIDGE_CHAT_URL:-http://127.0.0.1:$ORNITH_9B_PORT/v1/ornith-ai/Ornith-1.5-9B-GGUF}"
export EASEL_BRIDGE_MTP_CHAT_URL="${EASEL_BRIDGE_MTP_CHAT_URL:-http://127.0.0.1:$ORNITH_MTP_9B_PORT/v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8}"
export EASEL_BRIDGE_MOSS_URL="${EASEL_BRIDGE_MOSS_URL:-http://127.0.0.1:$MOSS_IMAGE_PORT/v1/openmoss/MOSS-Image-SFT-Q8_0}"
export EASEL_BRIDGE_VOXCPM_URL="${EASEL_BRIDGE_VOXCPM_URL:-http://127.0.0.1:${VOXCPM2_PORT:-8313}/v1/OpenBMB/VoxCPM2-GGUF}"

AUDIO_CPP_SERVICE_KEYS=(
  confucius4_tts index_tts25 midashenglm_gen qwen3_asr
  qwen3_tts_base seed_vc stable_audio_sfx voxcpm2
)
AUDIO_CPP_SERVICE_DIRS=(
  "$PROJECT_DIR/model-audio/tts_confucius4-tts"
  "$PROJECT_DIR/model-audio/tts_indextts2.5"
  "$PROJECT_DIR/model-audio/gen_midashenglm"
  "$PROJECT_DIR/model-audio/asr_qwen3-1.7b"
  "$PROJECT_DIR/model-audio/tts_qwen3-1.7b-base"
  "$PROJECT_DIR/model-audio/vc_seed-vc"
  "$PROJECT_DIR/model-audio/gen_stable-audio-3-small-sfx"
  "$PROJECT_DIR/model-audio/tts_voxcpm2"
)
AUDIO_CPP_SERVICE_ROUTES=(
  /v1/netease/Confucius4-TTS-GGUF
  /v1/indexTeam/IndexTTS2.5-GGUF
  /v1/mi/MiDashengLM-Gen-GGUF
  /v1/qwen/Qwen3-ASR-1.7B-GGUF
  /v1/qwen/Qwen3-TTS-12Hz-1.7B-Base-GGUF
  /v1/SeedVC/SeedVC-MLX-GGUF
  /v1/stabilityai/Stable-Audio-3-Small-SFX-GGUF
  /v1/OpenBMB/VoxCPM2-GGUF
)
AUDIO_CPP_SERVICE_HOSTS=(
  "${CONFUCIUS4_TTS_HOST:-$HOST}"
  "${INDEX_TTS25_HOST:-$HOST}"
  "${MIDASHENGLM_GEN_HOST:-$HOST}"
  "${QWEN3_ASR_HOST:-$HOST}"
  "${QWEN3_TTS_BASE_HOST:-$HOST}"
  "${SEED_VC_HOST:-$HOST}"
  "${STABLE_AUDIO_SFX_HOST:-$HOST}"
  "${VOXCPM2_HOST:-127.0.0.1}"
)
AUDIO_CPP_SERVICE_PORTS=(
  "${CONFUCIUS4_TTS_PORT:-8331}"
  "${INDEX_TTS25_PORT:-8312}"
  "${MIDASHENGLM_GEN_PORT:-8341}"
  "${QWEN3_ASR_PORT:-8321}"
  "${QWEN3_TTS_BASE_PORT:-8311}"
  "${SEED_VC_PORT:-8342}"
  "${STABLE_AUDIO_SFX_PORT:-8343}"
  "${VOXCPM2_PORT:-8313}"
)

mkdir -p \
  "$TIMBRE_STORAGE_DIR" \
  "$RUNTIME_CACHE_DIR" \
  "$AUDIO_CPP_TEMP_DIR" \
  "$SPATIAL_EXPORT_CACHE_DIR" \
  "$STEAM_AUDIO_RENDER_CACHE_DIR"

PORT_CHECK_ARGS=(
  'Control-plane' "$HOST" "$PORT"
  'Qwen3-TTS VoiceDesign' "$QWEN_VOICEDESIGN_HOST" "$QWEN_VOICEDESIGN_PORT"
  'Ornith API' "$ORNITH_9B_HOST" "$ORNITH_9B_PORT"
  'Ornith llama.cpp backend' "$ORNITH_9B_BACKEND_HOST" "$ORNITH_9B_LLAMA_CPP_PORT"
  'Ornith MTP 9B API' "$ORNITH_MTP_9B_HOST" "$ORNITH_MTP_9B_PORT"
  'Ornith MTP 9B llama.cpp backend' "$ORNITH_MTP_9B_BACKEND_HOST" "$ORNITH_MTP_9B_LLAMA_CPP_PORT"
  'MOSS-Image-SFT API' "$MOSS_IMAGE_HOST" "$MOSS_IMAGE_PORT"
  'FLUX.3 Klein Image API' "$FLUX3_IMAGE_HOST" "$FLUX3_IMAGE_PORT"
  'OpenAI-compatible bridge' '127.0.0.1' "$EASEL_BRIDGE_PORT"
)
for index in "${!AUDIO_CPP_SERVICE_KEYS[@]}"; do
  PORT_CHECK_ARGS+=(
    "${AUDIO_CPP_SERVICE_KEYS[$index]}"
    "${AUDIO_CPP_SERVICE_HOSTS[$index]}"
    "${AUDIO_CPP_SERVICE_PORTS[$index]}"
  )
done
uv run --no-sync --project "$MAIN_PROJECT_DIR" python "$MAIN_DIR/check_ports.py" \
  "${PORT_CHECK_ARGS[@]}"
if [[ ! -x "$SDCPP_BIN" ]]; then
  echo "找不到可执行的 stable-diffusion.cpp sd-cli：$SDCPP_BIN" >&2
  exit 1
fi
if [[ ! "$SDCPP_BIN_SHA256" =~ ^[a-fA-F0-9]{64}$ ]]; then
  echo "SDCPP_BIN_SHA256 必须是 sd-cli 的 64 位 SHA-256。" >&2
  exit 1
fi
MOSS_IMAGE_MODEL_ROOT="$HF_MIRROR_DIR/stable-diffusion-cpp-gguf/MOSS-Image-SFT-GGUF"
for model_component in \
  "$MOSS_IMAGE_MODEL_ROOT/MOSS-Image-SFT-Q8_0.gguf" \
  "$MOSS_IMAGE_MODEL_ROOT/Qwen3.5-9B-Q6_K.gguf" \
  "$MOSS_IMAGE_MODEL_ROOT/flux2-vae.safetensors" \
  "$MOSS_IMAGE_MODEL_ROOT/tokenizer.json"; do
  if [[ ! -s "$model_component" ]]; then
    echo "MOSS-Image-SFT 模型组件缺失或为空：$model_component" >&2
    exit 1
  fi
done
if [[ ! "$FLUX3_IMAGE_MAX_UPLOAD_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  echo "FLUX3_IMAGE_MAX_UPLOAD_BYTES 必须是正整数。" >&2
  exit 1
fi
FLUX3_IMAGE_MODEL_ROOT="$HF_MIRROR_DIR/stable-diffusion-cpp-gguf/FLUX-3-Image-Klein-GGUF"
for model_component in \
  "$FLUX3_IMAGE_MODEL_ROOT/flux3-image-klein-Q8_0.gguf" \
  "$FLUX3_IMAGE_MODEL_ROOT/Qwen3VL-4B-Instruct-Q8_0.gguf" \
  "$FLUX3_IMAGE_MODEL_ROOT/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf" \
  "$FLUX3_IMAGE_MODEL_ROOT/ae.safetensors"; do
  if [[ ! -s "$model_component" ]]; then
    echo "FLUX.3 Klein 模型组件缺失或为空：$model_component" >&2
    exit 1
  fi
done

if [[ "$ORNITH_9B_LLAMA_BASE_URL" != "http://127.0.0.1:$ORNITH_9B_LLAMA_CPP_PORT" ]]; then
  echo "ORNITH_9B_LLAMA_BASE_URL 必须与 Ornith 9B backend 的 127.0.0.1:$ORNITH_9B_LLAMA_CPP_PORT 一致。" >&2
  exit 1
fi
if [[ "$ORNITH_MTP_9B_LLAMA_BASE_URL" != "http://127.0.0.1:$ORNITH_MTP_9B_LLAMA_CPP_PORT" ]]; then
  echo "ORNITH_MTP_9B_LLAMA_BASE_URL 必须与 MTP backend 的 127.0.0.1:$ORNITH_MTP_9B_LLAMA_CPP_PORT 一致。" >&2
  exit 1
fi

if [[ -z "$LLAMA_SERVER_BIN" || ! -x "$LLAMA_SERVER_BIN" ]]; then
  echo "找不到可执行的 llama-server；请安装 llama.cpp 或设置 LLAMA_SERVER_BIN。" >&2
  exit 1
fi
if [[ ! -s "$ORNITH_9B_MODEL_FILE" ]]; then
  echo "Ornith 9B GGUF 缺失或为空：$ORNITH_9B_MODEL_FILE" >&2
  exit 1
fi
if [[ ! -s "$ORNITH_MTP_9B_MODEL_FILE" ]]; then
  echo "Ornith MTP 9B GGUF 缺失或为空：$ORNITH_MTP_9B_MODEL_FILE" >&2
  exit 1
fi
if [[ ! -s "$ORNITH_MTP_9B_MMPROJ_FILE" ]]; then
  echo "Ornith MTP 9B mmproj 缺失或为空：$ORNITH_MTP_9B_MMPROJ_FILE" >&2
  exit 1
fi
for setting in CONTEXT_SIZE BATCH_SIZE UBATCH_SIZE THREADS THREADS_BATCH N_GPU_LAYERS DRAFT_MAX CACHE_RAM_MIB SLEEP_IDLE_SECONDS STARTUP_TIMEOUT; do
  value_name="ORNITH_9B_${setting}"
  if [[ ! "${!value_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name 必须是正整数。" >&2
    exit 1
  fi
done
if [[ ! "$ORNITH_MTP_9B_SLEEP_IDLE_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ORNITH_MTP_9B_SLEEP_IDLE_SECONDS 必须是正整数；当前后端不接受 0。" >&2
  exit 1
fi
if [[ "$ORNITH_9B_SLEEP_CACHE" != "0" && "$ORNITH_9B_SLEEP_CACHE" != "1" ]]; then
  echo "ORNITH_9B_SLEEP_CACHE 必须是 0 或 1。" >&2
  exit 1
fi
if [[ "$ORNITH_9B_SPEC_TYPE" != "none" && "$ORNITH_9B_SPEC_TYPE" != "draft-mtp" ]]; then
  echo "ORNITH_9B_SPEC_TYPE 必须是 none 或 draft-mtp。" >&2
  exit 1
fi
if (( ORNITH_9B_CONTEXT_SIZE > 163840 )); then
  echo "ORNITH_9B_CONTEXT_SIZE 不得超过 163840。" >&2
  exit 1
fi
for value_name in ORNITH_9B_SLEEP_TIMEOUT ORNITH_9B_SLEEP_POLL_INTERVAL GPU_LOCK_WAIT_TIMEOUT; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+([.][0-9]+)?$ || ! "$value" =~ [1-9] ]]; then
    echo "$value_name 必须是正数。" >&2
    exit 1
  fi
done
for value_name in ORNITH_MTP_9B_STARTUP_TIMEOUT; do
  if [[ ! "${!value_name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name 必须是正整数。" >&2
    exit 1
  fi
done
for value_name in ORNITH_MTP_9B_SLEEP_TIMEOUT ORNITH_MTP_9B_SLEEP_POLL_INTERVAL; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+([.][0-9]+)?$ || ! "$value" =~ [1-9] ]]; then
    echo "$value_name 必须是正数。" >&2
    exit 1
  fi
done
if ! [[ "$CUDA_RELEASE_DELAY" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "CUDA_RELEASE_DELAY 必须是非负数。" >&2
  exit 1
fi
DEVICE_LIST="$("$LLAMA_SERVER_BIN" --list-devices 2>&1 || true)"
if ! grep -q 'CUDA0' <<<"$DEVICE_LIST"; then
  echo "llama-server 未提供 CUDA0 设备；请安装 CUDA 构建的 llama.cpp。" >&2
  printf '%s\n' "$DEVICE_LIST" >&2
  exit 1
fi
LLAMA_HELP="$("$LLAMA_SERVER_BIN" --help 2>&1 || true)"
REQUIRED_LLAMA_FLAGS=(--sleep-idle-seconds --load-mode --spec-type --spec-draft-n-max --ubatch-size --cache-ram --mmproj --mmproj-offload)
for required_flag in "${REQUIRED_LLAMA_FLAGS[@]}"; do
  if ! grep -q -- "$required_flag" <<<"$LLAMA_HELP"; then
    echo "llama-server 缺少所需参数：$required_flag。" >&2
    exit 1
  fi
done
if [[ -e "$GPU_LOCK_FILE.quarantine" ]]; then
  echo "GPU quarantine 标记存在：$GPU_LOCK_FILE.quarantine；请先运行 scripts/localai/recover_quarantine.py。" >&2
  exit 1
fi
uv run --no-sync --project "$MAIN_PROJECT_DIR" python "$PROJECT_DIR/scripts/sdcpp/manager.py" \
  --manifest "$SDCPP_MODEL_MANIFEST" validate --model MOSS-Image-SFT-Q8_0
uv run --no-sync --project "$MAIN_PROJECT_DIR" python "$PROJECT_DIR/scripts/sdcpp/manager.py" \
  --manifest "$SDCPP_MODEL_MANIFEST" validate --model FLUX-3-Image-Klein-GGUF-Q8

if [[ ! -x "$STEAM_AUDIO_RENDERER_BIN" ]]; then
  echo "警告：Steam Audio renderer 不存在或不可执行：$STEAM_AUDIO_RENDERER_BIN" >&2
  echo "      请先运行 scripts/build_steam_audio_renderer.sh；空间渲染将返回 503。" >&2
fi

echo "=================================================="
echo "   LLM Local Backend"
echo "=================================================="
echo "Control-plane project: $MAIN_PROJECT_DIR"
echo "Control-plane API:     http://$HOST:$PORT/v1/control"
echo "CPU audio export:      http://$HOST:$PORT/v1/audio/export"
echo "Spatial render:        http://$HOST:$PORT/v1/audio/spatial/render"
echo "VoiceDesign project:   $VOICEDESIGN_PROJECT_DIR"
echo "audio.cpp runtime:     $AUDIO_CPP_RUNTIME_PROJECT_DIR"
echo "VoiceDesign GGUF:      $QWEN_VOICEDESIGN_MODEL_FILE"
echo "Ornith GGUF:           $ORNITH_9B_MODEL_FILE"
echo "Ornith llama-server:   $LLAMA_SERVER_BIN (CUDA0, 127.0.0.1:$ORNITH_9B_LLAMA_CPP_PORT)"
echo "Ornith profile:        ctx=$ORNITH_9B_CONTEXT_SIZE batch=$ORNITH_9B_BATCH_SIZE/$ORNITH_9B_UBATCH_SIZE spec=$ORNITH_9B_SPEC_TYPE draft=$ORNITH_9B_DRAFT_MAX"
echo "Ornith sleep cache:    enabled=$ORNITH_9B_SLEEP_CACHE CPU budget=${ORNITH_9B_CACHE_RAM_MIB}MiB"
echo "Ornith API:            http://$ORNITH_9B_HOST:$ORNITH_9B_PORT/v1/ornith-ai/Ornith-1.5-9B-GGUF"
echo "Ornith MTP GGUF:       $ORNITH_MTP_9B_MODEL_FILE"
echo "Ornith MTP mmproj:     $ORNITH_MTP_9B_MMPROJ_FILE"
echo "Ornith MTP backend:    $LLAMA_SERVER_BIN (CUDA0, 127.0.0.1:$ORNITH_MTP_9B_LLAMA_CPP_PORT)"
echo "Ornith MTP API:        http://$ORNITH_MTP_9B_HOST:$ORNITH_MTP_9B_PORT/v1/ornith-ai/Ornith-1.5-9B-MTP-GGUF-Q8"
echo "MOSS-Image-SFT API:   http://$MOSS_IMAGE_HOST:$MOSS_IMAGE_PORT/v1/openmoss/MOSS-Image-SFT-Q8_0"
echo "FLUX.3 Klein API:     http://$FLUX3_IMAGE_HOST:$FLUX3_IMAGE_PORT/v1/openblack-forest-labs/FLUX-3-Image-Klein-GGUF-Q8"
echo "OpenAI-compatible:    http://127.0.0.1:$EASEL_BRIDGE_PORT/v1 (OpenCode / Easel)"
echo "VoiceDesign API:       http://$QWEN_VOICEDESIGN_HOST:$QWEN_VOICEDESIGN_PORT/v1/qwen/Qwen3-TTS-1.7B-VoiceDesign"
for index in "${!AUDIO_CPP_SERVICE_KEYS[@]}"; do
  echo "${AUDIO_CPP_SERVICE_KEYS[$index]} API: http://${AUDIO_CPP_SERVICE_HOSTS[$index]}:${AUDIO_CPP_SERVICE_PORTS[$index]}${AUDIO_CPP_SERVICE_ROUTES[$index]}"
done
echo "Timbre storage:        $TIMBRE_STORAGE_DIR"
echo "Steam Audio renderer:  $STEAM_AUDIO_RENDERER_BIN"
echo "GPU lock file:         $GPU_LOCK_FILE"
echo "GPU queue timeout:     ${GPU_LOCK_WAIT_TIMEOUT}s"
echo "=================================================="

is_enabled() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

cd "$PROJECT_DIR"

main_pid=""
voicedesign_pid=""
ornith_9b_pid=""
ornith_mtp_9b_pid=""
moss_image_pid=""
flux3_image_pid=""
easel_bridge_pid=""
ornith_9b_llama_server_pid=""
ornith_mtp_9b_llama_server_pid=""
ornith_9b_watch_pid=""
ornith_mtp_9b_watch_pid=""
ornith_9b_backend_start_attempted=0
ornith_mtp_9b_backend_start_attempted=0
audio_cpp_pids=()
localai_managed_started=0
telemetry_pid=""

cleanup() {
  local status=$?
  # 即使日志管道损坏或用户再次按 Ctrl+C，也必须继续停止所有已登记的进程组。
  trap '' INT TERM HUP PIPE
  trap - EXIT
  set +e
  if ! kill -0 "$START_LOG_PID" 2>/dev/null; then
    # 日志进程异常退出时直接追加到原日志，避免 shutdown helper 也因 BrokenPipe 中断。
    exec >> "$START_LOG_FILE" 2>&1
  fi
  local service_pids=("$easel_bridge_pid" "$main_pid" "$voicedesign_pid" "$ornith_9b_pid" "$ornith_mtp_9b_pid" "$moss_image_pid" "$flux3_image_pid" "$ornith_9b_watch_pid" "$ornith_mtp_9b_watch_pid" "${audio_cpp_pids[@]}")
  local recovery_attempt=1
  echo "[退出] 收到停止信号，开始停止服务进程组和 llama.cpp 后端。"

  for pid in "${service_pids[@]}"; do
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${service_pids[@]}"; do
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  done
  for pid in "${service_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  if [[ "$localai_managed_started" == "1" ]]; then
    if ! python3 "$PROJECT_DIR/scripts/localai/manager.py" stop; then
      echo "LocalAI 进程组停止未确认，请检查 GPU 隔离标记并运行 scripts/localai/manager.py status。" >&2
      [[ "$status" != "0" ]] || status=1
    fi
  fi

  local backend_models=(Ornith-1.5-9B-GGUF Ornith-1.5-9B-MTP-GGUF-Q8)
  local backend_hosts=("$ORNITH_9B_BACKEND_HOST" "$ORNITH_MTP_9B_BACKEND_HOST")
  local backend_ports=("$ORNITH_9B_LLAMA_CPP_PORT" "$ORNITH_MTP_9B_LLAMA_CPP_PORT")
  local backend_pid_files=("$ORNITH_9B_BACKEND_PID_FILE" "$ORNITH_MTP_9B_BACKEND_PID_FILE")
  local backend_attempted=("$ornith_9b_backend_start_attempted" "$ornith_mtp_9b_backend_start_attempted")
  local backend_pids=("$ornith_9b_llama_server_pid" "$ornith_mtp_9b_llama_server_pid")
  local backend_shutdown_args=()
  local backend_count="${#backend_models[@]}"
  local backend_index
  for ((backend_index = 0; backend_index < backend_count; backend_index++)); do
    local backend_pid="${backend_pids[$backend_index]}"
    if [[ -z "$backend_pid" && "${backend_attempted[$backend_index]}" == "1" && -s "${backend_pid_files[$backend_index]}" ]]; then
      read -r backend_pid < "${backend_pid_files[$backend_index]}" || true
    fi
    if [[ -n "$backend_pid" && "$backend_pid" =~ ^[1-9][0-9]*$ ]]; then
      backend_shutdown_args+=(
        --backend
        "${backend_models[$backend_index]}"
        "${backend_hosts[$backend_index]}"
        "${backend_ports[$backend_index]}"
        "$backend_pid"
        "$backend_pid"
        "$LLAMA_SERVER_BIN"
      )
    fi
  done
  if ((${#backend_shutdown_args[@]} > 0)); then
    echo "[退出] 正在通过同一 quarantine 事务确认所有 Ornith llama.cpp backend 停止。"
    if python3 -m llama_cpp_runtime.runtime shutdown-many \
      --lock-file "$GPU_LOCK_FILE" \
      --lock-timeout "$GPU_LOCK_WAIT_TIMEOUT" \
      --cleanup-timeout "${LLM_LOCALAI_CLEANUP_TIMEOUT:-30}" \
      --cuda-release-delay "$CUDA_RELEASE_DELAY" \
      "${backend_shutdown_args[@]}"; then
      for pid_file in "${backend_pid_files[@]}"; do
        rm -f -- "$pid_file"
      done
      echo "[退出] 已确认所有 Ornith llama.cpp backend 停止。"
    else
      echo "至少一个 Ornith llama.cpp backend 未确认停止，保留 GPU quarantine 标记。" >&2
      [[ "$status" != "0" ]] || status=1
    fi
  fi

  if [[ -e "$GPU_LOCK_FILE.quarantine" ]]; then
    recovery_attempt=1
    while [[ -e "$GPU_LOCK_FILE.quarantine" && "$recovery_attempt" -le 10 ]]; do
      if LLM_LOCALAI_ENABLED="$LLM_LOCALAI_ENABLED" \
        python3 "$PROJECT_DIR/scripts/localai/recover_quarantine.py"; then
        break
      fi
      if [[ -e "$GPU_LOCK_FILE.quarantine" ]]; then
        echo "[退出] GPU 隔离仍在收尾，1 秒后重试恢复（$recovery_attempt/10）。" >&2
        sleep 1
      fi
      recovery_attempt=$((recovery_attempt + 1))
    done
    if [[ -e "$GPU_LOCK_FILE.quarantine" ]]; then
      echo "GPU quarantine 恢复未确认，保留标记：$GPU_LOCK_FILE.quarantine" >&2
      [[ "$status" != "0" ]] || status=1
    fi
  fi
  # 服务和 backend 收尾后才停止采样，保留卸载阶段及结束快照。
  if [[ -n "$telemetry_pid" ]]; then
    kill -TERM -- "-$telemetry_pid" 2>/dev/null || true
    for _ in {1..60}; do
      kill -0 "$telemetry_pid" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL -- "-$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  exit "$status"
}

trap cleanup INT TERM HUP EXIT

if is_enabled "$MODEL_TELEMETRY_ENABLED"; then
  telemetry_root="${MODEL_TELEMETRY_ROOT:-$RUNTIME_CACHE_DIR/telemetry}"
  mkdir -p "$telemetry_root"
  MODEL_TELEMETRY_DIR="$(mktemp -d "$telemetry_root/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
  export MODEL_TELEMETRY_DIR
  # 固定入口只指向本次启动，旧批次仍独立保留。
  mkdir -p "$PROJECT_DIR/storage/.cache/runtime"
  printf '%s\n' "$MODEL_TELEMETRY_DIR" > "$PROJECT_DIR/storage/.cache/runtime/telemetry-current.tmp"
  mv "$PROJECT_DIR/storage/.cache/runtime/telemetry-current.tmp" "$PROJECT_DIR/storage/.cache/runtime/telemetry-current"
  echo "[性能记录] 已开启，目录：$MODEL_TELEMETRY_DIR"
  setsid python3 "$PROJECT_DIR/scripts/model_telemetry.py" --directory "$MODEL_TELEMETRY_DIR" \
    collect --owner-pid "$$" --interval "$MODEL_TELEMETRY_INTERVAL_SECONDS" --log "$START_LOG_FILE" &
  telemetry_pid=$!
else
  echo "[性能记录] 已关闭。"
fi

if [[ "${LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER,,}" != "localai" || "${LLM_LOCAL_AUDIO_CPP_PROVIDER,,}" != "localai" ]]; then
  echo "VoiceDesign 与 audio.cpp 模型服务固定使用 LocalAI provider。" >&2
  exit 1
fi
if ! is_enabled "$LLM_LOCALAI_ENABLED"; then
  echo "audio.cpp 模型服务必须启用 LLM_LOCALAI_ENABLED。" >&2
  exit 1
fi

python3 "$PROJECT_DIR/scripts/localai/manager.py" register-audio-cpp-candidates
if is_enabled "$LLM_LOCALAI_MANAGED"; then
  LLM_LOCALAI_OWNER_PID="$$" python3 "$PROJECT_DIR/scripts/localai/manager.py" start
  localai_managed_started=1
fi
uv run --no-sync --project "$MAIN_PROJECT_DIR" \
  python "$PROJECT_DIR/scripts/localai/preflight.py"

rm -f -- "$ORNITH_9B_BACKEND_PID_FILE"
rm -f -- "$ORNITH_MTP_9B_BACKEND_PID_FILE"
start_ornith_backend() {
  local model_alias="$1"
  local model_file="$2"
  local backend_host="$3"
  local backend_port="$4"
  local base_url="$5"
  local pid_file="$6"
  local startup_timeout="$7"
  local poll_interval="$8"
  local mmproj_file="${9:-}"
  local sleep_idle_seconds="$ORNITH_9B_SLEEP_IDLE_SECONDS"
  if [[ "$model_alias" == "Ornith-1.5-9B-MTP-GGUF-Q8" ]]; then
    sleep_idle_seconds="$ORNITH_MTP_9B_SLEEP_IDLE_SECONDS"
  fi
  local backend_args=(
    --model "$model_file"
    --host "$backend_host"
    --port "$backend_port"
    --alias "$model_alias"
    --device CUDA0
    --gpu-layers "$ORNITH_9B_N_GPU_LAYERS"
    --fit off
    --flash-attn on
    --jinja
    --ctx-size "$ORNITH_9B_CONTEXT_SIZE"
    --batch-size "$ORNITH_9B_BATCH_SIZE"
    --ubatch-size "$ORNITH_9B_UBATCH_SIZE"
    --threads "$ORNITH_9B_THREADS"
    --threads-batch "$ORNITH_9B_THREADS_BATCH"
    --spec-type "$ORNITH_9B_SPEC_TYPE"
    --spec-draft-n-max "$ORNITH_9B_DRAFT_MAX"
    --cache-ram "$ORNITH_9B_CACHE_RAM_MIB"
    --load-mode none
    --parallel 1
    --sleep-idle-seconds "$sleep_idle_seconds"
  )
  if [[ -n "$mmproj_file" && "$ORNITH_MTP_9B_VISION_ENABLED" == 1 ]]; then
    backend_args+=(--mmproj "$mmproj_file" --mmproj-offload)
  fi
  python3 -m llama_cpp_runtime.runtime startup \
    --lock-file "$GPU_LOCK_FILE" \
    --pid-file "$pid_file" \
    --host "$backend_host" \
    --port "$backend_port" \
    --model "$model_alias" \
    --base-url "$base_url" \
    --executable "$LLAMA_SERVER_BIN" \
    --lock-timeout "$GPU_LOCK_WAIT_TIMEOUT" \
    --timeout "$startup_timeout" \
    --interval "$poll_interval" \
    --cuda-release-delay "$CUDA_RELEASE_DELAY" \
    -- "${backend_args[@]}"
}

ornith_9b_backend_start_attempted=1
start_ornith_backend \
  Ornith-1.5-9B-GGUF \
  "$ORNITH_9B_MODEL_FILE" \
  "$ORNITH_9B_BACKEND_HOST" \
  "$ORNITH_9B_LLAMA_CPP_PORT" \
  "$ORNITH_9B_LLAMA_BASE_URL" \
  "$ORNITH_9B_BACKEND_PID_FILE" \
  "$ORNITH_9B_STARTUP_TIMEOUT" \
  "$ORNITH_9B_SLEEP_POLL_INTERVAL"
read -r ornith_9b_llama_server_pid < "$ORNITH_9B_BACKEND_PID_FILE"
if [[ ! "$ornith_9b_llama_server_pid" =~ ^[1-9][0-9]*$ ]]; then
  echo "Ornith 9B startup helper 未返回有效 backend PID。" >&2
  exit 1
fi
ornith_mtp_9b_backend_start_attempted=1
start_ornith_backend \
  Ornith-1.5-9B-MTP-GGUF-Q8 \
  "$ORNITH_MTP_9B_MODEL_FILE" \
  "$ORNITH_MTP_9B_BACKEND_HOST" \
  "$ORNITH_MTP_9B_LLAMA_CPP_PORT" \
  "$ORNITH_MTP_9B_LLAMA_BASE_URL" \
  "$ORNITH_MTP_9B_BACKEND_PID_FILE" \
  "$ORNITH_MTP_9B_STARTUP_TIMEOUT" \
  "$ORNITH_MTP_9B_SLEEP_POLL_INTERVAL" \
  "$ORNITH_MTP_9B_MMPROJ_FILE"
read -r ornith_mtp_9b_llama_server_pid < "$ORNITH_MTP_9B_BACKEND_PID_FILE"
if [[ ! "$ornith_mtp_9b_llama_server_pid" =~ ^[1-9][0-9]*$ ]]; then
  echo "Ornith MTP 9B startup helper 未返回有效 backend PID。" >&2
  exit 1
fi

setsid uv run --no-sync --project "$MAIN_PROJECT_DIR" python "$MAIN_DIR/main.py" &
main_pid=$!
QWEN_VOICEDESIGN_HOST="$QWEN_VOICEDESIGN_HOST" \
  QWEN_VOICEDESIGN_PORT="$QWEN_VOICEDESIGN_PORT" \
  HOST="$QWEN_VOICEDESIGN_HOST" PORT="$QWEN_VOICEDESIGN_PORT" \
  setsid uv run --no-sync --project "$VOICEDESIGN_PROJECT_DIR" \
  python "$VOICEDESIGN_PROJECT_DIR/main.py" &
voicedesign_pid=$!

for index in "${!AUDIO_CPP_SERVICE_KEYS[@]}"; do
  HOST="${AUDIO_CPP_SERVICE_HOSTS[$index]}" \
    PORT="${AUDIO_CPP_SERVICE_PORTS[$index]}" \
    setsid uv run --no-sync --project "$AUDIO_CPP_RUNTIME_PROJECT_DIR" \
    python "${AUDIO_CPP_SERVICE_DIRS[$index]}/main.py" &
  audio_cpp_pids+=("$!")
done

ORNITH_9B_HOST="$ORNITH_9B_HOST" \
  ORNITH_9B_PORT="$ORNITH_9B_PORT" \
  ORNITH_9B_LLAMA_BASE_URL="$ORNITH_9B_LLAMA_BASE_URL" \
  ORNITH_9B_BACKEND_HOST="$ORNITH_9B_BACKEND_HOST" \
  ORNITH_9B_BACKEND_PID="$ornith_9b_llama_server_pid" \
  ORNITH_9B_BACKEND_PGID="$ornith_9b_llama_server_pid" \
  ORNITH_9B_LLAMA_CPP_PORT="$ORNITH_9B_LLAMA_CPP_PORT" \
  setsid uv run --no-sync --project "$MAIN_PROJECT_DIR" \
  python "$ORNITH_9B_SERVICE_DIR/main.py" &
ornith_9b_pid=$!

ORNITH_MTP_9B_HOST="$ORNITH_MTP_9B_HOST" \
  ORNITH_MTP_9B_PORT="$ORNITH_MTP_9B_PORT" \
  ORNITH_MTP_9B_LLAMA_BASE_URL="$ORNITH_MTP_9B_LLAMA_BASE_URL" \
  ORNITH_MTP_9B_BACKEND_HOST="$ORNITH_MTP_9B_BACKEND_HOST" \
  ORNITH_MTP_9B_BACKEND_PID="$ornith_mtp_9b_llama_server_pid" \
  ORNITH_MTP_9B_BACKEND_PGID="$ornith_mtp_9b_llama_server_pid" \
  ORNITH_MTP_9B_LLAMA_CPP_PORT="$ORNITH_MTP_9B_LLAMA_CPP_PORT" \
  setsid uv run --no-sync --project "$MAIN_PROJECT_DIR" \
  python "$ORNITH_MTP_9B_SERVICE_DIR/main.py" &
ornith_mtp_9b_pid=$!

HOST="$MOSS_IMAGE_HOST" PORT="$MOSS_IMAGE_PORT" \
  setsid uv run --no-sync --project "$MAIN_PROJECT_DIR" \
  python "$MOSS_IMAGE_SERVICE_DIR/main.py" &
moss_image_pid=$!

HOST="$FLUX3_IMAGE_HOST" PORT="$FLUX3_IMAGE_PORT" \
  FLUX3_IMAGE_UPLOAD_DIR="$FLUX3_IMAGE_UPLOAD_DIR" \
  FLUX3_IMAGE_MAX_UPLOAD_BYTES="$FLUX3_IMAGE_MAX_UPLOAD_BYTES" \
  setsid uv run --no-sync --project "$MAIN_PROJECT_DIR" \
  python "$FLUX3_IMAGE_SERVICE_DIR/main.py" &
flux3_image_pid=$!

setsid uv run --no-sync --project "$MAIN_PROJECT_DIR" \
  python -m uvicorn easel_bridge.app:app --app-dir "$PROJECT_DIR" \
  --host 127.0.0.1 --port "$EASEL_BRIDGE_PORT" &
easel_bridge_pid=$!

setsid bash -c '
  backend_pid="$1"
  while kill -0 "$backend_pid" 2>/dev/null; do sleep 1; done
  exit 1
' _ "$ornith_9b_llama_server_pid" &
ornith_9b_watch_pid=$!
setsid bash -c '
  backend_pid="$1"
  while kill -0 "$backend_pid" 2>/dev/null; do sleep 1; done
  exit 1
' _ "$ornith_mtp_9b_llama_server_pid" &
ornith_mtp_9b_watch_pid=$!

wait -n "$easel_bridge_pid" "$main_pid" "$voicedesign_pid" "$ornith_9b_pid" "$ornith_mtp_9b_pid" "$moss_image_pid" "$flux3_image_pid" "$ornith_9b_watch_pid" "$ornith_mtp_9b_watch_pid" "${audio_cpp_pids[@]}"
