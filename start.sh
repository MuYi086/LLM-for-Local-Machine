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
ORNITH_SERVICE_DIR="$MODEL_TEXT_DIR/ornith_1_5_35b_a3b"
QWEN36_SERVICE_DIR="$MODEL_TEXT_DIR/qwen3_6_35b_a3b"
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
export ORNITH_MODEL_FILE="${ORNITH_MODEL_FILE:-$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf}"
export ORNITH_MMPROJ_FILE="${ORNITH_MMPROJ_FILE:-$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/mmproj-Ornith-1.5-35B-A3B-F16.gguf}"
export ORNITH_MTP_FILE="${ORNITH_MTP_FILE:-$HF_MIRROR_DIR/AtomicChat/Ornith-1.5-35B-A3B/mtp-Ornith-1.5-35B-A3B-Q8_0.gguf}"
export QWEN36_MODEL_FILE="${QWEN36_MODEL_FILE:-$HF_MIRROR_DIR/unsloth/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf}"
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
    local_cuda_llama_server="$HOME/.local/src/llama.cpp/build-cuda/bin/llama-server"
    if [[ -x "$local_cuda_llama_server" ]]; then
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
export ORNITH_HOST="${ORNITH_HOST:-$HOST}"
export ORNITH_PORT="${ORNITH_PORT:-8401}"
export LLAMA_CPP_PORT="${LLAMA_CPP_PORT:-8403}"
export ORNITH_LLAMA_BASE_URL="${ORNITH_LLAMA_BASE_URL:-http://127.0.0.1:$LLAMA_CPP_PORT}"
# 默认采用同权重、32K 上下文下的实测配置，记录见 tests/model_web_comparison/README.md。
export ORNITH_CONTEXT_SIZE="${ORNITH_CONTEXT_SIZE:-32768}"
export ORNITH_BATCH_SIZE="${ORNITH_BATCH_SIZE:-512}"
export ORNITH_THREADS="${ORNITH_THREADS:-6}"
export ORNITH_THREADS_BATCH="${ORNITH_THREADS_BATCH:-8}"
export ORNITH_CACHE_TYPE_K="${ORNITH_CACHE_TYPE_K:-q8_0}"
export ORNITH_CACHE_TYPE_V="${ORNITH_CACHE_TYPE_V:-q8_0}"
export ORNITH_LOAD_MODE="${ORNITH_LOAD_MODE:-none}"
export ORNITH_SPEC_TYPE="${ORNITH_SPEC_TYPE:-draft-mtp}"
export ORNITH_SPEC_DRAFT_N_MAX="${ORNITH_SPEC_DRAFT_N_MAX:-4}"
export ORNITH_SPEC_DRAFT_CACHE_TYPE_K="${ORNITH_SPEC_DRAFT_CACHE_TYPE_K:-f16}"
export ORNITH_SPEC_DRAFT_CACHE_TYPE_V="${ORNITH_SPEC_DRAFT_CACHE_TYPE_V:-f16}"
export ORNITH_N_GPU_LAYERS="${ORNITH_N_GPU_LAYERS:-99}"
# 独立 MTP 占用额外显存；23 层 CPU 专家搭配 Q8 主 KV 是本机串行复测选定的配置。
export ORNITH_N_CPU_MOE="${ORNITH_N_CPU_MOE:-23}"
export ORNITH_SLEEP_IDLE_SECONDS="${ORNITH_SLEEP_IDLE_SECONDS:-1}"
export ORNITH_STARTUP_TIMEOUT="${ORNITH_STARTUP_TIMEOUT:-1800}"
export ORNITH_SLEEP_TIMEOUT="${ORNITH_SLEEP_TIMEOUT:-300}"
export ORNITH_SLEEP_POLL_INTERVAL="${ORNITH_SLEEP_POLL_INTERVAL:-0.25}"
export ORNITH_BACKEND_PID_FILE="${ORNITH_BACKEND_PID_FILE:-$RUNTIME_CACHE_DIR/ornith-llama-server.pid}"
export QWEN36_HOST="${QWEN36_HOST:-127.0.0.1}"
export QWEN36_PORT="${QWEN36_PORT:-8402}"
export QWEN36_BACKEND_HOST="${QWEN36_BACKEND_HOST:-127.0.0.2}"
export QWEN36_LLAMA_CPP_PORT="${QWEN36_LLAMA_CPP_PORT:-8400}"
export QWEN36_LLAMA_BASE_URL="${QWEN36_LLAMA_BASE_URL:-http://$QWEN36_BACKEND_HOST:$QWEN36_LLAMA_CPP_PORT}"
export QWEN36_CONTEXT_SIZE="${QWEN36_CONTEXT_SIZE:-32768}"
export QWEN36_BATCH_SIZE="${QWEN36_BATCH_SIZE:-512}"
export QWEN36_THREADS="${QWEN36_THREADS:-8}"
export QWEN36_THREADS_BATCH="${QWEN36_THREADS_BATCH:-8}"
export QWEN36_CACHE_TYPE_K="${QWEN36_CACHE_TYPE_K:-q8_0}"
export QWEN36_CACHE_TYPE_V="${QWEN36_CACHE_TYPE_V:-q8_0}"
export QWEN36_LOAD_MODE="${QWEN36_LOAD_MODE:-none}"
export QWEN36_SPEC_DRAFT_N_MAX="${QWEN36_SPEC_DRAFT_N_MAX:-4}"
export QWEN36_SPEC_DRAFT_CACHE_TYPE_K="${QWEN36_SPEC_DRAFT_CACHE_TYPE_K:-f16}"
export QWEN36_SPEC_DRAFT_CACHE_TYPE_V="${QWEN36_SPEC_DRAFT_CACHE_TYPE_V:-f16}"
export QWEN36_N_GPU_LAYERS="${QWEN36_N_GPU_LAYERS:-99}"
export QWEN36_N_CPU_MOE="${QWEN36_N_CPU_MOE:-23}"
export QWEN36_SLEEP_IDLE_SECONDS="${QWEN36_SLEEP_IDLE_SECONDS:-1}"
export QWEN36_STARTUP_TIMEOUT="${QWEN36_STARTUP_TIMEOUT:-1800}"
export QWEN36_SLEEP_TIMEOUT="${QWEN36_SLEEP_TIMEOUT:-300}"
export QWEN36_SLEEP_POLL_INTERVAL="${QWEN36_SLEEP_POLL_INTERVAL:-0.25}"
export QWEN36_BACKEND_PID_FILE="${QWEN36_BACKEND_PID_FILE:-$RUNTIME_CACHE_DIR/qwen3-6-llama-server.pid}"

# 受管 API 改用其他端口时，网关默认跟随；显式上游 URL 仍优先。
export EASEL_BRIDGE_QWEN_URL="${EASEL_BRIDGE_QWEN_URL:-http://127.0.0.1:$QWEN36_PORT/v1/qwen/Qwen3.6-35B-A3B-MTP}"
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
  'Ornith API' "$ORNITH_HOST" "$ORNITH_PORT"
  'Ornith llama.cpp backend' '127.0.0.1' "$LLAMA_CPP_PORT"
  'Qwen3.6 API' "$QWEN36_HOST" "$QWEN36_PORT"
  'Qwen3.6 llama.cpp backend' "$QWEN36_BACKEND_HOST" "$QWEN36_LLAMA_CPP_PORT"
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

if [[ "$ORNITH_LLAMA_BASE_URL" != "http://127.0.0.1:$LLAMA_CPP_PORT" ]]; then
  echo "ORNITH_LLAMA_BASE_URL 必须与 Ornith backend 的 127.0.0.1:$LLAMA_CPP_PORT 一致。" >&2
  exit 1
fi
case "$QWEN36_BACKEND_HOST" in
  127.*|localhost) ;;
  *)
    echo "QWEN36_BACKEND_HOST 必须是 loopback 地址。" >&2
    exit 1
    ;;
esac
if [[ "$QWEN36_LLAMA_BASE_URL" != "http://$QWEN36_BACKEND_HOST:$QWEN36_LLAMA_CPP_PORT" ]]; then
  echo "QWEN36_LLAMA_BASE_URL 必须与 Qwen3.6 backend host 和 port 一致。" >&2
  exit 1
fi

if [[ -z "$LLAMA_SERVER_BIN" || ! -x "$LLAMA_SERVER_BIN" ]]; then
  echo "找不到可执行的 llama-server；请安装 llama.cpp 或设置 LLAMA_SERVER_BIN。" >&2
  exit 1
fi
if [[ ! -s "$ORNITH_MODEL_FILE" ]]; then
  echo "Ornith GGUF 缺失或为空：$ORNITH_MODEL_FILE" >&2
  exit 1
fi
if [[ ! -s "$ORNITH_MMPROJ_FILE" ]]; then
  echo "Ornith mmproj 缺失或为空：$ORNITH_MMPROJ_FILE" >&2
  exit 1
fi
case "$ORNITH_SPEC_TYPE" in
  none|draft-mtp) ;;
  *)
    echo "ORNITH_SPEC_TYPE 必须是 none 或 draft-mtp。" >&2
    exit 1
    ;;
esac
ORNITH_SPEC_ARGS=(--spec-type "$ORNITH_SPEC_TYPE")
if [[ "$ORNITH_SPEC_TYPE" == draft-mtp ]]; then
  if [[ ! -s "$ORNITH_MTP_FILE" ]]; then
    echo "Ornith MTP 缺失或为空：$ORNITH_MTP_FILE" >&2
    exit 1
  fi
  ORNITH_SPEC_ARGS+=(
    --spec-draft-model "$ORNITH_MTP_FILE"
    --spec-draft-n-max "$ORNITH_SPEC_DRAFT_N_MAX"
    --spec-draft-type-k "$ORNITH_SPEC_DRAFT_CACHE_TYPE_K"
    --spec-draft-type-v "$ORNITH_SPEC_DRAFT_CACHE_TYPE_V"
  )
fi
if [[ ! -s "$QWEN36_MODEL_FILE" ]]; then
  echo "Qwen3.6 GGUF 缺失或为空：$QWEN36_MODEL_FILE" >&2
  exit 1
fi
for model_prefix in ORNITH QWEN36; do
  for setting in CONTEXT_SIZE BATCH_SIZE THREADS THREADS_BATCH N_GPU_LAYERS N_CPU_MOE SLEEP_IDLE_SECONDS STARTUP_TIMEOUT; do
    value_name="${model_prefix}_${setting}"
    if [[ ! "${!value_name}" =~ ^[1-9][0-9]*$ ]]; then
      echo "$value_name 必须是正整数。" >&2
      exit 1
    fi
  done
  value_name="${model_prefix}_CONTEXT_SIZE"
  if (( ${!value_name} > 32768 )); then
    echo "$value_name 不得超过 32768。" >&2
    exit 1
  fi
done
for model_prefix in ORNITH QWEN36; do
  if [[ "$model_prefix" == ORNITH && "$ORNITH_SPEC_TYPE" == none ]]; then
    continue
  fi
  value_name="${model_prefix}_SPEC_DRAFT_N_MAX"
  if [[ ! "${!value_name}" =~ ^[0-9]+$ ]]; then
    echo "$value_name 必须是非负整数。" >&2
    exit 1
  fi
  for setting in SPEC_DRAFT_CACHE_TYPE_K SPEC_DRAFT_CACHE_TYPE_V; do
    value_name="${model_prefix}_${setting}"
    case "${!value_name}" in
      f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1) ;;
      *)
        echo "$value_name 不是支持的 KV 缓存类型。" >&2
        exit 1
        ;;
    esac
  done
done
for value_name in ORNITH_SLEEP_TIMEOUT ORNITH_SLEEP_POLL_INTERVAL \
  QWEN36_SLEEP_TIMEOUT QWEN36_SLEEP_POLL_INTERVAL GPU_LOCK_WAIT_TIMEOUT; do
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
REQUIRED_LLAMA_FLAGS=(--n-cpu-moe --sleep-idle-seconds --mmproj-offload --spec-type)
if [[ "$ORNITH_SPEC_TYPE" == draft-mtp ]]; then
  REQUIRED_LLAMA_FLAGS+=(--spec-draft-model)
fi
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
echo "Ornith GGUF:           $ORNITH_MODEL_FILE"
echo "Ornith mmproj:         $ORNITH_MMPROJ_FILE"
echo "Ornith speculative:    $ORNITH_SPEC_TYPE"
if [[ "$ORNITH_SPEC_TYPE" == draft-mtp ]]; then
  echo "Ornith MTP:            $ORNITH_MTP_FILE"
fi
echo "Ornith llama-server:   $LLAMA_SERVER_BIN (CUDA0, CPU MoE=$ORNITH_N_CPU_MOE, 127.0.0.1:$LLAMA_CPP_PORT)"
echo "Ornith API:            http://$ORNITH_HOST:$ORNITH_PORT/v1/AtomicChat/Ornith-1.5-35B-A3B-AD-Q5"
echo "Qwen3.6 GGUF:          $QWEN36_MODEL_FILE"
echo "Qwen3.6 llama-server:  $LLAMA_SERVER_BIN (CUDA0, CPU MoE=$QWEN36_N_CPU_MOE, MTP, $QWEN36_BACKEND_HOST:$QWEN36_LLAMA_CPP_PORT)"
echo "Qwen3.6 API:           http://$QWEN36_HOST:$QWEN36_PORT/v1/qwen/Qwen3.6-35B-A3B-MTP"
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
ornith_pid=""
qwen36_pid=""
moss_image_pid=""
flux3_image_pid=""
easel_bridge_pid=""
llama_server_pid=""
qwen36_llama_server_pid=""
llama_watch_pid=""
qwen36_llama_watch_pid=""
ornith_backend_start_attempted=0
qwen36_backend_start_attempted=0
audio_cpp_pids=()
localai_managed_started=0

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
  local service_pids=("$easel_bridge_pid" "$main_pid" "$voicedesign_pid" "$ornith_pid" "$qwen36_pid" "$moss_image_pid" "$flux3_image_pid" "$llama_watch_pid" "$qwen36_llama_watch_pid" "$llama_server_pid" "$qwen36_llama_server_pid" "${audio_cpp_pids[@]}")
  local recovery_attempt=1
  echo "[退出] 收到停止信号，开始停止服务进程组和两个 llama.cpp 后端。"

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

  if [[ -e "$GPU_LOCK_FILE.quarantine" ]]; then
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
  local backend_models=(Ornith-1.5-35B-A3B-AD-Q5 Qwen3.6-35B-A3B-MTP)
  local backend_hosts=(127.0.0.1 "$QWEN36_BACKEND_HOST")
  local backend_ports=("$LLAMA_CPP_PORT" "$QWEN36_LLAMA_CPP_PORT")
  local backend_pid_files=("$ORNITH_BACKEND_PID_FILE" "$QWEN36_BACKEND_PID_FILE")
  local backend_attempted=("$ornith_backend_start_attempted" "$qwen36_backend_start_attempted")
  local backend_pids=("$llama_server_pid" "$qwen36_llama_server_pid")
  local backend_count="${#backend_models[@]}"
  local backend_index
  for ((backend_index = 0; backend_index < backend_count; backend_index++)); do
    local backend_pid="${backend_pids[$backend_index]}"
    if [[ -z "$backend_pid" && "${backend_attempted[$backend_index]}" == "1" && -s "${backend_pid_files[$backend_index]}" ]]; then
      read -r backend_pid < "${backend_pid_files[$backend_index]}" || true
    fi
    if [[ -n "$backend_pid" && "$backend_pid" =~ ^[1-9][0-9]*$ && ! -e "$GPU_LOCK_FILE.quarantine" ]]; then
      echo "[退出] 正在停止 ${backend_models[$backend_index]}（${backend_hosts[$backend_index]}:${backend_ports[$backend_index]}，PID $backend_pid）。"
      if python3 -m ornith_1_5_35b_a3b.runtime shutdown \
        --lock-file "$GPU_LOCK_FILE" \
        --model "${backend_models[$backend_index]}" \
        --host "${backend_hosts[$backend_index]}" \
        --port "${backend_ports[$backend_index]}" \
        --pid "$backend_pid" \
        --pgid "$backend_pid" \
        --executable "$LLAMA_SERVER_BIN" \
        --lock-timeout "$GPU_LOCK_WAIT_TIMEOUT" \
        --cleanup-timeout "${LLM_LOCALAI_CLEANUP_TIMEOUT:-30}" \
        --cuda-release-delay "$CUDA_RELEASE_DELAY"; then
        rm -f -- "${backend_pid_files[$backend_index]}"
        echo "[退出] 已确认停止 ${backend_models[$backend_index]}。"
      else
        echo "${backend_models[$backend_index]} llama.cpp backend 未确认停止，保留 GPU quarantine 标记。" >&2
        [[ "$status" != "0" ]] || status=1
      fi
    fi
  done
  exit "$status"
}

trap cleanup INT TERM HUP EXIT

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

rm -f -- "$ORNITH_BACKEND_PID_FILE"
ornith_backend_start_attempted=1
python3 -m ornith_1_5_35b_a3b.runtime startup \
  --lock-file "$GPU_LOCK_FILE" \
  --pid-file "$ORNITH_BACKEND_PID_FILE" \
  --host 127.0.0.1 \
  --port "$LLAMA_CPP_PORT" \
  --model Ornith-1.5-35B-A3B-AD-Q5 \
  --base-url "$ORNITH_LLAMA_BASE_URL" \
  --executable "$LLAMA_SERVER_BIN" \
  --lock-timeout "$GPU_LOCK_WAIT_TIMEOUT" \
  --timeout "$ORNITH_STARTUP_TIMEOUT" \
  --interval "$ORNITH_SLEEP_POLL_INTERVAL" \
  --cuda-release-delay "$CUDA_RELEASE_DELAY" \
  -- \
  --model "$ORNITH_MODEL_FILE" \
  --mmproj "$ORNITH_MMPROJ_FILE" \
  "${ORNITH_SPEC_ARGS[@]}" \
  --host 127.0.0.1 \
  --port "$LLAMA_CPP_PORT" \
  --alias Ornith-1.5-35B-A3B-AD-Q5 \
  --device CUDA0 \
  --gpu-layers "$ORNITH_N_GPU_LAYERS" \
  --n-cpu-moe "$ORNITH_N_CPU_MOE" \
  --mmproj-offload \
  --flash-attn on \
  --jinja \
  --ctx-size "$ORNITH_CONTEXT_SIZE" \
  --batch-size "$ORNITH_BATCH_SIZE" \
  --threads "$ORNITH_THREADS" \
  --threads-batch "$ORNITH_THREADS_BATCH" \
  --cache-type-k "$ORNITH_CACHE_TYPE_K" \
  --cache-type-v "$ORNITH_CACHE_TYPE_V" \
  --load-mode "$ORNITH_LOAD_MODE" \
  --parallel 1 \
  --sleep-idle-seconds "$ORNITH_SLEEP_IDLE_SECONDS"
read -r llama_server_pid < "$ORNITH_BACKEND_PID_FILE"
if [[ ! "$llama_server_pid" =~ ^[1-9][0-9]*$ ]]; then
  echo "Ornith startup helper 未返回有效 backend PID。" >&2
  exit 1
fi

rm -f -- "$QWEN36_BACKEND_PID_FILE"
qwen36_backend_start_attempted=1
python3 -m ornith_1_5_35b_a3b.runtime startup \
  --lock-file "$GPU_LOCK_FILE" \
  --pid-file "$QWEN36_BACKEND_PID_FILE" \
  --host "$QWEN36_BACKEND_HOST" \
  --port "$QWEN36_LLAMA_CPP_PORT" \
  --model Qwen3.6-35B-A3B-MTP \
  --base-url "$QWEN36_LLAMA_BASE_URL" \
  --executable "$LLAMA_SERVER_BIN" \
  --lock-timeout "$GPU_LOCK_WAIT_TIMEOUT" \
  --timeout "$QWEN36_STARTUP_TIMEOUT" \
  --interval "$QWEN36_SLEEP_POLL_INTERVAL" \
  --cuda-release-delay "$CUDA_RELEASE_DELAY" \
  -- \
  --model "$QWEN36_MODEL_FILE" \
  --host "$QWEN36_BACKEND_HOST" \
  --port "$QWEN36_LLAMA_CPP_PORT" \
  --alias Qwen3.6-35B-A3B-MTP \
  --device CUDA0 \
  --gpu-layers "$QWEN36_N_GPU_LAYERS" \
  --n-cpu-moe "$QWEN36_N_CPU_MOE" \
  --spec-type draft-mtp \
  --spec-draft-n-max "$QWEN36_SPEC_DRAFT_N_MAX" \
  --spec-draft-type-k "$QWEN36_SPEC_DRAFT_CACHE_TYPE_K" \
  --spec-draft-type-v "$QWEN36_SPEC_DRAFT_CACHE_TYPE_V" \
  --flash-attn on \
  --jinja \
  --ctx-size "$QWEN36_CONTEXT_SIZE" \
  --batch-size "$QWEN36_BATCH_SIZE" \
  --threads "$QWEN36_THREADS" \
  --threads-batch "$QWEN36_THREADS_BATCH" \
  --cache-type-k "$QWEN36_CACHE_TYPE_K" \
  --cache-type-v "$QWEN36_CACHE_TYPE_V" \
  --load-mode "$QWEN36_LOAD_MODE" \
  --parallel 1 \
  --sleep-idle-seconds "$QWEN36_SLEEP_IDLE_SECONDS"
read -r qwen36_llama_server_pid < "$QWEN36_BACKEND_PID_FILE"
if [[ ! "$qwen36_llama_server_pid" =~ ^[1-9][0-9]*$ ]]; then
  echo "Qwen3.6 startup helper 未返回有效 backend PID。" >&2
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

ORNITH_HOST="$ORNITH_HOST" \
  ORNITH_PORT="$ORNITH_PORT" \
  ORNITH_LLAMA_BASE_URL="$ORNITH_LLAMA_BASE_URL" \
  ORNITH_BACKEND_HOST=127.0.0.1 \
  ORNITH_BACKEND_PID="$llama_server_pid" \
  ORNITH_BACKEND_PGID="$llama_server_pid" \
  LLAMA_CPP_PORT="$LLAMA_CPP_PORT" \
  setsid uv run --no-sync --project "$MAIN_PROJECT_DIR" \
  python "$ORNITH_SERVICE_DIR/main.py" &
ornith_pid=$!

QWEN36_HOST="$QWEN36_HOST" \
  QWEN36_PORT="$QWEN36_PORT" \
  QWEN36_LLAMA_BASE_URL="$QWEN36_LLAMA_BASE_URL" \
  QWEN36_BACKEND_HOST="$QWEN36_BACKEND_HOST" \
  QWEN36_LLAMA_CPP_PORT="$QWEN36_LLAMA_CPP_PORT" \
  QWEN36_BACKEND_PID="$qwen36_llama_server_pid" \
  QWEN36_BACKEND_PGID="$qwen36_llama_server_pid" \
  setsid uv run --no-sync --project "$MAIN_PROJECT_DIR" \
  python "$QWEN36_SERVICE_DIR/main.py" &
qwen36_pid=$!

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
' _ "$llama_server_pid" &
llama_watch_pid=$!

setsid bash -c '
  backend_pid="$1"
  while kill -0 "$backend_pid" 2>/dev/null; do sleep 1; done
  exit 1
' _ "$qwen36_llama_server_pid" &
qwen36_llama_watch_pid=$!

wait -n "$easel_bridge_pid" "$main_pid" "$voicedesign_pid" "$ornith_pid" "$qwen36_pid" "$moss_image_pid" "$flux3_image_pid" "$llama_watch_pid" "$qwen36_llama_watch_pid" "${audio_cpp_pids[@]}"
