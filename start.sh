#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="$PROJECT_DIR/main"
MAIN_PROJECT_DIR="$MAIN_DIR"
VOICEDESIGN_PROJECT_DIR="$PROJECT_DIR/model-audio/design_qwen3-tts-1.7b-voicedesign"
AUDIO_CPP_RUNTIME_PROJECT_DIR="$PROJECT_DIR/model-audio/audio_cpp_runtime"
AUDIO_CPP_RUNTIME_SRC="$AUDIO_CPP_RUNTIME_PROJECT_DIR/src"
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

export PYTHONPATH="$AUDIO_CPP_RUNTIME_SRC:$LLM_LOCAL_RUNTIME_SRC:$INFERENCE_GATEWAY_SRC${PYTHONPATH:+:$PYTHONPATH}"
export STORAGE_DIR="${STORAGE_DIR:-$PROJECT_DIR/storage}"
export RUNTIME_CACHE_DIR="${RUNTIME_CACHE_DIR:-$STORAGE_DIR/.cache/runtime}"
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
export TIMBRE_STORAGE_DIR="${TIMBRE_STORAGE_DIR:-$STORAGE_DIR/timbre}"

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
export QWEN_VOICEDESIGN_HOST="${QWEN_VOICEDESIGN_HOST:-$HOST}"
export QWEN_VOICEDESIGN_PORT="${QWEN_VOICEDESIGN_PORT:-8301}"

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
  "${VOXCPM2_HOST:-$HOST}"
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
audio_cpp_pids=()
localai_managed_started=0

cleanup() {
  local status=$?
  local pids=("$main_pid" "$voicedesign_pid" "${audio_cpp_pids[@]}")
  trap - INT TERM EXIT

  for pid in "${pids[@]}"; do
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${pids[@]}"; do
    if [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  if [[ "$localai_managed_started" == "1" ]]; then
    if ! python3 "$PROJECT_DIR/scripts/localai/manager.py" stop; then
      echo "LocalAI 进程组停止未确认，请检查 GPU 隔离标记并运行 scripts/localai/manager.py status。" >&2
      [[ "$status" != "0" ]] || status=1
    fi
  fi
  exit "$status"
}

trap cleanup INT TERM EXIT

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

wait -n "$main_pid" "$voicedesign_pid" "${audio_cpp_pids[@]}"
