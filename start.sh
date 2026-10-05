#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="$PROJECT_DIR/main"
# `--no-sync` 不会安装新增的本地 editable 依赖；显式暴露共享源码，确保部署前未重建
# 各服务虚拟环境时仍能启动 API。正式同步环境后该路径仍保持兼容且不覆盖 site-packages。
UNITALE_RUNTIME_SRC="$PROJECT_DIR/unitale_runtime/src"
if [[ ! -f "$UNITALE_RUNTIME_SRC/unitale_runtime/__init__.py" ]]; then
  echo "缺少共享运行时源码：$UNITALE_RUNTIME_SRC/unitale_runtime" >&2
  exit 1
fi
export PYTHONPATH="$UNITALE_RUNTIME_SRC${PYTHONPATH:+:$PYTHONPATH}"
# 所有服务共享同一个存储根目录；启动脚本只做路由和运行环境配置。
STORAGE_DIR="${STORAGE_DIR:-$PROJECT_DIR/storage}"
export STORAGE_DIR
MIMO_TTS_PROJECT_DIR="${MIMO_TTS_PROJECT_DIR:-$PROJECT_DIR/mimo_tts}"
# 控制面 wrapper 使用 Qwen3-TTS 的 uv 环境；重型 worker 使用各自项目环境。
MOSS_SOUNDEFFECT_PROJECT_DIR="${MOSS_SOUNDEFFECT_PROJECT_DIR:-$PROJECT_DIR/moss_soundEffect}"
QWEN3_TTS_PROJECT_DIR="${QWEN3_TTS_PROJECT_DIR:-$PROJECT_DIR/qwen3_tts}"
QWEN3_VOICEDESIGN_PROJECT_DIR="${QWEN3_VOICEDESIGN_PROJECT_DIR:-$PROJECT_DIR/qwen3_voiceDesign}"
MOSS_VOICEGENERATOR_PROJECT_DIR="${MOSS_VOICEGENERATOR_PROJECT_DIR:-$PROJECT_DIR/moss_voiceGenerator}"
MOSS_AUDIO_4B_THINKING_PROJECT_DIR="${MOSS_AUDIO_4B_THINKING_PROJECT_DIR:-$PROJECT_DIR/moss_audio_4b_thinking}"
CONFUCIUS4_TTS_PROJECT_DIR="${CONFUCIUS4_TTS_PROJECT_DIR:-$PROJECT_DIR/Confucius4_TTS}"
QWEN3_ASR_PROJECT_DIR="${QWEN3_ASR_PROJECT_DIR:-$PROJECT_DIR/Qwen3_ASR_1.7B}"
SEED_VC_PROJECT_DIR="${SEED_VC_PROJECT_DIR:-$PROJECT_DIR/seed-vc}"
STEP_AUDIO_EDITX_PROJECT_DIR="${STEP_AUDIO_EDITX_PROJECT_DIR:-$PROJECT_DIR/Step_Audio_EditX}"
LONGCAT_AUDIODIT_PROJECT_DIR="${LONGCAT_AUDIODIT_PROJECT_DIR:-$PROJECT_DIR/LongCat_AudioDiT_3.5B_bf16}"
DOTS_TTS_SOAR_PROJECT_DIR="${DOTS_TTS_SOAR_PROJECT_DIR:-$PROJECT_DIR/dots_tts_soar}"
STABLE_AUDIO_3_MEDIUM_PROJECT_DIR="${STABLE_AUDIO_3_MEDIUM_PROJECT_DIR:-$PROJECT_DIR/stable_audio_3_medium}"
ACESTEP_PROJECT_DIR="${ACESTEP_PROJECT_DIR:-$PROJECT_DIR/ace_step_1_5}"
VOXCPM2_PROJECT_DIR="${VOXCPM2_PROJECT_DIR:-$PROJECT_DIR/voxcpm2}"
FIRERED_TTS3_PROJECT_DIR="${FIRERED_TTS3_PROJECT_DIR:-$PROJECT_DIR/firered_tts3}"
TIGER_DNR_PROJECT_DIR="${TIGER_DNR_PROJECT_DIR:-$PROJECT_DIR/TIGER-DnR}"

export HF_MIRROR_DIR="${HF_MIRROR_DIR:-$HOME/hf-mirror}"
export QWEN_VOICEDESIGN_MODEL_DIR="${QWEN_VOICEDESIGN_MODEL_DIR:-$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
export TIGER_DNR_MODEL_DIR="${TIGER_DNR_MODEL_DIR:-$HF_MIRROR_DIR/JusperLee/TIGER-DnR}"
export TIGER_DNR_SOURCE_DIR="${TIGER_DNR_SOURCE_DIR:-$HOME/.local/share/tiger-dnr/TIGER}"
export MOSS_VOICEGENERATOR_MODEL_DIR="${MOSS_VOICEGENERATOR_MODEL_DIR:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-VoiceGenerator}"
export MOSS_AUDIO_TOKENIZER_PATH="${MOSS_AUDIO_TOKENIZER_PATH:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-Tokenizer}"
export MOSS_VOICEGENERATOR_REQUEST_TIMEOUT="${MOSS_VOICEGENERATOR_REQUEST_TIMEOUT:-900}"
export MOSS_AUDIO_4B_THINKING_MODEL_DIR="${MOSS_AUDIO_4B_THINKING_MODEL_DIR:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-4B-Thinking}"
export MOSS_AUDIO_4B_THINKING_DEPENDENCY_PATH="${MOSS_AUDIO_4B_THINKING_DEPENDENCY_PATH:-$HOME/tts-depency/MOSS-Audio}"
export MOSS_AUDIO_4B_THINKING_REQUEST_TIMEOUT="${MOSS_AUDIO_4B_THINKING_REQUEST_TIMEOUT:-900}"
export MOSS_AUDIO_4B_INSTRUCT_MODEL_DIR="${MOSS_AUDIO_4B_INSTRUCT_MODEL_DIR:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-Audio-4B-Instruct}"
export MOSS_AUDIO_4B_INSTRUCT_DEPENDENCY_PATH="${MOSS_AUDIO_4B_INSTRUCT_DEPENDENCY_PATH:-$HOME/tts-depency/MOSS-Audio}"
export MOSS_AUDIO_4B_INSTRUCT_REQUEST_TIMEOUT="${MOSS_AUDIO_4B_INSTRUCT_REQUEST_TIMEOUT:-900}"
export CONFUCIUS4_TTS_MODEL_DIR="${CONFUCIUS4_TTS_MODEL_DIR:-$HF_MIRROR_DIR/netease-youdao/Confucius4-TTS}"
export CONFUCIUS4_TTS_CODE_PATH="${CONFUCIUS4_TTS_CODE_PATH:-$HOME/tts-depency/Confucius4-TTS}"
export CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR="${CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR:-$HF_MIRROR_DIR/netease-youdao/facebook/w2v-bert-2.0}"
export CONFUCIUS4_TTS_VOCODER_MODEL_DIR="${CONFUCIUS4_TTS_VOCODER_MODEL_DIR:-$HF_MIRROR_DIR/netease-youdao/nv-community/bigvgan_v2_22khz_80band_256x}"
export CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT="${CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT:-$HF_MIRROR_DIR/netease-youdao/funasr/campplus/campplus_cn_common.bin}"
export CONFUCIUS4_TTS_REQUEST_TIMEOUT="${CONFUCIUS4_TTS_REQUEST_TIMEOUT:-900}"
export CONFUCIUS4_TTS_DEVICE="${CONFUCIUS4_TTS_DEVICE:-cuda:0}"
export QWEN3_ASR_MODEL_DIR="${QWEN3_ASR_MODEL_DIR:-$HF_MIRROR_DIR/Qwen/Qwen3-ASR-1.7B}"
export QWEN3_ASR_REQUEST_TIMEOUT="${QWEN3_ASR_REQUEST_TIMEOUT:-900}"
export SEED_VC_MODEL_DIR="${SEED_VC_MODEL_DIR:-$HF_MIRROR_DIR/Plachta/Seed-VC}"
export SEED_VC_CODE_PATH="${SEED_VC_CODE_PATH:-$HOME/tts-depency/seed-vc}"
export SEED_VC_WHISPER_MODEL_DIR="${SEED_VC_WHISPER_MODEL_DIR:-$HF_MIRROR_DIR/openai/whisper-small}"
export SEED_VC_VOCODER_MODEL_DIR="${SEED_VC_VOCODER_MODEL_DIR:-$HF_MIRROR_DIR/netease-youdao/nv-community/bigvgan_v2_22khz_80band_256x}"
export SEED_VC_F0_VOCODER_MODEL_DIR="${SEED_VC_F0_VOCODER_MODEL_DIR:-$HF_MIRROR_DIR/nvidia/bigvgan_v2_44khz_128band_512x}"
export SEED_VC_STYLE_ENCODER_CHECKPOINT="${SEED_VC_STYLE_ENCODER_CHECKPOINT:-$HF_MIRROR_DIR/netease-youdao/funasr/campplus/campplus_cn_common.bin}"
export SEED_VC_RMVPE_CHECKPOINT="${SEED_VC_RMVPE_CHECKPOINT:-$HF_MIRROR_DIR/lj1995/VoiceConversionWebUI/rmvpe.pt}"
export SEED_VC_REQUEST_TIMEOUT="${SEED_VC_REQUEST_TIMEOUT:-900}"
export SEED_VC_DEVICE="${SEED_VC_DEVICE:-cuda:0}"
export STEP_AUDIO_EDITX_MODEL_DIR="${STEP_AUDIO_EDITX_MODEL_DIR:-$HF_MIRROR_DIR/stepfun-ai/Step-Audio-EditX}"
export STEP_AUDIO_TOKENIZER_PATH="${STEP_AUDIO_TOKENIZER_PATH:-$HF_MIRROR_DIR/stepfun-ai/Step-Audio-Tokenizer}"
export STEP_AUDIO_EDITX_CODE_PATH="${STEP_AUDIO_EDITX_CODE_PATH:-$HOME/tts-depency/Step-Audio-EditX}"
export STEP_AUDIO_EDITX_REQUEST_TIMEOUT="${STEP_AUDIO_EDITX_REQUEST_TIMEOUT:-900}"
export STEP_AUDIO_EDITX_DTYPE="${STEP_AUDIO_EDITX_DTYPE:-bfloat16}"
export STEP_AUDIO_EDITX_MAX_MODEL_LEN="${STEP_AUDIO_EDITX_MAX_MODEL_LEN:-3072}"
export STEP_AUDIO_EDITX_GPU_MEMORY_UTILIZATION="${STEP_AUDIO_EDITX_GPU_MEMORY_UTILIZATION:-0.5}"
export STEP_AUDIO_EDITX_MAX_NUM_SEQS="${STEP_AUDIO_EDITX_MAX_NUM_SEQS:-1}"
export STEP_AUDIO_EDITX_COSYVOICE_DTYPE="${STEP_AUDIO_EDITX_COSYVOICE_DTYPE:-bfloat16}"
export STEP_AUDIO_EDITX_ENFORCE_EAGER="${STEP_AUDIO_EDITX_ENFORCE_EAGER:-1}"
export STEP_AUDIO_EDITX_COSYVOICE_CUDA_GRAPH="${STEP_AUDIO_EDITX_COSYVOICE_CUDA_GRAPH:-0}"
export MOSS_SOUNDEFFECT_CODE_PATH="${MOSS_SOUNDEFFECT_CODE_PATH:-$HOME/tts-depency/MOSS-TTS}"
export MOSS_SOUNDEFFECT_MODEL_DIR="${MOSS_SOUNDEFFECT_MODEL_DIR:-$HF_MIRROR_DIR/OpenMOSS-Team/MOSS-SoundEffect-v2.0}"
export MOSS_SOUNDEFFECT_DEVICE="${MOSS_SOUNDEFFECT_DEVICE:-cuda}"
export MOSS_SOUNDEFFECT_DTYPE="${MOSS_SOUNDEFFECT_DTYPE:-bfloat16}"
export MOSS_SOUNDEFFECT_DEFAULT_SECONDS="${MOSS_SOUNDEFFECT_DEFAULT_SECONDS:-10}"
export MOSS_SOUNDEFFECT_DEFAULT_STEPS="${MOSS_SOUNDEFFECT_DEFAULT_STEPS:-100}"
export MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE="${MOSS_SOUNDEFFECT_DEFAULT_CFG_SCALE:-4.0}"
export MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT="${MOSS_SOUNDEFFECT_DEFAULT_SIGMA_SHIFT:-5.0}"
export MOSS_SOUNDEFFECT_DEFAULT_SEED="${MOSS_SOUNDEFFECT_DEFAULT_SEED:-0}"
export MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO="${MOSS_SOUNDEFFECT_DISABLE_TORCHDYNAMO:-1}"
export STABLE_AUDIO_3_REPO_PATH="${STABLE_AUDIO_3_REPO_PATH:-$HOME/tts-depency/stable-audio-3}"
export STABLE_AUDIO_3_MEDIUM_MODEL_DIR="${STABLE_AUDIO_3_MEDIUM_MODEL_DIR:-$HF_MIRROR_DIR/stabilityai/stable-audio-3-medium}"
export STABLE_AUDIO_3_MEDIUM_DEVICE="${STABLE_AUDIO_3_MEDIUM_DEVICE:-cuda}"
export STABLE_AUDIO_3_MEDIUM_DTYPE="${STABLE_AUDIO_3_MEDIUM_DTYPE:-float16}"
export STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS="${STABLE_AUDIO_3_MEDIUM_DEFAULT_SECONDS:-7}"
export STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS="${STABLE_AUDIO_3_MEDIUM_DEFAULT_STEPS:-8}"
export STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE="${STABLE_AUDIO_3_MEDIUM_DEFAULT_CFG_SCALE:-1.0}"
export STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED="${STABLE_AUDIO_3_MEDIUM_DEFAULT_SEED:--1}"
export STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT="${STABLE_AUDIO_3_MEDIUM_REQUEST_TIMEOUT:-900}"
export STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN="${STABLE_AUDIO_3_MEDIUM_REQUIRE_FLASH_ATTN:-0}"
export ACESTEP_MODEL_DIR="${ACESTEP_MODEL_DIR:-$HF_MIRROR_DIR/ACE-Step/acestep-v15-xl-turbo-diffusers}"
export BGM_STORAGE_DIR="${BGM_STORAGE_DIR:-$STORAGE_DIR/bgm}"
export ACESTEP_OUTPUT_DIR="${ACESTEP_OUTPUT_DIR:-$BGM_STORAGE_DIR}"
export ACESTEP_DEVICE="${ACESTEP_DEVICE:-cuda}"
export ACESTEP_DTYPE="${ACESTEP_DTYPE:-bfloat16}"
export ACESTEP_OFFLOAD="${ACESTEP_OFFLOAD:-model}"
export ACESTEP_VAE_TILING="${ACESTEP_VAE_TILING:-1}"
export ACESTEP_DEFAULT_SECONDS="${ACESTEP_DEFAULT_SECONDS:-60}"
export ACESTEP_DEFAULT_STEPS="${ACESTEP_DEFAULT_STEPS:-8}"
export ACESTEP_DEFAULT_SEED="${ACESTEP_DEFAULT_SEED:--1}"
export ACESTEP_REQUEST_TIMEOUT="${ACESTEP_REQUEST_TIMEOUT:-1800}"
export QWEN3_TTS_MODEL_DIR="${QWEN3_TTS_MODEL_DIR:-$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
export VOXCPM2_MODEL_DIR="${VOXCPM2_MODEL_DIR:-$HF_MIRROR_DIR/openbmb/VoxCPM2}"
export FIRERED_TTS3_MODEL_DIR="${FIRERED_TTS3_MODEL_DIR:-$HF_MIRROR_DIR/drbaph/FireRedTTS3-bf16}"
export FIRERED_TTS3_CODE_PATH="${FIRERED_TTS3_CODE_PATH:-$HOME/tts-depency/FireRedTTS3}"
export QWEN_LIBS="${QWEN_LIBS:-$QWEN3_TTS_PROJECT_DIR/vendor/qwen_libs}"
export TIMBRE_STORAGE_DIR="${TIMBRE_STORAGE_DIR:-$STORAGE_DIR/timbre}"
export SOUNDEFFECT_STORAGE_DIR="${SOUNDEFFECT_STORAGE_DIR:-$STORAGE_DIR/soundEffect}"
export CLONE_STORAGE_DIR="${CLONE_STORAGE_DIR:-$STORAGE_DIR/clone}"
export CONFUCIUS4_TTS_OUTPUT_DIR="${CONFUCIUS4_TTS_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export SEED_VC_OUTPUT_DIR="${SEED_VC_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export TIGER_DNR_OUTPUT_DIR="${TIGER_DNR_OUTPUT_DIR:-$STORAGE_DIR/separation}"
export STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR="${STABLE_AUDIO_3_MEDIUM_OUTPUT_DIR:-$SOUNDEFFECT_STORAGE_DIR}"
export QWEN3_TTS_OUTPUT_DIR="${QWEN3_TTS_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export VOXCPM2_OUTPUT_DIR="${VOXCPM2_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export LONGCAT_AUDIODIT_OUTPUT_DIR="${LONGCAT_AUDIODIT_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export DOTS_TTS_SOAR_OUTPUT_DIR="${DOTS_TTS_SOAR_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export FIRERED_TTS3_REQUEST_TIMEOUT="${FIRERED_TTS3_REQUEST_TIMEOUT:-900}"
export STEP_AUDIO_EDITX_OUTPUT_DIR="${STEP_AUDIO_EDITX_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export PROMPTS_DIR="${PROMPTS_DIR:-$CLONE_STORAGE_DIR}"
export RUNTIME_CACHE_DIR="${RUNTIME_CACHE_DIR:-$STORAGE_DIR/.cache/runtime}"
export SPATIAL_EXPORT_CACHE_DIR="${SPATIAL_EXPORT_CACHE_DIR:-$RUNTIME_CACHE_DIR/spatial_exports}"
export SPATIAL_EXPORT_MAX_BYTES="${SPATIAL_EXPORT_MAX_BYTES:-536870912}"
export SPATIAL_EXPORT_TIMEOUT="${SPATIAL_EXPORT_TIMEOUT:-600}"
export SPATIAL_EXPORT_FFMPEG_BIN="${SPATIAL_EXPORT_FFMPEG_BIN:-ffmpeg}"
export STEAM_AUDIO_RENDER_CACHE_DIR="${STEAM_AUDIO_RENDER_CACHE_DIR:-$RUNTIME_CACHE_DIR/steam_audio_renders}"
export STEAM_AUDIO_RENDERER_BIN="${STEAM_AUDIO_RENDERER_BIN:-$PROJECT_DIR/steam_audio_renderer/build/steam-audio-render}"
export STEAM_AUDIO_SDK_DIR="${STEAM_AUDIO_SDK_DIR:-}"
export STEAM_AUDIO_HRTF_PATH="${STEAM_AUDIO_HRTF_PATH:-}"
export STEAM_AUDIO_RENDER_TIMEOUT="${STEAM_AUDIO_RENDER_TIMEOUT:-900}"
export STEAM_AUDIO_RENDER_MAX_ASSETS="${STEAM_AUDIO_RENDER_MAX_ASSETS:-500}"
export STEAM_AUDIO_RENDER_MAX_MANIFEST_BYTES="${STEAM_AUDIO_RENDER_MAX_MANIFEST_BYTES:-8388608}"
export STEAM_AUDIO_RENDER_MAX_BYTES="${STEAM_AUDIO_RENDER_MAX_BYTES:-2147483648}"
export STEAM_AUDIO_RENDER_THREADS="${STEAM_AUDIO_RENDER_THREADS:-4}"
export GPU_LOCK_FILE="${GPU_LOCK_FILE:-$RUNTIME_CACHE_DIR/gpu-runtime.lock}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
# LongCat 的克隆默认值集中在对应 API 顶部；Qwen3-TTS 的默认值
# 集中在 qwen3_tts/main.py；
# 此处只保留环境、模型路径等启动路由配置，避免覆盖 API 内可直接调试的值。
# VoxCPM2 的默认生成参数集中在 voxcpm2/main.py；此处不再写入默认值，
# 因此直接修改 API 顶部常量即可生效。外部显式设置的 VOXCPM2_* 环境变量会原样继承并覆盖 API 默认值。
# VoxCPM2 只使用仓库内独立 uv 项目，旧 Conda wrapper 已移除。
export LONGCAT_AUDIODIT_MODEL_DIR="${LONGCAT_AUDIODIT_MODEL_DIR:-$HF_MIRROR_DIR/drbaph/LongCat-AudioDiT-3.5B-bf16}"
export LONGCAT_AUDIODIT_REPO_PATH="${LONGCAT_AUDIODIT_REPO_PATH:-$HOME/tts-depency/LongCat-AudioDiT}"
export LONGCAT_AUDIODIT_TOKENIZER_PATH="${LONGCAT_AUDIODIT_TOKENIZER_PATH:-$HF_MIRROR_DIR/google/umt5-base}"
export DOTS_TTS_SOAR_MODEL_DIR="${DOTS_TTS_SOAR_MODEL_DIR:-$HF_MIRROR_DIR/rednote-hilab/dots.tts-soar}"
export QWEN3_TTS_USE_QWEN_LIBS="${QWEN3_TTS_USE_QWEN_LIBS:-0}"
export MOSS_SOUNDEFFECT_REQUEST_TIMEOUT="${MOSS_SOUNDEFFECT_REQUEST_TIMEOUT:-600}"
export CUDA_RELEASE_DELAY="${CUDA_RELEASE_DELAY:-2.0}"
export MIMO_BASE_URL="${MIMO_BASE_URL:-https://api.xiaomimimo.com/v1}"
export MIMO_MODEL="${MIMO_MODEL:-mimo-v2.5-tts-voicedesign}"
export MIMO_AUTH_HEADER="${MIMO_AUTH_HEADER:-api-key}"
export MIMO_TIMEOUT="${MIMO_TIMEOUT:-300}"
export MIMO_MAX_CHARS_PER_CHUNK="${MIMO_MAX_CHARS_PER_CHUNK:-300}"
export MIMO_PAUSE_MS="${MIMO_PAUSE_MS:-250}"
export MIMO_OPTIMIZE_TEXT_PREVIEW="${MIMO_OPTIMIZE_TEXT_PREVIEW:-0}"
export MIMO_MIN_REQUEST_INTERVAL_SECONDS="${MIMO_MIN_REQUEST_INTERVAL_SECONDS:-0}"
export MIMO_MAX_RETRIES="${MIMO_MAX_RETRIES:-3}"
export MIMO_RETRY_BASE_SECONDS="${MIMO_RETRY_BASE_SECONDS:-5}"
export MIMO_RETRY_MAX_SECONDS="${MIMO_RETRY_MAX_SECONDS:-60}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8300}"
export MIMO_TTS_HOST="${MIMO_TTS_HOST:-$HOST}"
export MIMO_TTS_PORT="${MIMO_TTS_PORT:-8303}"
export MIMO_TTS_PROXY_URL="${MIMO_TTS_PROXY_URL:-http://127.0.0.1:$MIMO_TTS_PORT/v1/mimo/timbre}"
export STEP_AUDIO_EDITX_HOST="${STEP_AUDIO_EDITX_HOST:-$HOST}"
export STEP_AUDIO_EDITX_PORT="${STEP_AUDIO_EDITX_PORT:-8331}"
export SOUNDEFFECT_HOST="${SOUNDEFFECT_HOST:-$HOST}"
export SOUNDEFFECT_PORT="${SOUNDEFFECT_PORT:-8312}"
export STABLE_AUDIO_3_MEDIUM_HOST="${STABLE_AUDIO_3_MEDIUM_HOST:-$HOST}"
export STABLE_AUDIO_3_MEDIUM_PORT="${STABLE_AUDIO_3_MEDIUM_PORT:-8311}"
export QWEN3_TTS_HOST="${QWEN3_TTS_HOST:-$HOST}"
export QWEN3_TTS_PORT="${QWEN3_TTS_PORT:-8321}"
export VOXCPM2_HOST="${VOXCPM2_HOST:-$HOST}"
export VOXCPM2_PORT="${VOXCPM2_PORT:-8322}"
export LONGCAT_AUDIODIT_HOST="${LONGCAT_AUDIODIT_HOST:-$HOST}"
export LONGCAT_AUDIODIT_PORT="${LONGCAT_AUDIODIT_PORT:-8323}"
export DOTS_TTS_SOAR_HOST="${DOTS_TTS_SOAR_HOST:-$HOST}"
export DOTS_TTS_SOAR_PORT="${DOTS_TTS_SOAR_PORT:-8324}"
export FIRERED_TTS3_TIMBRE_HOST="${FIRERED_TTS3_TIMBRE_HOST:-$HOST}"
export FIRERED_TTS3_TIMBRE_PORT="${FIRERED_TTS3_TIMBRE_PORT:-8304}"
export FIRERED_TTS3_CLONE_HOST="${FIRERED_TTS3_CLONE_HOST:-$HOST}"
export FIRERED_TTS3_CLONE_PORT="${FIRERED_TTS3_CLONE_PORT:-8325}"
export QWEN_VOICEDESIGN_HOST="${QWEN_VOICEDESIGN_HOST:-$HOST}"
export QWEN_VOICEDESIGN_PORT="${QWEN_VOICEDESIGN_PORT:-8301}"
export TIGER_DNR_HOST="${TIGER_DNR_HOST:-$HOST}"
export TIGER_DNR_PORT="${TIGER_DNR_PORT:-8351}"
export MOSS_VOICEGENERATOR_HOST="${MOSS_VOICEGENERATOR_HOST:-$HOST}"
export MOSS_VOICEGENERATOR_PORT="${MOSS_VOICEGENERATOR_PORT:-8302}"
export MOSS_AUDIO_4B_THINKING_HOST="${MOSS_AUDIO_4B_THINKING_HOST:-$HOST}"
export MOSS_AUDIO_4B_THINKING_PORT="${MOSS_AUDIO_4B_THINKING_PORT:-8341}"
export MOSS_AUDIO_4B_INSTRUCT_HOST="${MOSS_AUDIO_4B_INSTRUCT_HOST:-$HOST}"
export MOSS_AUDIO_4B_INSTRUCT_PORT="${MOSS_AUDIO_4B_INSTRUCT_PORT:-8342}"
export CONFUCIUS4_TTS_HOST="${CONFUCIUS4_TTS_HOST:-$HOST}"
export CONFUCIUS4_TTS_PORT="${CONFUCIUS4_TTS_PORT:-8361}"
export QWEN3_ASR_HOST="${QWEN3_ASR_HOST:-$HOST}"
export QWEN3_ASR_PORT="${QWEN3_ASR_PORT:-8371}"
export SEED_VC_HOST="${SEED_VC_HOST:-$HOST}"
export SEED_VC_PORT="${SEED_VC_PORT:-8381}"
export ACESTEP_HOST="${ACESTEP_HOST:-$HOST}"
export ACESTEP_PORT="${ACESTEP_PORT:-8313}"

export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$RUNTIME_CACHE_DIR/hf_modules}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$RUNTIME_CACHE_DIR/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUNTIME_CACHE_DIR/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_CACHE_DIR/xdg}"
# 先创建运行目录，避免服务第一次接收请求时才触发目录创建竞争。
mkdir -p "$TIMBRE_STORAGE_DIR" "$SOUNDEFFECT_STORAGE_DIR" "$BGM_STORAGE_DIR" "$CLONE_STORAGE_DIR" "$TIGER_DNR_OUTPUT_DIR" "$PROMPTS_DIR" "$SPATIAL_EXPORT_CACHE_DIR" "$STEAM_AUDIO_RENDER_CACHE_DIR" "$HF_MODULES_CACHE" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$(dirname "$GPU_LOCK_FILE")"

# Steam Audio 是 CPU 正式导出的可选能力；缺失时不阻断 17 个语音/音效进程，健康检查会明确报告 unavailable。
if [[ ! -x "$STEAM_AUDIO_RENDERER_BIN" ]]; then
  echo "警告：Steam Audio renderer 不存在或不可执行：$STEAM_AUDIO_RENDERER_BIN" >&2
  echo "      请先运行 scripts/build_steam_audio_renderer.sh；正式空间导出将返回 503。" >&2
fi

echo "=================================================="
echo "   Unitale AI local backend"
echo "=================================================="
echo "Control-plane uv project: $QWEN3_TTS_PROJECT_DIR"
echo "MiMo TTS uv project:      $MIMO_TTS_PROJECT_DIR"
echo "MOSS VoiceGenerator:  $MOSS_VOICEGENERATOR_MODEL_DIR"
echo "Step-Audio-EditX uv project: $STEP_AUDIO_EDITX_PROJECT_DIR"
echo "Step-Audio-EditX API: http://$STEP_AUDIO_EDITX_HOST:$STEP_AUDIO_EDITX_PORT"
echo "Step-Audio-EditX model: $STEP_AUDIO_EDITX_MODEL_DIR"
echo "Step-Audio tokenizer: $STEP_AUDIO_TOKENIZER_PATH"
echo "Step-Audio-EditX code: $STEP_AUDIO_EDITX_CODE_PATH"
echo "SoundEffect uv project: $MOSS_SOUNDEFFECT_PROJECT_DIR"
echo "SoundEffect source:  $MOSS_SOUNDEFFECT_CODE_PATH"
echo "SoundEffect model:   $MOSS_SOUNDEFFECT_MODEL_DIR"
echo "SoundEffect device:  $MOSS_SOUNDEFFECT_DEVICE ($MOSS_SOUNDEFFECT_DTYPE)"
echo "Stable Audio 3 Medium project: $STABLE_AUDIO_3_MEDIUM_PROJECT_DIR"
echo "Stable Audio 3 Medium model:  $STABLE_AUDIO_3_MEDIUM_MODEL_DIR"
echo "Stable Audio 3 source:        $STABLE_AUDIO_3_REPO_PATH"
echo "Stable Audio 3 Medium device: $STABLE_AUDIO_3_MEDIUM_DEVICE ($STABLE_AUDIO_3_MEDIUM_DTYPE)"
echo "ACE-Step uv project:           $ACESTEP_PROJECT_DIR"
echo "ACE-Step model:                $ACESTEP_MODEL_DIR"
echo "ACE-Step BGM storage:          $BGM_STORAGE_DIR"
echo "ACE-Step runtime:              $ACESTEP_DTYPE / offload=$ACESTEP_OFFLOAD"
echo "Qwen3-TTS uv project:  $QWEN3_TTS_PROJECT_DIR"
echo "Qwen3-TTS model:     $QWEN3_TTS_MODEL_DIR"
echo "Qwen VoiceDesign uv project: $QWEN3_VOICEDESIGN_PROJECT_DIR"
echo "Qwen VoiceDesign model:      $QWEN_VOICEDESIGN_MODEL_DIR"
echo "TIGER-DnR uv project: $TIGER_DNR_PROJECT_DIR"
echo "TIGER-DnR model:      $TIGER_DNR_MODEL_DIR"
echo "TIGER-DnR source:     $TIGER_DNR_SOURCE_DIR"
echo "MOSS VoiceGenerator uv project: $MOSS_VOICEGENERATOR_PROJECT_DIR"
echo "MOSS VoiceGenerator model:      $MOSS_VOICEGENERATOR_MODEL_DIR"
echo "MOSS Audio tokenizer:           $MOSS_AUDIO_TOKENIZER_PATH"
echo "MOSS-Audio-4B-Thinking uv project: $MOSS_AUDIO_4B_THINKING_PROJECT_DIR"
echo "MOSS-Audio-4B-Thinking model:      $MOSS_AUDIO_4B_THINKING_MODEL_DIR"
echo "MOSS-Audio upstream source:         $MOSS_AUDIO_4B_THINKING_DEPENDENCY_PATH"
echo "MOSS-Audio-4B-Instruct uv project:  $MOSS_AUDIO_4B_THINKING_PROJECT_DIR"
echo "MOSS-Audio-4B-Instruct model:       $MOSS_AUDIO_4B_INSTRUCT_MODEL_DIR"
echo "MOSS-Audio upstream source:         $MOSS_AUDIO_4B_INSTRUCT_DEPENDENCY_PATH"
echo "Confucius4-TTS uv project:           $CONFUCIUS4_TTS_PROJECT_DIR"
echo "Confucius4-TTS model:                $CONFUCIUS4_TTS_MODEL_DIR"
echo "Confucius4-TTS source:               $CONFUCIUS4_TTS_CODE_PATH"
echo "Confucius4-TTS Wav2Vec2-BERT:        $CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR"
echo "Confucius4-TTS BigVGAN:              $CONFUCIUS4_TTS_VOCODER_MODEL_DIR"
echo "Qwen3-ASR uv project:                $QWEN3_ASR_PROJECT_DIR"
echo "Qwen3-ASR model:                     $QWEN3_ASR_MODEL_DIR"
echo "VoxCPM2 uv project:  $VOXCPM2_PROJECT_DIR"
echo "VoxCPM2 model:       $VOXCPM2_MODEL_DIR"
echo "LongCat uv project:  $LONGCAT_AUDIODIT_PROJECT_DIR"
echo "LongCat model:       $LONGCAT_AUDIODIT_MODEL_DIR"
echo "LongCat repo:        $LONGCAT_AUDIODIT_REPO_PATH"
echo "LongCat tokenizer:   $LONGCAT_AUDIODIT_TOKENIZER_PATH"
echo "LongCat config:      managed by $LONGCAT_AUDIODIT_PROJECT_DIR/main.py"
echo "dots.tts-soar uv project: $DOTS_TTS_SOAR_PROJECT_DIR"
echo "dots.tts-soar model: $DOTS_TTS_SOAR_MODEL_DIR"
echo "dots.tts-soar config: managed by $DOTS_TTS_SOAR_PROJECT_DIR/main.py"
echo "FireRedTTS3 uv project: $FIRERED_TTS3_PROJECT_DIR"
echo "FireRedTTS3 model: $FIRERED_TTS3_MODEL_DIR"
echo "FireRedTTS3 official code: $FIRERED_TTS3_CODE_PATH"
echo "VoxCPM2 config:      managed by $VOXCPM2_PROJECT_DIR/main.py"
echo "Qwen3-TTS config:    managed by qwen3_tts/main.py"
echo "Qwen sidecar libs:   $QWEN_LIBS"
echo "MiMo base URL:       $MIMO_BASE_URL"
echo "MiMo model:          $MIMO_MODEL"
echo "MiMo API key:        $([[ -n "${MIMO_API_KEY:-}" ]] && echo configured || echo missing)"
echo "Storage root:        $STORAGE_DIR"
echo "Timbre storage:      $TIMBRE_STORAGE_DIR"
echo "SoundEffect storage: $SOUNDEFFECT_STORAGE_DIR"
echo "Clone storage:       $CLONE_STORAGE_DIR"
echo "TIGER-DnR storage:   $TIGER_DNR_OUTPUT_DIR"
echo "Reference audio dir: $PROMPTS_DIR"
echo "HF modules cache:    $HF_MODULES_CACHE"
echo "GPU lock file:       $GPU_LOCK_FILE"
echo "Spatial audio export: $SPATIAL_EXPORT_FFMPEG_BIN -> $SPATIAL_EXPORT_CACHE_DIR"
echo "Steam Audio renderer: $STEAM_AUDIO_RENDERER_BIN -> $STEAM_AUDIO_RENDER_CACHE_DIR"
echo "Main API:            http://$HOST:$PORT"
echo "Control route:       http://127.0.0.1:$PORT/v1/control"
echo "Audio export route:  http://127.0.0.1:$PORT/v1/audio/export"
echo "Formal spatial route: http://127.0.0.1:$PORT/v1/audio/spatial/render"
echo "MiMo TTS API:        http://$MIMO_TTS_HOST:$MIMO_TTS_PORT"
echo "MiMo TTS health:     http://127.0.0.1:$MIMO_TTS_PORT/v1/health"
echo "SoundEffect API:     http://$SOUNDEFFECT_HOST:$SOUNDEFFECT_PORT"
echo "SoundEffect health:  http://127.0.0.1:$SOUNDEFFECT_PORT/v1/health"
echo "Stable Audio 3 Medium API:    http://$STABLE_AUDIO_3_MEDIUM_HOST:$STABLE_AUDIO_3_MEDIUM_PORT"
echo "Stable Audio 3 Medium health: http://127.0.0.1:$STABLE_AUDIO_3_MEDIUM_PORT/v1/health"
echo "ACE-Step BGM API:             http://$ACESTEP_HOST:$ACESTEP_PORT"
echo "ACE-Step BGM health:           http://127.0.0.1:$ACESTEP_PORT/v1/health"
echo "Qwen3-TTS API:       http://$QWEN3_TTS_HOST:$QWEN3_TTS_PORT"
echo "Qwen3-TTS health:    http://127.0.0.1:$QWEN3_TTS_PORT/v1/health"
echo "Qwen VoiceDesign API: http://$QWEN_VOICEDESIGN_HOST:$QWEN_VOICEDESIGN_PORT"
echo "Qwen VoiceDesign health: http://127.0.0.1:$QWEN_VOICEDESIGN_PORT/v1/health"
echo "TIGER-DnR API: http://$TIGER_DNR_HOST:$TIGER_DNR_PORT"
echo "TIGER-DnR health: http://127.0.0.1:$TIGER_DNR_PORT/v1/health"
echo "MOSS VoiceGenerator API: http://$MOSS_VOICEGENERATOR_HOST:$MOSS_VOICEGENERATOR_PORT"
echo "MOSS VoiceGenerator health: http://127.0.0.1:$MOSS_VOICEGENERATOR_PORT/v1/health"
echo "MOSS-Audio-4B-Thinking API: http://$MOSS_AUDIO_4B_THINKING_HOST:$MOSS_AUDIO_4B_THINKING_PORT"
echo "MOSS-Audio-4B-Thinking health: http://127.0.0.1:$MOSS_AUDIO_4B_THINKING_PORT/v1/health"
echo "MOSS-Audio-4B-Instruct API: http://$MOSS_AUDIO_4B_INSTRUCT_HOST:$MOSS_AUDIO_4B_INSTRUCT_PORT"
echo "MOSS-Audio-4B-Instruct health: http://127.0.0.1:$MOSS_AUDIO_4B_INSTRUCT_PORT/v1/health"
echo "Confucius4-TTS API: http://$CONFUCIUS4_TTS_HOST:$CONFUCIUS4_TTS_PORT"
echo "Confucius4-TTS health: http://127.0.0.1:$CONFUCIUS4_TTS_PORT/v1/health"
echo "Qwen3-ASR API: http://$QWEN3_ASR_HOST:$QWEN3_ASR_PORT"
echo "Qwen3-ASR health: http://127.0.0.1:$QWEN3_ASR_PORT/v1/health"
echo "Seed-VC API: http://$SEED_VC_HOST:$SEED_VC_PORT"
echo "Seed-VC health: http://127.0.0.1:$SEED_VC_PORT/v1/health"
echo "VoxCPM2 API:         http://$VOXCPM2_HOST:$VOXCPM2_PORT"
echo "VoxCPM2 health:      http://127.0.0.1:$VOXCPM2_PORT/v1/health"
echo "LongCat health:      http://127.0.0.1:$LONGCAT_AUDIODIT_PORT/v1/health"
echo "dots.tts-soar health: http://127.0.0.1:$DOTS_TTS_SOAR_PORT/v1/health"
echo "FireRedTTS3 timbre health: http://127.0.0.1:$FIRERED_TTS3_TIMBRE_PORT/v1/health"
echo "FireRedTTS3 clone health: http://127.0.0.1:$FIRERED_TTS3_CLONE_PORT/v1/health"
echo "MOSS timbre route:   http://127.0.0.1:$MOSS_VOICEGENERATOR_PORT/v1/moss/timbre"
echo "MOSS-Audio understanding route: http://127.0.0.1:$MOSS_AUDIO_4B_THINKING_PORT/v1/mossAudioThinking/understand"
echo "MOSS-Audio Instruct route: http://127.0.0.1:$MOSS_AUDIO_4B_INSTRUCT_PORT/v1/mossAudioThinking/understand"
echo "Confucius4-TTS route: http://127.0.0.1:$CONFUCIUS4_TTS_PORT/v1/confucius4TTS/generate"
echo "Qwen3-ASR route: http://127.0.0.1:$QWEN3_ASR_PORT/v1/qwen3/asr"
echo "Seed-VC route: http://127.0.0.1:$SEED_VC_PORT/v1/seedVc/voiceConversion"
echo "MiMo timbre route:   http://127.0.0.1:$MIMO_TTS_PORT/v1/mimo/timbre"
echo "Step-Audio-EditX route: http://127.0.0.1:$STEP_AUDIO_EDITX_PORT/v1/stepAudioEditx/edit"
echo "MOSS sound-effect route: http://127.0.0.1:$SOUNDEFFECT_PORT/v1/moss/soundEffect"
echo "Stable Audio 3 Medium route: http://127.0.0.1:$STABLE_AUDIO_3_MEDIUM_PORT/v1/stableAudio/soundEffect"
echo "ACE-Step BGM route:     http://127.0.0.1:$ACESTEP_PORT/v1/aceStep/bgm"
echo "Qwen timbre route:   http://127.0.0.1:$QWEN_VOICEDESIGN_PORT/v1/qwen/timbre"
echo "TIGER-DnR route:     http://127.0.0.1:$TIGER_DNR_PORT/v1/tigerDnr/separate"
echo "Qwen3-TTS clone:     http://127.0.0.1:$QWEN3_TTS_PORT/v1/qwen/clone"
echo "VoxCPM2 clone:       http://127.0.0.1:$VOXCPM2_PORT/v1/voxcpm2/clone"
echo "LongCat clone:       http://127.0.0.1:$LONGCAT_AUDIODIT_PORT/v1/longCat/clone"
echo "dots.tts-soar clone: http://127.0.0.1:$DOTS_TTS_SOAR_PORT/v2/dotsTTS/clone"
echo "FireRedTTS3 timbre: http://127.0.0.1:$FIRERED_TTS3_TIMBRE_PORT/v1/FireRedTTS3/timbre"
echo "FireRedTTS3 clone: http://127.0.0.1:$FIRERED_TTS3_CLONE_PORT/v1/FireRedTTS3/clone"
echo ""
echo "模型接口与端口映射"
printf '%-24s %-6s %s\n' '服务' '端口' '最终接口'
printf '%-24s %-6s %s\n' '控制面' "$PORT" '/v1/control'
printf '%-24s %-6s %s\n' 'Qwen3-TTS VoiceDesign' "$QWEN_VOICEDESIGN_PORT" '/v1/qwen/timbre'
printf '%-24s %-6s %s\n' 'TIGER-DnR' "$TIGER_DNR_PORT" '/v1/tigerDnr/separate'
printf '%-24s %-6s %s\n' 'MOSS VoiceGenerator' "$MOSS_VOICEGENERATOR_PORT" '/v1/moss/timbre'
printf '%-24s %-6s %s\n' 'MOSS-Audio-4B-Thinking' "$MOSS_AUDIO_4B_THINKING_PORT" '/v1/mossAudioThinking/understand'
printf '%-24s %-6s %s\n' 'MOSS-Audio-4B-Instruct' "$MOSS_AUDIO_4B_INSTRUCT_PORT" '/v1/mossAudioThinking/understand'
printf '%-24s %-6s %s\n' 'Confucius4-TTS' "$CONFUCIUS4_TTS_PORT" '/v1/confucius4TTS/generate'
printf '%-24s %-6s %s\n' 'Qwen3-ASR-1.7B' "$QWEN3_ASR_PORT" '/v1/qwen3/asr'
printf '%-24s %-6s %s\n' 'Seed-VC' "$SEED_VC_PORT" '/v1/seedVc/voiceConversion'
printf '%-24s %-6s %s\n' 'MiMo TTS VoiceDesign' "$MIMO_TTS_PORT" '/v1/mimo/timbre'
printf '%-24s %-6s %s\n' 'Stable Audio 3 Medium' "$STABLE_AUDIO_3_MEDIUM_PORT" '/v1/stableAudio/soundEffect'
printf '%-24s %-6s %s\n' 'ACE-Step 1.5 XL Turbo' "$ACESTEP_PORT" '/v1/aceStep/bgm'
printf '%-24s %-6s %s\n' 'MOSS-SoundEffect v2' "$SOUNDEFFECT_PORT" '/v1/moss/soundEffect'
printf '%-24s %-6s %s\n' 'Qwen3-TTS Base' "$QWEN3_TTS_PORT" '/v1/qwen/clone'
printf '%-24s %-6s %s\n' 'VoxCPM2' "$VOXCPM2_PORT" '/v1/voxcpm2/clone'
printf '%-24s %-6s %s\n' 'LongCat-AudioDiT' "$LONGCAT_AUDIODIT_PORT" '/v1/longCat/clone'
printf '%-24s %-6s %s\n' 'dots.tts-soar' "$DOTS_TTS_SOAR_PORT" '/v2/dotsTTS/clone'
printf '%-24s %-6s %s\n' 'FireRedTTS3 Instruct' "$FIRERED_TTS3_TIMBRE_PORT" '/v1/FireRedTTS3/timbre'
printf '%-24s %-6s %s\n' 'FireRedTTS3 Base' "$FIRERED_TTS3_CLONE_PORT" '/v1/FireRedTTS3/clone'
printf '%-24s %-6s %s\n' 'Step-Audio-EditX' "$STEP_AUDIO_EDITX_PORT" '/v1/stepAudioEditx/edit'
echo "=================================================="

cd "$PROJECT_DIR"

main_pid=""
mimo_tts_pid=""
soundeffect_pid=""
stable_audio_3_medium_pid=""
acestep_pid=""
qwen3_tts_pid=""
voxcpm2_pid=""
longcat_audiodit_pid=""
dots_tts_soar_pid=""
firered_tts3_timbre_pid=""
firered_tts3_clone_pid=""
qwen_voicedesign_pid=""
tiger_dnr_pid=""
moss_voicegenerator_pid=""
moss_audio_4b_thinking_pid=""
moss_audio_4b_instruct_pid=""
confucius4_tts_pid=""
qwen3_asr_pid=""
seed_vc_pid=""
step_audio_editx_pid=""

cleanup() {
  local status=$?
  local pids=(
    "$main_pid"
    "$mimo_tts_pid"
    "$soundeffect_pid"
    "$stable_audio_3_medium_pid"
    "$acestep_pid"
    "$qwen3_tts_pid"
    "$qwen_voicedesign_pid"
    "$tiger_dnr_pid"
    "$moss_voicegenerator_pid"
    "$moss_audio_4b_thinking_pid"
    "$moss_audio_4b_instruct_pid"
    "$confucius4_tts_pid"
    "$qwen3_asr_pid"
    "$seed_vc_pid"
    "$step_audio_editx_pid"
    "$voxcpm2_pid"
    "$longcat_audiodit_pid"
    "$dots_tts_soar_pid"
    "$firered_tts3_timbre_pid"
    "$firered_tts3_clone_pid"
  )
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
  exit "$status"
}

trap cleanup INT TERM EXIT

# 主控制面负责 8300 控制路由，只使用 Qwen3-TTS uv 环境中的轻量 HTTP 依赖。
setsid uv run --no-sync --project "$QWEN3_TTS_PROJECT_DIR" python "$MAIN_DIR/main.py" &
main_pid=$!
# MiMo 是基于云端的 VoiceDesign 服务，拥有独立的健康检查和缓存。
MIMO_TTS_HOST="$MIMO_TTS_HOST" MIMO_TTS_PORT="$MIMO_TTS_PORT" \
  HOST="$MIMO_TTS_HOST" PORT="$MIMO_TTS_PORT" \
  setsid uv run --no-sync --project "$MIMO_TTS_PROJECT_DIR" \
  python "$MIMO_TTS_PROJECT_DIR/main.py" &
mimo_tts_pid=$!
# MOSS-SoundEffect uv 服务：使用最终约定的 8312 端口和路由。
HOST="$SOUNDEFFECT_HOST" PORT="$SOUNDEFFECT_PORT" \
  setsid uv run --no-sync --project "$MOSS_SOUNDEFFECT_PROJECT_DIR" \
  python "$MOSS_SOUNDEFFECT_PROJECT_DIR/main.py" &
soundeffect_pid=$!
# Stable Audio 3 Medium 已完整迁移到独立 uv 项目。
HOST="$STABLE_AUDIO_3_MEDIUM_HOST" PORT="$STABLE_AUDIO_3_MEDIUM_PORT" \
  setsid uv run --no-sync --project "$STABLE_AUDIO_3_MEDIUM_PROJECT_DIR" \
  python "$STABLE_AUDIO_3_MEDIUM_PROJECT_DIR/main.py" &
stable_audio_3_medium_pid=$!
# ACE-Step 1.5 是独立的一次性 worker uv 服务；启动前必须手动同步依赖，
# --no-sync 确保启动过程保持离线。
HOST="$ACESTEP_HOST" PORT="$ACESTEP_PORT" \
  setsid uv run --no-sync --project "$ACESTEP_PROJECT_DIR" \
  python "$ACESTEP_PROJECT_DIR/main.py" &
acestep_pid=$!
# Qwen3-TTS uv 服务：使用最终约定的 8321 端口和克隆路由。
HOST="$QWEN3_TTS_HOST" PORT="$QWEN3_TTS_PORT" setsid uv run --no-sync --project "$QWEN3_TTS_PROJECT_DIR" python "$QWEN3_TTS_PROJECT_DIR/main.py" &
qwen3_tts_pid=$!
# Qwen3-TTS VoiceDesign 独立 uv 服务：使用最终约定的 8301 端口。
QWEN_VOICEDESIGN_HOST="$QWEN_VOICEDESIGN_HOST" QWEN_VOICEDESIGN_PORT="$QWEN_VOICEDESIGN_PORT" \
  HOST="$QWEN_VOICEDESIGN_HOST" PORT="$QWEN_VOICEDESIGN_PORT" \
  setsid uv run --no-sync --project "$QWEN3_VOICEDESIGN_PROJECT_DIR" \
  python "$QWEN3_VOICEDESIGN_PROJECT_DIR/main.py" &
qwen_voicedesign_pid=$!
# TIGER-DnR 独立 uv 服务：以 8351 提供三 Stem 分离，模型在一次性 worker 中加载。
TIGER_DNR_HOST="$TIGER_DNR_HOST" TIGER_DNR_PORT="$TIGER_DNR_PORT" \
  HOST="$TIGER_DNR_HOST" PORT="$TIGER_DNR_PORT" \
  setsid uv run --no-sync --project "$TIGER_DNR_PROJECT_DIR" \
  python "$TIGER_DNR_PROJECT_DIR/main.py" &
tiger_dnr_pid=$!
# MOSS VoiceGenerator 独立 uv 服务：使用最终约定的 8302 端口。
MOSS_VOICEGENERATOR_HOST="$MOSS_VOICEGENERATOR_HOST" MOSS_VOICEGENERATOR_PORT="$MOSS_VOICEGENERATOR_PORT" \
  HOST="$MOSS_VOICEGENERATOR_HOST" PORT="$MOSS_VOICEGENERATOR_PORT" \
  setsid uv run --no-sync --project "$MOSS_VOICEGENERATOR_PROJECT_DIR" \
  python "$MOSS_VOICEGENERATOR_PROJECT_DIR/main.py" &
moss_voicegenerator_pid=$!
# MOSS-Audio-4B-Thinking 独立 uv 服务：上传音频并返回转写、描述或问答文本。
MOSS_AUDIO_4B_VARIANT="thinking" \
  MOSS_AUDIO_4B_THINKING_HOST="$MOSS_AUDIO_4B_THINKING_HOST" \
  MOSS_AUDIO_4B_THINKING_PORT="$MOSS_AUDIO_4B_THINKING_PORT" \
  HOST="$MOSS_AUDIO_4B_THINKING_HOST" PORT="$MOSS_AUDIO_4B_THINKING_PORT" \
  setsid uv run --no-sync --project "$MOSS_AUDIO_4B_THINKING_PROJECT_DIR" \
  python "$MOSS_AUDIO_4B_THINKING_PROJECT_DIR/main.py" &
moss_audio_4b_thinking_pid=$!
# MOSS-Audio-4B-Instruct 使用同一轻量 API/worker 结构，但显式切换到 Instruct 权重和 8342 端口。
MOSS_AUDIO_4B_VARIANT="instruct" \
  MOSS_AUDIO_4B_INSTRUCT_MODEL_DIR="$MOSS_AUDIO_4B_INSTRUCT_MODEL_DIR" \
  MOSS_AUDIO_4B_INSTRUCT_DEPENDENCY_PATH="$MOSS_AUDIO_4B_INSTRUCT_DEPENDENCY_PATH" \
  MOSS_AUDIO_4B_INSTRUCT_REQUEST_TIMEOUT="$MOSS_AUDIO_4B_INSTRUCT_REQUEST_TIMEOUT" \
  MOSS_AUDIO_4B_INSTRUCT_HOST="$MOSS_AUDIO_4B_INSTRUCT_HOST" \
  MOSS_AUDIO_4B_INSTRUCT_PORT="$MOSS_AUDIO_4B_INSTRUCT_PORT" \
  HOST="$MOSS_AUDIO_4B_INSTRUCT_HOST" PORT="$MOSS_AUDIO_4B_INSTRUCT_PORT" \
  setsid uv run --no-sync --project "$MOSS_AUDIO_4B_THINKING_PROJECT_DIR" \
  python "$MOSS_AUDIO_4B_THINKING_PROJECT_DIR/main.py" &
moss_audio_4b_instruct_pid=$!
# Confucius4-TTS 独立 uv 服务：使用 8361 端口提供参考音频零样本克隆。
CONFUCIUS4_TTS_HOST="$CONFUCIUS4_TTS_HOST" CONFUCIUS4_TTS_PORT="$CONFUCIUS4_TTS_PORT" \
  CONFUCIUS4_TTS_MODEL_DIR="$CONFUCIUS4_TTS_MODEL_DIR" \
  CONFUCIUS4_TTS_CODE_PATH="$CONFUCIUS4_TTS_CODE_PATH" \
  CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR="$CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR" \
  CONFUCIUS4_TTS_VOCODER_MODEL_DIR="$CONFUCIUS4_TTS_VOCODER_MODEL_DIR" \
  CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT="$CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT" \
  CONFUCIUS4_TTS_REQUEST_TIMEOUT="$CONFUCIUS4_TTS_REQUEST_TIMEOUT" \
  CONFUCIUS4_TTS_DEVICE="$CONFUCIUS4_TTS_DEVICE" \
  CONFUCIUS4_TTS_OUTPUT_DIR="$CONFUCIUS4_TTS_OUTPUT_DIR" \
  HOST="$CONFUCIUS4_TTS_HOST" PORT="$CONFUCIUS4_TTS_PORT" \
  setsid uv run --no-sync --project "$CONFUCIUS4_TTS_PROJECT_DIR" \
  python "$CONFUCIUS4_TTS_PROJECT_DIR/main.py" &
confucius4_tts_pid=$!
# Qwen3-ASR 独立 uv 服务：使用 8371 端口提供本地音频转写。
QWEN3_ASR_HOST="$QWEN3_ASR_HOST" QWEN3_ASR_PORT="$QWEN3_ASR_PORT" \
  QWEN3_ASR_MODEL_DIR="$QWEN3_ASR_MODEL_DIR" \
  QWEN3_ASR_REQUEST_TIMEOUT="$QWEN3_ASR_REQUEST_TIMEOUT" \
  HOST="$QWEN3_ASR_HOST" PORT="$QWEN3_ASR_PORT" \
  setsid uv run --no-sync --project "$QWEN3_ASR_PROJECT_DIR" \
  python "$QWEN3_ASR_PROJECT_DIR/main.py" &
qwen3_asr_pid=$!
# Seed-VC 独立 uv 服务：8381 音色转换，推理默认值集中在服务模块顶部。
SEED_VC_HOST="$SEED_VC_HOST" SEED_VC_PORT="$SEED_VC_PORT" \
  HOST="$SEED_VC_HOST" PORT="$SEED_VC_PORT" \
  setsid uv run --no-sync --project "$SEED_VC_PROJECT_DIR" \
  python "$SEED_VC_PROJECT_DIR/main.py" &
seed_vc_pid=$!
# Step-Audio-EditX 独立 uv 服务：完整提供上传、检查和编辑接口。
# 依赖由部署前手动执行 `uv sync --project Step_Audio_EditX --locked`；启动阶段不再联网解析。
STEP_AUDIO_EDITX_HOST="$STEP_AUDIO_EDITX_HOST" STEP_AUDIO_EDITX_PORT="$STEP_AUDIO_EDITX_PORT" \
  HOST="$STEP_AUDIO_EDITX_HOST" PORT="$STEP_AUDIO_EDITX_PORT" \
  setsid uv run --no-sync --project "$STEP_AUDIO_EDITX_PROJECT_DIR" \
  python "$STEP_AUDIO_EDITX_PROJECT_DIR/main.py" &
step_audio_editx_pid=$!
HOST="$VOXCPM2_HOST" PORT="$VOXCPM2_PORT" \
  setsid uv run --no-sync --project "$VOXCPM2_PROJECT_DIR" \
  python "$VOXCPM2_PROJECT_DIR/main.py" &
voxcpm2_pid=$!
# LongCat-AudioDiT uv 服务：使用最终约定的 8323 端口。
HOST="$LONGCAT_AUDIODIT_HOST" PORT="$LONGCAT_AUDIODIT_PORT" \
  setsid uv run --no-sync --project "$LONGCAT_AUDIODIT_PROJECT_DIR" \
  python "$LONGCAT_AUDIODIT_PROJECT_DIR/main.py" &
longcat_audiodit_pid=$!
# dots.tts-soar 使用独立 uv 项目，使用最终约定的 8324 端口。
HOST="$DOTS_TTS_SOAR_HOST" PORT="$DOTS_TTS_SOAR_PORT" \
  setsid uv run --no-sync --project "$DOTS_TTS_SOAR_PROJECT_DIR" \
    python "$DOTS_TTS_SOAR_PROJECT_DIR/main.py" &
dots_tts_soar_pid=$!
# FireRedTTS3 Instruct 音色设计服务：8304 只启动 Instruct worker 模式。
FIRERED_TTS3_MODE="timbre" FIRERED_TTS3_HOST="$FIRERED_TTS3_TIMBRE_HOST" \
  FIRERED_TTS3_PORT="$FIRERED_TTS3_TIMBRE_PORT" HOST="$FIRERED_TTS3_TIMBRE_HOST" \
  PORT="$FIRERED_TTS3_TIMBRE_PORT" setsid uv run --no-sync --project "$FIRERED_TTS3_PROJECT_DIR" \
  python "$FIRERED_TTS3_PROJECT_DIR/main.py" &
firered_tts3_timbre_pid=$!
# FireRedTTS3 Base 克隆服务：8325 只启动 Base worker 模式。
FIRERED_TTS3_MODE="clone" FIRERED_TTS3_HOST="$FIRERED_TTS3_CLONE_HOST" \
  FIRERED_TTS3_PORT="$FIRERED_TTS3_CLONE_PORT" HOST="$FIRERED_TTS3_CLONE_HOST" \
  PORT="$FIRERED_TTS3_CLONE_PORT" setsid uv run --no-sync --project "$FIRERED_TTS3_PROJECT_DIR" \
  python "$FIRERED_TTS3_PROJECT_DIR/main.py" &
firered_tts3_clone_pid=$!

wait -n "$main_pid" "$mimo_tts_pid" "$soundeffect_pid" "$stable_audio_3_medium_pid" "$acestep_pid" "$qwen3_tts_pid" "$qwen_voicedesign_pid" "$tiger_dnr_pid" "$moss_voicegenerator_pid" "$moss_audio_4b_thinking_pid" "$moss_audio_4b_instruct_pid" "$confucius4_tts_pid" "$qwen3_asr_pid" "$seed_vc_pid" "$step_audio_editx_pid" "$voxcpm2_pid" "$longcat_audiodit_pid" "$dots_tts_soar_pid" "$firered_tts3_timbre_pid" "$firered_tts3_clone_pid"
