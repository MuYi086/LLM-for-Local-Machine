#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="$PROJECT_DIR/main"
# `--no-sync` 不会安装新增的本地 editable 依赖；显式暴露共享源码，确保部署前未重建
# 各服务虚拟环境时仍能启动 API。正式同步环境后该路径仍保持兼容且不覆盖 site-packages。
LLM_LOCAL_RUNTIME_SRC="$PROJECT_DIR/llm_local_runtime/src"
if [[ ! -f "$LLM_LOCAL_RUNTIME_SRC/llm_local_runtime/__init__.py" ]]; then
  echo "缺少共享运行时源码：$LLM_LOCAL_RUNTIME_SRC/llm_local_runtime" >&2
  exit 1
fi
INFERENCE_GATEWAY_SRC="$PROJECT_DIR/inference_gateway/src"
LOCALAI_WORKER_BRIDGE_SRC="$PROJECT_DIR/localai_worker_bridge/src"
if [[ ! -f "$INFERENCE_GATEWAY_SRC/inference_gateway/__init__.py" ]]; then
  echo "缺少共享推理网关源码：$INFERENCE_GATEWAY_SRC/inference_gateway" >&2
  exit 1
fi
if [[ ! -f "$LOCALAI_WORKER_BRIDGE_SRC/localai_worker_bridge/__init__.py" ]]; then
  echo "缺少 LocalAI worker bridge 源码：$LOCALAI_WORKER_BRIDGE_SRC/localai_worker_bridge" >&2
  exit 1
fi
export PYTHONPATH="$LLM_LOCAL_RUNTIME_SRC:$INFERENCE_GATEWAY_SRC:$LOCALAI_WORKER_BRIDGE_SRC${PYTHONPATH:+:$PYTHONPATH}"
# 所有服务共享同一个存储根目录；启动脚本只做路由和运行环境配置。
STORAGE_DIR="${STORAGE_DIR:-$PROJECT_DIR/storage}"
export STORAGE_DIR
# 控制面 wrapper 使用 Qwen3-TTS 的 uv 环境；重型 worker 使用各自项目环境。
QWEN3_TTS_PROJECT_DIR="${QWEN3_TTS_PROJECT_DIR:-$PROJECT_DIR/model-audio/clone_qwen3-tts-1.7b-base}"
INDEXTTS25_PROJECT_DIR="${INDEXTTS25_PROJECT_DIR:-$PROJECT_DIR/model-audio/indexTTS-2.5}"
QWEN3_VOICEDESIGN_PROJECT_DIR="${QWEN3_VOICEDESIGN_PROJECT_DIR:-$PROJECT_DIR/model-audio/design_qwen3-tts-1.7b-voicedesign}"
CONFUCIUS4_TTS_PROJECT_DIR="${CONFUCIUS4_TTS_PROJECT_DIR:-$PROJECT_DIR/model-audio/clone_confucius-4}"
QWEN3_ASR_PROJECT_DIR="${QWEN3_ASR_PROJECT_DIR:-$PROJECT_DIR/model-audio/transcribe_qwen3-asr-1.7b}"
ACESTEP_PROJECT_DIR="${ACESTEP_PROJECT_DIR:-$PROJECT_DIR/model-audio/music_ace-step-1.5}"
YUE2_3B_PROJECT_DIR="${YUE2_3B_PROJECT_DIR:-$PROJECT_DIR/model-audio/music_yue2-3b}"
SHEETSAGE2_PROJECT_DIR="${SHEETSAGE2_PROJECT_DIR:-$PROJECT_DIR/model-audio/score_sheetsage-2}"
YUE2_VAE_PROJECT_DIR="${YUE2_VAE_PROJECT_DIR:-$PROJECT_DIR/model-audio/codec_yue2-vae}"
MERT_V2_FULLSONG_PROJECT_DIR="${MERT_V2_FULLSONG_PROJECT_DIR:-$PROJECT_DIR/model-audio/embedding_mert-v2-fullsong}"
VOXCPM2_PROJECT_DIR="${VOXCPM2_PROJECT_DIR:-$PROJECT_DIR/model-audio/clone_voxcpm-2}"
MSST_PROJECT_DIR="${MSST_PROJECT_DIR:-$PROJECT_DIR/model-audio/separate_msst-mega53}"
TIGER_DNR_PROJECT_DIR="${TIGER_DNR_PROJECT_DIR:-$PROJECT_DIR/model-audio/separate_tiger-dnr}"
MOSS_MUSIC_PROJECT_DIR="${MOSS_MUSIC_PROJECT_DIR:-$PROJECT_DIR/model-audio/analyse_moss-music-8b-Instruct}"
MOSS_SOUNDEFFECT_V2_PROJECT_DIR="${MOSS_SOUNDEFFECT_V2_PROJECT_DIR:-$PROJECT_DIR/model-audio/soundEffect-v2.0}"
SOULX_FLASHHEAD_PROJECT_DIR="${SOULX_FLASHHEAD_PROJECT_DIR:-$PROJECT_DIR/model-video/avatar_soulx-flashhead-1.3b}"
HUNYUAN_VIDEO_COMFY_PROJECT_DIR="${HUNYUAN_VIDEO_COMFY_PROJECT_DIR:-$PROJECT_DIR/model-video/HunyuanVideo-1.5-Comfy}"

export HF_MIRROR_DIR="${HF_MIRROR_DIR:-$HOME/hf-mirror}"
export INDEXTTS25_MODEL_DIR="${INDEXTTS25_MODEL_DIR:-$HF_MIRROR_DIR/IndexTeam/IndexTTS-2.5}"
export INDEXTTS25_CODE_PATH="${INDEXTTS25_CODE_PATH:-$HOME/LLM-depency/index-tts-2.5.0}"
export INDEXTTS25_W2V_BERT_MODEL_DIR="${INDEXTTS25_W2V_BERT_MODEL_DIR:-$HF_MIRROR_DIR/netease-youdao/facebook/w2v-bert-2.0}"
export INDEXTTS25_CAMPPLUS_PATH="${INDEXTTS25_CAMPPLUS_PATH:-$HF_MIRROR_DIR/netease-youdao/funasr/campplus/campplus_cn_common.bin}"
export INDEXTTS25_BIGVGAN_MODEL_DIR="${INDEXTTS25_BIGVGAN_MODEL_DIR:-$HF_MIRROR_DIR/netease-youdao/nv-community/bigvgan_v2_22khz_80band_256x}"
export MSST_MODEL_DIR="${MSST_MODEL_DIR:-$HF_MIRROR_DIR/oulianov/mvsep_mega_53}"
export MSST_SOURCE_DIR="${MSST_SOURCE_DIR:-$HOME/LLM-depency/Music-Source-Separation-Training}"
export MSST_OUTPUT_DIR="${MSST_OUTPUT_DIR:-$STORAGE_DIR/msst}"
export SOULX_FLASHHEAD_MODEL_DIR="${SOULX_FLASHHEAD_MODEL_DIR:-$HF_MIRROR_DIR/Soul-AILab/SoulX-FlashHead-1_3B}"
export SOULX_FLASHHEAD_WAV2VEC_DIR="${SOULX_FLASHHEAD_WAV2VEC_DIR:-$HF_MIRROR_DIR/Soul-AILab/wav2vec2-base-960h}"
export SOULX_FLASHHEAD_CODE_PATH="${SOULX_FLASHHEAD_CODE_PATH:-$HOME/LLM-depency/SoulX-FlashHead}"
export HUNYUAN_VIDEO_COMFY_MODEL_DIR="${HUNYUAN_VIDEO_COMFY_MODEL_DIR:-$HF_MIRROR_DIR/HunyuanVideo-1.5-Comfy}"
export HUNYUAN_VIDEO_COMFY_CODE_PATH="${HUNYUAN_VIDEO_COMFY_CODE_PATH:-$HOME/LLM-depency/ComfyUI}"
export MOSS_MUSIC_MODEL_DIR="${MOSS_MUSIC_MODEL_DIR:-$HF_MIRROR_DIR/OpenMOSS/moss-music}"
export MOSS_MUSIC_CAPTION_BIN="${MOSS_MUSIC_CAPTION_BIN:-$HOME/.local/share/moss-music/build/ace-caption}"
export MOSS_SOUNDEFFECT_V2_MODEL_DIR="${MOSS_SOUNDEFFECT_V2_MODEL_DIR:-$HF_MIRROR_DIR/OpenMOSS/MOSS-SoundEffect-v2.0}"
export MOSS_SOUNDEFFECT_V2_CODE_PATH="${MOSS_SOUNDEFFECT_V2_CODE_PATH:-$HOME/LLM-depency/MOSS-TTS/moss_soundeffect_v2}"
export MOSS_SOUNDEFFECT_V2_OUTPUT_DIR="${MOSS_SOUNDEFFECT_V2_OUTPUT_DIR:-$STORAGE_DIR/sound_effect_v2}"
export QWEN_VOICEDESIGN_MODEL_DIR="${QWEN_VOICEDESIGN_MODEL_DIR:-$HF_MIRROR_DIR/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
export TIGER_DNR_MODEL_DIR="${TIGER_DNR_MODEL_DIR:-$HF_MIRROR_DIR/JusperLee/TIGER-DnR}"
export TIGER_DNR_SOURCE_DIR="${TIGER_DNR_SOURCE_DIR:-$HOME/.local/share/tiger-dnr/TIGER}"
export CONFUCIUS4_TTS_MODEL_DIR="${CONFUCIUS4_TTS_MODEL_DIR:-$HF_MIRROR_DIR/netease-youdao/Confucius4-TTS}"
export CONFUCIUS4_TTS_CODE_PATH="${CONFUCIUS4_TTS_CODE_PATH:-$HOME/LLM-depency/Confucius4-TTS}"
export CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR="${CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR:-$HF_MIRROR_DIR/netease-youdao/facebook/w2v-bert-2.0}"
export CONFUCIUS4_TTS_VOCODER_MODEL_DIR="${CONFUCIUS4_TTS_VOCODER_MODEL_DIR:-$HF_MIRROR_DIR/netease-youdao/nv-community/bigvgan_v2_22khz_80band_256x}"
export CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT="${CONFUCIUS4_TTS_STYLE_ENCODER_CHECKPOINT:-$HF_MIRROR_DIR/netease-youdao/funasr/campplus/campplus_cn_common.bin}"
export CONFUCIUS4_TTS_REQUEST_TIMEOUT="${CONFUCIUS4_TTS_REQUEST_TIMEOUT:-900}"
export CONFUCIUS4_TTS_DEVICE="${CONFUCIUS4_TTS_DEVICE:-cuda:0}"
export QWEN3_ASR_MODEL_DIR="${QWEN3_ASR_MODEL_DIR:-$HF_MIRROR_DIR/Qwen/Qwen3-ASR-1.7B}"
export QWEN3_ASR_REQUEST_TIMEOUT="${QWEN3_ASR_REQUEST_TIMEOUT:-900}"
export LLM_LOCALAI_ENABLED="${LLM_LOCALAI_ENABLED:-false}"
export LLM_LOCALAI_MANAGED="${LLM_LOCALAI_MANAGED:-false}"
export LLM_LOCALAI_WORKER_BRIDGE_ENABLED="${LLM_LOCALAI_WORKER_BRIDGE_ENABLED:-false}"
export LLM_LOCALAI_BASE_URL="${LLM_LOCALAI_BASE_URL:-http://127.0.0.1:8400}"
export LLM_LOCALAI_BRIDGE_ADDRESS="${LLM_LOCALAI_BRIDGE_ADDRESS:-127.0.0.1:50052}"
export LLM_LOCAL_INFERENCE_PROVIDER="${LLM_LOCAL_INFERENCE_PROVIDER:-native}"
export LLM_LOCAL_QWEN3_TTS_PROVIDER="${LLM_LOCAL_QWEN3_TTS_PROVIDER:-$LLM_LOCAL_INFERENCE_PROVIDER}"
export LLM_LOCAL_QWEN3_TTS_MODEL="${LLM_LOCAL_QWEN3_TTS_MODEL:-llm-local.qwen3-tts.1p7b-base}"
export LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER="${LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER:-$LLM_LOCAL_INFERENCE_PROVIDER}"
export LLM_LOCAL_QWEN3_VOICEDESIGN_MODEL="${LLM_LOCAL_QWEN3_VOICEDESIGN_MODEL:-llm-local.qwen3-tts.1p7b-voicedesign}"
export LLM_LOCAL_SHEETSAGE2_PROVIDER="${LLM_LOCAL_SHEETSAGE2_PROVIDER:-$LLM_LOCAL_INFERENCE_PROVIDER}"
export LLM_LOCAL_SHEETSAGE2_MODEL="${LLM_LOCAL_SHEETSAGE2_MODEL:-llm-local.sheetsage2.worker.v1}"
export LLM_LOCAL_MERT_V2_FULLSONG_PROVIDER="${LLM_LOCAL_MERT_V2_FULLSONG_PROVIDER:-$LLM_LOCAL_INFERENCE_PROVIDER}"
export LLM_LOCAL_MERT_V2_FULLSONG_MODEL="${LLM_LOCAL_MERT_V2_FULLSONG_MODEL:-llm-local.mert-v2-fullsong.worker.v1}"
export LLM_LOCAL_YUE2_VAE_PROVIDER="${LLM_LOCAL_YUE2_VAE_PROVIDER:-$LLM_LOCAL_INFERENCE_PROVIDER}"
export LLM_LOCAL_YUE2_VAE_MODEL="${LLM_LOCAL_YUE2_VAE_MODEL:-llm-local.yue2-vae.worker.v1}"
export LLM_LOCAL_YUE2_3B_PROVIDER="${LLM_LOCAL_YUE2_3B_PROVIDER:-$LLM_LOCAL_INFERENCE_PROVIDER}"
export LLM_LOCAL_YUE2_3B_MODEL="${LLM_LOCAL_YUE2_3B_MODEL:-llm-local.yue2-3b.worker.v1}"
export YUE2_3B_REQUEST_TIMEOUT="${YUE2_3B_REQUEST_TIMEOUT:-1800}"
export LLM_LOCAL_TIGER_DNR_PROVIDER="${LLM_LOCAL_TIGER_DNR_PROVIDER:-$LLM_LOCAL_INFERENCE_PROVIDER}"
export LLM_LOCAL_TIGER_DNR_MODEL="${LLM_LOCAL_TIGER_DNR_MODEL:-llm-local.tiger-dnr.worker.v1}"
export LLM_LOCALAI_ARTIFACT_DIR="${LLM_LOCALAI_ARTIFACT_DIR:-${RUNTIME_CACHE_DIR:-$STORAGE_DIR/.cache/runtime}/localai_artifacts}"
export LLM_LOCALAI_CONNECT_TIMEOUT="${LLM_LOCALAI_CONNECT_TIMEOUT:-5}"
export LLM_LOCALAI_CLEANUP_TIMEOUT="${LLM_LOCALAI_CLEANUP_TIMEOUT:-30}"
export ACESTEP_MODEL_DIR="${ACESTEP_MODEL_DIR:-$HF_MIRROR_DIR/ACE-Step/acestep-v15-xl-turbo-diffusers}"
export YUE2_3B_MODEL_DIR="${YUE2_3B_MODEL_DIR:-$HF_MIRROR_DIR/m-a-p/YuE2-3B}"
export YUE2_VAE_MODEL_DIR="${YUE2_VAE_MODEL_DIR:-$HF_MIRROR_DIR/m-a-p/YuE2-Vae}"
export SHEETSAGE2_MODEL_DIR="${SHEETSAGE2_MODEL_DIR:-$HF_MIRROR_DIR/m-a-p/SheetSage2}"
export SHEETSAGE2_BASE_MODEL_DIR="${SHEETSAGE2_BASE_MODEL_DIR:-$HF_MIRROR_DIR/m-a-p/MERT-v2-FullSong}"
export MERT_V2_FULLSONG_MODEL_DIR="${MERT_V2_FULLSONG_MODEL_DIR:-$HF_MIRROR_DIR/m-a-p/MERT-v2-FullSong}"
export BGM_STORAGE_DIR="${BGM_STORAGE_DIR:-$STORAGE_DIR/bgm}"
export ACESTEP_OUTPUT_DIR="${ACESTEP_OUTPUT_DIR:-$BGM_STORAGE_DIR}"
export YUE2_3B_OUTPUT_DIR="${YUE2_3B_OUTPUT_DIR:-$BGM_STORAGE_DIR}"
export SHEETSAGE2_OUTPUT_DIR="${SHEETSAGE2_OUTPUT_DIR:-$STORAGE_DIR/sheetsage2}"
export YUE2_VAE_OUTPUT_DIR="${YUE2_VAE_OUTPUT_DIR:-$STORAGE_DIR/yue2_vae}"
export MERT_V2_FULLSONG_OUTPUT_DIR="${MERT_V2_FULLSONG_OUTPUT_DIR:-$STORAGE_DIR/mert_v2_fullsong}"
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
export QWEN_LIBS="${QWEN_LIBS:-$QWEN3_TTS_PROJECT_DIR/vendor/qwen_libs}"
export TIMBRE_STORAGE_DIR="${TIMBRE_STORAGE_DIR:-$STORAGE_DIR/timbre}"
export CLONE_STORAGE_DIR="${CLONE_STORAGE_DIR:-$STORAGE_DIR/clone}"
export CONFUCIUS4_TTS_OUTPUT_DIR="${CONFUCIUS4_TTS_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export TIGER_DNR_OUTPUT_DIR="${TIGER_DNR_OUTPUT_DIR:-$STORAGE_DIR/separation}"
export QWEN3_TTS_OUTPUT_DIR="${QWEN3_TTS_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export VOXCPM2_OUTPUT_DIR="${VOXCPM2_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export PROMPTS_DIR="${PROMPTS_DIR:-$CLONE_STORAGE_DIR}"
export RUNTIME_CACHE_DIR="${RUNTIME_CACHE_DIR:-$STORAGE_DIR/.cache/runtime}"
export INDEXTTS25_OUTPUT_DIR="${INDEXTTS25_OUTPUT_DIR:-$CLONE_STORAGE_DIR}"
export INDEXTTS25_WORKER_TMP_DIR="${INDEXTTS25_WORKER_TMP_DIR:-$RUNTIME_CACHE_DIR/index_tts25_worker}"
export INDEXTTS25_REQUEST_TIMEOUT="${INDEXTTS25_REQUEST_TIMEOUT:-900}"
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
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
# 集中在 model-audio/clone_qwen3-tts-1.7b-base/main.py；
# 此处只保留环境、模型路径等启动路由配置，避免覆盖 API 内可直接调试的值。
# VoxCPM2 的默认生成参数集中在 model-audio/clone_voxcpm-2/main.py；此处不再写入默认值，
# 因此直接修改 API 顶部常量即可生效。外部显式设置的 VOXCPM2_* 环境变量会原样继承并覆盖 API 默认值。
# VoxCPM2 只使用仓库内独立 uv 项目，旧 Conda wrapper 已移除。
export QWEN3_TTS_USE_QWEN_LIBS="${QWEN3_TTS_USE_QWEN_LIBS:-0}"
export CUDA_RELEASE_DELAY="${CUDA_RELEASE_DELAY:-2.0}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8300}"
export QWEN3_TTS_HOST="${QWEN3_TTS_HOST:-$HOST}"
export QWEN3_TTS_PORT="${QWEN3_TTS_PORT:-8321}"
export INDEXTTS25_HOST="${INDEXTTS25_HOST:-$HOST}"
export INDEXTTS25_PORT="${INDEXTTS25_PORT:-8323}"
export VOXCPM2_HOST="${VOXCPM2_HOST:-$HOST}"
export VOXCPM2_PORT="${VOXCPM2_PORT:-8322}"
export QWEN_VOICEDESIGN_HOST="${QWEN_VOICEDESIGN_HOST:-$HOST}"
export QWEN_VOICEDESIGN_PORT="${QWEN_VOICEDESIGN_PORT:-8301}"
export TIGER_DNR_HOST="${TIGER_DNR_HOST:-$HOST}"
export TIGER_DNR_PORT="${TIGER_DNR_PORT:-8351}"
export CONFUCIUS4_TTS_HOST="${CONFUCIUS4_TTS_HOST:-$HOST}"
export CONFUCIUS4_TTS_PORT="${CONFUCIUS4_TTS_PORT:-8361}"
export QWEN3_ASR_HOST="${QWEN3_ASR_HOST:-$HOST}"
export QWEN3_ASR_PORT="${QWEN3_ASR_PORT:-8371}"
export SOULX_FLASHHEAD_HOST="${SOULX_FLASHHEAD_HOST:-$HOST}"
export SOULX_FLASHHEAD_PORT="${SOULX_FLASHHEAD_PORT:-8391}"
export HUNYUAN_VIDEO_COMFY_HOST="${HUNYUAN_VIDEO_COMFY_HOST:-$HOST}"
export HUNYUAN_VIDEO_COMFY_PORT="${HUNYUAN_VIDEO_COMFY_PORT:-8392}"
export MOSS_MUSIC_HOST="${MOSS_MUSIC_HOST:-$HOST}"
export MOSS_MUSIC_PORT="${MOSS_MUSIC_PORT:-8310}"
export MOSS_SOUNDEFFECT_V2_HOST="${MOSS_SOUNDEFFECT_V2_HOST:-$HOST}"
export MOSS_SOUNDEFFECT_V2_PORT="${MOSS_SOUNDEFFECT_V2_PORT:-8362}"
export ACESTEP_HOST="${ACESTEP_HOST:-$HOST}"
export ACESTEP_PORT="${ACESTEP_PORT:-8313}"
export YUE2_3B_HOST="${YUE2_3B_HOST:-$HOST}"
export YUE2_3B_PORT="${YUE2_3B_PORT:-8314}"
export SHEETSAGE2_HOST="${SHEETSAGE2_HOST:-$HOST}"
export SHEETSAGE2_PORT="${SHEETSAGE2_PORT:-8315}"
export YUE2_VAE_HOST="${YUE2_VAE_HOST:-$HOST}"
export YUE2_VAE_PORT="${YUE2_VAE_PORT:-8316}"
export MERT_V2_FULLSONG_HOST="${MERT_V2_FULLSONG_HOST:-$HOST}"
export MERT_V2_FULLSONG_PORT="${MERT_V2_FULLSONG_PORT:-8317}"
export MERT_V2_FULLSONG_DEVICE="${MERT_V2_FULLSONG_DEVICE:-cuda:0}"
export MERT_V2_FULLSONG_REQUEST_TIMEOUT="${MERT_V2_FULLSONG_REQUEST_TIMEOUT:-900}"

export MSST_HOST="${MSST_HOST:-$HOST}"
export MSST_PORT="${MSST_PORT:-8380}"

# 先检查整组监听地址；重复启动或配置内端口冲突时不创建任何服务进程。
uv run --no-sync --project "$QWEN3_TTS_PROJECT_DIR" python "$MAIN_DIR/check_ports.py" \
  'Control-plane' "$HOST" "$PORT" \
  'SoulX-FlashHead' "$SOULX_FLASHHEAD_HOST" "$SOULX_FLASHHEAD_PORT" \
  'MOSS-Music' "$MOSS_MUSIC_HOST" "$MOSS_MUSIC_PORT" \
  'MOSS-SoundEffect-v2.0' "$MOSS_SOUNDEFFECT_V2_HOST" "$MOSS_SOUNDEFFECT_V2_PORT" \
  'ACE-Step' "$ACESTEP_HOST" "$ACESTEP_PORT" \
  'YuE2-3B' "$YUE2_3B_HOST" "$YUE2_3B_PORT" \
  'SheetSage2' "$SHEETSAGE2_HOST" "$SHEETSAGE2_PORT" \
  'YuE2-Vae' "$YUE2_VAE_HOST" "$YUE2_VAE_PORT" \
  'MERT-v2-FullSong' "$MERT_V2_FULLSONG_HOST" "$MERT_V2_FULLSONG_PORT" \
  'IndexTTS-2.5' "$INDEXTTS25_HOST" "$INDEXTTS25_PORT" \
  'Qwen3-TTS Base' "$QWEN3_TTS_HOST" "$QWEN3_TTS_PORT" \
  'Qwen VoiceDesign' "$QWEN_VOICEDESIGN_HOST" "$QWEN_VOICEDESIGN_PORT" \
  'MSST-Mega53' "$MSST_HOST" "$MSST_PORT" \
  'TIGER-DnR' "$TIGER_DNR_HOST" "$TIGER_DNR_PORT" \
  'Confucius4-TTS' "$CONFUCIUS4_TTS_HOST" "$CONFUCIUS4_TTS_PORT" \
  'Qwen3-ASR' "$QWEN3_ASR_HOST" "$QWEN3_ASR_PORT" \
  'VoxCPM2' "$VOXCPM2_HOST" "$VOXCPM2_PORT" \
  'HunyuanVideo-1.5-Comfy' "$HUNYUAN_VIDEO_COMFY_HOST" "$HUNYUAN_VIDEO_COMFY_PORT"

export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$RUNTIME_CACHE_DIR/hf_modules}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$RUNTIME_CACHE_DIR/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUNTIME_CACHE_DIR/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_CACHE_DIR/xdg}"
# 先创建运行目录，避免服务第一次接收请求时才触发目录创建竞争。

# Steam Audio 是 CPU 正式导出的可选能力；缺失时不阻断服务启动，健康检查会明确报告 unavailable。
if [[ ! -x "$STEAM_AUDIO_RENDERER_BIN" ]]; then
  echo "警告：Steam Audio renderer 不存在或不可执行：$STEAM_AUDIO_RENDERER_BIN" >&2
  echo "      请先运行 scripts/build_steam_audio_renderer.sh；正式空间导出将返回 503。" >&2
fi

echo "=================================================="
echo "   LLM Local Backend"
echo "=================================================="
echo "Control-plane uv project: $QWEN3_TTS_PROJECT_DIR"
echo "SoulX-FlashHead project:       $SOULX_FLASHHEAD_PROJECT_DIR"
echo "SoulX-FlashHead model:         $SOULX_FLASHHEAD_MODEL_DIR"
echo "HunyuanVideo-1.5 Comfy project: $HUNYUAN_VIDEO_COMFY_PROJECT_DIR"
echo "HunyuanVideo-1.5 Comfy model:   $HUNYUAN_VIDEO_COMFY_MODEL_DIR"
echo "HunyuanVideo-1.5 ComfyUI code:  $HUNYUAN_VIDEO_COMFY_CODE_PATH"
echo "SoulX-FlashHead wav2vec:       $SOULX_FLASHHEAD_WAV2VEC_DIR"
echo "MOSS-Music uv project:         $MOSS_MUSIC_PROJECT_DIR"
echo "MOSS-Music model:              $MOSS_MUSIC_MODEL_DIR"
echo "MOSS-Music API:                http://$MOSS_MUSIC_HOST:$MOSS_MUSIC_PORT/v1/openMoss/moss-music"
echo "MOSS-SoundEffect project:      $MOSS_SOUNDEFFECT_V2_PROJECT_DIR"
echo "MOSS-SoundEffect model:        $MOSS_SOUNDEFFECT_V2_MODEL_DIR"
echo "MOSS-SoundEffect source:       $MOSS_SOUNDEFFECT_V2_CODE_PATH"
echo "MOSS-SoundEffect API:          http://$MOSS_SOUNDEFFECT_V2_HOST:$MOSS_SOUNDEFFECT_V2_PORT/v1/moss/soundEffect-v2.0"
echo "ACE-Step uv project:           $ACESTEP_PROJECT_DIR"
echo "ACE-Step model:                $ACESTEP_MODEL_DIR"
echo "ACE-Step BGM storage:          $BGM_STORAGE_DIR"
echo "ACE-Step runtime:              $ACESTEP_DTYPE / offload=$ACESTEP_OFFLOAD"
echo "YuE2-3B uv project:            $YUE2_3B_PROJECT_DIR"
echo "YuE2-3B model:                 $YUE2_3B_MODEL_DIR"
echo "YuE2 VAE model:                $YUE2_VAE_MODEL_DIR"
echo "YuE2-3B BGM storage:           $YUE2_3B_OUTPUT_DIR"
echo "SheetSage2 uv project:         $SHEETSAGE2_PROJECT_DIR"
echo "SheetSage2 model:              $SHEETSAGE2_MODEL_DIR"
echo "MERT-v2 FullSong model:        $SHEETSAGE2_BASE_MODEL_DIR"
echo "SheetSage2 output:             $SHEETSAGE2_OUTPUT_DIR"
echo "YuE2-Vae uv project:           $YUE2_VAE_PROJECT_DIR"
echo "YuE2-Vae model:                $YUE2_VAE_MODEL_DIR"
echo "YuE2-Vae output:               $YUE2_VAE_OUTPUT_DIR"
echo "MERT-v2-FullSong uv project:   $MERT_V2_FULLSONG_PROJECT_DIR"
echo "MERT-v2-FullSong model:        $MERT_V2_FULLSONG_MODEL_DIR"
echo "MERT-v2-FullSong output:       $MERT_V2_FULLSONG_OUTPUT_DIR"
echo "Qwen3-TTS uv project:  $QWEN3_TTS_PROJECT_DIR"
echo "Qwen3-TTS model:     $QWEN3_TTS_MODEL_DIR"
echo "Qwen VoiceDesign uv project: $QWEN3_VOICEDESIGN_PROJECT_DIR"
echo "Qwen VoiceDesign model:      $QWEN_VOICEDESIGN_MODEL_DIR"
echo "MSST uv project: $MSST_PROJECT_DIR"
echo "MSST source/model: $MSST_SOURCE_DIR / $MSST_MODEL_DIR"
echo "MSST API: http://$MSST_HOST:$MSST_PORT/v1/msst/separate"
echo "TIGER-DnR uv project: $TIGER_DNR_PROJECT_DIR"
echo "TIGER-DnR model:      $TIGER_DNR_MODEL_DIR"
echo "TIGER-DnR source:     $TIGER_DNR_SOURCE_DIR"
echo "Confucius4-TTS uv project:           $CONFUCIUS4_TTS_PROJECT_DIR"
echo "Confucius4-TTS model:                $CONFUCIUS4_TTS_MODEL_DIR"
echo "Confucius4-TTS source:               $CONFUCIUS4_TTS_CODE_PATH"
echo "Confucius4-TTS Wav2Vec2-BERT:        $CONFUCIUS4_TTS_W2V_BERT_MODEL_DIR"
echo "Confucius4-TTS BigVGAN:              $CONFUCIUS4_TTS_VOCODER_MODEL_DIR"
echo "Qwen3-ASR uv project:                $QWEN3_ASR_PROJECT_DIR"
echo "Qwen3-ASR model:                     $QWEN3_ASR_MODEL_DIR"
echo "VoxCPM2 uv project:  $VOXCPM2_PROJECT_DIR"
echo "VoxCPM2 model:       $VOXCPM2_MODEL_DIR"
echo "VoxCPM2 config:      managed by $VOXCPM2_PROJECT_DIR/main.py"
echo "Qwen3-TTS config:    managed by model-audio/clone_qwen3-tts-1.7b-base/main.py"
echo "Qwen sidecar libs:   $QWEN_LIBS"
echo "LocalAI managed/base URL: $LLM_LOCALAI_MANAGED / $LLM_LOCALAI_BASE_URL"
echo "Storage root:        $STORAGE_DIR"
echo "Timbre storage:      $TIMBRE_STORAGE_DIR"
echo "Clone storage:       $CLONE_STORAGE_DIR"
echo "TIGER-DnR storage:   $TIGER_DNR_OUTPUT_DIR"
echo "Reference audio dir: $PROMPTS_DIR"
echo "HF modules cache:    $HF_MODULES_CACHE"
echo "GPU lock file:       $GPU_LOCK_FILE"
echo "Spatial audio export: $SPATIAL_EXPORT_FFMPEG_BIN -> $SPATIAL_EXPORT_CACHE_DIR"
echo "Steam Audio renderer: $STEAM_AUDIO_RENDERER_BIN -> $STEAM_AUDIO_RENDER_CACHE_DIR"
echo "Main API:            http://$HOST:$PORT"
echo "IndexTTS-2.5 project: $INDEXTTS25_PROJECT_DIR"
echo "IndexTTS-2.5 model:   $INDEXTTS25_MODEL_DIR"
echo "IndexTTS-2.5 source:  $INDEXTTS25_CODE_PATH"
echo "IndexTTS-2.5 API:     http://$INDEXTTS25_HOST:$INDEXTTS25_PORT/v1/index-team/indexTTS-2.5"
echo "Control route:       http://127.0.0.1:$PORT/v1/control"
echo "Audio export route:  http://127.0.0.1:$PORT/v1/audio/export"
echo "Formal spatial route: http://127.0.0.1:$PORT/v1/audio/spatial/render"
echo "ACE-Step BGM API:             http://$ACESTEP_HOST:$ACESTEP_PORT"
echo "ACE-Step BGM health:           http://127.0.0.1:$ACESTEP_PORT/v1/health"
echo "YuE2-3B music API:             http://$YUE2_3B_HOST:$YUE2_3B_PORT"
echo "YuE2-3B health:                http://127.0.0.1:$YUE2_3B_PORT/v1/health"
echo "SheetSage2 API:                http://$SHEETSAGE2_HOST:$SHEETSAGE2_PORT"
echo "SheetSage2 health:             http://127.0.0.1:$SHEETSAGE2_PORT/v1/health"
echo "YuE2-Vae API:                  http://$YUE2_VAE_HOST:$YUE2_VAE_PORT"
echo "YuE2-Vae health:               http://127.0.0.1:$YUE2_VAE_PORT/v1/health"
echo "MERT-v2-FullSong API:          http://$MERT_V2_FULLSONG_HOST:$MERT_V2_FULLSONG_PORT"
echo "MERT-v2-FullSong health:       http://127.0.0.1:$MERT_V2_FULLSONG_PORT/v1/health"
echo "Qwen3-TTS API:       http://$QWEN3_TTS_HOST:$QWEN3_TTS_PORT"
echo "Qwen3-TTS health:    http://127.0.0.1:$QWEN3_TTS_PORT/v1/health"
echo "Qwen VoiceDesign API: http://$QWEN_VOICEDESIGN_HOST:$QWEN_VOICEDESIGN_PORT"
echo "Qwen VoiceDesign health: http://127.0.0.1:$QWEN_VOICEDESIGN_PORT/v1/health"
echo "TIGER-DnR API: http://$TIGER_DNR_HOST:$TIGER_DNR_PORT"
echo "TIGER-DnR health: http://127.0.0.1:$TIGER_DNR_PORT/v1/health"
echo "Confucius4-TTS API: http://$CONFUCIUS4_TTS_HOST:$CONFUCIUS4_TTS_PORT"
echo "Confucius4-TTS health: http://127.0.0.1:$CONFUCIUS4_TTS_PORT/v1/health"
echo "Qwen3-ASR API: http://$QWEN3_ASR_HOST:$QWEN3_ASR_PORT"
echo "Qwen3-ASR health: http://127.0.0.1:$QWEN3_ASR_PORT/v1/health"
echo "VoxCPM2 API:         http://$VOXCPM2_HOST:$VOXCPM2_PORT"
echo "VoxCPM2 health:      http://127.0.0.1:$VOXCPM2_PORT/v1/health"
echo "Confucius4-TTS route: http://127.0.0.1:$CONFUCIUS4_TTS_PORT/v1/confucius4TTS/generate"
echo "Qwen3-ASR route: http://127.0.0.1:$QWEN3_ASR_PORT/v1/qwen3/asr"
echo "ACE-Step BGM route:     http://127.0.0.1:$ACESTEP_PORT/v1/aceStep/bgm"
echo "YuE2-3B music route:    http://127.0.0.1:$YUE2_3B_PORT/v1/m-a-p/YuE2"
echo "SheetSage2 route:       http://127.0.0.1:$SHEETSAGE2_PORT/v1/m-a-p/SheetSage2"
echo "YuE2-Vae route:         http://127.0.0.1:$YUE2_VAE_PORT/v1/m-a-p/YuE2-Vae"
echo "MERT-v2-FullSong route: http://127.0.0.1:$MERT_V2_FULLSONG_PORT/v1/m-a-p/MERT-v2-FullSong"
echo "SoulX-FlashHead route: http://127.0.0.1:$SOULX_FLASHHEAD_PORT/v1/soulX/flashHead"
echo "HunyuanVideo-1.5 route: http://127.0.0.1:$HUNYUAN_VIDEO_COMFY_PORT/v1/Tencent-Hunyuan/HunyuanVideo-1.5-Comfy"
echo "MOSS-Music route:      http://127.0.0.1:$MOSS_MUSIC_PORT/v1/openMoss/moss-music"
echo "MOSS-SoundEffect route: http://127.0.0.1:$MOSS_SOUNDEFFECT_V2_PORT/v1/moss/soundEffect-v2.0"
echo "Qwen timbre route:   http://127.0.0.1:$QWEN_VOICEDESIGN_PORT/v1/qwen/timbre"
echo "TIGER-DnR route:     http://127.0.0.1:$TIGER_DNR_PORT/v1/tigerDnr/separate"
echo "Qwen3-TTS clone:     http://127.0.0.1:$QWEN3_TTS_PORT/v1/qwen/clone"
echo "VoxCPM2 clone:       http://127.0.0.1:$VOXCPM2_PORT/v1/voxcpm2/clone"
echo ""
echo "模型接口与端口映射"
printf '%-24s %-6s %s\n' '服务' '端口' '最终接口'
printf '%-24s %-6s %s\n' '控制面' "$PORT" '/v1/control'
printf '%-24s %-6s %s\n' 'Qwen3-TTS VoiceDesign' "$QWEN_VOICEDESIGN_PORT" '/v1/qwen/timbre'
printf '%-24s %-6s %s\n' 'MOSS-Music-8B-Instruct' "$MOSS_MUSIC_PORT" '/v1/openMoss/moss-music'
printf '%-24s %-6s %s\n' 'MOSS-SoundEffect-v2.0' "$MOSS_SOUNDEFFECT_V2_PORT" '/v1/moss/soundEffect-v2.0'
printf '%-24s %-6s %s\n' 'MSST-Mega53' "$MSST_PORT" '/v1/msst/separate'
printf '%-24s %-6s %s\n' 'TIGER-DnR' "$TIGER_DNR_PORT" '/v1/tigerDnr/separate'
printf '%-24s %-6s %s\n' 'Confucius4-TTS' "$CONFUCIUS4_TTS_PORT" '/v1/confucius4TTS/generate'
printf '%-24s %-6s %s\n' 'Qwen3-ASR-1.7B' "$QWEN3_ASR_PORT" '/v1/qwen3/asr'
printf '%-24s %-6s %s\n' 'ACE-Step 1.5 XL Turbo' "$ACESTEP_PORT" '/v1/aceStep/bgm'
printf '%-24s %-6s %s\n' 'YuE2-3B' "$YUE2_3B_PORT" '/v1/m-a-p/YuE2'
printf '%-24s %-6s %s\n' 'SheetSage2' "$SHEETSAGE2_PORT" '/v1/m-a-p/SheetSage2'
printf '%-24s %-6s %s\n' 'YuE2-Vae' "$YUE2_VAE_PORT" '/v1/m-a-p/YuE2-Vae'
printf '%-24s %-6s %s\n' 'MERT-v2-FullSong' "$MERT_V2_FULLSONG_PORT" '/v1/m-a-p/MERT-v2-FullSong'
printf '%-24s %-6s %s\n' 'Qwen3-TTS Base' "$QWEN3_TTS_PORT" '/v1/qwen/clone'
printf '%-24s %-6s %s\n' 'IndexTTS-2.5' "$INDEXTTS25_PORT" '/v1/index-team/indexTTS-2.5'
printf '%-24s %-6s %s\n' 'VoxCPM2' "$VOXCPM2_PORT" '/v1/voxcpm2/clone'
printf '%-24s %-6s %s\n' 'SoulX-FlashHead-1.3B' "$SOULX_FLASHHEAD_PORT" '/v1/soulX/flashHead'
printf '%-24s %-6s %s\n' 'HunyuanVideo-1.5-Comfy' "$HUNYUAN_VIDEO_COMFY_PORT" '/v1/Tencent-Hunyuan/HunyuanVideo-1.5-Comfy'
echo "=================================================="

cd "$PROJECT_DIR"

soulx_flashhead_pid=""
hunyuan_video_comfy_pid=""
moss_music_pid=""
moss_soundeffect_v2_pid=""
main_pid=""
acestep_pid=""
yue2_3b_pid=""
sheetsage2_pid=""
yue2_vae_pid=""
mert_v2_fullsong_pid=""
qwen3_tts_pid=""
indextts25_pid=""
voxcpm2_pid=""
qwen_voicedesign_pid=""
tiger_dnr_pid=""
msst_pid=""
confucius4_tts_pid=""
qwen3_asr_pid=""
localai_managed_started=0

cleanup() {
  local status=$?
  local pids=(
    "$soulx_flashhead_pid"
    "$hunyuan_video_comfy_pid"
    "$moss_music_pid"
    "$moss_soundeffect_v2_pid"
    "$main_pid"
    "$acestep_pid"
    "$yue2_3b_pid"
    "$sheetsage2_pid"
    "$yue2_vae_pid"
    "$mert_v2_fullsong_pid"
    "$qwen3_tts_pid"
    "$indextts25_pid"
    "$qwen_voicedesign_pid"
    "$msst_pid"
    "$tiger_dnr_pid"
    "$confucius4_tts_pid"
    "$qwen3_asr_pid"
    "$voxcpm2_pid"
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
  if [[ "$localai_managed_started" == "1" ]]; then
    if ! python3 "$PROJECT_DIR/scripts/localai/manager.py" stop; then
      echo "LocalAI 进程组停止未确认，请检查 GPU 隔离标记并运行 scripts/localai/manager.py status。" >&2
      if [[ "$status" == "0" ]]; then
        status=1
      fi
    fi
  fi
  exit "$status"
}

trap cleanup INT TERM EXIT

if [[ "${LLM_LOCALAI_ENABLED,,}" == "true" || "${LLM_LOCALAI_ENABLED}" == "1" || \
  "${LLM_LOCALAI_ENABLED,,}" == "yes" || "${LLM_LOCALAI_ENABLED,,}" == "on" ]]; then
  if [[ "${LLM_LOCALAI_MANAGED,,}" == "true" || "${LLM_LOCALAI_MANAGED}" == "1" || \
    "${LLM_LOCALAI_MANAGED,,}" == "yes" || "${LLM_LOCALAI_MANAGED,,}" == "on" ]]; then
    LLM_LOCALAI_OWNER_PID="$$" python3 "$PROJECT_DIR/scripts/localai/manager.py" start
    localai_managed_started=1
  fi
  if [[ "${LLM_LOCAL_QWEN3_TTS_PROVIDER,,}" == "localai" || \
    "${LLM_LOCAL_QWEN3_VOICEDESIGN_PROVIDER,,}" == "localai" || \
    "${LLM_LOCAL_SHEETSAGE2_PROVIDER,,}" == "localai" || \
    "${LLM_LOCAL_MERT_V2_FULLSONG_PROVIDER,,}" == "localai" || \
    "${LLM_LOCAL_YUE2_VAE_PROVIDER,,}" == "localai" || \
    "${LLM_LOCAL_YUE2_3B_PROVIDER,,}" == "localai" || \
    "${LLM_LOCAL_TIGER_DNR_PROVIDER,,}" == "localai" ]]; then
    uv run --no-sync --project "$QWEN3_TTS_PROJECT_DIR" \
      python "$PROJECT_DIR/scripts/localai/preflight.py"
  fi
fi

# 主控制面负责 8300 控制路由，只使用 Qwen3-TTS uv 环境中的轻量 HTTP 依赖。
setsid uv run --no-sync --project "$QWEN3_TTS_PROJECT_DIR" python "$MAIN_DIR/main.py" &
main_pid=$!
# ACE-Step 1.5 是独立的一次性 worker uv 服务；启动前必须手动同步依赖，
# --no-sync 确保启动过程保持离线。
HOST="$ACESTEP_HOST" PORT="$ACESTEP_PORT" \
  setsid uv run --no-sync --project "$ACESTEP_PROJECT_DIR" \
  python "$ACESTEP_PROJECT_DIR/main.py" &
acestep_pid=$!
# YuE2-3B 是独立的一次性 worker uv 服务；服务使用本地 MOT/VAE 权重并通过 GPU 锁排队。
YUE2_3B_HOST="$YUE2_3B_HOST" YUE2_3B_PORT="$YUE2_3B_PORT" \
  YUE2_3B_MODEL_DIR="$YUE2_3B_MODEL_DIR" YUE2_VAE_MODEL_DIR="$YUE2_VAE_MODEL_DIR" \
  YUE2_3B_OUTPUT_DIR="$YUE2_3B_OUTPUT_DIR" \
  HOST="$YUE2_3B_HOST" PORT="$YUE2_3B_PORT" \
  setsid uv run --no-sync --project "$YUE2_3B_PROJECT_DIR" \
  python "$YUE2_3B_PROJECT_DIR/main.py" &
yue2_3b_pid=$!
# SheetSage2 独立 uv 服务：8315 接收音乐并返回 ABC、MIDI 与事件标注 ZIP。
SHEETSAGE2_HOST="$SHEETSAGE2_HOST" SHEETSAGE2_PORT="$SHEETSAGE2_PORT" \
  SHEETSAGE2_MODEL_DIR="$SHEETSAGE2_MODEL_DIR" \
  SHEETSAGE2_BASE_MODEL_DIR="$SHEETSAGE2_BASE_MODEL_DIR" \
  SHEETSAGE2_OUTPUT_DIR="$SHEETSAGE2_OUTPUT_DIR" \
  HOST="$SHEETSAGE2_HOST" PORT="$SHEETSAGE2_PORT" \
  setsid uv run --no-sync --project "$SHEETSAGE2_PROJECT_DIR" \
  python "$SHEETSAGE2_PROJECT_DIR/main.py" &
sheetsage2_pid=$!
# YuE2-Vae 在 8316 独立提供 acoustic latent 编码和 48 kHz 音频解码。
YUE2_VAE_HOST="$YUE2_VAE_HOST" YUE2_VAE_PORT="$YUE2_VAE_PORT" \
  YUE2_VAE_MODEL_DIR="$YUE2_VAE_MODEL_DIR" \
  YUE2_VAE_OUTPUT_DIR="$YUE2_VAE_OUTPUT_DIR" \
  HOST="$YUE2_VAE_HOST" PORT="$YUE2_VAE_PORT" \
  setsid uv run --no-sync --project "$YUE2_VAE_PROJECT_DIR" \
  python "$YUE2_VAE_PROJECT_DIR/main.py" &
yue2_vae_pid=$!
# MERT-v2-FullSong 独立提供完整歌曲音乐表征与逐帧特征，使用官方 Transformers 调用。
MERT_V2_FULLSONG_HOST="$MERT_V2_FULLSONG_HOST" MERT_V2_FULLSONG_PORT="$MERT_V2_FULLSONG_PORT" \
  MERT_V2_FULLSONG_MODEL_DIR="$MERT_V2_FULLSONG_MODEL_DIR" \
  MERT_V2_FULLSONG_OUTPUT_DIR="$MERT_V2_FULLSONG_OUTPUT_DIR" \
  MERT_V2_FULLSONG_DEVICE="$MERT_V2_FULLSONG_DEVICE" \
  MERT_V2_FULLSONG_REQUEST_TIMEOUT="$MERT_V2_FULLSONG_REQUEST_TIMEOUT" \
  HOST="$MERT_V2_FULLSONG_HOST" PORT="$MERT_V2_FULLSONG_PORT" \
  setsid uv run --no-sync --project "$MERT_V2_FULLSONG_PROJECT_DIR" \
  python "$MERT_V2_FULLSONG_PROJECT_DIR/main.py" &
mert_v2_fullsong_pid=$!
# MSST 重点音乐声部分离，独立 uv 环境与一次性 GPU worker。
setsid uv run --no-sync --project "$MSST_PROJECT_DIR" python "$MSST_PROJECT_DIR/main.py" &
msst_pid=$!
# Qwen3-TTS uv 服务：使用最终约定的 8321 端口和克隆路由。
HOST="$QWEN3_TTS_HOST" PORT="$QWEN3_TTS_PORT" setsid uv run --no-sync --project "$QWEN3_TTS_PROJECT_DIR" python "$QWEN3_TTS_PROJECT_DIR/main.py" &
qwen3_tts_pid=$!
# IndexTTS-2.5 使用独立 CUDA 12.8 uv 环境和一次性推理 worker。
INDEXTTS25_HOST="$INDEXTTS25_HOST" INDEXTTS25_PORT="$INDEXTTS25_PORT" \
  INDEXTTS25_MODEL_DIR="$INDEXTTS25_MODEL_DIR" INDEXTTS25_CODE_PATH="$INDEXTTS25_CODE_PATH" \
  INDEXTTS25_W2V_BERT_MODEL_DIR="$INDEXTTS25_W2V_BERT_MODEL_DIR" \
  INDEXTTS25_CAMPPLUS_PATH="$INDEXTTS25_CAMPPLUS_PATH" \
  INDEXTTS25_BIGVGAN_MODEL_DIR="$INDEXTTS25_BIGVGAN_MODEL_DIR" \
  INDEXTTS25_OUTPUT_DIR="$INDEXTTS25_OUTPUT_DIR" \
  INDEXTTS25_WORKER_TMP_DIR="$INDEXTTS25_WORKER_TMP_DIR" \
  INDEXTTS25_REQUEST_TIMEOUT="$INDEXTTS25_REQUEST_TIMEOUT" \
  HOST="$INDEXTTS25_HOST" PORT="$INDEXTTS25_PORT" \
  setsid uv run --no-sync --project "$INDEXTTS25_PROJECT_DIR" \
  python "$INDEXTTS25_PROJECT_DIR/main.py" &
indextts25_pid=$!
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
HOST="$VOXCPM2_HOST" PORT="$VOXCPM2_PORT" \
  setsid uv run --no-sync --project "$VOXCPM2_PROJECT_DIR" \
  python "$VOXCPM2_PROJECT_DIR/main.py" &
voxcpm2_pid=$!
# MOSS 音乐理解：一次性 GGUF worker；HTTP 进程不常驻加载模型。
HOST="$MOSS_MUSIC_HOST" PORT="$MOSS_MUSIC_PORT" \
  setsid uv run --no-sync --project "$MOSS_MUSIC_PROJECT_DIR" \
  python "$MOSS_MUSIC_PROJECT_DIR/main.py" &
moss_music_pid=$!
# MOSS-SoundEffect v2.0 按文本生成音效，每个请求在独立 worker 中加载官方 pipeline。
MOSS_SOUNDEFFECT_V2_HOST="$MOSS_SOUNDEFFECT_V2_HOST" \
  MOSS_SOUNDEFFECT_V2_PORT="$MOSS_SOUNDEFFECT_V2_PORT" \
  MOSS_SOUNDEFFECT_V2_MODEL_DIR="$MOSS_SOUNDEFFECT_V2_MODEL_DIR" \
  MOSS_SOUNDEFFECT_V2_CODE_PATH="$MOSS_SOUNDEFFECT_V2_CODE_PATH" \
  MOSS_SOUNDEFFECT_V2_OUTPUT_DIR="$MOSS_SOUNDEFFECT_V2_OUTPUT_DIR" \
  HOST="$MOSS_SOUNDEFFECT_V2_HOST" PORT="$MOSS_SOUNDEFFECT_V2_PORT" \
  setsid uv run --no-sync --project "$MOSS_SOUNDEFFECT_V2_PROJECT_DIR" \
  python "$MOSS_SOUNDEFFECT_V2_PROJECT_DIR/main.py" &
moss_soundeffect_v2_pid=$!
# SoulX 独立环境，仅请求时启动一次性视频 worker。
HOST="$SOULX_FLASHHEAD_HOST" PORT="$SOULX_FLASHHEAD_PORT" \
  setsid uv run --no-sync --project "$SOULX_FLASHHEAD_PROJECT_DIR" \
  python "$SOULX_FLASHHEAD_PROJECT_DIR/main.py" &
soulx_flashhead_pid=$!
# HunyuanVideo-1.5 Comfy 独立 uv API；ComfyUI 与权重只在一次性 worker 请求期间启动。
HUNYUAN_VIDEO_COMFY_HOST="$HUNYUAN_VIDEO_COMFY_HOST" \
  HUNYUAN_VIDEO_COMFY_PORT="$HUNYUAN_VIDEO_COMFY_PORT" \
  HUNYUAN_VIDEO_COMFY_MODEL_DIR="$HUNYUAN_VIDEO_COMFY_MODEL_DIR" \
  HUNYUAN_VIDEO_COMFY_CODE_PATH="$HUNYUAN_VIDEO_COMFY_CODE_PATH" \
  HOST="$HUNYUAN_VIDEO_COMFY_HOST" PORT="$HUNYUAN_VIDEO_COMFY_PORT" \
  setsid uv run --no-sync --project "$HUNYUAN_VIDEO_COMFY_PROJECT_DIR" \
  python "$HUNYUAN_VIDEO_COMFY_PROJECT_DIR/main.py" &
hunyuan_video_comfy_pid=$!
wait -n "$msst_pid" "$soulx_flashhead_pid" "$hunyuan_video_comfy_pid" "$moss_music_pid" "$moss_soundeffect_v2_pid" "$main_pid" "$acestep_pid" "$yue2_3b_pid" "$sheetsage2_pid" "$yue2_vae_pid" "$mert_v2_fullsong_pid" "$qwen3_tts_pid" "$indextts25_pid" "$voxcpm2_pid" "$qwen_voicedesign_pid" "$tiger_dnr_pid" "$confucius4_tts_pid" "$qwen3_asr_pid"
