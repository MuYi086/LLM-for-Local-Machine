#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$PROJECT_DIR/model-audio/spatial_steam-audio"
BUILD_DIR="${STEAM_AUDIO_RENDERER_BUILD_DIR:-$SOURCE_DIR/build}"
STEAM_AUDIO_SDK_DIR="${STEAM_AUDIO_SDK_DIR:-$HOME/.local/share/steam-audio/4.8.1}"
export STEAM_AUDIO_SDK_DIR

if [[ ! -f "$STEAM_AUDIO_SDK_DIR/include/phonon.h" ]]; then
  echo "Steam Audio SDK 缺少 include/phonon.h: $STEAM_AUDIO_SDK_DIR" >&2
  exit 2
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "未找到 cmake（要求 3.17 或更高版本）。" >&2
  exit 2
fi

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release --parallel
ctest --test-dir "$BUILD_DIR" -C Release --output-on-failure

echo "renderer: $BUILD_DIR/steam-audio-render"

