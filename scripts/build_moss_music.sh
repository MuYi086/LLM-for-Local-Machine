#!/usr/bin/env bash
# 构建音乐理解 CLI；权重和第三方源码均保留在仓库外，不在请求期间下载或编译。
set -euo pipefail
repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${MOSS_MUSIC_SOURCE_DIR:-$HOME/LLM-depency/HOT-Step-CPP}"
build_dir="${MOSS_MUSIC_BUILD_DIR:-$HOME/.local/share/moss-music/build}"
source_revision="2db8c0d5380ee5703e23affbcab74f4ef8e4fba7"
ggml_revision="8eb519bd4f2ad18fa9fc9ff1e46def9fd493fed0"
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$source_revision" ]] || \
   [[ "$(git -C "$source_dir/engine/ggml" rev-parse HEAD)" != "$ggml_revision" ]]; then
  echo "MOSS-Music 源码或 GGML 子模块 revision 不匹配，详见服务 README。" >&2
  exit 2
fi
cmake -S "$repository_dir/model-audio/analyse_moss-music-8b-Instruct" -B "$build_dir" \
  -DMOSS_MUSIC_SOURCE_DIR="$source_dir" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${MOSS_MUSIC_CUDA_ARCHITECTURES:-89-real}" \
  "${@}"
cmake --build "$build_dir" --target ace-caption --parallel "${MOSS_MUSIC_BUILD_JOBS:-4}"
echo "MOSS_MUSIC_CAPTION_BIN=$build_dir/ace-caption"
