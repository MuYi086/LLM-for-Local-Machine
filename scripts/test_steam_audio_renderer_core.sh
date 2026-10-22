#!/usr/bin/env bash
# 不依赖 Steam Audio SDK，编译并运行 Manifest 与语义映射核心测试。
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
renderer_dir="$repository_dir/model-audio/steam_audio_renderer"
test_build_dir="$(mktemp -d)"
trap 'rm -rf "$test_build_dir"' EXIT

g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -I "$renderer_dir/include" \
  "$renderer_dir/src/spatial_mapping.cpp" \
  "$renderer_dir/tests/spatial_mapping_test.cpp" \
  -o "$test_build_dir/spatial_mapping_test"
"$test_build_dir/spatial_mapping_test"

g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -I "$renderer_dir/include" \
  "$renderer_dir/src/json_value.cpp" \
  "$renderer_dir/src/render_manifest.cpp" \
  "$renderer_dir/tests/render_manifest_test.cpp" \
  -o "$test_build_dir/render_manifest_test"
"$test_build_dir/render_manifest_test"

g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -I "$renderer_dir/include" \
  "$renderer_dir/src/audio_io.cpp" \
  "$renderer_dir/tests/audio_io_test.cpp" \
  -o "$test_build_dir/audio_io_test"
"$test_build_dir/audio_io_test"
