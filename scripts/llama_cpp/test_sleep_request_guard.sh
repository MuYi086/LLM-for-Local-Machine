#!/usr/bin/env bash
# 在仓库外已有构建上验证真实 server_queue；不加载权重或调用 CUDA。
set -euo pipefail
source_dir="${1:?传入 llama.cpp 源码目录}"
build_dir="${2:?传入 llama.cpp 构建目录}"
mode="${3:-fixed}"
repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test_dir="$(mktemp -d)"
trap 'rm -rf "$test_dir"' EXIT
flags=()
if [[ "$mode" == fixed ]]; then flags+=(-DFIXED_REQUEST_GUARD); fi
c++ -std=c++17 -pthread "${flags[@]}" \
  -I"$source_dir/tools/server" -I"$source_dir/tools/mtmd" \
  -I"$source_dir/common" -I"$source_dir/include" -I"$source_dir/ggml/include" -I"$source_dir/vendor" \
  "$repository_dir/tests/llama_cpp/sleep_request_guard.cpp" \
  "$build_dir/tools/server/libserver-context.a" \
  -L"$build_dir/bin" -Wl,-rpath,"$build_dir/bin" \
  -lllama-common -lmtmd -lllama -lggml -lggml-base \
  -o "$test_dir/sleep-request-guard"
"$test_dir/sleep-request-guard"
