#!/usr/bin/env bash
# 不加载权重、不调用模型服务或 CUDA，链接仓库外既有后端代码验证缓存容量策略。
set -euo pipefail
source_dir="${1:?传入 llama.cpp 源码目录}"
build_dir="${2:?传入 llama.cpp 构建目录}"
repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test_dir="$(mktemp -d)"
trap 'rm -rf "$test_dir"' EXIT
c++ -std=c++17 -pthread \
  -I"$source_dir/tools/server" -I"$source_dir/tools/mtmd" \
  -I"$source_dir/common" -I"$source_dir/include" -I"$source_dir/ggml/include" -I"$source_dir/vendor" \
  "$repository_dir/tests/llama_cpp/sleep_prompt_cache.cpp" \
  "$build_dir/tools/server/libserver-context.a" \
  -L"$build_dir/bin" -Wl,-rpath,"$build_dir/bin" \
  -lllama-common -lmtmd -lllama -lggml -lggml-base \
  -o "$test_dir/sleep-prompt-cache"
"$test_dir/sleep-prompt-cache"
