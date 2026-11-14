#!/usr/bin/env bash
# 独立运行 Dagu；模型服务继续由仓库根目录的 start.sh 管理。
set -euo pipefail
repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DAGU_HOST=127.0.0.1
export DAGU_PORT="${DAGU_PORT:-18673}"
export DAGU_AUTH_MODE=builtin
export DAGU_DAGS_DIR="$repository_dir/dagu/dags"
export DAGU_HOME="${DAGU_HOME:-$HOME/.local/share/llm-dagu/server}"
export DAGU_JOB_ROOT="${DAGU_JOB_ROOT:-$HOME/.local/share/llm-dagu}"
if [[ "$#" -eq 0 ]]; then
    set -- start-all
fi
exec "${DAGU_BIN:-$HOME/.local/opt/llm-dagu/2.26.0/dagu}" --config "$repository_dir/dagu/config.yaml" "$@"
