#!/usr/bin/env bash
set -euo pipefail
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}"
export ALGORITHM=ggpo
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "${SCRIPT_DIR}/run_math.sh" "$@"
