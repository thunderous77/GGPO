#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
: "${DATA_ROOT:?Set DATA_ROOT to the prepared data directory}"
: "${MODEL_PATH:?Set MODEL_PATH to a base model or exported Hugging Face checkpoint}"
: "${EVAL_DIR:?Set EVAL_DIR to the output directory for this checkpoint}"
export ALGORITHM=grpo MODE=online
exec bash "${SCRIPT_DIR}/run_math.sh" \
    "data.val_files=[${DATA_ROOT}/math500.parquet,${DATA_ROOT}/aime24.parquet,${DATA_ROOT}/aime25.parquet,${DATA_ROOT}/olympiadbench.parquet,${DATA_ROOT}/minerva_math.parquet]" \
    trainer.val_only=true trainer.resume_mode=disable \
    "trainer.validation_data_dir=${EVAL_DIR}" "$@"
