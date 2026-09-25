#!/usr/bin/env bash
# Generate once per seed with MODE=collect; train each method with MODE=offline.
set -euo pipefail
export MODE=${MODE:-offline}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "${SCRIPT_DIR}/../run_math.sh" "$@"
