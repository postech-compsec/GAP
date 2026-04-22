#!/bin/bash
# Usage: `bash analysis/verify_claims.sh [pre-baked|fresh]`
# pre-baked: check the paper metrics against the shipped pre-baked data
# fresh: supplementary observed-summary report against those same paper metrics

set -euo pipefail

SOURCE="${1:-pre-baked}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
python -m analysis.verify_claims "$SOURCE"
