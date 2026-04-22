#!/bin/bash
# Usage: `bash analysis/generate_all.sh [pre-baked|fresh]`
# Generate all shipped tables and figures for one result tree.

set -e

SOURCE="${1:-pre-baked}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer the project-local venv unless the caller already chose one.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

echo "============================================================"
echo " GAP Analysis — generating tables and figures"
echo " Source: $SOURCE"
echo "============================================================"
echo

cd "$PROJECT_ROOT"

echo "--- RQ1 / Table 3: Baseline comparison ---"
python -m analysis.generate_rq1_table3 --source "$SOURCE" || echo "(skipped — no data)"
echo

echo "--- RQ1 / Figure 7: Successful attack-case analysis ---"
python -m analysis.generate_rq1_figure7 --source "$SOURCE" || echo "(skipped — no data)"
echo

echo "--- RQ2 / Table 4: Noise robustness ---"
python -m analysis.generate_rq2_table4 --source "$SOURCE" || echo "(skipped — no data)"
echo

echo "--- RQ3 / Table 5: Cross-platform transfer ---"
python -m analysis.generate_rq3_table5 --source "$SOURCE" || echo "(skipped — no data)"
echo

echo "--- RQ4: CI-Detector evasion ---"
python -m analysis.generate_rq4_analysis --source "$SOURCE" || echo "(skipped — no data)"
echo

echo "--- RQ5: Failsafe analysis over the shared attack-flight corpus ---"
python -m analysis.generate_rq5_analysis --source "$SOURCE" || echo "(skipped — no data)"
echo

echo "--- RQ6 / Table 6: Real-world flight trials ---"
python -m analysis.generate_rq6_table6 --source "$SOURCE" || echo "(skipped — no data)"
echo

echo "--- RQ6 / Figure 8: Real-world trajectories + bias ---"
python -m analysis.generate_rq6_figure8 --source "$SOURCE" || echo "(skipped — no ULog data)"
echo

echo "============================================================"
echo " Done. Check analysis/csv/ and analysis/figures/ for source-tagged output."
echo "============================================================"
