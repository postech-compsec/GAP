#!/bin/bash
# Usage: `./experiments/run_all.sh --mode approx|full [--raw-logs off|move]`
# Run the main fresh automated artifact path, then regenerate figures/tables.
set -euo pipefail

source "$(dirname "$0")/common.sh"

MODE=$(parse_mode "$@")
RAW_LOG_MODE=$(parse_raw_logs "$@")

echo "============================================================"
echo "  GAP Artifact Evaluation - Full Run"
echo "============================================================"
echo ""
echo "Mode: $MODE (see README for wall-clock estimates)"
echo "Raw logs: $RAW_LOG_MODE"
echo ""

"$SCRIPT_DIR/run_rq1.sh" --mode "$MODE" --raw-logs "$RAW_LOG_MODE"
"$SCRIPT_DIR/run_rq2.sh" --mode "$MODE" --raw-logs "$RAW_LOG_MODE"
"$SCRIPT_DIR/run_rq3.sh" --raw-logs "$RAW_LOG_MODE"

echo ""
echo "=== Generating tables and figures ==="
bash "$PROJECT_ROOT/analysis/generate_all.sh" fresh

echo ""
echo "============================================================"
echo "  All done. Output: $RESULTS_DIR"
echo "============================================================"
