#!/bin/bash
# Usage: `./experiments/run_rq1.sh --mode approx|full [--raw-logs off|move]`
# Run RQ1 baselines and GAP on PX4 jMAVSim.
set -euo pipefail

source "$(dirname "$0")/common.sh"

MODE=$(parse_mode "$@")
RAW_LOG_MODE=$(parse_raw_logs "$@")
OUTPUT_DIR="$RESULTS_DIR/rq1"
ensure_dir "$OUTPUT_DIR"
export GAP_RESULTS_DIR="$OUTPUT_DIR"
export GAP_LOG_DIR="$OUTPUT_DIR"
export GAP_RAW_LOG_MODE="$RAW_LOG_MODE"
export GAP_RAY_RUN_NAME="rq1_gap_px4_jmavsim"
unset GAP_PRIMARY_VARIANT

log_info "=== RQ1: Effectiveness of Trained Policy (Table 3) ==="
echo ""

NUM_TRIALS=$(get_num_trials "$MODE")
log_info "Mode: $MODE | Trials per worker: $NUM_TRIALS"
log_info "Raw logs: $RAW_LOG_MODE"
echo ""

START_TIME=$(date +%s)
FAILED=()

check_model_exists || exit 1

for case_num in 1 2 3; do
    log_info "[$case_num/4] Running baseline Case $case_num..."
    if (cd "$EVAL_BASELINE" && python3 baseline_test.py --case "$case_num" --speedup "$SPEEDUP"); then
        log_success "Baseline Case $case_num complete"
    else
        log_error "Baseline Case $case_num failed"
        FAILED+=("baseline-case$case_num")
    fi
    echo ""
done

log_info "[4/4] Running GAP (PX4-jMAVSim, $NUM_TRIALS trials per worker)..."
if (cd "$EVAL_PRIMARY" && python3 evaluate_multi_criteria.py \
        --checkpoint_path "$GAP_MODEL_PATH" \
        --num_trials "$NUM_TRIALS"); then
    log_success "GAP evaluation complete"
else
    log_error "GAP evaluation failed"
    FAILED+=("gap")
fi
echo ""

ELAPSED=$(($(date +%s) - START_TIME))

if [ ${#FAILED[@]} -gt 0 ]; then
    log_error "Some experiments failed: ${FAILED[*]}"
    log_error "RQ1 failed in $(elapsed_time $ELAPSED)"
    log_info "Results saved to: $OUTPUT_DIR"
    exit 1
fi
log_success "All RQ1 experiments completed successfully"
log_success "RQ1 complete in $(elapsed_time $ELAPSED)"
log_info "Results saved to: $OUTPUT_DIR"
