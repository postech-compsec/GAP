#!/bin/bash
# Usage: `./experiments/run_rq2.sh --mode approx|full [--raw-logs off|move]`
# Run RQ2 noise-condition sweeps on PX4 jMAVSim.
set -euo pipefail

source "$(dirname "$0")/common.sh"

MODE=$(parse_mode "$@")
RAW_LOG_MODE=$(parse_raw_logs "$@")
OUTPUT_DIR="$RESULTS_DIR/rq2"
ensure_dir "$OUTPUT_DIR"
export GAP_RESULTS_DIR="$OUTPUT_DIR"
export GAP_LOG_DIR="$OUTPUT_DIR"
export GAP_RAW_LOG_MODE="$RAW_LOG_MODE"
unset GAP_RAY_RUN_NAME
unset GAP_PRIMARY_VARIANT

log_info "=== RQ2: Robustness to Realistic Noise (Table 4) ==="
echo ""

NUM_TRIALS=$(get_num_trials "$MODE")
log_info "Mode: $MODE | Trials per worker: $NUM_TRIALS"
log_info "Raw logs: $RAW_LOG_MODE"
check_model_exists || exit 1

START_TIME=$(date +%s)
FAILED=()

declare -a CONDITIONS=(
    "none:0:0"
    "tracking:1:0"
    "delay-loss:0:1"
    "both:1:1"
)

i=0
for cond in "${CONDITIONS[@]}"; do
    i=$((i + 1))
    name="${cond%%:*}"
    rest="${cond#*:}"
    tracking="${rest%%:*}"
    delay_loss="${rest##*:}"

    log_info "[$i/4] Condition: $name (tracking=$tracking, delay_loss=$delay_loss)"
    if (cd "$EVAL_PRIMARY" && \
        NOISE_TRACKING="$tracking" \
        NOISE_DELAY_LOSS="$delay_loss" \
        GAP_PRIMARY_VARIANT="$name" \
        GAP_RAY_RUN_NAME="rq2_gap_px4_jmavsim_${name}" \
        python3 evaluate_multi_criteria.py \
            --checkpoint_path "$GAP_MODEL_PATH" \
            --num_trials "$NUM_TRIALS"); then
        log_success "Condition '$name' complete"
    else
        log_error "Condition '$name' failed"
        FAILED+=("rq2-$name")
    fi
    echo ""
done

ELAPSED=$(($(date +%s) - START_TIME))

if [ ${#FAILED[@]} -gt 0 ]; then
    log_error "Some conditions failed: ${FAILED[*]}"
    log_error "RQ2 failed in $(elapsed_time $ELAPSED)"
    log_info "Results saved to: $OUTPUT_DIR"
    exit 1
fi
log_success "RQ2 complete in $(elapsed_time $ELAPSED)"
log_info "Results saved to: $OUTPUT_DIR"
