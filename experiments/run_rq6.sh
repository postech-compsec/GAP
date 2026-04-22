#!/bin/bash
# Usage: `./experiments/run_rq6.sh --mode approx|full [--raw-logs off|move]`
# Run the supplementary RQ6 sim-to-real SITL evaluation.
set -euo pipefail

source "$(dirname "$0")/common.sh"

MODE=$(parse_mode "$@")
RAW_LOG_MODE=$(parse_raw_logs "$@")
OUTPUT_DIR="$RESULTS_DIR/rq6/sim_evaluation"
ensure_dir "$OUTPUT_DIR"
export GAP_RESULTS_DIR="$OUTPUT_DIR"
export GAP_LOG_DIR="$OUTPUT_DIR"
export GAP_RAW_LOG_MODE="$RAW_LOG_MODE"

SIM_TO_REAL_MODEL_PATH="$MODELS_DIR/sim-to-real_model"
if [ ! -d "$SIM_TO_REAL_MODEL_PATH" ]; then
    log_error "sim-to-real model not found at $SIM_TO_REAL_MODEL_PATH"
    log_info "Unpack the Zenodo archive into src/gap/models/; see README.md."
    exit 1
fi

log_info "=== RQ6 sim: Sim-to-Real Model SITL Evaluation ==="
echo ""

NUM_TRIALS=$(get_num_trials "$MODE")
log_info "Mode: $MODE | Trials per worker: $NUM_TRIALS"
log_info "Raw logs: $RAW_LOG_MODE"
log_info "Output: $OUTPUT_DIR"
echo ""

START_TIME=$(date +%s)

EVAL_SIM_TO_REAL_SIM="$SRC_DIR/evaluation/sim_to_real/sim"
if (cd "$EVAL_SIM_TO_REAL_SIM" && python3 evaluate_sim.py \
        --checkpoint_path "$SIM_TO_REAL_MODEL_PATH" \
        --num_trials "$NUM_TRIALS"); then
    log_success "Sim-to-real SITL evaluation complete"
else
    log_error "Sim-to-real SITL evaluation failed"
    exit 1
fi

ELAPSED=$(($(date +%s) - START_TIME))
log_success "RQ6 sim complete in $(elapsed_time $ELAPSED)"
log_info "Results saved to: $OUTPUT_DIR"
log_info ""
log_info "For RQ6 real (physical drone): see src/evaluation/sim_to_real/real/README.md"
