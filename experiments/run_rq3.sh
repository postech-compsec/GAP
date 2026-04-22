#!/bin/bash
# Usage: `./experiments/run_rq3.sh [--frames "singlecopter coaxcopter"] [--skip-gazebo] [--skip-ardupilot] [--raw-logs off|move]`
# Run RQ3 transfer tests on PX4 Gazebo and/or ArduPilot frames.
set -euo pipefail

source "$(dirname "$0")/common.sh"

FRAMES="${FRAMES:-quad hexa octa octaquad y6 dodeca-hexa tri singlecopter coaxcopter}"
SKIP_GAZEBO=0
SKIP_ARDUPILOT=0
RAW_LOG_MODE="move"
while [[ $# -gt 0 ]]; do
    case $1 in
        --frames)        FRAMES="$2"; shift 2;;
        --skip-gazebo)   SKIP_GAZEBO=1; shift;;
        --skip-ardupilot) SKIP_ARDUPILOT=1; shift;;
        --raw-logs)      RAW_LOG_MODE="$2"; shift 2;;
        *) shift;;
    esac
done
case "$RAW_LOG_MODE" in
    off|move) ;;
    *) log_error "Unsupported --raw-logs value: $RAW_LOG_MODE (use: off|move)"; exit 1 ;;
esac

OUTPUT_DIR="$RESULTS_DIR/rq3"
ensure_dir "$OUTPUT_DIR"
export GAP_RESULTS_DIR="$OUTPUT_DIR"
export GAP_LOG_DIR="$OUTPUT_DIR"
export GAP_RAW_LOG_MODE="$RAW_LOG_MODE"

log_info "=== RQ3: Cross-Platform Generalizability (Table 5) ==="
[ "$SKIP_GAZEBO" = "1" ] || log_info "  PX4-Gazebo: x500 @ 1x real time"
[ "$SKIP_ARDUPILOT" = "1" ] || log_info "  ArduPilot SITL frames: $FRAMES"
log_info "  Raw logs: $RAW_LOG_MODE"
echo ""

check_model_exists || exit 1

START_TIME=$(date +%s)
FAILED=()

# Gazebo ignores PX4_SIM_SPEED_FACTOR, so keep it at 1x.
if [ "$SKIP_GAZEBO" != "1" ]; then
    log_info "Testing PX4-Gazebo (x500, 1x real time)..."
    if (cd "$EVAL_CROSS_PLATFORM" && python3 cross_platform_test.py \
            --platform px4-gazebo \
            --checkpoint "$GAP_MODEL_PATH" \
            --speedup 1); then
        log_success "PX4-Gazebo complete"
    else
        log_error "PX4-Gazebo failed"
        FAILED+=("px4-gazebo")
    fi
    echo ""
fi

if [ "$SKIP_ARDUPILOT" != "1" ]; then
    for frame in $FRAMES; do
        log_info "Testing ArduPilot SITL ($frame)..."
        if (cd "$EVAL_CROSS_PLATFORM" && python3 cross_platform_test.py \
                --platform "$frame" \
                --checkpoint "$GAP_MODEL_PATH" \
                --speedup "$SPEEDUP"); then
            log_success "ArduPilot/$frame complete"
        else
            log_error "ArduPilot/$frame failed"
            FAILED+=("ardupilot-$frame")
        fi
        echo ""
    done
fi

ELAPSED=$(($(date +%s) - START_TIME))

if [ ${#FAILED[@]} -gt 0 ]; then
    log_error "Some platforms failed: ${FAILED[*]}"
    exit 1
fi
log_success "RQ3 complete in $(elapsed_time $ELAPSED)"
log_info "Results saved to: $OUTPUT_DIR"
