#!/bin/bash
# Shared helpers for experiment wrappers.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

# Prefer the project-local venv unless the caller already chose one.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

EVAL_PRIMARY="$SRC_DIR/evaluation/primary"
EVAL_BASELINE="$SRC_DIR/evaluation/baseline"
EVAL_CROSS_PLATFORM="$SRC_DIR/evaluation/cross_platform"

# Default to the bundled firmware submodules.
if [ "${GAP_USE_EXTERNAL_FIRMWARE:-0}" = "1" ]; then
    export PX4_ROOT="${PX4_ROOT:-$PROJECT_ROOT/GAP-PX4-Autopilot}"
    export ARDUPILOT_ROOT="${ARDUPILOT_ROOT:-$PROJECT_ROOT/GAP-ardupilot}"
else
    for var in PX4_ROOT ARDUPILOT_ROOT; do
        inherited="$(eval echo "\${$var:-}")"
        expected="$PROJECT_ROOT/GAP-$( [ "$var" = "PX4_ROOT" ] && echo PX4-Autopilot || echo ardupilot )"
        if [ -n "$inherited" ] && [ "$inherited" != "$expected" ]; then
            echo -e "\033[1;33m[WARN]\033[0m $var=$inherited in environment; overriding with bundled submodule."
            echo -e "\033[1;33m[WARN]\033[0m Set GAP_USE_EXTERNAL_FIRMWARE=1 to keep your override."
        fi
    done
    export PX4_ROOT="$PROJECT_ROOT/GAP-PX4-Autopilot"
    export ARDUPILOT_ROOT="$PROJECT_ROOT/GAP-ardupilot"
fi

# `src/` resolves `gap.*` and `evaluation.*`; repo root resolves `analysis.*`.
export PYTHONPATH="$SRC_DIR:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

MODELS_DIR="${MODELS_DIR:-$SRC_DIR/gap/models}"
GAP_MODEL_PATH="$MODELS_DIR/gap_model"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/fresh}"
export GAP_PX4_LOG_DIR="${GAP_PX4_LOG_DIR:-$RESULTS_DIR/flight-logs/px4/raw}"
export GAP_ARDUPILOT_LOG_DIR="${GAP_ARDUPILOT_LOG_DIR:-$RESULTS_DIR/flight-logs/ardupilot/raw}"
# Older scripts may still read GAP_ULOG_DIR.
export GAP_ULOG_DIR="$GAP_PX4_LOG_DIR"
export GAP_LOG_DIR="${GAP_LOG_DIR:-$RESULTS_DIR/log}"
export GAP_RAY_LOG_DIR="${GAP_RAY_LOG_DIR:-$RESULTS_DIR/ray_results}"
export GAP_MODELS_DIR="${GAP_MODELS_DIR:-$MODELS_DIR}"

SPEEDUP="${SPEEDUP:-4}"

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $*"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }

elapsed_time() {
    local seconds="$1"
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    if [ "$hours" -gt 0 ]; then
        printf "%dh %dm %ds" "$hours" "$minutes" "$secs"
    elif [ "$minutes" -gt 0 ]; then
        printf "%dm %ds" "$minutes" "$secs"
    else
        printf "%ds" "$secs"
    fi
}

ensure_dir() { mkdir -p "$1"; }

parse_mode() {
    local mode=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                if [[ $# -lt 2 ]]; then
                    log_error "Missing value after --mode (use: approx|full)"
                    return 1
                fi
                mode="$2"
                shift 2
                ;;
            --approx)
                mode="approx"
                shift
                ;;
            --full)
                mode="full"
                shift
                ;;
            *) shift;;
        esac
    done
    mode="${mode:-approx}"
    case "$mode" in
        approx|full)
            echo "$mode"
            ;;
        *)
            log_error "Unsupported mode: $mode (use: approx|full)"
            return 1
            ;;
    esac
}

parse_raw_logs() {
    local raw_logs="move"
    while [[ $# -gt 0 ]]; do
        case $1 in
            --raw-logs)
                raw_logs="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    case "$raw_logs" in
        off|move) echo "$raw_logs" ;;
        *) log_error "Unsupported --raw-logs value: $raw_logs (use: off|move)"; exit 1 ;;
    esac
}

get_num_trials() {
    local mode="$1"
    if [ "$mode" = "full" ]; then
        echo "100"
    else
        echo "2"
    fi
}

check_model_exists() {
    if [ ! -d "$GAP_MODEL_PATH" ]; then
        log_error "Model not found at $GAP_MODEL_PATH"
        log_info "Set MODELS_DIR to point to the checkpoint directory."
        return 1
    fi
    return 0
}
