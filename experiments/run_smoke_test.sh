#!/bin/bash
# Usage: `./experiments/run_smoke_test.sh`
# Verify imports, checkpoints, core analysis scripts, and pre-baked claim checks.
set -euo pipefail

_SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
_PROJECT_ROOT="$(dirname "$_SELF_DIR")"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$_PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$_PROJECT_ROOT/.venv/bin/activate"
fi

source "$(dirname "$0")/common.sh"

pass=0
fail=0
fail_list=()

_check() {
    local label="$1"; shift
    if "$@" >/tmp/gap_smoke.$$.log 2>&1; then
        log_success "$label"
        pass=$((pass + 1))
    else
        log_error "$label"
        sed 's/^/    /' /tmp/gap_smoke.$$.log
        fail=$((fail + 1))
        fail_list+=("$label")
    fi
    rm -f /tmp/gap_smoke.$$.log
}

log_info "=== GAP smoke test ==="
echo ""

_check "imports: gap, evaluation.common, evaluation.sim_to_real.common" \
    python3 -c "
import gap.asymmetric_rl_module
import evaluation.common.metrics
import evaluation.sim_to_real.common.config
import evaluation.sim_to_real.common.observation
"

_check "imports: cross-platform entrypoint bindings" \
    python3 -c "
import pathlib
import evaluation.cross_platform.cross_platform_test as m
assert m.Path is pathlib.Path
"

_check "entrypoints: py_compile" \
    python3 -m py_compile \
        src/evaluation/baseline/baseline_test.py \
        src/evaluation/cross_platform/cross_platform_test.py \
        src/evaluation/primary/evaluate_multi_criteria.py \
        src/evaluation/sim_to_real/sim/evaluate_sim.py \
        src/evaluation/ci_detector/ci_detector_test.py

_check "wrappers: parse_mode supports --approx/--full" \
    bash -lc "cd '$_PROJECT_ROOT' && source experiments/common.sh && [ \"\$(parse_mode --approx)\" = approx ] && [ \"\$(parse_mode --full)\" = full ]"

_check "checkpoint: gap_model/" \
    test -f "$MODELS_DIR/gap_model/rllib_checkpoint.json"
_check "checkpoint: sim-to-real_model/" \
    test -f "$MODELS_DIR/sim-to-real_model/rllib_checkpoint.json"

_check "analysis: generate_rq1_table3 (RQ1, pre-baked)" \
    python3 -m analysis.generate_rq1_table3 --source pre-baked

_check "analysis: generate_rq5_analysis (RQ5, pre-baked)" \
    python3 -m analysis.generate_rq5_analysis --source pre-baked

_check "analysis: generate_rq4_analysis (RQ4, pre-baked)" \
    python3 -m analysis.generate_rq4_analysis --source pre-baked

_check "analysis: generate_rq6_table6 (RQ6, pre-baked)" \
    python3 -m analysis.generate_rq6_table6 --source pre-baked

_check "analysis: verify_claims (pre-baked)" \
    bash analysis/verify_claims.sh pre-baked

echo ""
if [ "$fail" -gt 0 ]; then
    log_error "$fail of $((pass + fail)) checks failed:"
    for f in "${fail_list[@]}"; do echo "  - $f"; done
    exit 1
fi
log_success "All $pass checks passed. Artifact is installed correctly."
