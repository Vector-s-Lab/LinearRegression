#!/usr/bin/env bash
# One-click driver: run all experiments and regenerate tables/figures.
#
# Steps:
#   [1/4] Main benchmarks (4 datasets x 4 KPIs = 16 configs)
#   [2/4] Block-correlation multicollinearity (3 SNRs x 4 KPIs = 12 configs)
#   [3/4] Near-dependency multicollinearity (4 KPIs)
#   [4/4] Regenerate LaTeX tables and copy figures
#
# Usage:
#   bash run_all.sh                # run everything
#   bash run_all.sh --skip-main    # skip the main benchmarks
#   bash run_all.sh --skip-multi   # skip the block-correlation multicollinearity study
#   bash run_all.sh --skip-near    # skip the near-dependency multicollinearity study
#   bash run_all.sh --skip-assets  # skip table/figure regeneration
#
# Requires the dependencies listed in requirements.txt. If a conda env is
# present, it is activated automatically (override with CONDA_ENV=<name>).
#
# Each experiment writes to results/<auto_timestamped_dir>/. Per-task logs
# land in logs/run_all_<timestamp>/. A summary log records start/finish/wall
# time of each step. Failures in one step do not stop the rest.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Parse flags ---
RUN_MAIN=1
RUN_MULTI=1
RUN_NEAR=1
RUN_ASSETS=1
for arg in "$@"; do
    case "$arg" in
        --skip-main)   RUN_MAIN=0 ;;
        --skip-multi)  RUN_MULTI=0 ;;
        --skip-near)   RUN_NEAR=0 ;;
        --skip-assets) RUN_ASSETS=0 ;;
        -h|--help)
            sed -n '2,21p' "$0" | sed 's/^# //; s/^#//'
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg" >&2
            exit 1
            ;;
    esac
done

# --- Activate conda env (override CONDA_ENV to use a different env name) ---
if [[ -n "${CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# --- Logging ---
RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/run_all_${RUN_TS}"
mkdir -p "$LOG_DIR"
SUMMARY_LOG="$LOG_DIR/summary.log"

log() {
    local msg="[$(date +%H:%M:%S)] $*"
    echo "$msg" | tee -a "$SUMMARY_LOG"
}

run_step() {
    # run_step <name> <log_file> <command...>
    local name="$1"; shift
    local logf="$1"; shift
    log "START  $name"
    local t0
    t0=$(date +%s)
    if "$@" > "$logf" 2>&1; then
        local dt=$(( $(date +%s) - t0 ))
        log "OK     $name  (${dt}s)"
    else
        local rc=$?
        local dt=$(( $(date +%s) - t0 ))
        log "FAIL   $name  (rc=$rc, ${dt}s)  -> see $logf"
    fi
}

log "========== run_all.sh started =========="
log "Logs in $LOG_DIR/"
log "Flags: RUN_MAIN=$RUN_MAIN  RUN_MULTI=$RUN_MULTI  RUN_NEAR=$RUN_NEAR  RUN_ASSETS=$RUN_ASSETS"
log "Python: $(which python)"
log "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"

GLOBAL_T0=$(date +%s)

# --- 1. Main benchmarks (4 datasets x 4 KPIs = 16 configs) ---
if [[ "$RUN_MAIN" == "1" ]]; then
    log ""
    log "===== [1/4] Main benchmarks (16 configs) ====="
    for cfg in configs/*.yaml; do
        cfg_name=$(basename "$cfg" .yaml)
        run_step "main:${cfg_name}" "$LOG_DIR/main_${cfg_name}.log" \
            python -u run_benchmark.py "$cfg"
    done
fi

# --- 2. Block-correlation multicollinearity (3 SNRs x 4 KPIs = 12 configs) ---
if [[ "$RUN_MULTI" == "1" ]]; then
    log ""
    log "===== [2/4] Block-correlation multicollinearity (12 configs) ====="
    SNRS=("1.0" "2.38" "4.36")
    MKPIS=("r2" "adj_r2" "aic" "f_statistic")
    for snr in "${SNRS[@]}"; do
        for kpi in "${MKPIS[@]}"; do
            run_step "multi:snr${snr}_${kpi}" "$LOG_DIR/multi_snr${snr}_${kpi}.log" \
                python -u run_multicollinearity.py \
                    --kpi "$kpi" \
                    --n-features 8 --n-samples 1000 --snr "$snr" \
                    --levels 0.0 0.3 0.6 0.9 \
                    --seed 42 --n-runs 10
        done
    done
fi

# --- 3. Near-dependency multicollinearity (4 KPIs) ---
if [[ "$RUN_NEAR" == "1" ]]; then
    log ""
    log "===== [3/4] Near-dependency multicollinearity (4 KPIs) ====="
    NKPIS=("r2" "adj_r2" "aic" "f_statistic")
    for kpi in "${NKPIS[@]}"; do
        run_step "near:${kpi}" "$LOG_DIR/near_${kpi}.log" \
            python -u run_near_dependency.py \
                --kpi "$kpi" \
                --target-vifs 10 50 100 500 \
                --n-samples 1000 \
                --seed 42 --n-runs 10
    done
fi

# --- 4. Regenerate LaTeX tables and copy figures ---
if [[ "$RUN_ASSETS" == "1" ]]; then
    log ""
    log "===== [4/4] Regenerate tables and figures ====="
    run_step "assets" "$LOG_DIR/assets.log" \
        python -u generate_paper_assets.py
fi

GLOBAL_DT=$(( $(date +%s) - GLOBAL_T0 ))
log ""
log "========== run_all.sh finished in ${GLOBAL_DT}s =========="
log "Summary log: $SUMMARY_LOG"

# Print quick failure tally to stdout
FAIL_COUNT=$(grep -c "^\[..:..:..\] FAIL" "$SUMMARY_LOG" || true)
OK_COUNT=$(grep -c "^\[..:..:..\] OK" "$SUMMARY_LOG" || true)
echo ""
echo "Results: $OK_COUNT succeeded, $FAIL_COUNT failed"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
    echo "Failed steps (see $LOG_DIR/ for details):"
    grep "^\[..:..:..\] FAIL" "$SUMMARY_LOG" | sed 's/^/  /'
    exit 1
fi
