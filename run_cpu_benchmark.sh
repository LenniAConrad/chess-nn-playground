#!/usr/bin/env bash
# CPU inference benchmark across i018 / lc0_bt4_classifier (conv) /
# lc0_bt4_transformer at base / scale_up / scale_xl, batch=1/8/32.
#
# Runs INDEPENDENTLY of the GPU queue -- doesn't wait, doesn't touch CUDA.
# Conservative thread count so GPU dataloader workers aren't starved.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
THREADS="${CPU_BENCH_THREADS:-4}"
REPORT_DIR="$ROOT_DIR/reports/cpu_benchmark"
mkdir -p "$REPORT_DIR"
LAUNCHER_LOG="$REPORT_DIR/launcher.log"

log() { echo "[$(date -Is)] $*" | tee -a "$LAUNCHER_LOG"; }

log "CPU inference benchmark starting. Threads=$THREADS. CUDA disabled."
log "Independent of GPU queue -- runs immediately."

# Force CPU-only Torch in the subprocess.
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/benchmark_cpu_inference.py \
  --threads "$THREADS" \
  --output "$REPORT_DIR/results.md" 2>&1 | tee -a "$LAUNCHER_LOG"

log "Done. Report: $REPORT_DIR/results.md"
log "JSON: $REPORT_DIR/results.json"

# Roll the new CPU numbers into the aggregate report immediately.
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/reports/build_aggregate_report.py \
  >>"$LAUNCHER_LOG" 2>&1 || log "aggregate report rebuild failed (non-fatal)"
