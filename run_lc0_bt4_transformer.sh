#!/usr/bin/env bash
# Paper-grade benchmark run for the authentic LC0 BT4 transformer
# (encoder-only multi-head-attention trunk + puzzle_binary logit head).
#
# Runs LAST in the queue: waits for every other run_paper_ready_all.py AND the
# falsifier / i249 / bt4-primitive-mixer launchers to exit. Deterministic order:
#   hybrid -> falsifier -> i249 -> bt4-primitive-mixers -> lc0-bt4-transformer
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
CONFIG="$ROOT_DIR/configs/benchmarks/puzzle_binary/bench_lc0_bt4_transformer.yaml"
REPORT_DIR="$ROOT_DIR/reports/lc0_bt4_transformer"
RESULTS_DIR="$ROOT_DIR/results/lc0_bt4_transformer"
mkdir -p "$REPORT_DIR" "$RESULTS_DIR"
LAUNCHER_LOG="$REPORT_DIR/launcher.log"

log() { echo "[$(date -Is)] $*" | tee -a "$LAUNCHER_LOG"; }

rebuild_aggregate() {
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/reports/build_aggregate_report.py \
    >>"$LAUNCHER_LOG" 2>&1 || log "aggregate report rebuild failed (non-fatal)"
}

log "LC0 BT4 transformer benchmark launcher starting."
log "Authentic encoder-only transformer: base ~4.8M / scale_up ~16M / scale_xl ~38M params."
log "Will queue until GPU 0 is free (waits for all upstream launchers + run_paper_ready_all.py)."

while pgrep -f "run_paper_ready_all.py" >/dev/null 2>&1 \
   || pgrep -f "run_falsifier_i018.sh" >/dev/null 2>&1 \
   || pgrep -f "run_i249_fast.sh" >/dev/null 2>&1 \
   || pgrep -f "run_bt4_primitive_mixers.sh" >/dev/null 2>&1; do
  log "GPU still busy (an upstream pipeline or launcher is active). Sleeping 60s..."
  sleep 60
done

log "GPU 0 free. Launching paper-grade transformer benchmark: 3 seeds x 3 scales = 9 tasks."

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/run_paper_ready_all.py \
  "$CONFIG" \
  --no-benchmarks --no-ideas \
  --seeds 42,43,44 \
  --scale-variants base:1,scale_up:1.5,scale_xl:2 \
  --batch-size-caps base:256,scale_up:192,scale_xl:128 \
  --epochs 30 --min-epochs 15 --patience 8 \
  --jobs 1 --gpu-ids 0 \
  --results-dir "$RESULTS_DIR" \
  --report-dir "$REPORT_DIR" \
  --state-path "$REPORT_DIR/state.json" \
  --logs-dir "$REPORT_DIR/logs" \
  --generated-config-dir "$REPORT_DIR/generated_configs" \
  --event-log "$REPORT_DIR/events.jsonl" \
  --timeline "$REPORT_DIR/timeline.md" 2>&1 | tee -a "$LAUNCHER_LOG"

log "Transformer benchmark training done. Rebuilding aggregate report."
rebuild_aggregate

log "All done. Aggregate report: $ROOT_DIR/reports/aggregate_report.md"
