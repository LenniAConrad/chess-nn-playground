#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
REPORT_DIR="$ROOT_DIR/reports/paper_grade_top3"
RESULTS_DIR="$ROOT_DIR/results/paper_grade_top3"
mkdir -p "$REPORT_DIR" "$RESULTS_DIR"
LAUNCHER_LOG="$REPORT_DIR/launcher.log"

log() { echo "[$(date -Is)] $*" | tee -a "$LAUNCHER_LOG"; }

# Top 3 trunks from the 234-architecture scout (best val PR-AUC):
#   1. i193 exchange_then_king_dual_stream          (val_pr_auc=0.8901)
#   2. i011 vetoselect_positive_claim_abstention    (val_pr_auc=0.8721)
#   3. i018 oriented_tactical_sheaf_laplacian       (val_pr_auc=0.8678)
# Plus LC0 BT4 transformer reference.
TRUNK_CONFIGS=(
  "$ROOT_DIR/ideas/registry/i193_exchange_then_king_dual_stream/config.yaml"
  "$ROOT_DIR/ideas/registry/i011_vetoselect_positive_claim_abstention/config.yaml"
  "$ROOT_DIR/ideas/registry/i018_oriented_tactical_sheaf_laplacian/config.yaml"
)
BT4_CONFIG="$ROOT_DIR/configs/benchmarks/puzzle_binary/bench_lc0_bt4_classifier.yaml"

log "Paper-grade top-3 trunks + BT4 wrapper starting."
log "Will queue until GPU 0 is free, then run paper-grade with 3 seeds and 3 scales."
log "Trunks:"
for cfg in "${TRUNK_CONFIGS[@]}"; do log "  - $cfg"; done
log "BT4 reference: $BT4_CONFIG"
log "Seeds: 42,43,44   Scales: base,scale_up:1.5,scale_xl:2   -> 36 tasks total"

while pgrep -f "run_paper_ready_all.py.*primitive_pipeline" >/dev/null 2>&1; do
  log "Primitive scout still running. Sleeping 60s..."
  sleep 60
done

log "GPU 0 free. Launching paper-grade run."

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/run_paper_ready_all.py \
  "${TRUNK_CONFIGS[@]}" \
  "$BT4_CONFIG" \
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

log "Paper-grade run finished. Building extras (mean+/-std, scale curves, Pareto, heatmap)..."

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/reports/build_paper_grade_top3_extras.py \
  --state-path "$REPORT_DIR/state.json" \
  --output-dir "$REPORT_DIR/extras" 2>&1 | tee -a "$LAUNCHER_LOG" || log "Extras step failed (non-fatal)."

log "All done."
log "Open $REPORT_DIR/status.md first."
log "PDF: $REPORT_DIR/paper_report.pdf"
log "Extras HTML: $REPORT_DIR/extras/extras_report.html"
