#!/usr/bin/env bash
# Run i242 ablations A2/A3/A4 sequentially. Waits for any in-flight i242 run first.
set -uo pipefail
cd "$(dirname "$0")/.."

# Wait until no i242 training is in progress
while pgrep -f "train_model.*i242\|run_paper_ready.*i242" > /dev/null; do
  sleep 5
done
echo "no i242 training in progress, starting ablation sweep"

VENV=.venv/bin/python
RUNNER=scripts/run_paper_ready_all.py
COMMON_FLAGS=(
  --no-benchmarks --no-ideas
  --seeds 42 --scale-variants base:1
  --epochs 12 --min-epochs 6 --patience 3
  --shorten-training --monitor pr_auc
  --jobs 1 --gpu-ids 0 --timeout-minutes 30
)

for tag in A2_no_chess_bias A3_no_exchange A4_i193_hp; do
  cfg=/tmp/i242_ablations/${tag}.yaml
  out_results=results/architecture_scout_2026-05-11_i242_${tag}
  out_reports=reports/architecture_scout_2026-05-11_i242_${tag}
  mkdir -p "$out_results" "$out_reports"
  echo "=== launching ${tag} ==="
  PYTHONDONTWRITEBYTECODE=1 CHESS_NN_DISABLE_CPU_FALLBACK=1 \
    "$VENV" "$RUNNER" "$cfg" \
    "${COMMON_FLAGS[@]}" \
    --results-dir "$out_results" \
    --report-dir "$out_reports" \
    --state-path "$out_reports/state.json" \
    --logs-dir "$out_reports/logs" \
    --generated-config-dir "$out_reports/generated_configs" \
    --event-log "$out_reports/events.jsonl" \
    --timeline "$out_reports/timeline.md" \
    > "$out_reports/run.log" 2>&1
  rc=$?
  echo "=== ${tag} exit code: ${rc} ==="
done

echo "=== all ablations finished ==="
ls -la results/architecture_scout_2026-05-11_i242_A*/benchmark_*/metrics_final.json 2>/dev/null
