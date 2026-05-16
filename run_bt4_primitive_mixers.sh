#!/usr/bin/env bash
# Scout-grade training for all a###_bt4_*_mixer ideas: a BT4-style tower with
# each chess-research primitive (plus conv/attention baselines available
# separately) as the per-block spatial mixer.
#
# Queue order is deterministic: this waits for every other run_paper_ready_all.py
# AND the falsifier / i249 launchers to exit, so it runs last:
#   hybrid -> falsifier -> i249 -> bt4-primitive-mixers
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
REPORT_DIR="$ROOT_DIR/reports/bt4_primitive_mixers"
RESULTS_DIR="$ROOT_DIR/results/bt4_primitive_mixers"
mkdir -p "$REPORT_DIR" "$RESULTS_DIR"
LAUNCHER_LOG="$REPORT_DIR/launcher.log"

log() { echo "[$(date -Is)] $*" | tee -a "$LAUNCHER_LOG"; }

# Collect every generated bt4-primitive-mixer idea config.
mapfile -t CONFIGS < <(ls "$ROOT_DIR"/ideas/registry/a[0-9][0-9][0-9]_bt4_*_mixer/config.yaml 2>/dev/null | sort)

log "BT4 primitive-mixer scout launcher starting."
log "Found ${#CONFIGS[@]} a###_bt4_*_mixer idea configs."
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  log "ERROR: no bt4 primitive-mixer configs found. Run scripts/ideas/scaffold_bt4_primitive_mixers.py first."
  exit 1
fi

log "Will queue until GPU 0 is free: waits for any run_paper_ready_all.py AND"
log "the falsifier / i249 launchers to exit (deterministic queue order)."
while pgrep -f "run_paper_ready_all.py" >/dev/null 2>&1 \
   || pgrep -f "run_falsifier_i018.sh" >/dev/null 2>&1 \
   || pgrep -f "run_i249_fast.sh" >/dev/null 2>&1; do
  log "GPU still busy (a run_paper_ready_all.py or an upstream launcher is active). Sleeping 60s..."
  sleep 60
done

log "GPU 0 free. Launching scout-grade run: ${#CONFIGS[@]} ideas, seed 42, base scale."

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/run_paper_ready_all.py \
  "${CONFIGS[@]}" \
  --no-benchmarks --no-ideas \
  --seeds 42 \
  --scale-variants base:1 \
  --batch-size-caps base:256 \
  --epochs 15 --min-epochs 8 --patience 4 \
  --shorten-training \
  --monitor pr_auc \
  --jobs 1 --gpu-ids 0 \
  --results-dir "$RESULTS_DIR" \
  --report-dir "$REPORT_DIR" \
  --state-path "$REPORT_DIR/state.json" \
  --logs-dir "$REPORT_DIR/logs" \
  --generated-config-dir "$REPORT_DIR/generated_configs" \
  --event-log "$REPORT_DIR/events.jsonl" \
  --timeline "$REPORT_DIR/timeline.md" 2>&1 | tee -a "$LAUNCHER_LOG"

log "BT4 primitive-mixer scout done. Building leaderboard."

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" - <<PY 2>&1 | tee -a "$LAUNCHER_LOG"
import json
from pathlib import Path

RESULTS_DIR = Path("$RESULTS_DIR")
REPORT_DIR = Path("$REPORT_DIR")

rows = []
for run_dir in sorted(RESULTS_DIR.glob("idea_a*_bt4_*_mixer*")):
    m = run_dir / "metrics_final.json"
    if not m.exists():
        continue
    d = json.loads(m.read_text())
    pr = d.get("test_pr_auc")
    rows.append((pr if pr is not None else -1.0, run_dir.name, d.get("test_accuracy")))
rows.sort(reverse=True)

lines = ["# BT4 Primitive-Mixer Scout Leaderboard", ""]
lines.append("Each row: a BT4-style tower with one primitive as the per-block spatial mixer.")
lines.append("Scout grade: seed 42, base scale, shortened training. Filter, do not rank finely.")
lines.append("")
lines.append("Reference baselines from the paper-grade run (3-seed means):")
lines.append("- conv-equivalent (lc0_bt4 classifier): ~0.859 test PR-AUC")
lines.append("- i018 oriented_tactical_sheaf (base): ~0.875")
lines.append("")
lines.append("| rank | test PR-AUC | test acc | idea |")
lines.append("|---:|---:|---:|---|")
for i, (pr, name, acc) in enumerate(rows, 1):
    pr_s = "FAIL" if pr < 0 else f"{pr:.4f}"
    acc_s = "-" if acc is None else f"{acc:.3f}"
    lines.append(f"| {i} | {pr_s} | {acc_s} | {name} |")
out = REPORT_DIR / "leaderboard.md"
out.write_text("\n".join(lines) + "\n")
print(f"Wrote {out}  ({len(rows)} runs)")
PY

log "All done. Leaderboard: $REPORT_DIR/leaderboard.md"
