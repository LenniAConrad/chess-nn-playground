#!/usr/bin/env bash
# Train 4 hybrid models (i018 sheaf trunk + each of p013, p019, p023, p034)
# at 3 seeds each, base scale, and compare to baseline i018 mean.
#
# Queues after any other paper-ready run (waits for falsifier-i018 to finish).
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
CFG_DIR="$ROOT_DIR/configs/hybrid_i018_plus_primitive"
REPORT_DIR="$ROOT_DIR/reports/hybrid_i018"
RESULTS_DIR="$ROOT_DIR/results/hybrid_i018"
THESIS="$ROOT_DIR/ideas/registry/i018_oriented_tactical_sheaf_laplacian/math_thesis.md"
mkdir -p "$REPORT_DIR" "$RESULTS_DIR"
LAUNCHER_LOG="$REPORT_DIR/launcher.log"

log() { echo "[$(date -Is)] $*" | tee -a "$LAUNCHER_LOG"; }

CONFIGS=(
  "$CFG_DIR/i018_plus_p034.yaml"
  "$CFG_DIR/i018_plus_p013.yaml"
  "$CFG_DIR/i018_plus_p019.yaml"
  "$CFG_DIR/i018_plus_p023.yaml"
)

log "Hybrid launcher: 4 hybrids (i018 + p034 / p013 / p019 / p023)."
log "Will queue until no other run_paper_ready_all.py is running."

while pgrep -f "run_paper_ready_all.py" >/dev/null 2>&1; do
  log "Another run_paper_ready_all.py still active. Sleeping 60s..."
  sleep 60
done

log "GPU 0 free. Launching hybrid run (4 configs x 3 seeds = 12 tasks, base scale)."

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/run_paper_ready_all.py \
  "${CONFIGS[@]}" \
  --no-benchmarks --no-ideas \
  --seeds 42,43,44 \
  --scale-variants base:1 \
  --batch-size-caps base:256 \
  --epochs 20 --min-epochs 10 --patience 5 \
  --jobs 1 --gpu-ids 0 \
  --results-dir "$RESULTS_DIR" \
  --report-dir "$REPORT_DIR" \
  --state-path "$REPORT_DIR/state.json" \
  --logs-dir "$REPORT_DIR/logs" \
  --generated-config-dir "$REPORT_DIR/generated_configs" \
  --event-log "$REPORT_DIR/events.jsonl" \
  --timeline "$REPORT_DIR/timeline.md" 2>&1 | tee -a "$LAUNCHER_LOG"

log "Hybrid training done. Computing mean +/- std per hybrid and writing summary."

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" - <<PY 2>&1 | tee -a "$LAUNCHER_LOG"
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("$ROOT_DIR")
RESULTS_DIR = Path("$RESULTS_DIR")
REPORT_DIR = Path("$REPORT_DIR")
THESIS = Path("$THESIS")
BASELINE_DIR = ROOT / "results" / "paper_grade_top3"

def collect_baseline():
    rows = []
    for d in sorted(BASELINE_DIR.glob("idea_i018_*_seed*")):
        if "scale_up" in d.name or "scale_xl" in d.name:
            continue
        m = d / "metrics_final.json"
        if not m.exists():
            continue
        info = json.loads(m.read_text())
        pr = info.get("test_pr_auc")
        if pr is None:
            continue
        rows.append((d.name, float(pr)))
    return rows

def collect_hybrids():
    by_variant = defaultdict(list)
    for d in sorted(RESULTS_DIR.glob("benchmark_i018_plus_*_seed*")):
        # task_id form: benchmark_i018_plus_pXXX_<...>_seed42
        # extract the variant key from the directory name
        name = d.name.replace("benchmark_", "")
        # name like "i018_plus_p034_seed42" or "i018_plus_p013_sparse_delta_accumulator_seed42"
        # actually the runner uses path.stem of the source config -> "i018_plus_p034" so name is "i018_plus_p034_seed42"
        seed_idx = name.rfind("_seed")
        if seed_idx < 0:
            continue
        variant = name[:seed_idx]
        m = d / "metrics_final.json"
        if not m.exists():
            continue
        info = json.loads(m.read_text())
        pr = info.get("test_pr_auc")
        if pr is None:
            continue
        by_variant[variant].append((d.name, float(pr)))
    return by_variant

baseline_rows = collect_baseline()
hybrid_rows = collect_hybrids()

if not baseline_rows:
    print("ERROR: no baseline i018 runs found at", BASELINE_DIR)
    raise SystemExit(1)

baseline_prs = [r[1] for r in baseline_rows]
baseline_mean = statistics.fmean(baseline_prs)
baseline_std = statistics.pstdev(baseline_prs) if len(baseline_prs) > 1 else 0.0

lines = []
lines.append("# Hybrid i018 + Primitive Results")
lines.append("")
lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
lines.append("")
lines.append(f"**Baseline i018 (base scale, 3 seeds)**: test_pr_auc = **{baseline_mean:.4f} +/- {baseline_std:.4f}** (n={len(baseline_prs)})")
lines.append("")
lines.append("Each hybrid grafts one primitive onto the i018 sheaf trunk via gated-logit fusion:")
lines.append("`final = sheaf_logit + sigmoid(gate) * primitive_logit`")
lines.append("")
lines.append("## Summary")
lines.append("")
lines.append("| Hybrid | n | mean test_pr_auc | std | delta vs baseline | verdict |")
lines.append("|---|---:|---:|---:|---:|---|")

verdicts = []
ranking = []
for variant in sorted(hybrid_rows.keys()):
    prs = [r[1] for r in hybrid_rows[variant]]
    if not prs:
        continue
    mean = statistics.fmean(prs)
    std = statistics.pstdev(prs) if len(prs) > 1 else 0.0
    delta = mean - baseline_mean
    if delta >= 0.005:
        v = "+lift (>=0.005)"
    elif delta <= -0.005:
        v = "-regression (>=0.005)"
    elif abs(delta) < 0.002:
        v = "wash (|delta|<0.002)"
    else:
        v = "small +" if delta > 0 else "small -"
    ranking.append((mean, variant, std, delta, v, len(prs)))
    verdicts.append((variant, mean, std, delta, v))

ranking.sort(reverse=True)
for mean, variant, std, delta, v, n in ranking:
    lines.append(f"| {variant} | {n} | {mean:.4f} | {std:.4f} | {delta:+.4f} | {v} |")

lines.append("")
lines.append("## Per-seed details")
lines.append("")
lines.append("```")
lines.append(f"# baseline i018 (n={len(baseline_rows)})")
for name, pr in baseline_rows:
    lines.append(f"  test_pr_auc={pr:.4f}  {name}")
for variant in sorted(hybrid_rows.keys()):
    lines.append(f"# {variant} (n={len(hybrid_rows[variant])})")
    for name, pr in hybrid_rows[variant]:
        lines.append(f"  test_pr_auc={pr:.4f}  {name}")
lines.append("```")
lines.append("")
lines.append("## Verdict summary")
lines.append("")
any_lift = any(v[3] >= 0.005 for v in verdicts)
if any_lift:
    best = max(verdicts, key=lambda x: x[3])
    lines.append(f"At least one hybrid shows a real lift over baseline. Best: **{best[0]}** with delta={best[3]:+.4f} (mean={best[1]:.4f} +/- {best[2]:.4f}).")
else:
    lines.append("No hybrid shows a clear (>=0.005 PR-AUC) lift over baseline i018. Adding these primitives to the sheaf trunk does not help on this benchmark at base scale.")
lines.append("")

results_path = REPORT_DIR / "results.md"
results_path.write_text("\n".join(lines))
print(f"Wrote summary to {results_path}")

# Also append a short TL;DR section to math_thesis.md so the i018 thesis page
# carries the result without requiring a separate file lookup.
tldr = []
tldr.append("")
tldr.append("## Hybrid Primitive Composition Results (auto-generated)")
tldr.append("")
tldr.append(f"Last run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
tldr.append("")
tldr.append(f"Baseline i018 (base scale, 3 seeds): test_pr_auc = {baseline_mean:.4f} +/- {baseline_std:.4f}")
tldr.append("")
tldr.append("Each hybrid grafts one primitive via gated-logit fusion.")
tldr.append("")
tldr.append("| Hybrid | mean test_pr_auc | delta vs baseline | verdict |")
tldr.append("|---|---:|---:|---|")
for mean, variant, std, delta, v, n in ranking:
    tldr.append(f"| {variant} | {mean:.4f} | {delta:+.4f} | {v} |")
tldr.append("")
tldr.append(f"Full details: [reports/hybrid_i018/results.md]({(REPORT_DIR / 'results.md').relative_to(ROOT)})")
tldr.append("")

text = THESIS.read_text()
marker = "## Hybrid Primitive Composition Results (auto-generated)"
if marker in text:
    text = text.split(marker)[0].rstrip() + "\n"
THESIS.write_text(text + "\n".join(tldr))
print(f"Appended TL;DR to {THESIS}")
PY

log "All done."
log "Full report: $REPORT_DIR/results.md"
log "Math thesis updated: $THESIS"
