#!/usr/bin/env bash
# Train i249 oriented_tactical_sheaf_fast (3 seeds x 3 scales) and compare its
# speed + accuracy against the i018 baseline already in results/paper_grade_top3/.
#
# Queues after any other run_paper_ready_all.py (waits for hybrid-i018 to finish).
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
CONFIG="$ROOT_DIR/ideas/registry/i249_oriented_tactical_sheaf_fast/config.yaml"
REPORT_DIR="$ROOT_DIR/reports/i249_fast"
RESULTS_DIR="$ROOT_DIR/results/i249_fast"
mkdir -p "$REPORT_DIR" "$RESULTS_DIR"
LAUNCHER_LOG="$REPORT_DIR/launcher.log"

log() { echo "[$(date -Is)] $*" | tee -a "$LAUNCHER_LOG"; }

log "i249 oriented_tactical_sheaf_fast launcher starting."
log "Will queue until no run_paper_ready_all.py is active AND the falsifier"
log "launcher (run_falsifier_i018.sh) has exited -- keeps the queue order"
log "hybrid -> falsifier -> i249 deterministic, no GPU race."

while pgrep -f "run_paper_ready_all.py" >/dev/null 2>&1 \
   || pgrep -f "run_falsifier_i018.sh" >/dev/null 2>&1; do
  log "GPU still busy (run_paper_ready_all.py or falsifier launcher active). Sleeping 60s..."
  sleep 60
done

log "GPU 0 free. Launching i249 run (3 seeds x 3 scales = 9 tasks)."

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/run_paper_ready_all.py \
  "$CONFIG" \
  --no-benchmarks --no-ideas \
  --seeds 42,43,44 \
  --scale-variants base:1,scale_up:1.5,scale_xl:2 \
  --batch-size-caps base:256,scale_up:192,scale_xl:128 \
  --epochs 20 --min-epochs 10 --patience 5 \
  --jobs 1 --gpu-ids 0 \
  --results-dir "$RESULTS_DIR" \
  --report-dir "$REPORT_DIR" \
  --state-path "$REPORT_DIR/state.json" \
  --logs-dir "$REPORT_DIR/logs" \
  --generated-config-dir "$REPORT_DIR/generated_configs" \
  --event-log "$REPORT_DIR/events.jsonl" \
  --timeline "$REPORT_DIR/timeline.md" 2>&1 | tee -a "$LAUNCHER_LOG"

log "i249 training done. Comparing speed + accuracy vs i018 baseline."

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" - <<PY 2>&1 | tee -a "$LAUNCHER_LOG"
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("$ROOT_DIR")
FAST_DIR = Path("$RESULTS_DIR")
BASELINE_DIR = ROOT / "results" / "paper_grade_top3"
REPORT_DIR = Path("$REPORT_DIR")

SCALES = ["base", "scale_up", "scale_xl"]

def scale_of(name: str) -> str:
    if "scale_xl" in name:
        return "scale_xl"
    if "scale_up" in name:
        return "scale_up"
    return "base"

def collect(dirpath: Path, match: str):
    by_scale = defaultdict(list)
    for run_dir in sorted(dirpath.glob(match)):
        m = run_dir / "metrics_final.json"
        if not m.exists():
            continue
        info = json.loads(m.read_text())
        pr = info.get("test_pr_auc")
        if pr is None:
            continue
        sp = run_dir / "speed_summary.json"
        fit_s = train_sps = None
        if sp.exists():
            s = json.loads(sp.read_text())
            fit_s = s.get("fit_elapsed_seconds")
            train_sps = (s.get("train") or {}).get("samples_per_second")
        by_scale[scale_of(run_dir.name)].append(
            {"pr": float(pr), "fit_s": fit_s, "train_sps": train_sps, "name": run_dir.name}
        )
    return by_scale

base = collect(BASELINE_DIR, "idea_i018_oriented_tactical_sheaf_laplacian_*seed*")
fast = collect(FAST_DIR, "idea_i249_oriented_tactical_sheaf_fast_*seed*")

def agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None, None
    return statistics.fmean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

lines = []
lines.append("# i249 oriented_tactical_sheaf_fast -- Speed + Accuracy vs i018")
lines.append("")
lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
lines.append("")
lines.append("i249 is a pure execution optimization of i018 -- same math, same parameters,")
lines.append("numerically verified identical (logits ~1e-8, gradients ~1e-10). The only")
lines.append("intended difference is GPU wall-clock. So test PR-AUC should match within seed")
lines.append("noise, and the training throughput should improve.")
lines.append("")
lines.append("| Scale | model | n | test PR-AUC (mean+/-std) | train samp/s (mean) | fit seconds (mean) | speedup |")
lines.append("|---|---|---:|---:|---:|---:|---:|")
for scale in SCALES:
    b = base.get(scale, [])
    f = fast.get(scale, [])
    bpr_m, bpr_s = agg(b, "pr")
    fpr_m, fpr_s = agg(f, "pr")
    bsps_m, _ = agg(b, "train_sps")
    fsps_m, _ = agg(f, "train_sps")
    bfit_m, _ = agg(b, "fit_s")
    ffit_m, _ = agg(f, "fit_s")
    def fmt(x, p=4):
        return "n/a" if x is None else f"{x:.{p}f}"
    speedup = "n/a"
    if bsps_m and fsps_m:
        speedup = f"{fsps_m / bsps_m:.2f}x"
    elif bfit_m and ffit_m and ffit_m > 0:
        speedup = f"{bfit_m / ffit_m:.2f}x"
    if b:
        lines.append(f"| {scale} | i018 baseline | {len(b)} | {fmt(bpr_m)} +/- {fmt(bpr_s)} | {fmt(bsps_m,1)} | {fmt(bfit_m,1)} | - |")
    if f:
        lines.append(f"| {scale} | i249 fast | {len(f)} | {fmt(fpr_m)} +/- {fmt(fpr_s)} | {fmt(fsps_m,1)} | {fmt(ffit_m,1)} | {speedup} |")
    if b and f and bpr_m is not None and fpr_m is not None:
        lines.append(f"| {scale} | **PR-AUC delta** | | **{fpr_m - bpr_m:+.4f}** | | | |")

lines.append("")
lines.append("## Per-seed detail")
lines.append("")
lines.append("```")
for label, data in [("i018 baseline", base), ("i249 fast", fast)]:
    lines.append(f"# {label}")
    for scale in SCALES:
        for r in data.get(scale, []):
            sps = f"{r['train_sps']:.0f}" if r["train_sps"] else "n/a"
            fit = f"{r['fit_s']:.0f}s" if r["fit_s"] else "n/a"
            lines.append(f"  {scale:<9} pr={r['pr']:.4f}  train_sps={sps:>8}  fit={fit:>8}  {r['name']}")
lines.append("```")
lines.append("")
lines.append("## Verdict")
lines.append("")
# overall verdict
all_pr_deltas = []
all_speedups = []
for scale in SCALES:
    b = base.get(scale, []); f = fast.get(scale, [])
    bpr_m, _ = agg(b, "pr"); fpr_m, _ = agg(f, "pr")
    bsps_m, _ = agg(b, "train_sps"); fsps_m, _ = agg(f, "train_sps")
    if bpr_m is not None and fpr_m is not None:
        all_pr_deltas.append(fpr_m - bpr_m)
    if bsps_m and fsps_m:
        all_speedups.append(fsps_m / bsps_m)
if all_pr_deltas:
    max_abs_delta = max(abs(d) for d in all_pr_deltas)
    if max_abs_delta < 0.005:
        lines.append(f"- **Accuracy preserved**: max |PR-AUC delta| across scales = {max_abs_delta:.4f} (< 0.005, within seed noise).")
    else:
        lines.append(f"- **WARNING**: max |PR-AUC delta| = {max_abs_delta:.4f} -- larger than expected for a numerically-identical variant; investigate.")
if all_speedups:
    lines.append(f"- **Speed**: training throughput speedup = {statistics.fmean(all_speedups):.2f}x mean across scales (range {min(all_speedups):.2f}x - {max(all_speedups):.2f}x).")
else:
    lines.append("- Speed: no speed_summary.json throughput found; check fit_elapsed_seconds column.")
lines.append("")

out = REPORT_DIR / "speed_report.md"
out.write_text("\n".join(lines))
print(f"Wrote {out}")
print("\n".join(lines[-6:]))
PY

log "All done."
log "Speed report: $REPORT_DIR/speed_report.md"
