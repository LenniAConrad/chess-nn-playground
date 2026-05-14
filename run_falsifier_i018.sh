#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
CONFIG="$ROOT_DIR/ideas/registry/i018_oriented_tactical_sheaf_laplacian/config_falsifier.yaml"
THESIS="$ROOT_DIR/ideas/registry/i018_oriented_tactical_sheaf_laplacian/math_thesis.md"
REPORT_DIR="$ROOT_DIR/reports/falsifier_i018"
RESULTS_DIR="$ROOT_DIR/results/falsifier_i018"
mkdir -p "$REPORT_DIR" "$RESULTS_DIR"
LAUNCHER_LOG="$REPORT_DIR/launcher.log"

log() { echo "[$(date -Is)] $*" | tee -a "$LAUNCHER_LOG"; }

log "Falsifier launcher for i018 oriented_tactical_sheaf_laplacian."
log "Will queue until GPU 0 is free, then train 3 seeds at base scale with scrambled relations."

# Wait for both paper-grade-top3 AND the primitive_pipeline scout (if anything ressurects).
while pgrep -f "run_paper_ready_all.py" >/dev/null 2>&1; do
  log "Another run_paper_ready_all.py still active. Sleeping 60s..."
  sleep 60
done

log "GPU 0 free. Launching falsifier run (3 seeds, base scale)."

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/run_paper_ready_all.py \
  "$CONFIG" \
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

log "Falsifier training done. Computing mean +/- std vs baseline and updating math_thesis.md."

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" - <<PY 2>&1 | tee -a "$LAUNCHER_LOG"
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("$ROOT_DIR")
REPORT_DIR = Path("$REPORT_DIR")
RESULTS_DIR = Path("$RESULTS_DIR")
THESIS = Path("$THESIS")
BASELINE_DIR = ROOT / "results" / "paper_grade_top3"

def collect(dirpath: Path, label: str):
    rows = []
    for run_dir in sorted(dirpath.glob("idea_i018_*_seed*")):
        if "scale_up" in run_dir.name or "scale_xl" in run_dir.name:
            continue
        m = run_dir / "metrics_final.json"
        if not m.exists():
            continue
        d = json.loads(m.read_text())
        pr = d.get("test_pr_auc")
        if pr is None:
            continue
        rows.append((run_dir.name, float(pr), d.get("test_accuracy")))
    if not rows:
        return None
    prs = [r[1] for r in rows]
    return {
        "label": label,
        "runs": rows,
        "mean": statistics.fmean(prs),
        "std": statistics.pstdev(prs) if len(prs) > 1 else 0.0,
        "n": len(prs),
    }

baseline = collect(BASELINE_DIR, "Baseline (real chess geometry, scramble_relations=false)")
falsifier = collect(RESULTS_DIR, "Falsifier (degree-preserving random masks, scramble_relations=true)")

if baseline is None:
    print("ERROR: no baseline runs found at", BASELINE_DIR)
    raise SystemExit(1)
if falsifier is None:
    print("ERROR: no falsifier runs found at", RESULTS_DIR)
    raise SystemExit(1)

delta = falsifier["mean"] - baseline["mean"]
abs_delta = abs(delta)
verdict_unchanged = abs_delta < 0.01
verdict_strong_support = (-delta) >= 0.02

if verdict_strong_support:
    verdict = "**THESIS SUPPORTED**: falsifier drops by >= 0.02 PR-AUC; real chess geometry is doing real work."
elif verdict_unchanged:
    verdict = "**THESIS REJECTED**: falsifier within 0.01 PR-AUC of baseline; the sheaf math does not need real chess relations."
elif -delta > 0:
    verdict = f"**INCONCLUSIVE (modest support)**: falsifier drops by {-delta:+.4f}; small effect, between the strong-support and rejection thresholds."
else:
    verdict = f"**INCONCLUSIVE (falsifier improves)**: scrambling raised PR-AUC by {delta:+.4f}; check for confounds (training instability, init effect)."

block = []
block.append("")
block.append("## Falsifier Results (auto-generated)")
block.append("")
block.append(f"Last run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
block.append("")
block.append("Falsifier from this thesis: *\"replacing real relation masks with degree-preserving random masks; if performance is unchanged the typed-relation thesis is rejected and the family must not be re-scaled.\"*")
block.append("")
block.append(f"Setup: 3 seeds (42, 43, 44), base scale, 20 epochs, single-seed configs identical to the paper-grade baseline runs except `scramble_relations: true` (per-(batch, relation) random column permutation; preserves per-source out-degree exactly).")
block.append("")
block.append("| Variant | n | Test PR-AUC (mean) | Test PR-AUC (std) |")
block.append("|---|---:|---:|---:|")
block.append(f"| {baseline['label']} | {baseline['n']} | {baseline['mean']:.4f} | {baseline['std']:.4f} |")
block.append(f"| {falsifier['label']} | {falsifier['n']} | {falsifier['mean']:.4f} | {falsifier['std']:.4f} |")
block.append(f"| **Delta (falsifier - baseline)** | | **{delta:+.4f}** | |")
block.append("")
block.append("### Per-seed details")
block.append("")
block.append("```")
for label, info in [("baseline", baseline), ("falsifier", falsifier)]:
    block.append(f"# {label}")
    for name, pr, acc in info["runs"]:
        block.append(f"  test_pr_auc={pr:.4f}  test_acc={acc if acc is not None else 'NA'}  {name}")
block.append("```")
block.append("")
block.append("### Verdict")
block.append("")
block.append(verdict)
block.append("")
block.append("Thresholds:")
block.append("- strong support: falsifier drops by >= 0.02 PR-AUC")
block.append("- rejection: falsifier within 0.01 PR-AUC of baseline")
block.append("- between: inconclusive")
block.append("")

text = THESIS.read_text()
marker = "## Falsifier Results (auto-generated)"
if marker in text:
    text = text.split(marker)[0].rstrip() + "\n"
THESIS.write_text(text + "\n".join(block))
print("Appended falsifier results to", THESIS)
print(verdict)
PY

log "All done."
log "Math thesis updated: $THESIS"
log "Falsifier launcher log: $LAUNCHER_LOG"
