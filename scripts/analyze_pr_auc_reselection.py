#!/usr/bin/env python
"""Read-only PR-AUC reselection report.

For each completed run under `--results-root`, identify which validation epoch
would have been chosen if best-checkpoint selection had used PR AUC instead of
F1. Compares against the actually-selected epoch (driven by the trainer's
hardcoded F1 monitor at the time of the run).

This is REPORT-ONLY. It does NOT modify metrics_final.json, run_summary.md, or
any leaderboard. Only val PR AUC at the alternative epoch can be reported,
because per-epoch checkpoints are not retained — so we cannot re-evaluate test
PR AUC for an epoch other than the one whose checkpoint exists on disk.

Output:
  <out-root>/pr_auc_reselection_report.json
  <out-root>/pr_auc_reselection_report.md

Usage:
  scripts/analyze_pr_auc_reselection.py \
    --results-root _archive/paper_ready_all_2026-05-09/results/paper_ready_all \
    --out-root reports/audits
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _argmax_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    eligible = [r for r in rows if isinstance(r.get(key), (int, float))]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r[key])


def analyze_run(run_dir: Path) -> dict[str, Any] | None:
    history = _load_json(run_dir / "metrics_history.json")
    if not history:
        return None
    val_rows = history.get("val") or history.get("validation") or []
    if not val_rows:
        return None

    f1_best = _argmax_by(val_rows, "f1")
    pr_best = _argmax_by(val_rows, "pr_auc")
    if f1_best is None or pr_best is None:
        return None

    final = _load_json(run_dir / "metrics_final.json") or {}
    metadata = _load_json(run_dir / "run_metadata.json") or {}

    return {
        "run_dir": run_dir.as_posix(),
        "run_name": run_dir.name,
        "selected_epoch_f1": int(f1_best.get("epoch", -1)),
        "selected_val_f1": float(f1_best.get("f1")),
        "selected_val_pr_auc_at_f1_epoch": (
            float(f1_best.get("pr_auc")) if f1_best.get("pr_auc") is not None else None
        ),
        "would_select_epoch_pr_auc": int(pr_best.get("epoch", -1)),
        "would_select_val_pr_auc": float(pr_best.get("pr_auc")),
        "would_select_val_f1_at_pr_auc_epoch": (
            float(pr_best.get("f1")) if pr_best.get("f1") is not None else None
        ),
        "epoch_diff": int(pr_best.get("epoch", 0)) - int(f1_best.get("epoch", 0)),
        "val_pr_auc_lift": float(pr_best.get("pr_auc")) - (
            float(f1_best.get("pr_auc")) if f1_best.get("pr_auc") is not None else float("nan")
        ),
        # Test PR AUC of the f1-selected checkpoint (this IS in the archive)
        "test_pr_auc_at_f1_epoch": (
            float(final.get("test_pr_auc")) if final.get("test_pr_auc") is not None else None
        ),
        "test_pr_auc_alternative_unavailable_reason": (
            "Per-epoch checkpoints not retained; only checkpoint_best.pt (f1-best) and "
            "checkpoint_last.pt are saved. To get test PR AUC at the would-be epoch, "
            "retrain with training.monitor: pr_auc."
        ),
        "n_epochs": len(val_rows),
        "monitor_used": (
            metadata["training"].get("monitor")
            if isinstance(metadata.get("training"), dict) else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path,
                        help="Root containing per-run subdirectories with metrics_history.json")
    parser.add_argument("--out-root", required=True, type=Path,
                        help="Where to write pr_auc_reselection_report.{json,md}")
    parser.add_argument("--filter", default=None,
                        help="Only include run names containing this substring")
    args = parser.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"--results-root does not exist: {args.results_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped = 0
    for run_dir in sorted(args.results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if args.filter and args.filter not in run_dir.name:
            continue
        result = analyze_run(run_dir)
        if result is None:
            skipped += 1
            continue
        rows.append(result)

    rows.sort(key=lambda r: r["val_pr_auc_lift"], reverse=True)

    report = {
        "results_root": args.results_root.as_posix(),
        "n_runs": len(rows),
        "n_skipped_no_history": skipped,
        "summary": {
            "n_with_lift": sum(1 for r in rows if r["val_pr_auc_lift"] > 1e-6),
            "n_no_change": sum(1 for r in rows if abs(r["val_pr_auc_lift"]) <= 1e-6),
            "n_negative_lift": sum(1 for r in rows if r["val_pr_auc_lift"] < -1e-6),
            "mean_val_pr_auc_lift": (
                sum(r["val_pr_auc_lift"] for r in rows) / len(rows) if rows else 0.0
            ),
            "max_val_pr_auc_lift": max((r["val_pr_auc_lift"] for r in rows), default=0.0),
            "epoch_diff_distribution": dict(
                Counter(r["epoch_diff"] for r in rows).most_common(15)
            ),
        },
        "runs": rows,
    }

    json_out = args.out_root / "pr_auc_reselection_report.json"
    md_out = args.out_root / "pr_auc_reselection_report.md"
    json_out.write_text(json.dumps(report, indent=2, sort_keys=False))

    lines = [
        "# PR-AUC reselection report (READ-ONLY, val-only)",
        "",
        "**Question:** if best-checkpoint selection had used PR AUC instead of F1, ",
        "which val epoch would have been chosen, and how much higher would *val* PR AUC have been?",
        "",
        "**Important caveat:** test PR AUC at the would-be epoch CANNOT be reported here.",
        "Per-epoch checkpoints are not retained, so the only test_pr_auc available is at",
        "the F1-best epoch (i.e., the existing `checkpoint_best.pt`). To get true",
        "corrected test PR AUC, retrain with `training.monitor: pr_auc`.",
        "",
        f"- results-root: `{report['results_root']}`",
        f"- runs analyzed: {report['n_runs']}  (skipped {report['n_skipped_no_history']} without metrics_history.json)",
        f"- runs where PR-AUC reselection picks a *different* epoch: {report['summary']['n_with_lift'] + report['summary']['n_negative_lift']}",
        f"- mean val PR AUC lift from reselection: {report['summary']['mean_val_pr_auc_lift']:+.4f}",
        f"- max  val PR AUC lift: {report['summary']['max_val_pr_auc_lift']:+.4f}",
        "",
        "## Top 25 by val PR AUC lift",
        "",
        "| run | f1-epoch | pr_auc-epoch | Δepoch | val PR AUC (f1-sel) | val PR AUC (pr-sel) | lift | test PR AUC (f1-sel ckpt only) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows[:25]:
        test_pr = f"{r['test_pr_auc_at_f1_epoch']:.4f}" if r['test_pr_auc_at_f1_epoch'] is not None else "-"
        f1_at = r["selected_val_pr_auc_at_f1_epoch"]
        f1_at_str = f"{f1_at:.4f}" if f1_at is not None else "-"
        lines.append(
            f"| `{r['run_name']}` | {r['selected_epoch_f1']} | {r['would_select_epoch_pr_auc']} | "
            f"{r['epoch_diff']:+d} | {f1_at_str} | {r['would_select_val_pr_auc']:.4f} | "
            f"{r['val_pr_auc_lift']:+.4f} | {test_pr} |"
        )
    md_out.write_text("\n".join(lines) + "\n")

    print(f"Wrote {json_out} and {md_out}")
    print(f"Runs analyzed: {len(rows)}; mean val PR AUC lift: {report['summary']['mean_val_pr_auc_lift']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
