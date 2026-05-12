#!/usr/bin/env python
"""Consolidate per-run scout metrics into a single shippable JSON + CSV.

Reads every run directory under `--results-root` (default the combined
scout view), pulls the key fields out of `metrics_final.json` and
`run_metadata.json`, and writes one flat row per run.  Output goes to
`reports/audits/scout_all_runs.{json,csv}` — small enough to commit and
self-explanatory enough that a reader can answer ``what worked and how
well'' without re-running the audit pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "run_name", "timestamp", "model_name", "input_encoding",
    "architecture_scale", "num_params", "estimated_flops_per_position",
    "epochs_trained", "best_epoch", "best_score",
    "test_accuracy", "test_precision", "test_recall", "test_f1",
    "test_roc_auc", "test_pr_auc",
    "val_accuracy", "val_pr_auc",
    "train_samples_per_second", "test_samples_per_second",
    "fit_elapsed_seconds",
    "batch_size", "learning_rate", "epochs_configured",
    "git_commit", "config_hash",
]


def _pluck(d: dict, *path, default=None) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def extract_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics_final.json"
    md_path = run_dir / "run_metadata.json"
    if not (metrics_path.exists() and md_path.exists()):
        return None
    try:
        m = json.loads(metrics_path.read_text())
        md = json.loads(md_path.read_text())
    except Exception:
        return None

    # metrics_final.json sometimes nests test / validation under top-level
    # keys; sometimes it stores flat test metrics directly.  Handle both.
    test = m.get("test") or m
    val = m.get("validation") or {}

    return {
        "run_name": md.get("run_name") or run_dir.name,
        "timestamp": md.get("timestamp"),
        "model_name": md.get("model_name"),
        "input_encoding": md.get("input_encoding"),
        "architecture_scale": _pluck(md, "architecture_scale", "variant"),
        "num_params": md.get("num_params"),
        "estimated_flops_per_position": _pluck(md, "complexity", "estimated_flops_per_position"),
        "epochs_trained": md.get("best_epoch") + 1 if md.get("best_epoch") is not None else None,
        "best_epoch": md.get("best_epoch"),
        "best_score": md.get("best_score"),
        "test_accuracy":  test.get("accuracy"),
        "test_precision": test.get("precision"),
        "test_recall":    test.get("recall"),
        "test_f1":        test.get("f1"),
        "test_roc_auc":   test.get("roc_auc"),
        "test_pr_auc":    test.get("pr_auc"),
        "val_accuracy":   val.get("accuracy"),
        "val_pr_auc":     val.get("pr_auc"),
        "train_samples_per_second": _pluck(md, "speed", "train_samples_per_second"),
        "test_samples_per_second":  _pluck(md, "speed", "final_eval", "test", "samples_per_second"),
        "fit_elapsed_seconds":      _pluck(md, "speed", "fit_elapsed_seconds"),
        "batch_size":      _pluck(md, "training", "batch_size"),
        "learning_rate":   _pluck(md, "training", "learning_rate"),
        "epochs_configured": _pluck(md, "training", "epochs"),
        "git_commit":      md.get("git_commit"),
        "config_hash":     md.get("config_hash"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, default=Path("_scout_combined_view"))
    p.add_argument("--out-json", type=Path,
                   default=Path("reports/audits/scout_all_runs.json"))
    p.add_argument("--out-csv", type=Path,
                   default=Path("reports/audits/scout_all_runs.csv"))
    args = p.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"--results-root does not exist: {args.results_root}")

    rows = []
    for d in sorted(args.results_root.iterdir()):
        if not d.is_dir():
            continue
        r = extract_run(d)
        if r is not None:
            rows.append(r)

    # Sort by test_pr_auc descending so the top of the file is the leaderboard.
    rows.sort(key=lambda r: (r.get("test_pr_auc") or -1.0), reverse=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {args.out_json} ({len(rows)} runs)")

    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
