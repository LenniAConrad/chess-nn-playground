#!/usr/bin/env python
"""Matched-recall false-positive and worst-slice analysis (read-only).

For each puzzle_binary run under `--results-root`, picks a probability threshold
to match a target recall, then reports:
  - total false positives at that threshold
  - "near-puzzle" false positives (true_fine_label == 1 -> verified_near_puzzle)
  - per-slice test accuracy on `crtk_difficulty`, `crtk_eval_bucket`, `crtk_phase`

This is the metric set elevated by docs/reliable_training_protocol.md and is the
"right scoreboard" for selective/abstention models like i011 vetoselect and i012
dykstra_lcp.

Output: <out-root>/matched_recall_fp_report.{json,md}

Usage:
  scripts/analyze_matched_recall_fp.py \\
    --results-root _archive/paper_ready_all_2026-05-09/results/paper_ready_all \\
    --out-root reports/audits \\
    --recall-targets 0.80 0.85
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_predictions(run_dir: Path) -> pd.DataFrame | None:
    candidate_names = ("predictions_test.parquet", "predictions_val.parquet")
    for name in candidate_names:
        p = run_dir / name
        if p.exists():
            try:
                df = pd.read_parquet(p)
                df.attrs["source_file"] = name
                return df
            except Exception:
                continue
    return None


def _is_puzzle_binary_run(df: pd.DataFrame) -> bool:
    # puzzle_binary: positive class is true_fine_label == 2, AND the test set
    # contains fine_label == 1 (verified-near-puzzle negatives). coarse_binary
    # runs have n_near == 0 by construction, so we filter them out: they would
    # trivially top the near_FP_rate ranking with 0/0.
    if "true_fine_label" not in df.columns:
        return False
    if "prob_1" not in df.columns and "probabilities" not in df.columns:
        return False
    if "true_label" not in df.columns:
        return False
    y_true = df["true_label"].to_numpy(dtype=int)
    fine = df["true_fine_label"].to_numpy(dtype=int)
    n_near = int(((y_true == 0) & (fine == 1)).sum())
    if n_near == 0:
        return False
    return True


def _positive_prob(df: pd.DataFrame) -> np.ndarray:
    if "prob_1" in df.columns:
        return df["prob_1"].to_numpy(dtype=float)
    # fall back to second element of `probabilities`
    return np.array([p[1] for p in df["probabilities"]], dtype=float)


def _threshold_at_recall(probs: np.ndarray, y_true: np.ndarray, target_recall: float) -> float:
    """Return the largest threshold at which recall >= target_recall."""
    pos_mask = y_true == 1
    n_pos = int(pos_mask.sum())
    if n_pos == 0:
        return float("nan")
    pos_probs = np.sort(probs[pos_mask])[::-1]  # descending
    idx = int(np.ceil(target_recall * n_pos)) - 1
    idx = max(0, min(idx, n_pos - 1))
    return float(pos_probs[idx])


def analyze_run(run_dir: Path, recall_targets: list[float]) -> dict[str, Any] | None:
    df = _load_predictions(run_dir)
    if df is None or not _is_puzzle_binary_run(df):
        return None

    y_true = df["true_label"].to_numpy(dtype=int) if "true_label" in df.columns else None
    if y_true is None:
        # derive from fine label
        y_true = (df["true_fine_label"].to_numpy(dtype=int) == 2).astype(int)

    probs = _positive_prob(df)
    fine = df["true_fine_label"].to_numpy(dtype=int)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    n_near = int(((y_true == 0) & (fine == 1)).sum())  # verified-near-puzzle negatives
    n_far = int(((y_true == 0) & (fine == 0)).sum())   # other negatives

    # Per-recall-target threshold sweep
    per_target: list[dict[str, Any]] = []
    for target in recall_targets:
        thr = _threshold_at_recall(probs, y_true, target)
        if np.isnan(thr):
            per_target.append({"target_recall": target, "threshold": None})
            continue
        pred_pos = probs >= thr
        tp = int(((y_true == 1) & pred_pos).sum())
        fp = int(((y_true == 0) & pred_pos).sum())
        fn = int(((y_true == 1) & ~pred_pos).sum())
        tn = int(((y_true == 0) & ~pred_pos).sum())
        fp_near = int(((y_true == 0) & (fine == 1) & pred_pos).sum())
        fp_far = int(((y_true == 0) & (fine == 0) & pred_pos).sum())
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        per_target.append({
            "target_recall": target,
            "threshold": float(thr),
            "achieved_recall": float(recall),
            "precision": float(precision),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fp_near_puzzle": fp_near,
            "fp_far_negative": fp_far,
            "near_puzzle_fp_rate": fp_near / max(n_near, 1),
            "far_negative_fp_rate": fp_far / max(n_far, 1),
        })

    # Per-slice accuracy at each target threshold (using the first target threshold for now)
    primary_target = recall_targets[0]
    primary_thr = per_target[0].get("threshold")
    # Single-valued slice columns
    slice_keys = [c for c in ("crtk_difficulty", "crtk_eval_bucket", "crtk_phase") if c in df.columns]
    # Multi-valued: pipe-separated lists of tactic motifs (e.g. "fork|hanging").
    # Promotion/underpromotion live here per docs/reliable_training_protocol.md.
    motif_col = "crtk_tactic_motifs" if "crtk_tactic_motifs" in df.columns else None
    slices: dict[str, dict[str, dict[str, float | int]]] = {}
    if primary_thr is not None:
        pred_pos = probs >= primary_thr
        correct_at_thr = (pred_pos.astype(int) == y_true)
        for col in slice_keys:
            slice_stats: dict[str, dict[str, float | int]] = {}
            for value, group in df.groupby(col):
                mask = df[col] == value
                if mask.sum() == 0:
                    continue
                acc = float(correct_at_thr[mask].mean())
                slice_stats[str(value)] = {
                    "n": int(mask.sum()),
                    "accuracy_at_target_recall": acc,
                }
            slices[col] = slice_stats
        # Tactic-motif slices: each row's motif column is "|"-separated; slice by
        # motif membership rather than by exact value. Always emit promotion and
        # underpromotion explicitly (even if zero rows) so the audit table is stable.
        if motif_col is not None:
            motif_strs = df[motif_col].fillna("").astype(str).to_numpy()
            tokenized = [set(s.split("|")) - {""} for s in motif_strs]
            # Universe = all motifs that appear at least once + the two we always want
            universe: set[str] = set()
            for s in tokenized:
                universe.update(s)
            for required in ("promotion", "underpromotion"):
                universe.add(required)
            motif_stats: dict[str, dict[str, float | int]] = {}
            for motif in sorted(universe):
                mask = np.array([motif in s for s in tokenized], dtype=bool)
                n = int(mask.sum())
                if n == 0:
                    motif_stats[motif] = {"n": 0, "accuracy_at_target_recall": float("nan")}
                    continue
                acc = float(correct_at_thr[mask].mean())
                # Also compute near-puzzle FP rate restricted to this motif slice
                neg_in_slice = mask & (y_true == 0)
                near_neg_in_slice = neg_in_slice & (fine == 1)
                fp_in_slice = neg_in_slice & pred_pos
                near_fp_in_slice = near_neg_in_slice & pred_pos
                motif_stats[motif] = {
                    "n": n,
                    "n_negative": int(neg_in_slice.sum()),
                    "n_near_negative": int(near_neg_in_slice.sum()),
                    "accuracy_at_target_recall": acc,
                    "fp": int(fp_in_slice.sum()),
                    "near_fp": int(near_fp_in_slice.sum()),
                    "near_fp_rate": (
                        float(near_fp_in_slice.sum() / max(near_neg_in_slice.sum(), 1))
                    ),
                }
            slices[motif_col] = motif_stats

    # Identify worst slices
    worst_slices: list[dict[str, Any]] = []
    for col, stats in slices.items():
        for value, s in stats.items():
            if s["n"] < 50:  # ignore tiny slices
                continue
            worst_slices.append({"slice_dim": col, "slice_value": value,
                                 "n": s["n"], "accuracy": s["accuracy_at_target_recall"]})
    worst_slices.sort(key=lambda r: r["accuracy"])

    return {
        "run_dir": run_dir.as_posix(),
        "run_name": run_dir.name,
        "predictions_file": df.attrs.get("source_file"),
        "n_samples": int(len(df)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_near_puzzle_negative": n_near,
        "n_far_negative": n_far,
        "per_recall_target": per_target,
        "slice_accuracies_at_target_recall": slices,
        "primary_target_recall": primary_target,
        "worst_5_slices": worst_slices[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path,
                        help="Root containing per-run subdirectories with predictions parquet files")
    parser.add_argument("--out-root", required=True, type=Path,
                        help="Where to write matched_recall_fp_report.{json,md}")
    parser.add_argument("--filter", default=None,
                        help="Only include run names containing this substring")
    parser.add_argument("--recall-targets", nargs="+", type=float,
                        default=[0.80, 0.85],
                        help="Target recall levels to evaluate at (default 0.80 0.85)")
    args = parser.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"--results-root does not exist: {args.results_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    skipped = 0
    for run_dir in sorted(args.results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if args.filter and args.filter not in run_dir.name:
            continue
        try:
            r = analyze_run(run_dir, args.recall_targets)
        except Exception as exc:
            print(f"  ! {run_dir.name}: {type(exc).__name__}: {exc}")
            r = None
        if r is None:
            skipped += 1
            continue
        results.append(r)

    # Rank by near-puzzle FP rate at first target recall (lower is better)
    def _key(r: dict[str, Any]) -> float:
        pt = r["per_recall_target"][0]
        return pt.get("near_puzzle_fp_rate", float("inf"))
    results.sort(key=_key)

    report = {
        "results_root": args.results_root.as_posix(),
        "n_runs": len(results),
        "n_skipped": skipped,
        "recall_targets": args.recall_targets,
        "ranked_by": f"near_puzzle_fp_rate at recall {args.recall_targets[0]}",
        "runs": results,
    }
    json_out = args.out_root / "matched_recall_fp_report.json"
    md_out = args.out_root / "matched_recall_fp_report.md"
    json_out.write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# Matched-recall false-positive and worst-slice report",
        "",
        f"- results-root: `{report['results_root']}`",
        f"- runs analyzed: {report['n_runs']} (skipped {report['n_skipped']} non-puzzle_binary or missing predictions)",
        f"- recall targets: {args.recall_targets}",
        f"- ranking: lower `near_puzzle_fp_rate` at recall {args.recall_targets[0]} is better",
        "",
        f"## Top 25 by lowest near-puzzle FP rate at recall {args.recall_targets[0]}",
        "",
        "| run | n_pos | n_near | recall | precision | total_FP | near_FP | near_FP_rate | far_FP_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results[:25]:
        pt = r["per_recall_target"][0]
        if pt.get("threshold") is None:
            continue
        lines.append(
            f"| `{r['run_name']}` | {r['n_positive']} | {r['n_near_puzzle_negative']} | "
            f"{pt['achieved_recall']:.3f} | {pt['precision']:.3f} | {pt['fp']} | "
            f"{pt['fp_near_puzzle']} | {pt['near_puzzle_fp_rate']:.3f} | {pt['far_negative_fp_rate']:.3f} |"
        )
    if len(args.recall_targets) > 1:
        lines.extend([
            "",
            f"## Same table at recall {args.recall_targets[1]}",
            "",
            "| run | recall | precision | total_FP | near_FP | near_FP_rate | far_FP_rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        # re-sort by second target
        results2 = sorted(results, key=lambda r: r["per_recall_target"][1].get("near_puzzle_fp_rate", float("inf")))
        for r in results2[:25]:
            pt = r["per_recall_target"][1]
            if pt.get("threshold") is None:
                continue
            lines.append(
                f"| `{r['run_name']}` | {pt['achieved_recall']:.3f} | {pt['precision']:.3f} | "
                f"{pt['fp']} | {pt['fp_near_puzzle']} | {pt['near_puzzle_fp_rate']:.3f} | {pt['far_negative_fp_rate']:.3f} |"
            )

    lines.extend(["", "## Worst slices (top 5 per run, sample size >= 50)", ""])
    for r in results[:10]:
        if not r.get("worst_5_slices"):
            continue
        lines.append(f"### `{r['run_name']}`  (slice acc evaluated at recall {r['primary_target_recall']})")
        lines.append("")
        lines.append("| slice | value | n | accuracy |")
        lines.append("|---|---|---:|---:|")
        for s in r["worst_5_slices"]:
            lines.append(f"| {s['slice_dim']} | {s['slice_value']} | {s['n']} | {s['accuracy']:.3f} |")
        lines.append("")

    # Dedicated promotion / underpromotion table (these are critical per
    # docs/reliable_training_protocol.md and are not single-valued slices).
    lines.extend([
        "",
        "## Promotion / underpromotion tactic-motif slices (top 25 by near-puzzle FP rejection)",
        "",
        "| run | motif | n | n_near_neg | accuracy@recall | near_FP | near_FP_rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    motif_rows: list[tuple[float, str]] = []
    for r in results:
        slices = r.get("slice_accuracies_at_target_recall", {}) or {}
        motif_slice = slices.get("crtk_tactic_motifs", {}) if isinstance(slices, dict) else {}
        for motif in ("promotion", "underpromotion"):
            s = motif_slice.get(motif) if isinstance(motif_slice, dict) else None
            if not isinstance(s, dict) or s.get("n", 0) == 0:
                continue
            row = (
                f"| `{r['run_name']}` | {motif} | {s['n']} | {s.get('n_near_negative', 0)} | "
                f"{s['accuracy_at_target_recall']:.3f} | {s.get('near_fp', 0)} | "
                f"{s.get('near_fp_rate', float('nan')):.3f} |"
            )
            sort_key = s.get("near_fp_rate", float("inf"))
            motif_rows.append((sort_key, row))
    motif_rows.sort(key=lambda x: x[0])
    for _, row in motif_rows[:25]:
        lines.append(row)

    md_out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_out} and {md_out}")
    print(f"Runs analyzed: {len(results)}, skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
