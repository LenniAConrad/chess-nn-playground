#!/usr/bin/env python
"""Per-class performance matrix on puzzle_binary runs (read-only).

For each (model_group, slice) pair, computes per-slice **test PR AUC** and
**accuracy at a target overall recall** using existing predictions parquets.

Slices computed:
  - crtk_difficulty (5 levels: very_easy / easy / medium / hard / very_hard)
  - crtk_phase (opening / middlegame / endgame)
  - crtk_eval_bucket (9 levels)
  - crtk_to_move (white / black)
  - crtk_tactic_motifs (multi-valued: hanging / fork / pin / skewer /
    overload / discovered_attack / mate_in_1 / promotion / underpromotion)

Aggregation:
  - Runs are grouped by their family name (everything before "_seedNN").
  - Per-slice metrics are computed per-seed, then averaged with std across seeds.
  - Skips groups with <2 seeds (no std possible).

Output:
  <out-root>/per_class_benchmark.{json,md}

Usage:
  scripts/analyze_per_class_benchmark.py \\
    --results-root _combined_view \\
    --out-root reports/audits \\
    --recall-target 0.80
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


SINGLE_VALUED_SLICES = ("crtk_difficulty", "crtk_phase", "crtk_eval_bucket", "crtk_to_move")
MOTIF_COLUMN = "crtk_tactic_motifs"
INTERESTING_MOTIFS = (
    "hanging", "fork", "pin", "skewer", "overload",
    "discovered_attack", "mate_in_1", "promotion", "underpromotion",
)


def _load_predictions(run_dir: Path) -> pd.DataFrame | None:
    for name in ("predictions_test.parquet", "predictions_val.parquet"):
        p = run_dir / name
        if p.exists():
            try:
                df = pd.read_parquet(p)
                df.attrs["source_file"] = name
                return df
            except Exception:
                continue
    return None


def _is_puzzle_binary(df: pd.DataFrame) -> bool:
    if "true_fine_label" not in df.columns or "true_label" not in df.columns:
        return False
    if "prob_1" not in df.columns and "probabilities" not in df.columns:
        return False
    y = df["true_label"].to_numpy(dtype=int)
    fine = df["true_fine_label"].to_numpy(dtype=int)
    n_near = int(((y == 0) & (fine == 1)).sum())
    return n_near > 0


def _positive_prob(df: pd.DataFrame) -> np.ndarray:
    if "prob_1" in df.columns:
        return df["prob_1"].to_numpy(dtype=float)
    return np.array([p[1] for p in df["probabilities"]], dtype=float)


def _threshold_at_recall(probs: np.ndarray, y_true: np.ndarray, target_recall: float) -> float:
    pos = probs[y_true == 1]
    if pos.size == 0:
        return float("nan")
    pos_sorted = np.sort(pos)[::-1]
    idx = int(np.ceil(target_recall * pos.size)) - 1
    return float(pos_sorted[max(0, min(idx, pos.size - 1))])


def _slice_metrics(probs_slice: np.ndarray, y_slice: np.ndarray, thr: float) -> dict[str, float]:
    n = int(y_slice.size)
    n_pos = int((y_slice == 1).sum())
    n_neg = int((y_slice == 0).sum())
    out: dict[str, float] = {"n": float(n), "n_pos": float(n_pos), "n_neg": float(n_neg)}
    # PR AUC restricted to slice (only meaningful with both classes present)
    if n_pos > 0 and n_neg > 0:
        try:
            out["pr_auc"] = float(average_precision_score(y_slice, probs_slice))
        except Exception:
            out["pr_auc"] = float("nan")
    else:
        out["pr_auc"] = float("nan")
    # Accuracy at the global recall-0.80 threshold restricted to this slice
    if n > 0 and not np.isnan(thr):
        pred = (probs_slice >= thr).astype(int)
        out["accuracy_at_target_recall"] = float((pred == y_slice).mean())
        # Recall and precision restricted to the slice (fixed-threshold)
        tp = int(((y_slice == 1) & (pred == 1)).sum())
        fp = int(((y_slice == 0) & (pred == 1)).sum())
        fn = int(((y_slice == 1) & (pred == 0)).sum())
        out["slice_recall"] = float(tp / max(tp + fn, 1))
        out["slice_precision"] = float(tp / max(tp + fp, 1))
    return out


def analyze_run(run_dir: Path, target_recall: float) -> dict[str, Any] | None:
    df = _load_predictions(run_dir)
    if df is None or not _is_puzzle_binary(df):
        return None
    y = df["true_label"].to_numpy(dtype=int)
    probs = _positive_prob(df)
    thr = _threshold_at_recall(probs, y, target_recall)

    overall = _slice_metrics(probs, y, thr)
    overall["threshold_at_target_recall"] = thr

    per_slice: dict[str, dict[str, dict[str, float]]] = {}
    for col in SINGLE_VALUED_SLICES:
        if col not in df.columns:
            continue
        per_value: dict[str, dict[str, float]] = {}
        for value, group in df.groupby(col):
            mask = (df[col] == value).to_numpy()
            if mask.sum() == 0:
                continue
            per_value[str(value)] = _slice_metrics(probs[mask], y[mask], thr)
        per_slice[col] = per_value

    if MOTIF_COLUMN in df.columns:
        motif_strs = df[MOTIF_COLUMN].fillna("").astype(str).to_numpy()
        tokenized = [set(s.split("|")) - {""} for s in motif_strs]
        per_motif: dict[str, dict[str, float]] = {}
        for motif in INTERESTING_MOTIFS:
            mask = np.array([motif in s for s in tokenized], dtype=bool)
            if mask.sum() == 0:
                per_motif[motif] = {"n": 0.0, "pr_auc": float("nan")}
                continue
            per_motif[motif] = _slice_metrics(probs[mask], y[mask], thr)
        per_slice[MOTIF_COLUMN] = per_motif

    return {
        "run_dir": run_dir.as_posix(),
        "run_name": run_dir.name,
        "predictions_file": df.attrs.get("source_file"),
        "overall": overall,
        "per_slice": per_slice,
    }


def _group_name(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", run_name)


def _aggregate(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--recall-target", type=float, default=0.80)
    parser.add_argument("--filter", default=None, help="Only include run names containing this substring")
    parser.add_argument("--min-seeds", type=int, default=2,
                        help="Skip groups with fewer than this many seeds (default 2)")
    args = parser.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"--results-root does not exist: {args.results_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for run_dir in sorted(args.results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if args.filter and args.filter not in run_dir.name:
            continue
        try:
            r = analyze_run(run_dir, args.recall_target)
        except Exception as exc:
            print(f"  ! {run_dir.name}: {type(exc).__name__}: {exc}")
            r = None
        if r is None:
            skipped += 1
            continue
        by_group[_group_name(r["run_name"])].append(r)

    # Build per-group aggregated view
    aggregated: list[dict[str, Any]] = []
    for group, runs in by_group.items():
        if len(runs) < args.min_seeds:
            continue
        # Overall test PR AUC mean
        overall_pr = [r["overall"]["pr_auc"] for r in runs]
        overall_acc = [r["overall"].get("accuracy_at_target_recall", float("nan")) for r in runs]
        # Per-slice aggregation
        per_slice_agg: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        # Collect union of all slice values across seeds (in case one seed missed a value)
        slice_dims = set()
        for r in runs:
            slice_dims.update(r["per_slice"].keys())
        for dim in slice_dims:
            values_seen: set[str] = set()
            for r in runs:
                values_seen.update(r["per_slice"].get(dim, {}).keys())
            agg_for_dim: dict[str, dict[str, dict[str, float]]] = {}
            for v in values_seen:
                pr_vals = []
                acc_vals = []
                n_vals = []
                for r in runs:
                    cell = r["per_slice"].get(dim, {}).get(v)
                    if not cell:
                        continue
                    pr_vals.append(cell.get("pr_auc", float("nan")))
                    acc_vals.append(cell.get("accuracy_at_target_recall", float("nan")))
                    n_vals.append(cell.get("n", 0))
                agg_for_dim[v] = {
                    "pr_auc": _aggregate(pr_vals),
                    "accuracy_at_target_recall": _aggregate(acc_vals),
                    "n_per_seed_mean": float(np.mean(n_vals)) if n_vals else 0.0,
                }
            per_slice_agg[dim] = agg_for_dim
        aggregated.append({
            "group": group,
            "n_seeds": len(runs),
            "overall_pr_auc": _aggregate(overall_pr),
            "overall_accuracy_at_target_recall": _aggregate(overall_acc),
            "per_slice": per_slice_agg,
        })

    # Sort by overall PR AUC mean (descending)
    aggregated.sort(key=lambda x: x["overall_pr_auc"]["mean"] if not np.isnan(x["overall_pr_auc"]["mean"]) else -1, reverse=True)

    report = {
        "results_root": args.results_root.as_posix(),
        "recall_target": args.recall_target,
        "n_runs_analyzed": sum(len(rs) for rs in by_group.values()),
        "n_runs_skipped": skipped,
        "n_groups": len(aggregated),
        "groups": aggregated,
    }
    json_out = args.out_root / "per_class_benchmark.json"
    md_out = args.out_root / "per_class_benchmark.md"
    json_out.write_text(json.dumps(report, indent=2, default=str))

    # Build the markdown
    lines: list[str] = [
        "# Per-class benchmark report (puzzle_binary)",
        "",
        f"- results-root: `{report['results_root']}`",
        f"- recall target for accuracy@recall metric: **{args.recall_target}**",
        f"- runs analyzed: {report['n_runs_analyzed']} (skipped {report['n_runs_skipped']} non-puzzle_binary)",
        f"- groups (3+ seeds): {report['n_groups']}",
        "",
        "All numeric cells are 3-seed mean (± std). Higher PR AUC is better.",
        "Per-slice PR AUC is computed restricted to that slice's rows.",
        "",
    ]

    # Pick a focused subset to render in the matrices (top 12 by overall PR AUC)
    focus = aggregated[:12]

    def render_matrix(title: str, dim: str, values: list[str], metric: str = "pr_auc") -> list[str]:
        out = ["", f"## {title}", ""]
        header = "| Group | overall | " + " | ".join(values) + " |"
        sep = "|---|---:|" + "---:|" * len(values)
        out.append(header)
        out.append(sep)
        for g in focus:
            cells = [f"`{g['group']}`"]
            ov = g["overall_pr_auc"]
            cells.append(f"{ov['mean']:.3f} ± {ov['std']:.3f}")
            slc = g["per_slice"].get(dim, {})
            for v in values:
                cell = slc.get(v)
                if cell is None or np.isnan(cell[metric]["mean"]):
                    cells.append("-")
                else:
                    m = cell[metric]
                    cells.append(f"{m['mean']:.3f} ± {m['std']:.3f}")
            out.append("| " + " | ".join(cells) + " |")
        return out

    lines += render_matrix(
        "Matrix 1 — model × difficulty (test PR AUC, restricted to slice)",
        "crtk_difficulty",
        ["very_easy", "easy", "medium", "hard", "very_hard"],
    )
    lines += render_matrix(
        "Matrix 2 — model × phase (test PR AUC)",
        "crtk_phase",
        ["opening", "middlegame", "endgame"],
    )
    lines += render_matrix(
        "Matrix 3 — model × eval_bucket (test PR AUC)",
        "crtk_eval_bucket",
        ["crushing_white", "winning_white", "clear_white", "slight_white",
         "equal", "slight_black", "clear_black", "winning_black", "crushing_black"],
    )
    lines += render_matrix(
        "Matrix 4 — model × tactic_motif (test PR AUC, restricted to positions tagged with each motif)",
        MOTIF_COLUMN,
        list(INTERESTING_MOTIFS),
    )
    lines += render_matrix(
        "Matrix 5 — model × to_move (test PR AUC)",
        "crtk_to_move",
        ["white", "black"],
    )

    # "Strengths" section: who's the best on each slice value
    lines += ["", "## Per-slice winners (best 3-seed mean PR AUC for each slice value)", ""]
    lines.append("| Slice dim | Slice value | Best group | PR AUC mean ± std | Margin to 2nd |")
    lines.append("|---|---|---|---:|---:|")
    for dim in (["crtk_difficulty", "crtk_phase", "crtk_eval_bucket", "crtk_to_move", MOTIF_COLUMN]):
        # Collect (value, group, mean, std) tuples
        per_value_rankings: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
        for g in aggregated:
            for v, cell in g["per_slice"].get(dim, {}).items():
                m = cell["pr_auc"]
                if np.isnan(m["mean"]):
                    continue
                per_value_rankings[v].append((g["group"], m["mean"], m["std"]))
        for v, ranking in per_value_rankings.items():
            ranking.sort(key=lambda t: t[1], reverse=True)
            if not ranking:
                continue
            top = ranking[0]
            margin = (top[1] - ranking[1][1]) if len(ranking) > 1 else float("nan")
            margin_str = f"+{margin:.3f}" if not np.isnan(margin) else "-"
            lines.append(f"| {dim} | {v} | `{top[0]}` | {top[1]:.3f} ± {top[2]:.3f} | {margin_str} |")

    md_out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_out} and {md_out}")
    print(f"Groups (3+ seeds): {report['n_groups']}; runs analyzed: {report['n_runs_analyzed']}; skipped: {report['n_runs_skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
