#!/usr/bin/env python
"""Evaluate i242 against the rest of the scout pool.

Reads i242's metrics_final.json + predictions_val.parquet,
compares against the full scout combined view, and reports
test PR AUC ranking + per-slice champions.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


I242_RUN = Path("results/architecture_scout_2026-05-11_i242/idea_i242_chess_decomposed_attention_seed42")
COMBINED = Path("_scout_combined_view")


def slice_pr_auc(probs, y):
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    try:
        return float(average_precision_score(y, probs))
    except Exception:
        return float("nan")


def analyze_predictions(run_dir: Path):
    for name in ("predictions_test.parquet", "predictions_val.parquet"):
        p = run_dir / name
        if p.exists():
            df = pd.read_parquet(p); break
    else:
        return None
    y = df["true_label"].to_numpy(dtype=int)
    fine = df["true_fine_label"].to_numpy(dtype=int) if "true_fine_label" in df.columns else None
    probs = df["prob_1"].to_numpy(dtype=float) if "prob_1" in df.columns else \
        np.array([p[1] for p in df["probabilities"]], dtype=float)
    overall = slice_pr_auc(probs, y)
    # Slice analysis
    slices = {}
    for col, name in [("crtk_difficulty", "difficulty"),
                       ("crtk_eval_bucket", "eval_bucket"),
                       ("crtk_phase", "phase")]:
        if col not in df.columns: continue
        slices[name] = {}
        for v in df[col].dropna().unique():
            mask = (df[col] == v).to_numpy()
            slices[name][str(v)] = slice_pr_auc(probs[mask], y[mask])
    # Motifs
    if "crtk_tactic_motifs" in df.columns:
        slices["motif"] = {}
        motifs = df["crtk_tactic_motifs"].fillna("").astype(str).to_numpy()
        tok = [set(s.split("|")) - {""} for s in motifs]
        for m in ("hanging", "fork", "pin", "skewer", "overload", "discovered_attack",
                  "mate_in_1", "promotion", "underpromotion"):
            mask = np.array([m in t for t in tok], dtype=bool)
            if mask.sum() > 0:
                slices["motif"][m] = slice_pr_auc(probs[mask], y[mask])
    return overall, slices, fine, probs, y


def main():
    if not I242_RUN.exists():
        print(f"i242 run dir missing: {I242_RUN}")
        return 1

    mf = I242_RUN / "metrics_final.json"
    rm = I242_RUN / "run_metadata.json"
    if not mf.exists():
        print("metrics_final.json missing — training did not complete")
        return 1
    m = json.loads(mf.read_text())
    md = json.loads(rm.read_text())
    print(f"=== i242 results ===")
    print(f"  params (num_params from metadata): {md.get('num_params'):,}")
    print(f"  test PR AUC:  {m.get('test_pr_auc'):.4f}")
    print(f"  val PR AUC:   {m.get('best_score'):.4f}")
    print(f"  test F1:      {m.get('test_f1'):.4f}")
    print(f"  test accuracy: {m.get('test_accuracy'):.4f}")
    print(f"  samples/sec (test): {m.get('test_samples_per_second'):.0f}")
    print()

    # Per-slice on i242
    overall, slices, fine, probs, y = analyze_predictions(I242_RUN)
    print(f"=== i242 per-slice PR AUC ===")
    for dim_name, dim in slices.items():
        print(f"  -- {dim_name} --")
        for v, pr in sorted(dim.items(), key=lambda kv: -kv[1] if kv[1] == kv[1] else 0):
            print(f"     {v:<25s} {pr:.4f}")

    # Compare to top of combined view scout
    print()
    print(f"=== Position in scout leaderboard ===")
    ranking = []
    i242_pr = m["test_pr_auc"]
    for d in sorted(COMBINED.iterdir()):
        if not d.is_dir(): continue
        mff = d / "metrics_final.json"
        if not mff.exists(): continue
        try: mm = json.loads(mff.read_text())
        except: continue
        pr = mm.get("test_pr_auc")
        if pr is None: continue
        ranking.append((re.sub(r"_seed\d+$", "", d.name), float(pr)))
    ranking.append(("idea_i242_chess_decomposed_attention", i242_pr))
    ranking.sort(key=lambda r: r[1], reverse=True)
    pos = next(i for i, r in enumerate(ranking, 1) if r[0] == "idea_i242_chess_decomposed_attention")
    print(f"  rank: #{pos} of {len(ranking)} models")
    print()
    print("  top-10 + i242 context:")
    for i, (name, pr) in enumerate(ranking[:10], 1):
        marker = " <-- i242" if name == "idea_i242_chess_decomposed_attention" else ""
        print(f"   {i:>3d}. {pr:.4f}  {name}{marker}")
    if pos > 10:
        for i in range(max(0, pos - 3), min(len(ranking), pos + 3)):
            name, pr = ranking[i]
            marker = " <-- i242" if name == "idea_i242_chess_decomposed_attention" else ""
            print(f"   {i+1:>3d}. {pr:.4f}  {name}{marker}")

    # Verdict
    print()
    i193_pr = next((pr for name, pr in ranking if name == "idea_i193_exchange_then_king_dual_stream"), None)
    bench_lc0_pr = next((pr for name, pr in ranking if name == "benchmark_bench_lc0_bt4_classifier"), None)
    if i193_pr:
        delta_i193 = i242_pr - i193_pr
        print(f"i242 vs i193 (parent dual-stream): {delta_i193:+.4f} PR AUC")
    if bench_lc0_pr:
        delta_lc0 = i242_pr - bench_lc0_pr
        print(f"i242 vs bench_lc0_bt4_classifier:  {delta_lc0:+.4f} PR AUC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
