#!/usr/bin/env python
"""Render a paper-quality multi-panel per-slice heatmap.

For each completed run under `--results-root`, computes per-slice **test PR AUC**
from `predictions_val.parquet` (or `predictions_test.parquet` if present), then
draws a stacked heatmap with the slice dimensions side-by-side and the top-N
models on the y-axis (sorted by overall test PR AUC).

Hard requirements:
  - Only includes runs with `metrics_final.json` (filters out mid-training).
  - Aggregates across seeds if multiple per group are present (mean PR AUC).
  - Requires explicit --results-root (no hidden default).

Usage:
  scripts/plot_per_class_heatmap.py \\
    --results-root _scout_combined_view \\
    --out reports/audits/scout_heatmap.png \\
    --top-n 30
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


SINGLE_VALUED_SLICES = ("crtk_difficulty", "crtk_phase", "crtk_eval_bucket", "crtk_to_move")
MOTIF_COLUMN = "crtk_tactic_motifs"
MOTIFS = (
    "hanging", "fork", "pin", "skewer", "overload",
    "discovered_attack", "mate_in_1", "promotion", "underpromotion",
)

DIFF_ORDER = ("very_easy", "easy", "medium", "hard", "very_hard")
PHASE_ORDER = ("opening", "middlegame", "endgame")
EVAL_ORDER = ("crushing_white", "winning_white", "clear_white", "slight_white",
              "equal", "slight_black", "clear_black", "winning_black", "crushing_black")
MOVE_ORDER = ("white", "black")


def _load_predictions(run_dir: Path) -> pd.DataFrame | None:
    for name in ("predictions_test.parquet", "predictions_val.parquet"):
        p = run_dir / name
        if p.exists():
            try: return pd.read_parquet(p)
            except: continue
    return None


def _is_puzzle_binary(df: pd.DataFrame) -> bool:
    if "true_fine_label" not in df.columns or "true_label" not in df.columns: return False
    if "prob_1" not in df.columns and "probabilities" not in df.columns: return False
    y = df["true_label"].to_numpy(dtype=int)
    fine = df["true_fine_label"].to_numpy(dtype=int)
    return int(((y == 0) & (fine == 1)).sum()) > 0


def _positive_prob(df):
    if "prob_1" in df.columns: return df["prob_1"].to_numpy(dtype=float)
    return np.array([p[1] for p in df["probabilities"]], dtype=float)


def _slice_pr_auc(probs, y):
    if (y == 1).sum() == 0 or (y == 0).sum() == 0: return float("nan")
    try: return float(average_precision_score(y, probs))
    except: return float("nan")


def analyze_run(run_dir: Path) -> dict | None:
    if not (run_dir / "metrics_final.json").exists():
        return None  # Skip mid-training runs
    df = _load_predictions(run_dir)
    if df is None or not _is_puzzle_binary(df): return None
    y = df["true_label"].to_numpy(dtype=int)
    probs = _positive_prob(df)
    overall = _slice_pr_auc(probs, y)

    per_slice = {}
    for col in SINGLE_VALUED_SLICES:
        if col not in df.columns: continue
        per_slice[col] = {}
        for v in df[col].dropna().unique():
            mask = (df[col] == v).to_numpy()
            per_slice[col][str(v)] = _slice_pr_auc(probs[mask], y[mask])

    if MOTIF_COLUMN in df.columns:
        motif_strs = df[MOTIF_COLUMN].fillna("").astype(str).to_numpy()
        tokenized = [set(s.split("|")) - {""} for s in motif_strs]
        per_slice[MOTIF_COLUMN] = {}
        for motif in MOTIFS:
            mask = np.array([motif in s for s in tokenized], dtype=bool)
            per_slice[MOTIF_COLUMN][motif] = _slice_pr_auc(probs[mask], y[mask]) if mask.sum() > 0 else float("nan")

    # Encoding from config_resolved.yaml (best signal)
    enc = "?"
    cfg_p = run_dir / "config_resolved.yaml"
    if cfg_p.exists():
        try:
            import yaml
            cfg = yaml.safe_load(cfg_p.read_text())
            enc = cfg.get("data", {}).get("encoding", "?")
        except: pass

    return {"name": run_dir.name, "encoding": enc, "overall": overall, "per_slice": per_slice}


def short_name(name: str) -> str:
    """Compact label for y-axis."""
    name = re.sub(r"_seed\d+$", "", name)
    name = name.replace("benchmark_bench_", "B/")
    name = name.replace("idea_", "")
    # Trim "_network", "_classifier", "_bottleneck" from end for compactness
    name = re.sub(r"_(network|classifier|bottleneck)$", "", name)
    if len(name) > 50: name = name[:47] + "..."
    return name


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--also-pdf", action="store_true", default=True)
    args = p.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"--results-root does not exist: {args.results_root}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from {args.results_root}...")
    runs = []
    for d in sorted(args.results_root.iterdir()):
        if not d.is_dir(): continue
        try: r = analyze_run(d)
        except Exception as exc:
            print(f"  ! {d.name}: {type(exc).__name__}: {exc}")
            continue
        if r is None: continue
        runs.append(r)
    print(f"  {len(runs)} runs loaded (with metrics_final.json + predictions parquet)")

    # Aggregate across seeds within a group (drop _seedNN)
    by_group = defaultdict(list)
    for r in runs:
        g = re.sub(r"_seed\d+$", "", r["name"])
        by_group[g].append(r)

    agg = []
    for g, lst in by_group.items():
        overall = np.nanmean([r["overall"] for r in lst])
        encs = {r["encoding"] for r in lst}
        enc = next(iter(encs)) if len(encs) == 1 else "mixed"
        per_slice = {}
        # Combine slices
        slice_dims = set()
        for r in lst: slice_dims.update(r["per_slice"].keys())
        for dim in slice_dims:
            per_slice[dim] = {}
            values = set()
            for r in lst: values.update(r["per_slice"].get(dim, {}).keys())
            for v in values:
                vals = [r["per_slice"].get(dim, {}).get(v, float("nan")) for r in lst]
                vals = [x for x in vals if not np.isnan(x)]
                per_slice[dim][v] = float(np.mean(vals)) if vals else float("nan")
        agg.append({"group": g, "n_seeds": len(lst), "overall": float(overall),
                    "encoding": enc, "per_slice": per_slice})

    agg.sort(key=lambda x: x["overall"], reverse=True)
    top = agg[:args.top_n]
    print(f"  {len(agg)} groups, plotting top {len(top)}")

    # Build matrices for each panel
    panels = [
        ("Difficulty", "crtk_difficulty", DIFF_ORDER),
        ("Phase", "crtk_phase", PHASE_ORDER),
        ("Engine eval bucket", "crtk_eval_bucket", EVAL_ORDER),
        ("Tactic motif", MOTIF_COLUMN, MOTIFS),
        ("Side to move", "crtk_to_move", MOVE_ORDER),
    ]

    # Plot
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 7,
    })

    # Width-to-data ratio: scale columns by count, fixed model column on left
    col_counts = [len(p[2]) for p in panels]
    label_col_width = 6.0
    per_col_width = 0.45
    panel_widths = [c * per_col_width for c in col_counts]
    overview_col_width = 0.45  # one column for overall PR AUC
    total_w = label_col_width + overview_col_width + sum(panel_widths) + 0.6  # margins
    total_h = max(8.0, len(top) * 0.30 + 2.0)

    fig = plt.figure(figsize=(total_w, total_h), dpi=150)
    # Width ratios: [labels] [overall] [panel1] [panel2] ...
    width_ratios = [label_col_width, overview_col_width] + panel_widths
    gs = fig.add_gridspec(
        nrows=2, ncols=len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=[1, 0.04],
        wspace=0.05, hspace=0.10,
    )

    # Compute global vmin/vmax for color consistency across panels
    all_vals = []
    for g in top:
        all_vals.append(g["overall"])
        for dim, _, vals in [(p[1], p[0], p[2]) for p in panels]:
            for v in vals:
                x = g["per_slice"].get(dim, {}).get(v, float("nan"))
                if not np.isnan(x): all_vals.append(x)
    vmin = max(0.30, np.percentile(all_vals, 5))  # clip very low end for contrast
    vmax = min(1.00, np.percentile(all_vals, 99))
    cmap = plt.get_cmap("viridis")

    # === LABELS column (model names + encoding tag) ===
    ax_label = fig.add_subplot(gs[0, 0])
    ax_label.set_xlim(0, 1); ax_label.set_ylim(len(top), 0)
    ax_label.axis("off")
    enc_color = {"lc0_bt4_112": "#0e6ba8", "simple_18": "#a82e0e", "mixed": "gray", "?": "gray"}
    for i, g in enumerate(top):
        # encoding chip
        ax_label.add_patch(plt.Rectangle((0.0, i + 0.1), 0.04, 0.8,
                                         facecolor=enc_color.get(g["encoding"], "gray"),
                                         edgecolor="none"))
        ax_label.text(0.06, i + 0.5, short_name(g["group"]), ha="left", va="center", fontsize=7)
        # rank number
        ax_label.text(0.99, i + 0.5, f"#{i+1}", ha="right", va="center", fontsize=7, color="#666")
    ax_label.set_title("Model (rank by overall PR AUC)", loc="left", fontsize=10)

    # === OVERALL column ===
    ax_overall = fig.add_subplot(gs[0, 1])
    overall_arr = np.array([g["overall"] for g in top]).reshape(-1, 1)
    ax_overall.imshow(overall_arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for i, v in enumerate(overall_arr.flatten()):
        ax_overall.text(0, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < (vmin + vmax) / 2 else "black", fontsize=7, fontweight="bold")
    ax_overall.set_xticks([0]); ax_overall.set_xticklabels(["overall"], rotation=90)
    ax_overall.set_yticks([])

    # === PANELS ===
    panel_ax_list = []
    for pi, (panel_title, dim, vals) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 2 + pi])
        panel_ax_list.append(ax)
        # Build matrix: row=model, col=value
        M = np.full((len(top), len(vals)), np.nan)
        for ri, g in enumerate(top):
            slc = g["per_slice"].get(dim, {})
            for ci, v in enumerate(vals):
                x = slc.get(v, float("nan"))
                if not np.isnan(x): M[ri, ci] = x
        ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        # Cell annotations
        for ri in range(M.shape[0]):
            for ci in range(M.shape[1]):
                v = M[ri, ci]
                if np.isnan(v): continue
                color = "white" if v < (vmin + vmax) / 2 else "black"
                ax.text(ci, ri, f"{v:.2f}", ha="center", va="center", fontsize=6, color=color)
        ax.set_xticks(range(len(vals)))
        # Make labels readable for long values
        ax.set_xticklabels(vals, rotation=60, ha="right", fontsize=7)
        ax.set_yticks([])
        ax.set_title(panel_title, fontsize=10)
        # Mark winners on each column with a thin border
        for ci in range(M.shape[1]):
            col = M[:, ci]
            if np.all(np.isnan(col)): continue
            best_row = int(np.nanargmax(col))
            rect = plt.Rectangle((ci - 0.5, best_row - 0.5), 1, 1, fill=False,
                                 edgecolor="red", linewidth=1.2, zorder=10)
            ax.add_patch(rect)

    # === COLORBAR ===
    ax_cb = fig.add_subplot(gs[1, 1:])
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cb.set_label("Test PR AUC (slice-restricted; cell value = mean over seeds)", fontsize=9)

    # === Title and legend ===
    n_simple = sum(1 for g in top if g["encoding"] == "simple_18")
    n_lc0 = sum(1 for g in top if g["encoding"] == "lc0_bt4_112")
    title = (f"Architecture scout — per-slice PR AUC (top {len(top)} of {len(agg)} models)\n"
             f"Single seed (42), base scale, puzzle_binary, monitor=pr_auc, max 12 epochs. "
             f"Red box = column winner. "
             f"Left bar: blue=lc0_bt4_112 ({n_lc0})  red=simple_18 ({n_simple})")
    fig.suptitle(title, fontsize=12, y=0.995)

    # Caption
    fig.text(0.01, 0.005,
             f"Data: {args.results_root.as_posix()} | Generated by scripts/plot_per_class_heatmap.py | "
             f"All cells are PR AUC (sklearn average_precision_score) restricted to the slice's rows.",
             ha="left", fontsize=6, color="#666")

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out}")
    if args.also_pdf:
        pdf = args.out.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight")
        print(f"Wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
