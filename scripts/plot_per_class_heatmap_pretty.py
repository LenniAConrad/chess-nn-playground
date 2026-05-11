#!/usr/bin/env python
"""Pretty paper-quality per-slice heatmap with model info table on the left.

Adds compared-to the basic version:
  - Sequential refined-green colormap (lighter low -> deep green high)
  - Side info table: rank, encoding badge, model name, params, inference
    samples/sec, MFLOPs/position
  - Amber-bordered cell for column winners (warm contrast against green)
  - Larger fonts, more whitespace, cleaner panel separators
  - Designed for top-N=15 (1 page A4 portrait)

Usage:
  scripts/plot_per_class_heatmap_pretty.py \\
    --results-root _scout_combined_view \\
    --out reports/audits/scout_heatmap_pretty.png \\
    --top-n 15
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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


SINGLE_VALUED_SLICES = ("crtk_difficulty", "crtk_phase", "crtk_eval_bucket", "crtk_to_move")
MOTIF_COLUMN = "crtk_tactic_motifs"
MOTIFS = ("hanging", "fork", "pin", "skewer", "overload",
          "discovered_attack", "mate_in_1", "promotion", "underpromotion")

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


def _safe_load_json(p: Path) -> dict:
    if not p.exists(): return {}
    try: return json.loads(p.read_text())
    except: return {}


def analyze_run(run_dir: Path) -> dict | None:
    if not (run_dir / "metrics_final.json").exists(): return None
    df = _load_predictions(run_dir)
    if df is None or not _is_puzzle_binary(df): return None
    metadata = _safe_load_json(run_dir / "run_metadata.json")
    metrics = _safe_load_json(run_dir / "metrics_final.json")
    complexity = _safe_load_json(run_dir / "complexity_estimate.json")

    y = df["true_label"].to_numpy(dtype=int)
    probs = _positive_prob(df)
    overall = _slice_pr_auc(probs, y)

    per_slice: dict[str, dict[str, float]] = {}
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

    return {
        "name": run_dir.name,
        "encoding": metadata.get("input_encoding", "?"),
        "num_params": metadata.get("num_params"),
        "test_samples_per_second": metrics.get("test_samples_per_second")
            or metrics.get("val_samples_per_second")
            or metrics.get("samples_per_second"),
        "mflops_per_position": complexity.get("estimated_mflops_per_position"),
        "overall": overall,
        "per_slice": per_slice,
    }


_TRIM_SUFFIXES = ("_network", "_classifier", "_bottleneck", "_net", "_model")

def short_name(name: str) -> str:
    name = re.sub(r"_seed\d+$", "", name)
    name = name.replace("benchmark_bench_", "B/")
    name = name.replace("idea_", "")
    # Iteratively strip common suffixes (handles ..._bottleneck_network etc.)
    changed = True
    while changed:
        changed = False
        for s in _TRIM_SUFFIXES:
            if name.endswith(s):
                name = name[: -len(s)]
                changed = True
    if len(name) > 46: name = name[:43] + "..."
    return name


def fmt_params(n) -> str:
    if n is None: return "—"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.0f}k"
    return str(int(n))


def fmt_speed(n) -> str:
    if n is None: return "—"
    if n >= 1000: return f"{n/1000:.1f}k/s"
    return f"{n:.0f}/s"


def fmt_mflops(n) -> str:
    if n is None: return "—"
    if n >= 1000: return f"{n/1000:.1f}G"
    if n >= 1: return f"{n:.1f}M"
    return f"{n*1000:.0f}k"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--top-n", type=int, default=15)
    args = p.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"--results-root does not exist: {args.results_root}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    runs = []
    for d in sorted(args.results_root.iterdir()):
        if not d.is_dir(): continue
        try: r = analyze_run(d)
        except Exception as exc:
            print(f"  ! {d.name}: {type(exc).__name__}: {exc}"); continue
        if r is not None: runs.append(r)
    print(f"Loaded {len(runs)} runs")

    # Aggregate by group
    by_group = defaultdict(list)
    for r in runs:
        by_group[re.sub(r"_seed\d+$", "", r["name"])].append(r)
    agg = []
    for g, lst in by_group.items():
        per_slice: dict[str, dict[str, float]] = {}
        slice_dims = set()
        for r in lst: slice_dims.update(r["per_slice"].keys())
        for dim in slice_dims:
            per_slice[dim] = {}
            values = set()
            for r in lst: values.update(r["per_slice"].get(dim, {}).keys())
            for v in values:
                vs = [r["per_slice"][dim].get(v, float("nan")) for r in lst if dim in r["per_slice"]]
                vs = [x for x in vs if not np.isnan(x)]
                per_slice[dim][v] = float(np.mean(vs)) if vs else float("nan")
        encs = {r["encoding"] for r in lst}
        agg.append({
            "group": g, "n_seeds": len(lst),
            "overall": float(np.nanmean([r["overall"] for r in lst])),
            "encoding": next(iter(encs)) if len(encs) == 1 else "mixed",
            "num_params": int(np.nanmean([r["num_params"] or 0 for r in lst])) or None,
            "test_samples_per_second": float(np.nanmean([r["test_samples_per_second"] or 0 for r in lst])) or None,
            "mflops_per_position": float(np.nanmean([r["mflops_per_position"] or 0 for r in lst])) or None,
            "per_slice": per_slice,
        })
    agg.sort(key=lambda x: x["overall"], reverse=True)
    top = agg[:args.top_n]

    panels = [
        ("Difficulty", "crtk_difficulty", DIFF_ORDER),
        ("Phase",      "crtk_phase",      PHASE_ORDER),
        ("Engine eval bucket", "crtk_eval_bucket", EVAL_ORDER),
        ("Tactic motif", MOTIF_COLUMN, MOTIFS),
        ("Side to move", "crtk_to_move", MOVE_ORDER),
    ]

    # Register bundled Inter font (Inter is a professional, paper-grade sans
    # designed by Rasmus Andersson; widely used in modern publications).
    import matplotlib.font_manager as fm
    from pathlib import Path as _P
    _fonts_dir = _P("assets/fonts")
    if _fonts_dir.exists():
        for _p in _fonts_dir.glob("*.otf"):
            fm.fontManager.addfont(str(_p))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Liberation Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#1B3F2F",
        "text.color": "#14181a",
        "xtick.color": "#1B3F2F",
        "ytick.color": "#1B3F2F",
    })

    # Compute color range. Diverging colormap centered at MEDIAN of all visible cells.
    all_vals = [g["overall"] for g in top]
    for g in top:
        for _, dim, vals in [(p[0], p[1], p[2]) for p in panels]:
            for v in vals:
                x = g["per_slice"].get(dim, {}).get(v)
                if x is not None and not np.isnan(x): all_vals.append(x)
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 98))
    vmid = float(np.median(all_vals))
    # Sequential refined-green palette: matches the report's design.
    # Slightly desaturated for reading comfort. Lighter = lower PR AUC,
    # deeper green = higher.
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("refined_green", [
        "#fbfdfb",   # almost white
        "#e9f1ea",
        "#d0e3d6",
        "#aecbb6",
        "#84ad93",
        "#5b8e72",
        "#3b7155",
        "#1f5340",
        "#0f2c20",   # near-black green
    ])

    # Layout widths — calibrated for Inter at fontsize 9.
    # Inter has wider character widths than DejaVu Sans at fontsize 9 because
    # of its design (slightly wider glyphs for clarity). Measure longest name
    # and give a generous per-character coefficient + a buffer so names don't
    # clip into the adjacent column.
    longest = max(len(short_name(g["group"])) for g in top)
    model_col_w = round(longest * 0.062 + 0.20, 2)

    info_cols = [
        ("#",         0.22),
        ("",          0.18),                # encoding chip (no header)
        ("model",     model_col_w),
        ("params",    0.50),
        ("speed",     0.60),
        ("FLOPs/pos", 0.55),
    ]
    info_total_w = sum(w for _, w in info_cols)
    overall_col_w = 0.55
    per_col_w = 0.48
    panel_widths = [len(p[2]) * per_col_w for p in panels]
    total_w = info_total_w + overall_col_w + sum(panel_widths) + 0.20
    total_h = max(6.4, len(top) * 0.42 + 1.1)

    fig = plt.figure(figsize=(total_w, total_h), dpi=150, facecolor="white")
    width_ratios = [w for _, w in info_cols] + [overall_col_w] + panel_widths
    gs = fig.add_gridspec(
        nrows=1, ncols=len(width_ratios),
        width_ratios=width_ratios,
        wspace=0.05,
        left=0.004, right=0.998, top=0.86, bottom=0.10,
    )

    # ---- INFO COLUMNS ----
    enc_color = {"lc0_bt4_112": "#1b3f2f", "simple_18": "#8aab8d", "mixed": "#1B3F2F", "?": "#1B3F2F"}
    info_axes = []
    for ci, (label, _) in enumerate(info_cols):
        ax = fig.add_subplot(gs[0, ci])
        ax.set_xlim(0, 1); ax.set_ylim(len(top), 0)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        if label:
            ax.set_title(label, fontsize=8.5, color="#444", pad=6, fontweight="normal")
        info_axes.append(ax)

    for i, g in enumerate(top):
        # row stripe (covers info area only, every other row for readability)
        for ci in range(len(info_cols)):
            if i % 2 == 0:
                info_axes[ci].add_patch(plt.Rectangle((0, i), 1, 1, facecolor="#f5f5f5",
                                                       edgecolor="none", zorder=0))
        # rank
        info_axes[0].text(0.5, i + 0.5, f"{i+1}", ha="center", va="center",
                          fontsize=10, fontweight="bold", color="#222")
        # encoding chip
        info_axes[1].add_patch(plt.Rectangle((0.15, i + 0.18), 0.7, 0.64,
                                              facecolor=enc_color.get(g["encoding"], "gray"),
                                              edgecolor="white", linewidth=0.5))
        # model name (left-aligned, small inset)
        info_axes[2].text(0.03, i + 0.5, short_name(g["group"]),
                          ha="left", va="center", fontsize=9, color="#111")
        # params
        info_axes[3].text(0.5, i + 0.5, fmt_params(g["num_params"]),
                          ha="center", va="center", fontsize=9, color="#222")
        # speed
        info_axes[4].text(0.5, i + 0.5, fmt_speed(g["test_samples_per_second"]),
                          ha="center", va="center", fontsize=9, color="#222")
        # mflops
        info_axes[5].text(0.5, i + 0.5, fmt_mflops(g["mflops_per_position"]),
                          ha="center", va="center", fontsize=9, color="#222")

    # ---- OVERALL column (heatmap) ----
    ax_overall = fig.add_subplot(gs[0, len(info_cols)])
    overall_arr = np.array([g["overall"] for g in top]).reshape(-1, 1)
    im = ax_overall.imshow(overall_arr, aspect="auto", cmap=cmap, norm=norm)
    for i, v in enumerate(overall_arr.flatten()):
        # text color flips depending on color intensity
        rgba = cmap(norm(v))
        lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
        text_color = "white" if lum < 0.55 else "#111"
        ax_overall.text(0, i, f"{v:.3f}", ha="center", va="center",
                        color=text_color, fontsize=9, fontweight="bold")
    ax_overall.set_xticks([0]); ax_overall.set_xticklabels(["overall"], rotation=0, fontsize=9)
    ax_overall.set_yticks([])
    ax_overall.set_title("PR AUC", fontsize=10, pad=8)
    for spine in ax_overall.spines.values(): spine.set_visible(False)

    # ---- PANELS ----
    for pi, (panel_title, dim, vals) in enumerate(panels):
        ax = fig.add_subplot(gs[0, len(info_cols) + 1 + pi])
        M = np.full((len(top), len(vals)), np.nan)
        for ri, g in enumerate(top):
            slc = g["per_slice"].get(dim, {})
            for ci, v in enumerate(vals):
                x = slc.get(v, float("nan"))
                if not np.isnan(x): M[ri, ci] = x
        ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)

        for ri in range(M.shape[0]):
            for ci in range(M.shape[1]):
                v = M[ri, ci]
                if np.isnan(v): continue
                rgba = cmap(norm(v))
                lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                text_color = "white" if lum < 0.55 else "#111"
                ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color)

        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals, rotation=55, ha="right", fontsize=8)
        ax.set_yticks([])
        ax.set_title(panel_title, fontsize=10, pad=8)
        for spine in ax.spines.values(): spine.set_visible(False)

        # Column-winner highlight: small ">" glyph in the cell, stroked for
        # readability across the full white -> deepforest colormap range.
        import matplotlib.patheffects as path_effects
        for ci in range(M.shape[1]):
            col = M[:, ci]
            if np.all(np.isnan(col)): continue
            best_row = int(np.nanargmax(col))
            cell_v = M[best_row, ci]
            depth = norm(cell_v) if not np.isnan(cell_v) else 0.0
            txt_color = "white" if depth > 0.55 else "#0F2C20"
            stroke_color = "#0F2C20" if depth > 0.55 else "white"
            t = ax.text(ci + 0.32, best_row, ">", color=txt_color,
                        fontsize=8.5, fontweight="bold",
                        ha="center", va="center", zorder=10)
            t.set_path_effects([
                path_effects.Stroke(linewidth=1.6, foreground=stroke_color),
                path_effects.Normal(),
            ])

    # ---- HEADERS / TITLE ----
    n_simple = sum(1 for g in top if g["encoding"] == "simple_18")
    n_lc0 = sum(1 for g in top if g["encoding"] == "lc0_bt4_112")
    title = (f"Architecture scout — top {len(top)} of {len(agg)} models   "
             f"(seed 42, base scale, puzzle_binary, 12 epochs max)")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

    sub = (f"encoding: deep-green = lc0_bt4_112 ({n_lc0})  ·  light-green = simple_18 ({n_simple})       "
           f"\">\" marker = column winner       "
           f"color: paler = lower PR AUC, deeper green = higher")
    fig.text(0.5, 0.92, sub, ha="center", fontsize=8.5, color="#1B3F2F")

    fig.text(0.005, 0.012,
             f"data: {args.results_root.as_posix()}   ·   "
             f"params + speed + FLOPs from run_metadata.json + complexity_estimate.json",
             ha="left", fontsize=6, color="#1B3F2F")

    fig.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    print(f"Wrote {args.out}")
    pdf = args.out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    print(f"Wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
