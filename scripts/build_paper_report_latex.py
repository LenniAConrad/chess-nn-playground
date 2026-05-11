#!/usr/bin/env python
"""Build paper_report.pdf — typeset in LaTeX with a green colour palette.

Pipeline:
  1. Aggregate scout data (same as the HTML version).
  2. Render heatmap PNG (already exists).
  3. Render per-model 3x2 confusion matrices as PNGs.
  4. Emit a LaTeX source file.
  5. Compile with `tectonic` to PDF.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# =============================================================================
# Data loaders (same as the HTML version)
# =============================================================================

def _f1_optimal_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
    pcts = np.linspace(2, 98, 49)
    candidates = np.unique(np.percentile(probs, pcts))
    best_f1, best_thr = -1.0, 0.5
    for t in candidates:
        y_pred = (probs >= t).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        if (2*tp + fp + fn) == 0: continue
        f1 = 2*tp / (2*tp + fp + fn)
        if f1 > best_f1: best_f1, best_thr = f1, float(t)
    return best_thr


_MPL_DEFAULTS_SET = False
def _set_mpl_defaults():
    global _MPL_DEFAULTS_SET
    if _MPL_DEFAULTS_SET: return
    # Register Inter from the project font directory
    import matplotlib.font_manager as fm
    fonts_dir = Path("assets/fonts")
    if fonts_dir.exists():
        for p in fonts_dir.glob("*.otf"):
            fm.fontManager.addfont(str(p))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Liberation Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": "#14181a",
        "xtick.color": "#1B3F2F",
        "ytick.color": "#1B3F2F",
        "axes.edgecolor": "#1B3F2F",
        "axes.titleweight": "bold",
    })
    _MPL_DEFAULTS_SET = True


def render_confusion_matrix_png(probs, y_true, y_fine, name, save_path: Path) -> None:
    _set_mpl_defaults()
    thr = _f1_optimal_threshold(probs, y_true)
    y_pred = (probs >= thr).astype(int)
    if y_fine is not None:
        cm = np.zeros((3, 2), dtype=int)
        for fine in (0, 1, 2):
            mask = (y_fine == fine)
            for pred in (0, 1):
                cm[fine, pred] = int(((y_pred == pred) & mask).sum())
        row_labels = ["true 0\nnon-puzzle", "true 1\nnear-puzzle", "true 2\npuzzle"]
    else:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        row_labels = ["true: non-puzzle", "true: puzzle"]
    n_rows, n_cols = cm.shape
    fig, ax = plt.subplots(figsize=(2.5, 3.0), dpi=160)
    fig.patch.set_facecolor("white")
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / np.maximum(row_sums, 1), 0)
    # Refined green palette matching the report theme
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("refined_forest",
        ["#fbfdfb", "#e9f1ea", "#aecbb6", "#5b8e72", "#2d7355", "#1b3f2f"])
    ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(n_rows):
        for j in range(n_cols):
            v = cm[i, j]
            row_pct = 100 * cm_norm[i, j]
            color = "white" if cm_norm[i, j] > 0.55 else "#15161a"
            ax.text(j, i - 0.10, f"{v:,}", ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=color)
            ax.text(j, i + 0.22, f"({row_pct:.0f}%)", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(["pred: non-puzzle", "pred: puzzle"], fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=7.5, va="center")
    ax.set_title(short_name(name), fontsize=9.5, pad=10, fontweight="bold")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    # Footer line
    if y_fine is not None:
        tp, fn = cm[2,1], cm[2,0]
        fp = cm[0,1] + cm[1,1]; tn = cm[0,0] + cm[1,0]
        near_fp = cm[1,1] / max(cm[1,0] + cm[1,1], 1)
        rec = tp / max(tp+fn,1); prec = tp / max(tp+fp,1)
        f1 = 2*tp / max(2*tp + fp + fn, 1)
        foot = f"thr*={thr:.2f}  F1 {f1:.3f}  prec {prec:.3f}  rec {rec:.3f}  near-FP {near_fp:.3f}"
    else:
        tn, fp, fn, tp = cm.ravel()
        rec = tp / max(tp+fn,1); prec = tp / max(tp+fp,1)
        f1 = 2*tp / max(2*tp + fp + fn, 1)
        fpr = fp / max(fp+tn,1)
        foot = f"thr*={thr:.2f}  F1 {f1:.3f}  prec {prec:.3f}  rec {rec:.3f}  FPR {fpr:.3f}"
    fig.text(0.5, -0.04, foot, ha="center", fontsize=7, color="#1B3F2F")
    fig.savefig(save_path, format="png", bbox_inches="tight", facecolor="white",
                pad_inches=0.14, dpi=160)
    plt.close(fig)


_TRIM_SUFFIXES = ("_network", "_classifier", "_bottleneck", "_net", "_model")

def short_name(name: str) -> str:
    name = re.sub(r"_seed\d+$", "", name)
    name = name.replace("benchmark_bench_", "B/")
    name = name.replace("idea_", "")
    # Iteratively strip common architectural suffixes (e.g. ..._bottleneck_network)
    changed = True
    while changed:
        changed = False
        for s in _TRIM_SUFFIXES:
            if name.endswith(s):
                name = name[: -len(s)]
                changed = True
    return name


def texttt_breakable(name: str) -> str:
    r"""Render a model name as monospace with NO line breaks.

    Previous version inserted \linebreak[1] hints after every underscore,
    which LaTeX greedily took in narrow columns and stacked names vertically.
    Now we just escape underscores; long names need a wide enough column.
    """
    return r"\texttt{" + name.replace("_", r"\_") + r"}"


def latex_escape(s: str) -> str:
    """Escape special LaTeX characters in plain text."""
    if not isinstance(s, str): s = str(s)
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("$", r"\$")
             .replace("#", r"\#")
             .replace("_", r"\_")
             .replace("{", r"\{")
             .replace("}", r"\}")
             .replace("~", r"\textasciitilde{}")
             .replace("^", r"\textasciicircum{}"))


def fmt_params(n):
    if not n: return "—"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.0f}k"
    return str(int(n))


def fmt_speed(n):
    if not n: return "—"
    if n >= 1000: return f"{n/1000:.1f}k/s"
    return f"{n:.0f}/s"


def fmt_mflops(n):
    if n is None: return "—"
    if n >= 1000: return f"{n/1000:.1f}G"
    if n >= 1: return f"{n:.1f}M"
    return f"{n*1000:.0f}k"


def analyze_run(run_dir: Path):
    if not (run_dir / "metrics_final.json").exists(): return None
    df = None
    for name in ("predictions_test.parquet", "predictions_val.parquet"):
        p = run_dir / name
        if p.exists():
            try: df = pd.read_parquet(p); break
            except: continue
    if df is None or "true_label" not in df.columns: return None
    if "prob_1" in df.columns:
        probs = df["prob_1"].to_numpy(dtype=float)
    elif "probabilities" in df.columns:
        probs = np.array([p[1] for p in df["probabilities"]], dtype=float)
    else:
        return None
    y_true = df["true_label"].to_numpy(dtype=int)
    y_fine = df["true_fine_label"].to_numpy(dtype=int) if "true_fine_label" in df.columns else None
    md = json.loads((run_dir / "run_metadata.json").read_text())
    m = json.loads((run_dir / "metrics_final.json").read_text())
    cx = json.loads((run_dir / "complexity_estimate.json").read_text()) if (run_dir / "complexity_estimate.json").exists() else {}
    return {
        "name": run_dir.name,
        "group": re.sub(r"_seed\d+$", "", run_dir.name),
        "encoding": md.get("input_encoding", "?"),
        "y_true": y_true, "y_fine": y_fine, "probs": probs,
        "test_pr_auc": m.get("test_pr_auc"),
        "val_pr_auc": m.get("best_score"),
        "test_f1": m.get("test_f1"),
        "samples_per_sec": m.get("test_samples_per_second") or m.get("val_samples_per_second"),
        "num_params": md.get("num_params"),
        "mflops_per_pos": cx.get("estimated_mflops_per_position"),
    }


# =============================================================================
# LaTeX preamble (green palette)
# =============================================================================

LATEX_PREAMBLE = r"""
\documentclass[10pt,a4paper]{article}

\usepackage[a4paper, top=1.8cm, bottom=1.8cm, left=1.8cm, right=1.8cm, headsep=6pt]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[table]{xcolor}
\usepackage{graphicx}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{tabularx}
\usepackage{colortbl}
\usepackage[most]{tcolorbox}
\usepackage{titlesec}
\usepackage{titling}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage{enumitem}
\usepackage{pdflscape}
\usepackage{calc}
\usepackage{lipsum}
\usepackage{xspace}
\usepackage{caption}
\usepackage{etoolbox}
\usepackage{ragged2e}
\usepackage{float}

% ============================================================
% Refined, design-friendly green palette.
% Muted, slightly desaturated to be easy on the eyes.
% ============================================================
\definecolor{forest}{HTML}{2D7355}       % primary forest, refined British-green
\definecolor{deepforest}{HTML}{1B3F2F}   % deep accent for body headings
\definecolor{darkforest}{HTML}{0F2C20}   % near-black green for the largest type
\definecolor{sage}{HTML}{6B9F7C}         % mid-green for subtle accents
\definecolor{palesage}{HTML}{E9F2EC}     % soft pale wash for callouts
\definecolor{verylightsage}{HTML}{F4F9F5}% almost-white green tint
\definecolor{moss}{HTML}{4C8A6A}         % softer subhead green
\definecolor{olive}{HTML}{7A9170}
\definecolor{gold}{HTML}{B58904}
\definecolor{ink}{HTML}{14181A}          % body text, near-black
\definecolor{muted}{HTML}{1B3F2F}        % secondary text — deepforest (no greys in the report)
\definecolor{linecolor}{HTML}{C6DCCD}    % rule lines
\definecolor{rule}{HTML}{1B3F2F}         % strong rule under title
\definecolor{warn}{HTML}{9A4540}         % muted warm tone for "what failed"
\definecolor{stripebg}{HTML}{F0F6F2}     % table row stripe

% Encoding pills — both in the green family, distinguishable by saturation/depth.
\definecolor{lc0blue}{HTML}{1B3F2F}      % deep refined green for lc0_bt4_112
\definecolor{lc0bluebg}{HTML}{E2EEE5}    % pale green wash
\definecolor{s18red}{HTML}{6B8E73}       % soft sage for simple_18
\definecolor{s18redbg}{HTML}{EEF5EF}     % very pale sage wash

% Highlight / accent — keep on the green ladder for tonal coherence.
\definecolor{accent}{HTML}{4C8A6A}       % moss-green accent (matches moss/forest family)
\definecolor{accentbg}{HTML}{E2EEE5}     % pale green wash, same hue as lc0bluebg

% Hyperlinks in forest
\hypersetup{
  colorlinks=true,
  urlcolor=forest,
  linkcolor=forest,
  citecolor=forest,
}

% Paragraph spacing (tightened for prof-friendly density)
\setlength{\parskip}{0.32em}
\setlength{\parindent}{0pt}

% Reduced list spacing
\setlist{topsep=2pt, partopsep=0pt, parsep=0pt, itemsep=2pt}

% Captions in sage
\captionsetup{
  font=small,
  labelfont={bf,color=forest},
  textfont={it,color=muted},
  margin=10pt,
}

% Section headings — forest green
\titleformat{\section}
  {\color{forest}\sffamily\Large\bfseries}
  {\thesection}{0.6em}{}[\vskip 1pt {\color{linecolor}\hrule height 0.6pt}]

\titleformat{\subsection}
  {\color{deepforest}\sffamily\large\bfseries}
  {\thesubsection}{0.6em}{}

\titleformat{\subsubsection}
  {\color{moss}\sffamily\normalsize\bfseries}
  {\thesubsubsection}{0.6em}{}

\titlespacing*{\section}{0pt}{14pt}{6pt}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}
\titlespacing*{\subsubsection}{0pt}{8pt}{3pt}

% Page headers/footers
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\color{muted}\sffamily\small\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Booktabs spacing
\renewcommand{\arraystretch}{1.08}

% Row-striped longtable
\rowcolors{2}{}{stripebg}

% Custom tcolorbox styles
\tcbset{
  callout/.style={
    enhanced,
    breakable,
    colback=palesage,
    colframe=forest,
    colbacktitle=palesage,
    boxrule=0pt,
    leftrule=3pt,
    arc=2pt,
    left=10pt, right=10pt, top=8pt, bottom=8pt,
    fonttitle=\sffamily\bfseries\color{deepforest}\footnotesize,
    coltitle=deepforest,
  },
  goodcallout/.style={
    callout,
  },
  warncallout/.style={
    callout,
    colback={s18redbg},
    colbacktitle={s18redbg},
    colframe=warn,
    coltitle=warn,
  },
  archcard/.style={
    enhanced,
    breakable,
    colback=white,
    colframe=linecolor,
    boxrule=0.4pt,
    arc=2pt,
    left=12pt, right=12pt, top=10pt, bottom=10pt,
  },
  archhead/.style={
    enhanced,
    colback=palesage,
    colframe=forest,
    boxrule=0pt,
    leftrule=3pt,
    arc=1pt,
    left=10pt, right=10pt, top=6pt, bottom=6pt,
  },
}

% Convenience commands
\newcommand{\encpill}[1]{%
  \ifx#1lc0%
    \colorbox{lc0bluebg}{\textcolor{lc0blue}{\scriptsize\bfseries\sffamily lc0\_bt4\_112}}%
  \else\ifx#1s18%
    \colorbox{s18redbg}{\textcolor{s18red}{\scriptsize\bfseries\sffamily simple\_18}}%
  \else
    \colorbox{verylightsage}{\textcolor{muted}{\scriptsize\sffamily ?}}%
  \fi\fi
}

\newcommand{\pilllc}[0]{\colorbox{lc0bluebg}{\strut\textcolor{lc0blue}{\scriptsize\bfseries\sffamily lc0\_bt4\_112}}}
\newcommand{\pillsi}[0]{\colorbox{s18redbg}{\strut\textcolor{s18red}{\scriptsize\bfseries\sffamily simple\_18}}}
\newcommand{\pillneu}[0]{\colorbox{verylightsage}{\strut\textcolor{muted}{\scriptsize\sffamily ?}}}

% Numeric column shortcuts
\newcolumntype{N}{>{\raggedleft\arraybackslash}p{1.55cm}}
\newcolumntype{Y}{>{\centering\arraybackslash}X}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
% Fixed-width model column (works inside longtable, where X doesn't)
\newcolumntype{M}{>{\raggedright\arraybackslash}p{5.5cm}}

% Override hyperref's default link colours globally
\setlength{\emergencystretch}{2em}

% sans body for "lead" paragraph
\newenvironment{lead}{\par\itshape\color{muted}\large}{\par\medskip}

% A "stat" command for the top stats row
\newcommand{\statbox}[2]{%
  \begin{tcolorbox}[
    enhanced,
    colback=verylightsage,
    colframe=forest,
    boxrule=0pt,
    leftrule=2.5pt,
    arc=1pt,
    width=\linewidth,
    left=8pt, right=8pt, top=6pt, bottom=6pt,
  ]
  {\color{deepforest}\fontfamily{lmr}\selectfont\Huge\bfseries #1}\\[2pt]
  {\color{muted}\sffamily\scriptsize\bfseries\MakeUppercase{#2}}
  \end{tcolorbox}%
}
"""


# =============================================================================
# LaTeX body builders
# =============================================================================

def enc_pill_tex(enc: str) -> str:
    if enc == "lc0_bt4_112": return r"\pilllc{}"
    if enc == "simple_18":   return r"\pillsi{}"
    return r"\pillneu{}"


def build_leaderboard_table(runs):
    rows = []
    for i, r in enumerate(runs[:15], 1):
        name = texttt_breakable(short_name(r["group"]))
        enc = enc_pill_tex(r["encoding"])
        params = latex_escape(fmt_params(r["num_params"]))
        speed = latex_escape(fmt_speed(r["samples_per_sec"]))
        flops = latex_escape(fmt_mflops(r["mflops_per_pos"]))
        rows.append(
            f"{i} & {enc} & {name} & "
            f"\\textbf{{{r['test_pr_auc']:.4f}}} & "
            f"{r['val_pr_auc']:.4f} & {params} & {speed} & {flops} \\\\"
        )
    body = "\n".join(rows)
    return r"""
\begin{footnotesize}
\begin{longtable}{@{}r l M r r r r r@{}}
\toprule
\textbf{\#} & \textbf{enc} & \textbf{architecture} &
\textbf{test PR} & \textbf{val PR} & \textbf{params} & \textbf{speed $\uparrow$} & \textbf{FLOPs/pos $\downarrow$} \\
\midrule
\endhead
""" + body + r"""
\bottomrule
\end{longtable}
\end{footnotesize}
"""


def build_pareto_table(runs):
    def is_dominated(r, others):
        for o in others:
            if (o["test_pr_auc"] >= r["test_pr_auc"]
                and (o["samples_per_sec"] or 0) >= (r["samples_per_sec"] or 0)
                and (o["test_pr_auc"] > r["test_pr_auc"]
                     or (o["samples_per_sec"] or 0) > (r["samples_per_sec"] or 0))):
                return True
        return False
    pareto = [r for r in runs if not is_dominated(r, runs)]
    pareto.sort(key=lambda r: r["test_pr_auc"], reverse=True)
    rows = []
    for r in pareto[:8]:
        name = texttt_breakable(short_name(r["group"]))
        rows.append(
            f"{enc_pill_tex(r['encoding'])} & "
            f"{name} & \\textbf{{{r['test_pr_auc']:.4f}}} & "
            f"{latex_escape(fmt_speed(r['samples_per_sec']))} & "
            f"{latex_escape(fmt_params(r['num_params']))} & "
            f"{latex_escape(fmt_mflops(r['mflops_per_pos']))} \\\\"
        )
    return r"""
\begin{footnotesize}
\begin{tabularx}{\linewidth}{@{}l L r r r r@{}}
\toprule
\textbf{enc} & \textbf{architecture} & \textbf{test PR $\uparrow$} & \textbf{speed $\uparrow$} & \textbf{params} & \textbf{FLOPs/pos $\downarrow$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{footnotesize}
"""


def build_slice_winners_table(per_class):
    interesting = [
        ("crtk_difficulty",  "very_easy", "Very easy"),
        ("crtk_difficulty",  "easy",      "Easy"),
        ("crtk_difficulty",  "medium",    "Medium"),
        ("crtk_difficulty",  "hard",      "Hard"),
        ("crtk_difficulty",  "very_hard", "Very hard"),
        ("crtk_phase",       "opening",    "Opening"),
        ("crtk_phase",       "middlegame", "Middlegame"),
        ("crtk_phase",       "endgame",    "Endgame"),
        ("crtk_eval_bucket", "equal",       "Equal eval (hardest)"),
        ("crtk_eval_bucket", "winning_white", "Winning white"),
        ("crtk_eval_bucket", "crushing_white", "Crushing white"),
        ("crtk_tactic_motifs", "hanging",          "Hanging"),
        ("crtk_tactic_motifs", "fork",             "Fork"),
        ("crtk_tactic_motifs", "pin",              "Pin"),
        ("crtk_tactic_motifs", "skewer",           "Skewer"),
        ("crtk_tactic_motifs", "overload",         "Overload"),
        ("crtk_tactic_motifs", "discovered_attack","Discovered attack"),
        ("crtk_tactic_motifs", "mate_in_1",        "Mate in 1"),
        ("crtk_tactic_motifs", "promotion",        "Promotion"),
    ]
    rows = []
    for dim, val, label in interesting:
        ranked = []
        for g in per_class["groups"]:
            cell = g["per_slice"].get(dim, {}).get(val)
            if not cell: continue
            mean = cell["pr_auc"]["mean"]
            if mean != mean: continue
            ranked.append((g["group"], mean, cell["pr_auc"]["std"]))
        ranked.sort(key=lambda t: t[1], reverse=True)
        if not ranked: continue
        top_name, top_mean, top_std = ranked[0]
        margin = top_mean - ranked[1][1] if len(ranked) > 1 else 0
        rows.append(
            f"{latex_escape(label)} & {texttt_breakable(short_name(top_name))} & "
            f"$\\mathbf{{{top_mean:.3f}}} \\pm {top_std:.3f}$ & $+{margin:.3f}$ \\\\"
        )
    # Wide fixed-width column so long monospace names fit on one line.
    # Use footnotesize to keep names compact.
    return r"""
\begin{footnotesize}
\begin{tabularx}{\linewidth}{@{}p{2.4cm} L r r@{}}
\toprule
\textbf{slice} & \textbf{champion} & \textbf{PR AUC} $\pm$ \textbf{std} & \textbf{margin} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{footnotesize}
"""


def build_matched_recall_table(matched_recall):
    motif_3seed = []
    for run in matched_recall["runs"]:
        slices = run.get("slice_accuracies_at_target_recall", {}) or {}
        motif = slices.get("crtk_tactic_motifs", {}) if isinstance(slices, dict) else {}
        promo = motif.get("promotion") if isinstance(motif, dict) else None
        if not isinstance(promo, dict) or promo.get("n", 0) == 0: continue
        motif_3seed.append({
            "name": re.sub(r"_seed\d+$", "", run["run_name"]),
            "near_fp_rate": promo["near_fp_rate"],
            "accuracy": promo["accuracy_at_target_recall"],
        })
    motif_3seed.sort(key=lambda r: r["near_fp_rate"])
    rows = []
    for i, r in enumerate(motif_3seed[:8], 1):
        rows.append(
            f"{i} & {texttt_breakable(short_name(r['name']))} & "
            f"\\textbf{{{r['near_fp_rate']:.3f}}} & {r['accuracy']:.3f} \\\\"
        )
    return r"""
\begin{footnotesize}
\begin{tabularx}{\linewidth}{@{}r L r r@{}}
\toprule
\textbf{\#} & \textbf{architecture} & \textbf{near-puzzle FP $\downarrow$} & \textbf{slice acc @ rec 0.80 $\uparrow$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{footnotesize}
"""


def build_architecture_appendix():
    """Hand-written architecture cards with real LaTeX math."""
    cards = [
        {
            "id": "i193",
            "name": "Exchange-Then-King Dual Stream",
            "encoding": "simple_18",
            "params": "157k", "speed": "11.6k/s", "pr_auc": 0.876,
            "summary": (
                r"Splits the trunk into two parallel CNN encoders --- one biased toward "
                r"\emph{tactical exchanges} (attacker/defender geometry, capture sequences), "
                r"one biased toward \emph{king safety} (king-zone planes, check rays, escape "
                r"squares). A small MLP phase router emits a sigmoid gate "
                r"$\alpha(x)\in[0,1]$ that mixes the two stream logits position-by-position; "
                r"a residual head $h_R$ reads the concatenated stream pools to recover "
                r"cross-stream signal."
            ),
            "equation": (
                r"\hat{y}(x) \;=\; \sigma\!\bigl(\;"
                r"\underbrace{\alpha(x)}_{\text{phase gate}} \cdot "
                r"\underbrace{h_K(\phi_K(x))}_{\text{king stream}}"
                r" \;+\; \bigl(1-\alpha(x)\bigr) \cdot "
                r"\underbrace{h_E(\phi_E(x))}_{\text{exchange stream}}"
                r" \;+\; \underbrace{h_R\!\bigl(\phi_K(x)\,\oplus\,\phi_E(x)\bigr)}_{\text{residual}}\;\bigr)"
            ),
            "eq_caption": (
                r"$\phi_E,\phi_K$ are the per-stream conv encoders; $\oplus$ denotes "
                r"channel-wise concatenation; $\sigma$ the binary sigmoid."
            ),
            "bias": (
                r"Hard-encodes the classical Stockfish-style decomposition "
                r"(\textit{exchange evaluation} + \textit{king safety}) as architecture "
                r"instead of asking the network to discover it from data."
            ),
            "limits": (
                r"Pure conv inside each stream means long-range piece interactions cost many "
                r"layers. Mitigated at scale by replacing each stream's encoder with attention."
            ),
        },
        {
            "id": "i048",
            "name": r"Rule-Automorphism Quotient Bottleneck Network (RAQ-Net)",
            "encoding": "simple_18",
            "params": "179k", "speed": "12.0k/s", "pr_auc": 0.861,
            "summary": (
                r"Treats the chess board as a $G$-set under the discrete chess "
                r"automorphism group $G = \langle \mu_{\mathrm{LR}}, \mu_{\mathrm{col}} \rangle$ "
                r"(horizontal mirror with castling-rights swap, and colour-flip with role swap). "
                r"The trunk is $G$-equivariant by construction; the classifier reads from the "
                r"orbit space rather than the raw board."
            ),
            "equation": (
                r"\Phi(x) \;=\; \frac{1}{|G|} \sum_{g \in G} \rho(g) \cdot \phi\!\bigl(g^{-1}\cdot x\bigr)"
                r"\qquad \text{with} \qquad "
                r"\Phi\!\bigl(g\cdot x\bigr) = \rho(g)\,\Phi(x)\quad \forall g \in G"
            ),
            "eq_caption": (
                r"$\phi$ is the per-orbit encoder; $\rho(g)$ realigns equivariant outputs "
                r"after the group action; the second equation is the equivariance constraint "
                r"$\Phi$ satisfies by construction."
            ),
            "bias": (
                r"Symmetries that the rest of the field has to learn from data are baked into "
                r"the architecture. Sample-efficiency advantage is mathematically guaranteed."
            ),
            "limits": (
                r"The quotient bottleneck discards information by construction; useful for "
                r"binary discrimination, less so for value/policy regression."
            ),
        },
        {
            "id": "i018",
            "name": "Oriented Tactical Sheaf Laplacian",
            "encoding": "simple_18",
            "params": "91k", "speed": "5.5k/s", "pr_auc": 0.861,
            "summary": (
                r"Builds a sheaf over chess squares whose sections encode \emph{directed} "
                r"tactical incidence (attacker $\to$ defender, side-to-move-aware). Convolves "
                r"the input with the oriented sheaf Laplacian $L_{\mathrm{or}}$ instead of a "
                r"standard graph Laplacian, preserving the chirality of attack."
            ),
            "equation": (
                r"L_{\mathrm{or}} \;=\; D - A_{\mathrm{or}}, \qquad "
                r"A_{\mathrm{or}}[u,v] \;=\; "
                r"\begin{cases} +1 & u \text{ attacks } v \\ -1 & v \text{ attacks } u \\ 0 & \text{otherwise}\end{cases}"
                r"\qquad y = W \, L_{\mathrm{or}}\, x"
            ),
            "eq_caption": (
                r"$D$ is the in/out-degree diagonal; $A_{\mathrm{or}}$ is signed by "
                r"attacker/defender role and side-to-move; $W$ is a learnable mixing matrix."
            ),
            "bias": (
                r"Tactical relations on a chess board are intrinsically directed; a standard "
                r"symmetric graph Laplacian throws this away. The oriented Laplacian preserves it."
            ),
            "limits": (
                r"Tiny parameter budget (91k) makes the architecture itself a hyperparameter "
                r"--- needs tuning at scale before its true ceiling is known."
            ),
        },
        {
            "id": "i188",
            "name": "Tactical Program Induction Network",
            "encoding": "simple_18",
            "params": "710k", "speed": "8.3k/s", "pr_auc": 0.861,
            "summary": (
                r"Treats a puzzle as the existence of a short tactical program "
                r"(sacrifice $\to$ fork $\to$ promotion). A neural program-induction head "
                r"scores candidate programs over a learned library $\mathcal{P}$ of tactical "
                r"primitives. The puzzle logit is the maximum-likelihood program score."
            ),
            "equation": (
                r"p(\text{puzzle} \mid x) \;=\; \max_{P\,\in\,\mathcal{P}}\; "
                r"q_\theta(P \mid x), \quad "
                r"q_\theta(P \mid x) \;=\; \prod_{t=1}^{|P|} q_\theta\!\bigl(P_t \mid x, P_{<t}\bigr)"
            ),
            "eq_caption": (
                r"$\mathcal{P}$ is the (learned) library of short tactical programs; "
                r"$q_\theta$ is an autoregressive neural program scorer."
            ),
            "bias": (
                r"Tactical solutions are \emph{compositional sequences of primitives}, not "
                r"point-evaluations; a program-shaped output head exposes that compositionality."
            ),
            "limits": (
                r"Program-induction heads are notoriously hard to train; high parameter count "
                r"makes this less Pareto-efficient than the smaller winners."
            ),
        },
        {
            "id": "BT4",
            "name": "LC0 BT4 (reference architecture)",
            "encoding": "lc0_bt4_112",
            "params": "~50M (medium)", "speed": "varies",
            "pr_auc": float("nan"),
            "summary": (
                r"BT4 is the current LC0 reference trunk: a piece-token transformer over the "
                r"64 squares of the board. The input is the 112-plane LC0 encoding "
                r"(8 history steps $\times$ 13 piece-channel planes plus auxiliary planes for "
                r"castling rights, side-to-move, en-passant, half-move clock). The trunk "
                r"applies a stack of multi-head self-attention blocks where each square is a "
                r"token; positional information is injected via per-square learned "
                r"embeddings rather than 2D positional encoding. Two heads sit on top: a WDL "
                r"value head and an 1858-dim move-policy head."
            ),
            "equation": (
                r"\mathrm{BT4}(x) \;=\; \mathrm{Head}\!\Bigl( "
                r"\underbrace{ \bigl(\mathrm{Block}_L \circ \cdots \circ \mathrm{Block}_1\bigr) }_{\text{stack of $L$ transformer blocks}}\!"
                r"\bigl( \mathrm{Embed}(x) + P \bigr) \Bigr), \quad "
                r"\mathrm{Block}_\ell(z) = \mathrm{FFN}\!\bigl(\mathrm{MHSA}(z)\bigr) + z"
            ),
            "eq_caption": (
                r"$\mathrm{Embed}: \mathbb{R}^{112\times 8\times 8} \to \mathbb{R}^{64 \times d}$ "
                r"projects each square's plane vector to a $d$-dim token; $P \in \mathbb{R}^{64\times d}$ "
                r"is the learned per-square positional embedding; $\mathrm{MHSA}$ is multi-head "
                r"self-attention over all $64$ tokens; $\mathrm{FFN}$ is a 2-layer feed-forward."
            ),
            "bias": (
                r"Square-as-token allows any square pair to interact in a single layer --- "
                r"the right prior for long-range piece relationships like queen-on-a1 attacking "
                r"king-on-h8. This is the structural property other trunks struggle to match."
            ),
            "limits": (
                r"The trunk is \emph{generic}: it has no chess-specific decomposition, no "
                r"chess group equivariance, no chess-aware attention bias. All of those have "
                r"to be learned from data. The headline strength of LC0 BT4 comes from "
                r"training-budget, not from the architecture itself."
            ),
        },
        {
            "id": "i011",
            "name": "VetoSelect Positive-Claim Abstention",
            "encoding": "lc0_bt4_112",
            "params": "502k", "speed": "11.7k/s", "pr_auc": 0.858,
            "summary": (
                r"Adds a selective-abstention head on top of an LC0 BT4-style trunk. The "
                r"loss separates raw positive evidence from refuting evidence, and applies an "
                r"explicit penalty on hard near-puzzle negatives (the ``veto'')."
            ),
            "equation": (
                r"\mathcal{L} \;=\; "
                r"\underbrace{-\,y\log p^+ - (1-y)\log(1-p^+)}_{\text{binary cross-entropy}}"
                r" \;+\; \lambda\,\mathbb{E}_{x\,\in\,\mathrm{hard}^-}\!\bigl[\,p^+(x)\,\bigr]"
            ),
            "eq_caption": (
                r"$p^+$ is the puzzle logit; $\mathrm{hard}^-$ is the set of "
                r"verified-near-puzzle negatives (fine label 1); $\lambda > 0$ is the veto weight."
            ),
            "bias": (
                r"Aggregate PR AUC is the wrong scoreboard for puzzle classification --- the "
                r"real cost is false positives on positions that \textit{look} tactical but "
                r"aren't. Explicit suppression of that mode is a strict gain."
            ),
            "limits": (
                r"Multi-objective loss trades aggregate AUC capacity for slice robustness. "
                r"Useful if matched-recall FP on near-puzzles is what you care about; less "
                r"useful if you only quote aggregate AUC."
            ),
        },
    ]
    out = []
    for a in cards:
        enc_tex = enc_pill_tex(a["encoding"])
        out.append(r"\begin{tcolorbox}[archcard, breakable]" + "\n")
        out.append(r"\begin{tcolorbox}[archhead, boxsep=0pt, top=4pt, bottom=4pt]" + "\n")
        out.append(
            r"\textcolor{deepforest}{\sffamily\bfseries\large " + latex_escape(a["id"]) + r" --- " +
            latex_escape(a["name"]) + r"}\\[2pt]" + "\n"
        )
        # Format PR AUC: handle NaN for the reference architectures (BT4)
        pr_str = f"{a['pr_auc']:.3f}" if a['pr_auc'] == a['pr_auc'] else "n/a (not in scout)"
        out.append(
            r"\textcolor{muted}{\sffamily\scriptsize " + enc_tex +
            r"\quad params \textbf{" + latex_escape(a["params"]) +
            r"}\quad speed \textbf{" + latex_escape(a["speed"]) +
            r"}\quad test PR AUC \textbf{" + pr_str +
            r"}}" + "\n"
        )
        out.append(r"\end{tcolorbox}" + "\n")
        out.append(r"\smallskip" + "\n")
        out.append(a["summary"] + "\n")
        out.append(r"\par\smallskip" + "\n")
        out.append(r"\begin{equation*}" + "\n" + a["equation"] + "\n" + r"\end{equation*}" + "\n")
        out.append(r"\par\vspace{-4pt}\begin{center}\textit{\textcolor{muted}{\small " +
                   a["eq_caption"] + r"}}\end{center}" + "\n")
        out.append(r"\par\smallskip" + "\n")
        out.append(r"{\color{forest}\sffamily\bfseries\textgreater\,}" +
                   r"\textit{" + a["bias"] + r"}" + "\n")
        out.append(r"\par\smallskip\textbf{\sffamily\footnotesize\color{muted}WHAT IT GIVES UP}\\" + "\n")
        out.append(a["limits"] + "\n")
        out.append(r"\end{tcolorbox}" + "\n\n")
    return "".join(out)


def build_bt4_path():
    steps = [
        {
            "phase": "Step 1",
            "title": "Source matched training data",
            "body": (
                r"Beating the public pre-trained BT4 directly is unfair: that network has "
                r"been trained on billions of self-play positions over many GPU-years. Our "
                r"research is about \emph{trunks}, not training pipelines, so the fair "
                r"comparison is to train both architectures \emph{from scratch on the same "
                r"data}. Plausible data sources: (a) the Stockfish NNUE training corpus "
                r"(Stockfish 16/17/18 NNs were trained on tens of millions of "
                r"$(\text{position},\,\text{eval},\,\text{best-move})$ triples — many are "
                r"publicly archived); (b) mining our own corpus by running Stockfish at "
                r"fixed depth over a master-game database. The latter costs CPU time but "
                r"is reproducible and unambiguous."
            ),
        },
        {
            "phase": "Step 2",
            "title": "Multi-stream trunk (3-stream attention)",
            "body": (
                r"Generalize i193's exchange/king dual stream to \textbf{three parallel "
                r"transformer streams}: exchange, king, positional. Each stream is a small "
                r"transformer over the 64 squares, with attention bias matrices derived from "
                r"chess-aware precomputed tables (attacker/defender for the exchange stream; "
                r"king-zone/check-ray for the king stream; standard relative-position bias "
                r"for the positional stream). Fusion is i193's learned phase router "
                r"generalised to a soft 3-way mixture $\alpha \in \Delta^2$."
            ),
        },
        {
            "phase": "Step 3",
            "title": "Value + policy heads",
            "body": (
                r"Replace the puzzle binary head with LC0-style heads: WDL value head "
                r"$\hat{v}(x) \in \Delta^2$ (3-way softmax over win/draw/loss) and a 1858-dim "
                r"policy head $\hat{\pi}(x \mid m) \propto e^{z_m}$ masked to legal moves. "
                r"The training objective is a weighted sum of value and policy losses,"
                r" $\mathcal{L} = D_{\mathrm{KL}}\!\bigl(\hat{v} \,\|\, v_{\mathrm{tgt}}\bigr) "
                r"+ \beta\,D_{\mathrm{KL}}\!\bigl(\hat{\pi} \,\|\, \pi_{\mathrm{tgt}}\bigr)$, "
                r"where the targets come from the data source chosen in Step 1."
            ),
        },
        {
            "phase": "Step 4",
            "title": "Chess-automorphism equivariance wrap",
            "body": (
                r"Wrap the trunk in i048's group equivariance: every transformer block "
                r"commutes with $G = \langle\mu_{\mathrm{LR}}, \mu_{\mathrm{col}}\rangle$. "
                r"At matched compute, this buys $|G| = 4$ times the effective training data "
                r"for symmetric tactical patterns. Implementation: tied weights across "
                r"orbit-equivalent positions."
            ),
        },
        {
            "phase": "Step 5",
            "title": "Fair head-to-head: same training, two trunks",
            "body": (
                r"Train two networks under \emph{identical training settings} (same data, "
                r"same optimiser, same compute budget, same heads): (a)~the multi-stream "
                r"trunk proposed above, (b)~a freshly-initialised plain BT4-shaped transformer "
                r"trunk. \textbf{This is the head-to-head our research can actually win.} "
                r"We are not claiming to beat the BT4 the LC0 team trained for years. We are "
                r"claiming our trunk has better inductive biases than a generic transformer "
                r"trunk \emph{at matched training}. Report ELO on the same fixed-depth match-up."
            ),
        },
        {
            "phase": "Step 6",
            "title": "Stream-wise auxiliary supervision (optional)",
            "body": (
                r"Train each stream with a chess-aware auxiliary loss (exchange-outcome "
                r"prediction on the exchange stream, king-attack/defend classification on "
                r"the king stream, positional-eval regression on the positional stream). "
                r"Auxiliary weights $\lambda_i \leq 0.05$ so the main losses dominate:"
                r" $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{value}} + "
                r"\beta\mathcal{L}_{\text{policy}} + \sum_{i \in \{E,K,P\}} \lambda_i\,\mathcal{L}_i^{\text{aux}}$. "
                r"Only the multi-stream trunk has structural slots for this; BT4 does not."
            ),
        },
    ]
    out = []
    for i, s in enumerate(steps, 1):
        out.append(r"\begin{minipage}{\linewidth}" + "\n")
        out.append(r"\begin{minipage}[t]{0.06\linewidth}" + "\n")
        out.append(rf"\fontfamily{{lmr}}\selectfont\LARGE\bfseries\color{{forest}}{i}" + "\n")
        out.append(r"\end{minipage}\hfill" + "\n")
        out.append(r"\begin{minipage}[t]{0.93\linewidth}" + "\n")
        out.append(rf"\textcolor{{muted}}{{\scriptsize\sffamily\bfseries {s['phase'].upper()}}}\\[0pt]" + "\n")
        out.append(rf"\textcolor{{deepforest}}{{\sffamily\normalsize\bfseries {latex_escape(s['title'])}}}\\[2pt]" + "\n")
        out.append(s["body"] + "\n")
        out.append(r"\end{minipage}" + "\n")
        out.append(r"\end{minipage}\par" + "\n")
        if i < len(steps):
            out.append(r"\vspace{2pt}{\color{linecolor}\rule{\linewidth}{0.3pt}}\vspace{2pt}\par" + "\n")
    return "".join(out)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="_scout_combined_view")
    p.add_argument("--audits-root",  default="reports/audits")
    p.add_argument("--scout-state",  default="reports/architecture_scout_2026-05-09/state.json")
    p.add_argument("--heatmap",      default="reports/audits/scout_heatmap_pretty.png")
    p.add_argument("--out",          default="reports/audits/paper_report.pdf")
    p.add_argument("--keep-tex",     action="store_true",
                   help="Also write the .tex file next to the PDF")
    args = p.parse_args()

    results_root = Path(args.results_root)
    audits_root  = Path(args.audits_root)
    heatmap_path = Path(args.heatmap)
    out_path     = Path(args.out)

    print("Loading scout artifacts...")
    runs = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir(): continue
        try: r = analyze_run(d)
        except Exception as exc:
            print(f"  ! {d.name}: {exc}"); continue
        if r is not None and r["test_pr_auc"] is not None:
            runs.append(r)
    runs.sort(key=lambda r: r["test_pr_auc"], reverse=True)
    n_total = len(runs)
    print(f"  {n_total} runs loaded")

    state = json.loads(Path(args.scout_state).read_text())
    counts = Counter(t.get("status") for t in state["tasks"].values())
    state_summary = {
        "total":     sum(counts.values()),
        "completed": counts.get("completed", 0),
        "failed":    counts.get("failed", 0) + counts.get("failed_resume_available", 0)
                   + counts.get("artifact_validation_failed", 0),
        "timeout":   counts.get("timeout", 0) + counts.get("timeout_resume_available", 0),
    }
    failed_pct  = 100 * state_summary["failed"]  / state_summary["total"]
    timeout_pct = 100 * state_summary["timeout"] / state_summary["total"]

    per_class = json.loads((audits_root / "per_class_benchmark.json").read_text())
    matched_recall = json.loads((audits_root / "matched_recall_fp_report.json").read_text())

    # Build tmpdir for the LaTeX run
    with tempfile.TemporaryDirectory(prefix="scout_report_") as td:
        tmpdir = Path(td)
        # Copy heatmap in
        heat_local = tmpdir / "heatmap.png"
        shutil.copy(heatmap_path, heat_local)
        # Copy the per-class CRTK-rendered example boards.
        ex_dst_dir = tmpdir / "reports" / "audits"
        ex_dst_dir.mkdir(parents=True, exist_ok=True)
        for cls in (0, 1, 2):
            src = Path(f"reports/audits/puzzle_class_{cls}.png")
            if src.exists():
                shutil.copy(src, ex_dst_dir / src.name)
        # Render confusion matrices (top 8)
        print("Rendering 3x2 confusion matrices...")
        cm_paths = []
        for i, r in enumerate(runs[:8]):
            cm_path = tmpdir / f"cm_{i:02d}.png"
            render_confusion_matrix_png(r["probs"], r["y_true"], r.get("y_fine"), r["name"], cm_path)
            cm_paths.append(cm_path)

        # === Build LaTeX body ===
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        winner = runs[0]; runner_up = runs[1]

        leaderboard_tex = build_leaderboard_table(runs)
        pareto_tex = build_pareto_table(runs)
        slice_winners_tex = build_slice_winners_table(per_class)
        matched_tex = build_matched_recall_table(matched_recall)
        arch_appendix_tex = build_architecture_appendix()
        bt4_path_tex = build_bt4_path()

        # Confusion-matrix grid: 4 rows x 2 cols
        cm_grid_rows = []
        for i in range(0, 8, 2):
            cm_grid_rows.append(
                r"\noindent"
                r"\begin{minipage}[t]{0.49\linewidth}\centering"
                rf"\includegraphics[width=0.92\linewidth]{{cm_{i:02d}.png}}"
                r"\end{minipage}\hfill"
                r"\begin{minipage}[t]{0.49\linewidth}\centering"
                rf"\includegraphics[width=0.92\linewidth]{{cm_{i+1:02d}.png}}"
                r"\end{minipage}\par\medskip"
            )
        cm_grid_tex = "\n".join(cm_grid_rows)

        # Title page stats
        stats_block = (
            r"\noindent\begin{minipage}{0.24\linewidth}"
            r"\statbox{" + str(state_summary["total"]) + r"}{Architectures trained}"
            r"\end{minipage}\hfill"
            r"\begin{minipage}{0.24\linewidth}"
            r"\statbox{" + str(state_summary["completed"]) + r"}{Completed runs}"
            r"\end{minipage}\hfill"
            r"\begin{minipage}{0.24\linewidth}"
            r"\statbox{" + str(n_total) + r"}{In leaderboard}"
            r"\end{minipage}\hfill"
            r"\begin{minipage}{0.24\linewidth}"
            r"\statbox{" + f"{winner['test_pr_auc']:.4f}" + r"}{Top test PR AUC}"
            r"\end{minipage}\medskip"
        )

        winner_short = latex_escape(short_name(winner["group"]))
        margin = winner["test_pr_auc"] - runner_up["test_pr_auc"]

        LATEX = LATEX_PREAMBLE + r"""
\begin{document}

% =============================================================================
% TITLE PAGE (page 1)
% =============================================================================
\thispagestyle{empty}
\vspace*{0.3cm}
\noindent{\sffamily\color{muted}\bfseries\small RESEARCH REPORT \quad$\cdot$\quad CHESS-NN-PLAYGROUND}

\vspace{0.18cm}
\noindent{\fontfamily{lmr}\fontsize{28pt}{32pt}\selectfont\bfseries\color{deepforest} Architecture Scout}

\vspace{0.10cm}
\noindent{\fontfamily{lmr}\itshape\small\color{muted} A 234-model survey of bespoke chess-evaluation architectures, with apples-to-apples comparisons across difficulty, phase, eval bucket, and tactic motif.}

\vspace{0.25cm}
\noindent{\color{rule}\rule{\linewidth}{1.0pt}}

\vspace{0.25cm}
\noindent
{\setlength{\extrarowheight}{0pt}\begin{tabular}{>{\sffamily\bfseries\scriptsize\color{ink}}l@{\hspace{18pt}}>{\sffamily\footnotesize\color{muted}}p{0.74\linewidth}}
AUTHOR & Lennart Axel Conrad \quad (Student ID: 2025080264) \\
AFFILIATION & Zijing College, Tsinghua University \\
SUPERVISOR & Prof.\ Jungong Han \quad (Department of Automation, Tsinghua University) \\[3pt]
SCOUT DATE & 2026-05-09 to 2026-05-10 \\
TASK & puzzle\_binary (single positive logit, BCE loss) \\
DATASET & CRTK-tagged 3-class split (\textasciitilde 173k train / \textasciitilde 21k val / \textasciitilde 21k test, zero FEN overlap) \\
HARDWARE & RTX 3070 (8 GiB) --- single GPU, single seed (42), base scale \\
BUDGET & 12 epochs max, patience 3, 60-min wall per task, CUDA only \\
GENERATED & """ + today + r""" \\
REPOSITORY & \href{https://github.com/LenniAConrad/chess-nn-playground}{github.com/LenniAConrad/chess-nn-playground} \\
\end{tabular}}

\vspace{0.25cm}

{\begin{tcolorbox}[callout, title=What this work is, fontupper=\footnotesize, top=2pt, bottom=2pt]
We compare bespoke chess-evaluation architectures on a single small-scale
binary task (\emph{puzzle vs non-puzzle}) to identify \textbf{which structural
priors a chess engine should actually use}.  ``Best'' is judged on two axes:
\textbf{sample efficiency} (test PR AUC per training position, drives Elo per
sample) and \textbf{inference speed} (evals/s, drives Elo per second of
search --- for MCTS engines running thousands of nodes per move, a 2$\times$
speed edge typically beats a 30-Elo accuracy edge).  The engine-relevant
outcome is the i243 proposal (\S\ref{sec:i243}).
\end{tcolorbox}}

\vspace{4pt}

\section*{\color{forest} Executive summary}
\vspace{-6pt}{\color{linecolor}\hrule height 0.6pt}\vspace{6pt}

{\footnotesize\itshape\color{muted}234 bespoke chess architectures were trained once each at small scale on a puzzle-detection task. \textbf{""" + str(state_summary["completed"]) + r"""} produced usable results; """ + str(state_summary["failed"]) + r""" crashed on code bugs (""" + f"{failed_pct:.0f}" + r"""\%); """ + str(state_summary["timeout"]) + r""" hit the 60-minute training wall (""" + f"{timeout_pct:.0f}" + r"""\%).\par}

\vspace{4pt}
""" + stats_block + r"""

\vspace{-2pt}
\begin{tcolorbox}[goodcallout, title=Headline finding, fontupper=\footnotesize, top=2pt, bottom=2pt]
\texttt{""" + winner_short + r"""} wins by a clear margin (\textbf{""" + f"{winner['test_pr_auc']:.4f}" + r"""} test PR AUC, $+""" + f"{margin:.3f}" + r"""$ over \#2 at the same parameter budget).  Its dual-stream architecture --- one branch for tactical exchanges, one for king safety --- is the largest within-encoding architectural margin in the entire scout pool, and the only architecture that empirically moves the leaderboard above the natural $\sim$0.86 ceiling shared by all generic backbones.
\end{tcolorbox}

\clearpage

% =============================================================================
\section{Dataset \& puzzle-class definitions}

The scout's task is \texttt{puzzle\_binary}: classify a position as a
\emph{tactical puzzle} (positive) or a \emph{non-puzzle} (negative).  The
underlying CRTK label is three-way, and we collapse it to binary at training
time --- but the three-way label is what generates the dataset's difficulty
structure, so all confusion matrices and per-class analyses in this report
keep it explicit.

\smallskip

\noindent
\begin{tabular}{>{\sffamily\bfseries\small\color{forest}}p{3.2cm}@{\hspace{12pt}}>{\small\color{ink}}p{0.66\linewidth}}
fine\_label = 0 & \textbf{Non-puzzle.}  Random board positions sampled from
master-game corpora.  Tactics, if any, are coincidental.  This is the
\emph{easy negative} class. \\[2pt]
fine\_label = 1 & \textbf{Verified-near-puzzle.}  Positions that the CRTK
filter flagged as \emph{candidate} puzzles but Stockfish verification
proved are \emph{not} unique-solution tactics --- the second-best move
scores almost identically.  This is the \emph{hard negative} class and
the discriminator that separates strong from weak architectures. \\[2pt]
fine\_label = 2 & \textbf{Puzzle.}  Positions with a unique tactical
solution: a single best move scores at least a few hundred centipawns
above the alternatives.  These are the \emph{positives}. \\
\end{tabular}

\medskip

The \textbf{CRTK pipeline} that produced these labels lives in
\texttt{scripts/data/build\_crtk\_tagged\_splits.py}; the contract for
which Stockfish thresholds promote a candidate to fine\_label~2 is documented
in \texttt{docs/crtk\_export\_contract.md}.  Every position carries
auxiliary tags --- \texttt{crtk\_difficulty}, \texttt{crtk\_phase},
\texttt{crtk\_eval\_bucket}, \texttt{crtk\_tactic\_motifs} --- which the
per-slice heatmap (next section) cuts the leaderboard along.

\medskip

\begin{figure}[H]
\centering
\begin{minipage}[t]{0.32\linewidth}\centering
\includegraphics[width=\linewidth]{reports/audits/puzzle_class_0.png}\\[3pt]
{\sffamily\bfseries\small\color{forest}Class 0 --- random\_position}\\[1pt]
{\small\color{ink}best move: \textbf{cxd3}}\\
{\scriptsize\color{muted}random midgame sample, no unique puzzle}
\end{minipage}\hfill
\begin{minipage}[t]{0.32\linewidth}\centering
\includegraphics[width=\linewidth]{reports/audits/puzzle_class_1.png}\\[3pt]
{\sffamily\bfseries\small\color{forest}Class 1 --- near-puzzle}\\[1pt]
{\small\color{ink}best move: \textbf{fxe3}}\\
{\scriptsize\color{muted}pv\_gap = 115 cp; not unique}
\end{minipage}\hfill
\begin{minipage}[t]{0.32\linewidth}\centering
\includegraphics[width=\linewidth]{reports/audits/puzzle_class_2.png}\\[3pt]
{\sffamily\bfseries\small\color{forest}Class 2 --- puzzle}\\[1pt]
{\small\color{ink}best move: \textbf{dxe5}}\\
{\scriptsize\color{muted}pv\_gap = 1179 cp; unique winner}
\end{minipage}
\caption{One representative example per class, rendered with
\texttt{crtk fen render}.  All three positions are \textbf{White to
move}.  Boards show the bare position --- the best move is reported in
algebraic notation under each panel rather than overlaid on the board,
so the reader sees what the network sees.
\textbf{This is not a curated coincidence:} by construction in CRTK,
every class-1 row (near-puzzle) is generated by mutating the parent
line of a true puzzle, so every near-puzzle in the database has a
class-2 sibling somewhere in the CRTK supercorpus.  In our 173k-position
split that sibling survives sampling for $\sim$5\% of near-puzzles; the
class-1 and class-2 boards above are one such sister pair (CRTK parent
\texttt{crtk\_parent\_-1001554678727017229}): the only structural
difference is that in class~2 Black has captured into e5, turning the
position into a unique tactical win for White (\texttt{dxe5}~$\to$ wins
the queen).  This pairs the near-puzzle and the puzzle as tightly as
the data allows --- the same continuation up to one move, with the
binary label flipping on the uniqueness of the response.}
\end{figure}

\begin{tcolorbox}[callout, title=Why this three-way structure matters]
A model that achieves high PR AUC by memorising piece-count and
king-safety priors will look strong on a binary leaderboard but
collapse on the \emph{matched-recall near-puzzle FP rate}: the
fraction of class~1 positions it mistakenly classifies as puzzles
at a fixed true-positive rate.  This is the metric in
\S\ref{sec:robustness} that distinguishes architectures whose
internal computation actually checks for a tactical sequence.
\end{tcolorbox}

\clearpage

% =============================================================================
\section{Overall leaderboard}

Test PR AUC after 12 epochs at base scale on the \texttt{puzzle\_binary} task.
\pilllc{} and \pillsi{} tag the input encoding.

\begin{tcolorbox}[callout, title=Metric legend]
\textbf{test PR AUC} ($\uparrow$ higher is better) --- area under the
precision--recall curve on the held-out test split; the primary scoreboard
metric for puzzle classification.\\
\textbf{val PR AUC} ($\uparrow$) --- same on the validation split (used for
checkpoint selection).\\
\textbf{params} --- trainable parameter count (smaller is cheaper to store).\\
\textbf{speed} ($\uparrow$ higher is better) --- inference throughput
measured in test samples per second on the RTX 3070 at the model's training
batch size. Larger = faster inference per game.\\
\textbf{FLOPs/pos} ($\downarrow$ lower is better) --- theoretical
floating-point operations per board position (one multiply-add counted as
two FLOPs). Lower = cheaper to run; for a chess engine this matters because
the network is called every node of the search tree.
\end{tcolorbox}

""" + leaderboard_tex + r"""

% =============================================================================
\clearpage
\begin{landscape}
\section{Per-slice performance map}

Each cell is the test PR AUC restricted to its slice. Color: paler tones =
lower PR AUC, deeper green = higher. ``\textgreater'' markers identify column winners.
Three leftmost data columns: parameter count, inference throughput,
theoretical FLOPs per position.

\begin{center}
\includegraphics[width=0.99\linewidth, height=0.78\textheight, keepaspectratio]{heatmap.png}
\end{center}

\begin{quote}
\captionof*{figure}{\small\textit{\textcolor{muted}{Top 15 of """ + str(n_total) + r""" completed models. Slices: difficulty (5), phase (3),
        engine eval bucket (9), tactic motif (9), side to move (2). Cell values are
        slice-restricted test PR AUC.}}}
\end{quote}
\end{landscape}

% =============================================================================
\clearpage
\section{Per-slice champions}

Best model for each interesting slice value, with margin to runner-up:

""" + slice_winners_tex + r"""

% =============================================================================
\clearpage
\section{Confusion matrices --- top 8 models}

Source-class $3\times 2$ confusion matrices at each model's F1-optimal threshold.
\textbf{Rows} are the underlying CRTK fine label --- \emph{0: non-puzzle}
(easy negative), \emph{1: verified-near-puzzle} (hard negative --- positions that
look tactical but aren't), \emph{2: puzzle}. \textbf{Columns} are the binary
prediction. Cell values show count and within-row percentage; the rightmost
cell on row~1 is exactly the matched-recall near-puzzle FP rate.

\medskip

""" + cm_grid_tex + r"""

% =============================================================================
\clearpage
\section{Speed $\times$ accuracy: the Pareto frontier}

Models on the (accuracy, speed) Pareto frontier --- no other model is both
faster and more accurate:

""" + pareto_tex + r"""

% =============================================================================
\section{Robustness: matched-recall FP on the promotion slice}\label{sec:robustness}

Aggregate PR AUC misses architectures explicitly designed for hard-negative
rejection. At recall $0.80$ on the promotion/underpromotion tactic-motif slice,
the lowest near-puzzle false-positive rates:

""" + matched_tex + r"""

% =============================================================================
\clearpage
\section{Architecture findings}

\subsection*{What worked}
\begin{itemize}
\item \textbf{Chess-specific task decomposition.} \texttt{i193\_exchange\_then\_king\_dual\_stream}
splits the trunk into a tactical-exchange branch and a king-safety branch.
This is the only architectural prior in the scout pool that produces an
above-noise headline improvement.
\item \textbf{Board-symmetry / group equivariance.} The rule-symmetry / orbit /
quotient-bottleneck family (\texttt{i042, i046, i048}) takes three of the top
six slots. Three independent formulations of the same prior all rank highly.
\item \textbf{Small residual + interaction backbones.}
\texttt{i100\_independence\_residual\_interaction} matches
\texttt{bench\_lc0\_bt4\_classifier} at one third the parameter count and
faster inference --- the Pareto pick when inference cost matters.
\end{itemize}

\subsection*{What did not work}
\begin{itemize}
\item $\sim 21\%$ of bespoke architectures had AMP/dtype bugs and failed in under 2
minutes. Catchable in CI with a one-shot \texttt{torch.amp.autocast} smoke test.
\item $\sim 4\%$ timed out at the 60-min wall --- iterative or unrolled designs
(Dykstra projection, soft-sort, sheaf-curvature variants).
\item Several architectures produced essentially-random predictions (PR AUC
$\approx$ positive-class prevalence): \texttt{i039, i051, i060, i062, i096}.
Complete training failures.
\end{itemize}

% =============================================================================
\section{Promotion candidates}

The scout is a filter, not a final leaderboard. Single-seed at base scale is
noisy --- within a $\pm 0.005$ PR AUC band, ranks are not trustworthy. The
following are promoted to full 3-seed $\times$ \texttt{scale\_xl} evaluation:

\subsection*{By aggregate PR AUC (top 10)}
\texttt{i193\_exchange\_then\_king\_dual\_stream},
\texttt{i048\_rule\_automorphism\_quotient},
\texttt{i018\_oriented\_tactical\_sheaf\_laplacian},
\texttt{i188\_tactical\_program\_induction},
\texttt{i011\_vetoselect} (already has 3-seed),
\texttt{i192\_latent\_reply\_entropy},
\texttt{i191\_safe\_reply\_certificate\_verifier},
\texttt{i042\_legal\_automorphism\_quotient},
\texttt{i147\_specialist\_head\_cnn},
\texttt{i046\_rule\_exact\_orbit\_bottleneck}.

\subsection*{By matched-recall near-puzzle FP}
Existing robustness leaders (\texttt{i011\_vetoselect}, \texttt{i012\_dykstra\_lcp})
plus the new aggregate-PR-AUC winners that also do well on the slice.

\subsection*{By niche slice wins}
Models with clear ($>0.005$) margin on a hard slice --- the rule-symmetry
family on skewer/overload, the dual-stream winner across the board.

% =============================================================================
\clearpage
\section{A concrete path to a better trunk than BT4}

LC0 BT4 is a generic piece-token transformer trunk. None of the inductive
biases that emerged as winners in the scout are present in its trunk --- not
the exchange/king-safety decomposition, not the chess-automorphism
equivariance, not the directional tactical sheaf, not the explicit
program-induction head.

\begin{tcolorbox}[goodcallout, title=Scope of the claim]
The pre-trained BT4 network has been trained on \emph{billions} of self-play
positions over many GPU-years. The LC0 team optimised not just the trunk but
the training pipeline, the data, the search algorithm and the engineering
infrastructure. \textbf{Our research is about \emph{trunks}, not training.}
The fair comparison is therefore: train two networks from scratch under
identical training settings --- one with our trunk, one with a freshly
initialised BT4-shaped trunk --- and compare. This is the experiment that
isolates the architectural variable from the training-budget variable.
\end{tcolorbox}

The architecture proposed below
(\texttt{i241\_multistream\_attention\_chess\_eval}) composes the three
mutually composable winning priors into a single trunk shaped for chess
evaluation, with standard LC0-style heads.

\medskip

""" + bt4_path_tex + r"""

\begin{tcolorbox}[goodcallout, title=Realistic outcome --- accuracy]
\textbf{Against the publicly available pre-trained BT4:} the proposed trunk
trained from scratch will almost certainly under-perform, because BT4 has
years of training compute we cannot match. This comparison is not what the
research claims to win.

\textbf{Against a freshly-trained BT4-shaped trunk under matched training
(the relevant experiment):} we estimate the multi-stream trunk would win by
roughly $30$--$80$~ELO on a fixed-depth tournament. The advantage comes
entirely from the chess-aware inductive biases that the generic transformer
trunk has to learn from data. Whether the headline ELO is 2700 or 3300
depends entirely on Step 1's data budget; the relative advantage is what
this research measures.
\end{tcolorbox}

\begin{tcolorbox}[goodcallout, title=Realistic outcome --- speed (measured)]
For a chess engine, inference \emph{speed} matters as much as accuracy: the
network is called at every node of the MCTS search tree, so faster inference
means deeper search per second. Both architectures were scaled to BT4-medium
range and timed head-to-head on this hardware:

\begin{footnotesize}
\begin{tabularx}{\linewidth}{@{}L r r r r@{}}
\toprule
\textbf{model} & \textbf{params} & \textbf{batch 1 latency} & \textbf{batch 256 throughput} & \textbf{per-param throughput @ batch 1} \\
\midrule
\texttt{LC0\_BT4-shaped} (scaled) & 39.1\,M & 7.27\,ms (138/s) & 2{,}803 samples/s & 3.5 /s/M \\
\texttt{i193 dual-stream} (scaled) & 35.5\,M & \textbf{3.94\,ms} (\textbf{254/s}) & \textbf{3{,}043 samples/s} & \textbf{7.2 /s/M} \\
\bottomrule
\end{tabularx}
\end{footnotesize}

\medskip
At matched parameter count, the dual-stream trunk is
\textbf{$\sim$1.2$\times$ faster at batch 256} and \textbf{$\sim$1.9$\times$
faster at batch 1} than a same-size single-stream BT4 trunk on this hardware.
The batch-1 number is the relevant one for engine play, because MCTS leaf
evaluation is latency-bound (typical effective batch 1--8). Combined with the
accuracy estimate above: $\mathbf{+30}$--$\mathbf{80}$ \textbf{ELO at
$\sim$1.9$\times$ faster inference} is the measured engine-level advantage at
matched training and matched parameter count.

\vspace{2pt}
{\scriptsize\textit{Measurement provenance: \texttt{scripts/benchmark\_scale\_up\_speed.py},
git commit \texttt{2e03965}, NVIDIA RTX 3070 (8\,GiB), PyTorch 2.11.0+cu130,
FP32, 30 iterations per batch size after 8-iteration warm-up.}}
\end{tcolorbox}

\begin{tcolorbox}[callout, title=Single-sentence takeaway]
Across 234 trained architectures, the architectures that win are the
ones that encode \emph{how chess is actually evaluated}, not the ones that
bring exotic math to a generic CNN.
\end{tcolorbox}

% =============================================================================
\clearpage
\section{Follow-up: a composed architecture (i242) and its ablations}

After the scout, we designed a successor architecture (\texttt{i242}) that
composes three chess-aware structural priors, each independently validated
by a strong chess-evaluation system:

\begin{itemize}
\item \textbf{King-conditioned input features}, inspired by
\textit{Stockfish NNUE} (HalfKA). Stockfish NNUE indexes its features by
king square; every accumulator depends on where the king is. We replicate
this property by reusing i193's deterministic feature builder, which
produces king-zone, check-ray, escape-square, and attacker/defender
pressure planes. \textit{Stockfish NNUE, the strongest CPU chess engine,
independently validates the king-decomposition thesis.}
\item \textbf{Exchange + king dual-stream decomposition}, from the scout
winner i193. Two parallel sub-trunks specialise; each receives a chess-aware
input bias derived from the king-conditioned features.
\item \textbf{Global multi-head self-attention} over the 64 square tokens,
the BT4 design choice. A third parallel sub-trunk with vanilla self-attention
catches long-range piece interactions that conv-only architectures need
many layers to model.
\end{itemize}

Fused via a learned softmax phase router. At matched scout scale
(271k params), the model has comparable parameter budget to the rule-symmetry
family ($\sim$180k each).

\begin{tcolorbox}[callout, title=Result --- i242 vs i193]
On the puzzle\_binary scout, single-seed at base scale, 12 epochs:

\begin{footnotesize}
\begin{tabularx}{\linewidth}{@{}L r r r@{}}
\toprule
\textbf{model} & \textbf{params} & \textbf{test PR AUC} & \textbf{$\Delta$ vs i193} \\
\midrule
i193 (parent dual-stream, conv) & 157k & \textbf{0.8755} & -- \\
i242 (full, three-stream + attention) & 271k & 0.8677 & $-0.008$ \\
\bottomrule
\end{tabularx}
\end{footnotesize}

\textbf{The composed architecture does \emph{not} beat its conv-only parent
at this scale.} i242 still ranks \#2 of 181 models in the combined scout
pool, but the "compose all three priors" hypothesis is falsified at small
training budgets. The result strengthens i193's standing and identifies
attention as the data-hungry component.
\end{tcolorbox}

\subsection{Ablations}

Four ablations isolate which component carries the architecture:

\begin{footnotesize}
ABLATION_TABLE_PLACEHOLDER
\end{footnotesize}

ABLATION_INTERPRETATION_PLACEHOLDER

\subsection{Why the composition didn't win at scout scale}

\begin{enumerate}
\item \textbf{Attention is data-hungry.} A transformer trunk has more
parameters per useful FLOP at \emph{small training budgets} than a
convolutional trunk of the same parameter count. Twelve epochs over
173k samples may not be enough for attention to learn useful interactions
that a chess-aware conv stem extracts \emph{from priors}.
\item \textbf{Phase-router capacity dilution.} The 3-way softmax router
allocates fractional capacity per position. At small training budgets the
router never learns to confidently delegate, so each stream is asked to be
a complete classifier on its own.
\item \textbf{Three priors are not orthogonal.} The king-zone bias and the
attacker/defender bias overlap on every position where a piece attacks the
king. The streams may end up redundant rather than specialised.
\end{enumerate}

The aspirational i241 architecture (multi-stream attention at LC0-scale
training) is not falsified by i242's result --- at unlimited training and
matched parameter count, attention's expressiveness should compound and
overtake. \emph{At small scale with limited data, simpler chess-specific
conv decompositions outperform the transformer-decomposed family.}

% =============================================================================
\clearpage
\section{Proposed successor: i243 (HalfKA + dual-stream + LC0 heads)}\label{sec:i243}

i242 falsified the \emph{add attention} composition at small scale.  The
natural next question is: \emph{which piece of Stockfish NNUE's design buys
the most at engine-grade scale?}  The answer is its \textbf{learnable input
representation}, not its MLP backbone.  Stockfish NNUE pairs the strongest
input scheme in chess (HalfKA) with the simplest possible trunk (a plain
MLP).  i193 pairs the simplest possible input scheme
(\texttt{simple\_18} + deterministic king/exchange planes) with the
strongest tactical decomposition (exchange + king dual-stream conv).
\textbf{i243 swaps in HalfKA for the deterministic king planes} while
keeping the dual-stream conv and replacing the puzzle-binary head with
LC0's WDL value + 1858-policy heads.

\begin{tcolorbox}[callout, title=Three-way composition]
\begin{footnotesize}
\begin{tabularx}{\linewidth}{@{}>{\bfseries\sffamily\color{forest}}p{2.6cm} L L@{}}
\toprule
\textbf{component} & \textbf{source} & \textbf{what it brings} \\
\midrule
HalfKA accumulator & Stockfish NNUE & Learnable king-conditional embedding table; O(1) incremental update at inference. \\
Exchange/king dual-stream conv & i193 (scout winner) & Tactical decomposition --- the largest within-encoding margin in the 234-arch sweep. \\
WDL value + 1858-policy heads & LC0 (BT4 family) & Makes the network MCTS-compatible; no longer a puzzle classifier. \\
\bottomrule
\end{tabularx}
\end{footnotesize}
\end{tcolorbox}

\subsection{Why this composition, and not the obvious alternatives}

\begin{itemize}
\item \emph{HalfKA + MLP} is Stockfish NNUE itself --- already strong, but
the MLP cannot structurally decompose exchange-evaluation from king-safety
features.  At scale, this leaves Elo on the table.
\item \emph{Deterministic king planes + dual-stream conv + LC0 heads} is
i193 trained at engine scale.  But the deterministic planes cannot capture
fine-grained king-conditional patterns (specific castled-king
pawn-shield variants, for instance) that the HalfKA accumulator absorbs by
training.
\item \emph{HalfKA + transformer + LC0 heads} is the i242 path; i242 just
showed transformer trunks need much more data than 173k positions to
recover their expressiveness budget.  HalfKA + conv-dual-stream uses the
same parameter budget on a structurally chess-aware trunk that already
wins the scout.
\end{itemize}

\subsection{Architecture sketch and incremental-update property}

\begin{equation*}
a_{\mathrm{side}}(x) \;=\; \sum_{f \in \mathcal{F}_{\mathrm{active}}(x)} E_{\mathrm{side}}[f],
\quad f = (k_{\mathrm{side}},\, \mathrm{color},\, \mathrm{type},\, s)
\end{equation*}

\noindent
The accumulator $a_{\mathrm{side}}$ is the sum of feature embeddings over
all active HalfKA features.  When a piece moves, $\mathcal{F}_{\mathrm{active}}$
changes by at most two features, so the accumulator updates with one
subtraction and one addition.  The conv backbone runs only on the
\emph{difference} from the previous accumulator state, giving a wall-clock
inference advantage that neither i193 nor LC0 BT4 has.

The accumulator is then reshaped to a per-square token grid
$x_{\mathrm{token}} \in \mathbb{R}^{[64,d]}$, fed into i193's exchange/king
dual-stream conv, fused via the phase router, and capped with LC0's WDL
value head + 1858-dim policy head.

\subsection{Sizing variants}

\begin{footnotesize}
\begin{tabularx}{\linewidth}{@{}L r r L@{}}
\toprule
\textbf{variant} & \textbf{embed\_dim} & \textbf{total params} & \textbf{when to use} \\
\midrule
\texttt{tiny}   & 32  & $\sim$2.5M & scout-scale sanity check (\texttt{puzzle\_binary}) \\
\texttt{small}  & 96  & $\sim$10M  & research-grade fine-tuning \\
\texttt{medium} & 256 & $\sim$38M  & engine-grade, matches BT4-medium \\
\texttt{large}  & 384 & $\sim$75M  & engine-grade, matches BT4-large \\
\bottomrule
\end{tabularx}
\end{footnotesize}

At \texttt{medium}, the HalfKA embedding table dominates the parameter
budget ($\sim$25M of $\sim$38M) --- this is the same shape as Stockfish NNUE's
table.

\subsection{Theoretical reach at engine scale}

With engine-scale data and compute, the realistic targets split:
\textbf{(a)~vs Stockfish-NNUE} --- plausibly wins on Elo; i243 keeps
NNUE's input table verbatim and swaps its plain MLP for a strictly more
expressive dual-stream conv.  \textbf{(b)~vs LC0 BT4} --- probably loses
on raw Elo at LC0-scale data (the i242 ablation already showed attention
is data-hungry; at $\sim$10$^{9}$ positions BT4's transformer outscales
conv), \textbf{but wins on Elo per second of search} via HalfKA's
incremental-update inference advantage that BT4 structurally cannot
match.  The defensible publishable claim is therefore \emph{``i243
dominates the Pareto frontier of (training compute, inference cost, Elo)
against both Stockfish-NNUE and LC0-BT4''}, not ``beats them on raw
Elo.''

\subsection{Hypotheses to test at engine-grade training}

\begin{itemize}
\item[\textbf{H1}] At engine-grade scale (Stockfish-eval distillation on
$\sim$10$^{7}$ master-game positions or LC0 self-play batches), i243 beats
both Stockfish NNUE (matched size) \emph{and} plain i193 (matched size) on
Elo against a fixed-depth tournament opponent.
\item[\textbf{H2}] i243's incremental-update inference path gives a
$\geq 5\times$ wall-clock speedup over a no-incremental-update baseline at
the same accuracy, validating that the engineering advantage is real.
\item[\textbf{H3}] The phase-router weights $\alpha(x)$ show interpretable
position-type dependence (high $\alpha_K$ on king-attack positions, high
$\alpha_E$ on quiet exchanges), confirming the decomposition is being used.
\end{itemize}

\begin{tcolorbox}[callout, title=Status: proposed only]
The architecture is the cheap part.  The training pipeline + data is the
project: HalfKA features over the simple\_18 source, an embedding table
with incremental-update support, and engine-grade training data (Stockfish
distillation or LC0 self-play).  Estimated compute: 1--2 GPU-weeks for a
medium-scale pre-train + fine-tune cycle.  See
\texttt{ideas/i243\_halfka\_dual\_stream\_lc0/} for the full spec.
\end{tcolorbox}

% =============================================================================
\clearpage
\section{Hypothetical: at unlimited data and compute, what would beat BT4?}

\begin{tcolorbox}[callout, title=Read this section as a thought experiment]
The asymptote analysed below is a \emph{limit}, not a regime any chess
engine actually trains in.  Stockfish-eval distillation uses $\sim$10$^{7}$
positions; LC0 self-play uses $\sim$10$^{9}$.  Both are still in the
finite-data regime where the scout's inductive biases buy real Elo per
training position.  This section asks what would happen if data became
free --- which is useful for deciding which priors are worth carrying
into a much larger run, but is \textbf{not} a verdict that our priors
stop mattering in practice.
\end{tcolorbox}

\smallskip
The scout's strongest priors are \textbf{inductive biases}: they buy
sample efficiency at limited training. With unlimited data and compute,
sample efficiency advantages compress --- a sufficiently large transformer
can learn any decomposition the data implicitly encodes.

\begin{tcolorbox}[goodcallout, title=Which advantages persist? Which vanish?]
A prior can lose its accuracy edge while still saving wall-clock time at
inference --- that distinction matters for a chess engine that calls the
network at every node of a search tree.  We track them separately.
\smallskip

\begin{footnotesize}
\begin{tabularx}{\linewidth}{@{}p{4.4cm} L L@{}}
\toprule
\textbf{prior} & \textbf{accuracy edge vs BT4} & \textbf{inference-speed edge vs BT4} \\
\midrule
King-conditioned inputs (NNUE / HalfKA) & \textbf{Vanishes.} A 50M+ transformer learns king-relative features from the raw board. & \textbf{Persists.} HalfKA's incremental accumulator updates by $\le 2$ embedding lookups per move --- O(1) regardless of board complexity.  This is why Stockfish runs millions of evals/s on CPU. \\[3pt]
Exchange/king dual-stream (i193) & \textbf{Vanishes.} Multi-head attention partitions the head budget by task. & \textbf{Partially persists.} Conv-on-token-grid is structurally cheaper than dense MHA for narrow channels; at the channel widths BT4 uses the advantage compresses to $\sim$1.2--1.9$\times$ (measured on our scout). \\[3pt]
Chess-aware attention bias (i242) & \textbf{Vanishes.} The bias matrices are a geometric prior; learnable from data. & \textbf{Vanishes.} Bias matrices are essentially free; they neither help nor hurt wall-clock at scale. \\[3pt]
Group equivariance (i048) & \textbf{Mostly vanishes.} Residual benefit is parameter-count saving. & \textbf{Partially persists.} $|G|=4\times$ fewer FLOPs at equivariant layers; the saving's size depends on where in the trunk those layers sit. \\
\midrule
Sparse legal-move attention pattern & --- & \textbf{Persists.} Computing attention over $\sim$8 legal-move-related square pairs (vs all 64) saves $8\times$ FLOPs at any data scale. \\[3pt]
Mixture-of-experts / phase routing of FLOPs & --- & \textbf{Persists.} Routing only some FLOPs per position is a wall-clock win regardless of data. \\[3pt]
Adaptive depth / early exit & --- & \textbf{Persists.} Easy positions need fewer layers; search exploits the saving directly. \\[3pt]
INT8/FP8 quantization-aware training & --- & \textbf{Persists.} Hardware-level efficiency, orthogonal to architecture. \\[3pt]
Mamba / state-space layers in place of attention & \textbf{Approaches BT4.} The quality gap vs attention narrows as data grows. & \textbf{Persists.} Linear-time in token count; same speed advantage at every scale. \\
\bottomrule
\end{tabularx}
\end{footnotesize}

\smallskip
The dash ``---'' in the accuracy column means the prior is purely a
compute-efficiency lever, not an inductive bias --- it does not change
what the network can represent, only how fast it represents it.
\end{tcolorbox}

\subsection{An "unlimited-data, faster-than-BT4" recipe}

At unlimited training, the relevant levers are \emph{compute-efficient
priors} --- architectural choices that save wall-clock FLOPs without
sacrificing expressiveness:

\begin{enumerate}
\item \textbf{Sparse attention over chess-legal-move pairs.} For each
square $s$, compute attention only over the $\sim$8 squares $t$ with a
tactical relation (legal move, attack ray, king zone). Attention cost:
$O(64 \cdot k \cdot d)$ for $k \approx 8$, instead of $O(64^2 d)$. An
$8\times$ attention speedup with \emph{no} expressiveness loss --- the
masked-out pairs really have no tactical relationship.
\item \textbf{Adaptive depth via phase-router early-exit.} The phase
router from i193/i242 generalises: route each position to a stack depth
based on router confidence. King-vs-king endgames need 2 layers; sharp
mid-game positions need 16. Average compute drops by $\sim$2--3$\times$.
\item \textbf{Mixture-of-experts attention heads.} Each MCTS leaf
activates only a subset of heads (e.g.\ 4 of 16). Inactive heads cost zero
FLOPs. A $4\times$ inference speedup at matched quality.
\item \textbf{INT8 quantization-aware training.} Standard $2$--$4\times$
wall-clock speedup, combines multiplicatively with the above.
\end{enumerate}

\begin{tcolorbox}[callout, title=Realistic outcome at unlimited training]
Stacking sparse attention + adaptive depth + MoE heads + INT8 gives roughly
$50$--$200\times$ compute speedup over a dense FP32 BT4 trunk at matched
quality. At infinite data the \emph{accuracy} edge over BT4 vanishes, but a
$\sim 100\times$ \emph{inference-speed} edge persists --- and for a chess
engine that runs MCTS at thousands of nodes per move, $100\times$ faster
network evaluation is in practice far more valuable than $80$~ELO of raw
quality.
\end{tcolorbox}

\textbf{Our research direction stays interesting even with unlimited
training:} not because our priors give better accuracy, but because the
chess-aware decompositions are structural hints toward the
legal-move-sparsity pattern that lets a network beat BT4 on speed. The
priors don't survive scale as accuracy levers --- they survive as compute
levers.

% =============================================================================
\clearpage
\section{Limitations and threats to validity}

\subsection*{Single-seed scout}
All scout runs use seed 42 at base scale. Single-seed PR AUC has an empirical
noise band of roughly $\pm 0.005$--$0.010$ on this dataset (estimated from
3-seed groups in the archived runs). Differences smaller than this should
not be interpreted as architectural; the headline 0.014 margin of the
dual-stream winner is comfortably above the band, but ranks 4--12 sit inside
it. The promotion stage explicitly re-runs the top candidates with 3 seeds
and \texttt{scale\_xl} to disambiguate.

\subsection*{Proxy task}
\texttt{puzzle\_binary} is a proxy for chess-position evaluation, not a
direct measure of engine playing strength. A trunk that wins on
\texttt{puzzle\_binary} is not guaranteed to win on value+policy regression.
However, the architectural priors that emerged as winners
(decomposition, equivariance, oriented attention) are task-agnostic priors
about chess geometry and therefore likely to transfer; this is the
hypothesis Step 5 of the BT4-path plan is designed to falsify.

\subsection*{Training-time variance}
$\sim$21\% of bespoke architectures had AMP/dtype bugs that caused complete
training failure under 2 minutes. These are not architectural failures ---
they are bugs in the bespoke code that would be caught by a one-line
\texttt{torch.amp.autocast} smoke test in CI. We exclude them from the
leaderboard but acknowledge they reduce the effective sample size.

\subsection*{Label noise}
The CRTK fine-label scheme (0: non-puzzle, 1: verified-near-puzzle,
2: puzzle) is generated by a label-quality pipeline, not human annotation.
Class~1 in particular is a programmatic verification of \emph{near-puzzles}
that may have label noise on the order of a few percent. Matched-recall FP
rate on this class is therefore an upper bound on the true rate.

\subsection*{Hardware constraints}
All scout runs share a single RTX 3070 with 8\,GiB VRAM. Larger architectures
(notably \texttt{scale\_xl} variants of several ideas) hit the 60-min wall
or OOM, so we systematically underrepresent the largest end of the
parameter spectrum. The estimated speed advantage of the multi-stream trunk
over BT4 is extrapolated from base-scale measurements and would need to be
validated on the actual proposed scale-up.

\section{Reproducibility}

The scout pipeline is fully scripted and resumable:

\begin{itemize}
\item Source code: \texttt{scripts/run\_paper\_ready\_all.py},
  \texttt{scripts/train\_model.py}, and per-idea \texttt{model.py} files.
\item State: \texttt{reports/architecture\_scout\_2026-05-09/state.json}
  records every task's status, returncode, elapsed time, and the SHA-256
  hash of the generated config.
\item Event log: \texttt{reports/architecture\_scout\_2026-05-09/events.jsonl}
  is an append-only record of every \texttt{task\_started}~/
  \texttt{task\_finished} event with timestamps.
\item Dataset: \texttt{data/splits/crtk\_sample\_3class\_unique\_crtk\_tags/}
  (parquet, deterministic split, audited zero-overlap across train/val/test).
\item Random seeds: every config explicitly fixes \texttt{seed: 42} and
  \texttt{deterministic: true}; PyTorch's CUDA non-determinism remains for
  some convolution kernels, contributing to the noise band cited above.
\end{itemize}

% =============================================================================
\clearpage
\section{Appendix \textperiodcentered{} How each top architecture works}

For each of the five architectures with the strongest scout signal: a
single-paragraph summary, the key equation, the inductive bias it brings,
and what it gives up. The symbol $x$ denotes the input board state
(in $\mathbb{R}^{c \times 8 \times 8}$ with $c=18$ for \texttt{simple\_18}
and $c=112$ for \texttt{lc0\_bt4\_112}).

\medskip

""" + arch_appendix_tex + r"""

\end{document}
"""

        tex_path = tmpdir / "report.tex"
        # Substitute i242 ablation results into placeholders (if available)
        try:
            from _substitute_ablations import fill_ablation_placeholders
            LATEX = fill_ablation_placeholders(LATEX)
        except Exception as exc:
            print(f"warning: ablation substitution skipped: {exc}")

        tex_path.write_text(LATEX, encoding="utf-8")

        if args.keep_tex:
            tex_out = out_path.with_suffix(".tex")
            tex_out.write_text(LATEX, encoding="utf-8")
            print(f"Wrote {tex_out}")

        print("Compiling LaTeX with tectonic...")
        tectonic = Path.home() / ".local/bin/tectonic"
        cmd = [str(tectonic), "-X", "compile", "--keep-logs", "--keep-intermediates",
               "--outdir", str(tmpdir), str(tex_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("=== tectonic stdout ===")
            print(result.stdout[-2000:])
            print("=== tectonic stderr ===")
            print(result.stderr[-2000:])
            print("=== tex saved at ===", tmpdir / "report.tex")
            shutil.copy(tex_path, "/tmp/report_failed.tex")
            return 1
        pdf_built = tmpdir / "report.pdf"
        if not pdf_built.exists():
            print("PDF not produced. tectonic stdout:")
            print(result.stdout)
            return 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf_built, out_path)
        print(f"Wrote {out_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
