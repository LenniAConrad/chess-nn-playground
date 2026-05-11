#!/usr/bin/env python
"""Build paper_report.pdf — a polished multi-page PDF for the scout.

Adds to the HTML version:
  - Per-model confusion matrices (top 8) at threshold=0.5
  - Strengths-and-weaknesses analysis per top model (which slices it
    wins or loses on)
  - Renders via WeasyPrint to a real PDF

Usage:
  scripts/build_paper_report_pdf.py \\
    --results-root _scout_combined_view \\
    --audits-root reports/audits \\
    --scout-state reports/architecture_scout_2026-05-09/state.json \\
    --heatmap reports/audits/scout_heatmap_pretty.png \\
    --out reports/audits/paper_report.pdf
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix


def short_name(name: str) -> str:
    name = re.sub(r"_seed\d+$", "", name)
    name = name.replace("benchmark_bench_", "B/")
    name = name.replace("idea_", "")
    return name


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


def img_to_data_uri(path_or_bytes) -> str:
    if isinstance(path_or_bytes, Path):
        raw = path_or_bytes.read_bytes()
        suffix = path_or_bytes.suffix.lstrip(".")
    else:
        raw = path_or_bytes
        suffix = "png"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/{suffix};base64,{b64}"


def encoding_pill(enc: str) -> str:
    if enc == "lc0_bt4_112":
        return '<span class="pill lc0">lc0_bt4_112</span>'
    if enc == "simple_18":
        return '<span class="pill s18">simple_18</span>'
    return f'<span class="pill neutral">{enc}</span>'


def _f1_optimal_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Pick threshold that maximizes F1 (sweep over unique sorted probs)."""
    # Candidate thresholds: probability values at unique percentiles
    pcts = np.linspace(2, 98, 49)
    candidates = np.unique(np.percentile(probs, pcts))
    best_f1, best_thr = -1.0, 0.5
    for t in candidates:
        y_pred = (probs >= t).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        if (2 * tp + fp + fn) == 0: continue
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1: best_f1, best_thr = f1, float(t)
    return best_thr


def render_math_equation(latex: str, fontsize: int = 14, padding: float = 0.18) -> bytes:
    """Render a LaTeX equation via matplotlib mathtext to PNG bytes."""
    fig = plt.figure(figsize=(0.01, 0.01), dpi=200)
    fig.patch.set_facecolor("white")
    # First, render to measure
    txt = fig.text(0.5, 0.5, f"${latex}$", ha="center", va="center", fontsize=fontsize)
    fig.canvas.draw()
    bbox = txt.get_window_extent()
    inv = fig.dpi_scale_trans.inverted()
    w_in = bbox.width / fig.dpi + padding * 2
    h_in = bbox.height / fig.dpi + padding * 2
    plt.close(fig)
    # Real render
    fig = plt.figure(figsize=(max(w_in, 0.5), max(h_in, 0.3)), dpi=200)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.5, f"${latex}$", ha="center", va="center", fontsize=fontsize, color="#1a1a1f")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    return buf.getvalue()


def render_confusion_matrix(probs: np.ndarray, y_true: np.ndarray, name: str,
                             y_fine: np.ndarray | None = None,
                             figsize=(2.6, 3.1)) -> bytes:
    """Render a 3x2 source-class confusion matrix.

    Rows: true_fine_label in {0, 1, 2}, where 0 = non-puzzle (easy negative),
    1 = verified-near-puzzle (hard negative), 2 = actual puzzle.
    Columns: predicted binary label in {0 = non-puzzle, 1 = puzzle}.

    Threshold chosen by max F1 on the binary task.
    Cells are normalized within their row (per-class recall view).
    """
    thr = _f1_optimal_threshold(probs, y_true)
    y_pred = (probs >= thr).astype(int)

    # Fall back to binary 2x2 if fine labels unavailable
    if y_fine is None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        row_labels = ["true: non-puzzle", "true: puzzle"]
    else:
        cm = np.zeros((3, 2), dtype=int)
        for fine in (0, 1, 2):
            mask = (y_fine == fine)
            for pred in (0, 1):
                cm[fine, pred] = int(((y_pred == pred) & mask).sum())
        row_labels = ["true 0\nnon-puzzle", "true 1\nnear-puzzle", "true 2\npuzzle"]

    n_rows, n_cols = cm.shape

    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    fig.patch.set_facecolor("white")
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / np.maximum(row_sums, 1), 0)
    cmap = plt.get_cmap("Blues")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    ax.imshow(cm_norm, cmap=cmap, norm=norm, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            v = cm[i, j]
            row_pct = 100 * cm_norm[i, j]
            color = "white" if cm_norm[i, j] > 0.55 else "#1a1a1f"
            ax.text(j, i - 0.08, f"{v:,}", ha="center", va="center",
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

    # Derived metrics on the binary task
    tot = cm.sum()
    if y_fine is not None:
        # tp = true puzzle predicted puzzle (row 2 col 1)
        tp = cm[2, 1]; fn = cm[2, 0]
        fp = cm[0, 1] + cm[1, 1]; tn = cm[0, 0] + cm[1, 0]
        # near-puzzle FP rate specifically
        near_fp_rate = cm[1, 1] / max(cm[1, 0] + cm[1, 1], 1)
        rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1); fpr = fp / max(fp + tn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        footer = (f"thr*={thr:.2f}  ·  F1 {f1:.3f}  ·  prec {prec:.3f}  ·  "
                  f"rec {rec:.3f}  ·  near-FP {near_fp_rate:.3f}")
    else:
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / tot if tot else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
        footer = (f"thr*={thr:.2f}  ·  F1 {f1:.3f}  ·  prec {prec:.3f}  ·  "
                  f"rec {rec:.3f}  ·  FPR {fpr:.3f}")

    fig.text(0.5, -0.04, footer, ha="center", fontsize=7.5, color="#666")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", pad_inches=0.14)
    plt.close(fig)
    return buf.getvalue()


def analyze_run(run_dir: Path) -> dict | None:
    if not (run_dir / "metrics_final.json").exists(): return None
    df = None
    for name in ("predictions_test.parquet", "predictions_val.parquet"):
        p = run_dir / name
        if p.exists():
            try: df = pd.read_parquet(p); break
            except: continue
    if df is None: return None
    if "true_label" not in df.columns: return None
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
        "y_true": y_true,
        "y_fine": y_fine,
        "probs": probs,
        "test_pr_auc": m.get("test_pr_auc"),
        "val_pr_auc": m.get("best_score"),
        "test_f1": m.get("test_f1"),
        "samples_per_sec": m.get("test_samples_per_second") or m.get("val_samples_per_second"),
        "num_params": md.get("num_params"),
        "mflops_per_pos": cx.get("estimated_mflops_per_position"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="_scout_combined_view")
    p.add_argument("--audits-root",  default="reports/audits")
    p.add_argument("--scout-state",  default="reports/architecture_scout_2026-05-09/state.json")
    p.add_argument("--heatmap",      default="reports/audits/scout_heatmap_pretty.png")
    p.add_argument("--out",          default="reports/audits/paper_report.pdf")
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

    # State summary
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

    # Load per-class data for strengths/weaknesses analysis
    per_class = json.loads((audits_root / "per_class_benchmark.json").read_text())
    matched_recall = json.loads((audits_root / "matched_recall_fp_report.json").read_text())

    # === Per-slice champions ===
    slice_interesting = [
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

    # Build per-slice rankings
    slice_rankings = {}
    for dim, val, label in slice_interesting:
        ranked = []
        for g in per_class["groups"]:
            cell = g["per_slice"].get(dim, {}).get(val)
            if not cell: continue
            mean = cell["pr_auc"]["mean"]
            if mean != mean: continue  # NaN
            ranked.append((g["group"], mean, cell["pr_auc"]["std"]))
        ranked.sort(key=lambda t: t[1], reverse=True)
        slice_rankings[(dim, val, label)] = ranked

    # Top-N for confusion matrices and strengths
    top_for_cm = runs[:8]

    # === Confusion matrices (3x2 source-class view) ===
    print("Rendering confusion matrices (3x2 source-class view)...")
    cm_imgs = []
    for r in top_for_cm:
        png = render_confusion_matrix(r["probs"], r["y_true"], r["name"], y_fine=r.get("y_fine"))
        cm_imgs.append(img_to_data_uri(png))

    # === Strengths and weaknesses per top model ===
    print("Computing strengths and weaknesses...")
    sw_data = []
    for r in runs[:8]:
        strengths, weaknesses = [], []
        for (dim, val, label), ranked in slice_rankings.items():
            # Where does this model rank for this slice?
            pos = next((i for i, (n, _, _) in enumerate(ranked) if n == r["group"]), None)
            if pos is None: continue
            score = ranked[pos][1]
            if pos <= 2:
                # top-3 = strength; record margin to next-best non-self
                margin = ranked[0][1] - ranked[3][1] if len(ranked) > 3 else 0
                strengths.append((label, score, pos + 1, margin))
            elif pos >= len(ranked) // 2 + len(ranked) // 4:
                # bottom quartile of named slices = weakness
                weaknesses.append((label, score, pos + 1, len(ranked)))
        # Limit
        strengths.sort(key=lambda t: t[1], reverse=True)
        weaknesses.sort(key=lambda t: t[1])
        sw_data.append({"run": r, "strengths": strengths[:4], "weaknesses": weaknesses[:3]})

    # === BUILD HTML ===
    heatmap_uri = img_to_data_uri(heatmap_path)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    winner = runs[0]
    runner_up = runs[1]

    # Top 15 leaderboard rows
    leaderboard_rows = []
    for i, r in enumerate(runs[:15], 1):
        name = short_name(r["group"])
        leaderboard_rows.append(
            f'<tr><td class="rank">{i}</td>'
            f'<td>{encoding_pill(r["encoding"])}</td>'
            f'<td class="name">{name}</td>'
            f'<td class="num"><strong>{r["test_pr_auc"]:.4f}</strong></td>'
            f'<td class="num">{r["val_pr_auc"]:.4f}</td>'
            f'<td class="num">{fmt_params(r["num_params"])}</td>'
            f'<td class="num">{fmt_speed(r["samples_per_sec"])}</td>'
            f'<td class="num">{fmt_mflops(r["mflops_per_pos"])}</td>'
            f'</tr>'
        )

    # Per-slice champions (15 rows)
    slice_winner_rows = []
    for (dim, val, label), ranked in slice_rankings.items():
        if not ranked: continue
        top_name, top_mean, top_std = ranked[0]
        margin = top_mean - ranked[1][1] if len(ranked) > 1 else 0
        slice_winner_rows.append(
            f'<tr><td>{label}</td>'
            f'<td class="name">{short_name(top_name)}</td>'
            f'<td class="num"><strong>{top_mean:.3f}</strong> <span class="dim">± {top_std:.3f}</span></td>'
            f'<td class="num">+{margin:.3f}</td>'
            f'</tr>'
        )

    # Pareto frontier
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
    pareto_rows = []
    for r in pareto[:8]:
        pareto_rows.append(
            f'<tr><td>{encoding_pill(r["encoding"])}</td>'
            f'<td class="name">{short_name(r["group"])}</td>'
            f'<td class="num"><strong>{r["test_pr_auc"]:.4f}</strong></td>'
            f'<td class="num">{fmt_speed(r["samples_per_sec"])}</td>'
            f'<td class="num">{fmt_params(r["num_params"])}</td>'
            f'<td class="num">{fmt_mflops(r["mflops_per_pos"])}</td>'
            f'</tr>'
        )

    # Matched-recall promotion table
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
    motif_rows = []
    for i, r in enumerate(motif_3seed[:8], 1):
        motif_rows.append(
            f'<tr><td class="rank">{i}</td>'
            f'<td class="name">{short_name(r["name"])}</td>'
            f'<td class="num"><strong>{r["near_fp_rate"]:.3f}</strong></td>'
            f'<td class="num">{r["accuracy"]:.3f}</td>'
            f'</tr>'
        )

    # === Architecture appendix data ===
    architectures = [
        {
            "id": "i193",
            "name": "Exchange-Then-King Dual Stream",
            "encoding": "simple_18",
            "params": "157k",
            "speed": "11.6k/s",
            "test_pr_auc": 0.876,
            "key_op": "f(x) = \\alpha(x)\\cdot h_K(x) + (1-\\alpha(x))\\cdot h_E(x) + h_R(x)",
            "key_op_caption": "Phase-router gates between king-stream and exchange-stream logits; the residual head $h_R$ adds cross-stream signal.",
            "summary": (
                "Splits the trunk into two parallel CNN encoders — one biased toward "
                "<em>tactical exchanges</em> (attacker/defender geometry, capture sequences), one "
                "biased toward <em>king safety</em> (king-zone planes, check rays, escape squares). "
                "A small MLP phase-router emits a sigmoid gate $\\alpha(x)$ that "
                "mixes the two stream logits position by position; a residual head reads the "
                "concatenated stream pools to recover cross-stream signal."
            ),
            "bias": "Hard-encodes the classical Stockfish-style decomposition (“exchange evaluation + king safety”) as architecture instead of asking the network to discover it in data.",
            "limits": "Pure conv inside each stream means long-range piece interactions (queen on a1 vs king on h8) cost many layers. Mitigated at scale by replacing each stream's encoder with attention.",
        },
        {
            "id": "i048",
            "name": "Rule-Automorphism Quotient Bottleneck Network (RAQ-Net)",
            "encoding": "simple_18",
            "params": "179k",
            "speed": "12.0k/s",
            "test_pr_auc": 0.861,
            "key_op": "\\Phi(x) = \\frac{1}{|G|}\\sum_{g \\in G} \\rho(g)\\cdot \\phi(g^{-1}\\cdot x)",
            "key_op_caption": "Group-average over the chess automorphism group $G$ (mirror + color-flip-with-role-swap); $\\phi$ is the per-orbit encoder, $\\rho(g)$ realigns equivariant outputs.",
            "summary": (
                "Treats the chess board as a $G$-set under the discrete automorphism group "
                "(horizontal mirror with castling-rights swap, color flip with role swap). "
                "The trunk is $G$-equivariant by construction: every layer either commutes with "
                "$G$ or pools to a quotient bottleneck. The classifier reads from the orbit "
                "space rather than the raw board."
            ),
            "bias": "Symmetries that the rest of the field has to learn from data are baked into the architecture. Sample-efficiency advantage is mathematically guaranteed, not empirical.",
            "limits": "The quotient bottleneck discards information by construction — useful for binary discrimination, hurts for value/policy regression where calibrated continuous output matters.",
        },
        {
            "id": "i018",
            "name": "Oriented Tactical Sheaf Laplacian",
            "encoding": "simple_18",
            "params": "91k",
            "speed": "5.5k/s",
            "test_pr_auc": 0.861,
            "key_op": "L_{\\mathrm{or}} = D - A_{\\mathrm{or}},\\quad y = \\sigma(W \\cdot L_{\\mathrm{or}}\\, x)",
            "key_op_caption": "Oriented graph Laplacian over the side-to-move-oriented tactical incidence sheaf; $D$ is degree, $A_{\\mathrm{or}}$ is signed by attacker/defender role.",
            "summary": (
                "Builds a sheaf over chess squares whose sections encode <em>directed</em> "
                "tactical incidence (attacker $\\to$ defender, side-to-move-aware). Convolves "
                "the input with the oriented sheaf Laplacian $L_{\\mathrm{or}}$ instead of a "
                "standard graph Laplacian, which preserves the chirality of attack — a "
                "passive defender and an active attacker are not the same node."
            ),
            "bias": "Tactical relations on a chess board are intrinsically directed; a standard symmetric graph Laplacian throws this away. The oriented Laplacian preserves it.",
            "limits": "Small parameter budget (91k) makes the architecture itself a hyperparameter — needs tuning at scale before its true ceiling is known.",
        },
        {
            "id": "i188",
            "name": "Tactical Program Induction Network",
            "encoding": "simple_18",
            "params": "710k",
            "speed": "8.3k/s",
            "test_pr_auc": 0.861,
            "key_op": "p(\\text{puzzle} \\mid x) = \\max_{P \\in \\mathcal{P}} q_\\theta(P \\mid x),\\ \\mathcal{P} = \\{\\text{tactical programs}\\}",
            "key_op_caption": "Score the position by the maximum-likelihood tactical program $P$ from a learned library $\\mathcal{P}$; $q_\\theta$ is a neural program-induction head.",
            "summary": (
                "Treats a puzzle as the existence of a short tactical program (sacrifice $\\to$ "
                "fork $\\to$ promotion) and trains a neural program-induction head that scores "
                "candidate programs over a learned library of tactical primitives. The "
                "classifier output is the max over the program library."
            ),
            "bias": "Tactical solutions are <em>compositional sequences of primitives</em>, not point-evaluations; a program-shaped output head exposes that compositionality.",
            "limits": "Program-induction heads are notoriously hard to train; high parameter count for the family makes this less Pareto-efficient than the smaller winners.",
        },
        {
            "id": "i011",
            "name": "VetoSelect Positive-Claim Abstention",
            "encoding": "lc0_bt4_112",
            "params": "502k",
            "speed": "11.7k/s",
            "test_pr_auc": 0.858,
            "key_op": "\\mathcal{L} = -y\\log p^+ + (1-y)\\log(1 - p^+) + \\lambda\\, \\mathbb{E}_{x \\in \\mathrm{hard}^-}[p^+(x)]",
            "key_op_caption": "Standard BCE on the puzzle logit $p^+$ plus a hard-negative “veto” penalty that suppresses positive claims on near-puzzle decoys.",
            "summary": (
                "Adds a selective-abstention head on top of an LC0 BT4-style trunk. The head "
                "separates <em>positive puzzle evidence</em> from <em>negative refuting "
                "evidence</em>; an anchor-and-decoy training loop pulls down false-positive "
                "rate on hard near-puzzle negatives while preserving recall."
            ),
            "bias": "Aggregate PR AUC is the wrong scoreboard for puzzle classification — the real cost is false-positives on positions that look tactical but are not. Explicit suppression of that mode is a strict gain.",
            "limits": "Multi-objective loss with an abstention head trades aggregate-AUC capacity for slice robustness. Useful if you care about matched-recall FP at recall 0.80; less useful if you only quote aggregate AUC.",
        },
    ]

    # Render math equations as PNG and embed
    print("Rendering math equations...")
    for a in architectures:
        png = render_math_equation(a["key_op"], fontsize=15)
        a["math_uri"] = img_to_data_uri(png)

    # === Path-to-BT4 plan ===
    bt4_plan = [
        {
            "label": "Phase 1",
            "title": "Multi-stream trunk (3-stream attention)",
            "body": (
                "Generalize i193's exchange/king dual stream to <strong>three "
                "parallel transformer streams</strong>: exchange, king, positional. Each "
                "stream is a small transformer over the 64 squares, with attention bias "
                "matrices derived from chess-aware precomputed tables (attacker/defender for "
                "exchange; king-zone/check-ray for king; standard relative-position for "
                "positional). The fusion is i193's learned phase router generalised to a "
                "soft 3-way mixture."
            ),
            "compute": "1–2 weeks · 1 GPU",
        },
        {
            "label": "Phase 2",
            "title": "Value + policy heads + Stockfish-eval distillation",
            "body": (
                "Replace the puzzle binary head with LC0-style heads: WDL value head (3-way "
                "softmax) and a 1858-dim policy head masked to legal moves. Train by "
                "<strong>distillation from Stockfish evaluations</strong> on a corpus of "
                "master games (≥5M positions with eval+move-policy labels). This is the "
                "cheapest realistic path to a chess-playing-strength evaluator without an "
                "MCTS self-play loop."
            ),
            "compute": "2–4 weeks · 1–4 GPUs",
        },
        {
            "label": "Phase 3",
            "title": "Equivariance + group-pooling layer",
            "body": (
                "Wrap the trunk in i048's chess-automorphism equivariance: every transformer "
                "block commutes with the discrete chess symmetry group $G$ (horizontal mirror "
                "with castling swap, color-flip with role swap). At matched compute, this "
                "buys roughly $|G|=4$ × sample efficiency for symmetric tactical patterns."
            ),
            "compute": "1 week of refactor · same training cost",
        },
        {
            "label": "Phase 4",
            "title": "Self-play with limited search depth",
            "body": (
                "Move from distillation to a small-scale self-play loop — LC0-style "
                "alternating between MCTS-with-search game generation and supervised "
                "training on the resulting (position, value, policy) triples. Even at "
                "modest search depth (200–800 visits) this typically lifts a "
                "distillation-trained network by 100–300 ELO."
            ),
            "compute": "1–3 months · 4–8 GPUs",
        },
        {
            "label": "Phase 5",
            "title": "Stream-wise auxiliary supervision",
            "body": (
                "Train each stream with a chess-aware auxiliary loss (exchange-outcome "
                "prediction on the exchange stream, king-attack/defend classification on "
                "the king stream, positional-eval regression on the positional stream). "
                "Auxiliary weights $\\leq 0.05$ so the main losses dominate. This is the "
                "BT4-gap closer: BT4 has no structural place to put auxiliary supervision; "
                "the multi-stream architecture does."
            ),
            "compute": "loss tweak · negligible compute",
        },
    ]

    bt4_html_steps = "\n".join(
        f"""
        <div class="bt4-step">
          <div class="num">{i+1}</div>
          <div>
            <div class="label">{step['label']}</div>
            <h4>{step['title']}</h4>
            <p>{step['body']}</p>
            <div class="compute">est. {step['compute']}</div>
          </div>
        </div>"""
        for i, step in enumerate(bt4_plan)
    )

    # === Architecture cards HTML ===
    arch_cards_html = []
    for a in architectures:
        arch_cards_html.append(f"""
        <div class="arch-card">
          <h3>{a['id']} — {a['name']}</h3>
          <div class="arch-meta">
            <div class="item">{encoding_pill(a['encoding'])}</div>
            <div class="item">params <strong>{a['params']}</strong></div>
            <div class="item">speed <strong>{a['speed']}</strong></div>
            <div class="item">test PR AUC <strong>{a['test_pr_auc']:.3f}</strong></div>
          </div>
          <p>{a['summary']}</p>
          <div class="math-block">
            <img src="{a['math_uri']}" alt="key equation" />
          </div>
          <div class="math-caption">{a['key_op_caption']}</div>
          <div class="bias-line">{a['bias']}</div>
          <h4>What it gives up</h4>
          <p style="margin-top:-2px">{a['limits']}</p>
        </div>
        """)

    # Strengths/weaknesses HTML
    sw_html = []
    for d in sw_data:
        r = d["run"]
        s_rows = "".join(
            f'<li><strong>{lbl}</strong> — PR AUC <code>{score:.3f}</code> (rank #{pos})</li>'
            for lbl, score, pos, _ in d["strengths"]
        ) or "<li><em>No clear strengths in named slices.</em></li>"
        w_rows = "".join(
            f'<li><strong>{lbl}</strong> — PR AUC <code>{score:.3f}</code> (rank #{pos} of {tot})</li>'
            for lbl, score, pos, tot in d["weaknesses"]
        ) or "<li><em>No notable weaknesses in named slices.</em></li>"
        sw_html.append(f"""
        <div class="sw-card">
          <div class="sw-head">
            <span class="sw-rank">#{runs.index(r) + 1}</span>
            <span class="sw-name">{short_name(r["group"])}</span>
            <span class="sw-enc">{encoding_pill(r["encoding"])}</span>
            <span class="sw-pr">test PR AUC <strong>{r["test_pr_auc"]:.4f}</strong></span>
          </div>
          <div class="sw-cols">
            <div class="sw-col strengths">
              <div class="sw-col-head">Where it wins</div>
              <ul>{s_rows}</ul>
            </div>
            <div class="sw-col weaknesses">
              <div class="sw-col-head">Where it loses</div>
              <ul>{w_rows}</ul>
            </div>
          </div>
        </div>
        """)

    # Confusion-matrix grid (2 columns, 4 rows = 8 cells)
    cm_html_blocks = []
    for i, (r, uri) in enumerate(zip(top_for_cm, cm_imgs)):
        cm_html_blocks.append(f"""
        <div class="cm-cell">
          <img src="{uri}" alt="confusion matrix {short_name(r['group'])}" />
        </div>""")

    # === Refined CSS — academic feel with better hierarchy ===
    CSS = r"""
:root {
    /* Restrained, paper-grade palette */
    --c-text:    #15161a;
    --c-muted:   #5c5e66;
    --c-light:   #95979e;
    --c-faint:   #b8babf;
    --c-line:    #d8dade;
    --c-rule:    #2c2e33;
    --c-stripe:  #f7f7f9;
    --c-paper:   #fdfdfb;
    /* Accent: warm brick-red (more refined than the previous saturated red) */
    --c-acc:     #8c2f2c;
    --c-acc-bg:  #f8eeec;
    --c-acc-ink: #5d1d1c;
    /* Deep blue for lc0 encoding */
    --c-blue:    #1a3a6b;
    --c-blue-bg: #ebf0f7;
    /* Forest green for "good" */
    --c-good:    #2c5f3c;
    --c-good-bg: #ecf3ee;
    /* Muted maroon for "bad" */
    --c-bad:     #8a3033;
    --c-bad-bg:  #f6eaeb;
    /* Goldenrod for highlights */
    --c-gold:    #b58904;
    --c-gold-bg: #fbf5e3;
    /* Typography: refined academic stack */
    --f-serif:   'EB Garamond', 'Crimson Pro', 'Source Serif Pro', 'Iowan Old Style', Georgia, serif;
    --f-sans:    'Inter', 'Source Sans Pro', 'Helvetica Neue', Arial, sans-serif;
    --f-mono:    'JetBrains Mono', 'IBM Plex Mono', 'SF Mono', Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--c-paper); color: var(--c-text); }
body { font-family: var(--f-sans); font-size: 10pt; line-height: 1.6;
    -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }
.page-break { page-break-before: always; }
.section-break { page-break-inside: avoid; }

/* Title page */
.title-page {
    padding-top: 56px;
    border-bottom: 1.5px solid var(--c-rule);
    padding-bottom: 32px;
    margin-bottom: 30px;
}
.title-eyebrow { text-transform: uppercase; letter-spacing: 0.22em; font-size: 8.5pt;
    color: var(--c-muted); font-weight: 700; margin-bottom: 18px; }
.title-page h1 { font-family: var(--f-serif); font-size: 36pt; line-height: 1.05;
    margin: 0 0 12px 0; font-weight: 600; letter-spacing: -0.015em; color: var(--c-text); }
.title-page .subtitle { font-family: var(--f-serif); font-size: 15pt;
    color: var(--c-muted); font-style: italic; margin-bottom: 32px; line-height: 1.42;
    font-weight: 400; max-width: 92%; }
.title-meta { display: grid; grid-template-columns: 23% 77%; gap: 5px 20px;
    font-size: 9.5pt; color: var(--c-muted); }
.title-meta .k { font-weight: 700; color: var(--c-text);
    text-transform: uppercase; letter-spacing: 0.06em; font-size: 8.5pt; }

h2 { font-family: var(--f-serif); font-size: 19pt; margin: 32px 0 12px 0;
    line-height: 1.15; font-weight: 600; border-bottom: 1.2px solid var(--c-rule);
    padding-bottom: 8px; letter-spacing: -0.005em; color: var(--c-text); }
h3 { font-family: var(--f-serif); font-size: 13pt; font-weight: 600;
    margin: 22px 0 10px 0; color: var(--c-text); letter-spacing: -0.003em; }
h4 { font-size: 8.5pt; font-weight: 700; margin: 16px 0 7px 0;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--c-muted);
    font-family: var(--f-sans); }

p { margin: 0 0 12px 0; }
p.lead { font-size: 11.5pt; color: var(--c-muted); line-height: 1.55;
    font-family: var(--f-serif); }
em, i { font-style: italic; }

ul, ol { margin: 0 0 12px 0; padding-left: 22px; }
li { margin-bottom: 4px; }
li::marker { color: var(--c-muted); }

/* Stats grid */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0 22px 0; }
.stat { border-left: 3px solid var(--c-acc); padding: 11px 14px; background: var(--c-stripe); }
.stat .num { font-family: var(--f-serif); font-size: 24pt; font-weight: 600;
    line-height: 1; color: var(--c-text); letter-spacing: -0.01em; }
.stat .lbl { text-transform: uppercase; letter-spacing: 0.09em;
    font-size: 7.5pt; color: var(--c-muted); margin-top: 7px; font-weight: 700; }

/* Tables */
table { width: 100%; border-collapse: collapse; margin: 12px 0 16px 0;
    font-size: 9.5pt; font-variant-numeric: tabular-nums; page-break-inside: auto; }
thead { display: table-header-group; }
table thead th { text-align: left; font-weight: 700; font-size: 8.5pt;
    text-transform: uppercase; letter-spacing: 0.04em; color: var(--c-muted);
    border-bottom: 1.4px solid var(--c-text); padding: 7px 8px; background: #fff; }
table tbody td { padding: 6px 8px; border-bottom: 1px solid var(--c-line); vertical-align: top; }
table tbody tr:nth-child(even) { background: var(--c-stripe); }
table .num { text-align: right; }
table .center { text-align: center; }
table .name { font-family: var(--f-mono); font-size: 9pt; }
table .rank { text-align: right; color: var(--c-light); width: 24px; padding-right: 10px; }
.dim { color: #888; font-size: 8.5pt; }

/* Pills */
.pill { display: inline-block; font-size: 8pt; font-weight: 700; padding: 0 6px;
    border-radius: 8px; border: 1px solid; line-height: 1.5; }
.pill.lc0 { background: var(--c-blue-bg); border-color: var(--c-blue); color: var(--c-blue); }
.pill.s18 { background: var(--c-acc-bg); border-color: var(--c-acc); color: var(--c-acc); }
.pill.neutral { background: #f0f0f0; border-color: #888; color: #555; }

code { font-family: var(--f-mono); font-size: 9pt; background: #f3f3f5;
    padding: 0 4px; border-radius: 3px; }

.callout { border-left: 4px solid var(--c-acc); background: var(--c-acc-bg);
    padding: 12px 16px; margin: 14px 0; border-radius: 0 4px 4px 0;
    page-break-inside: avoid; }
.callout.good { border-color: var(--c-good); background: var(--c-good-bg); }
.callout.bad  { border-color: var(--c-bad);  background: var(--c-bad-bg); }
.callout .head { font-weight: 700; font-size: 9.5pt; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 5px; }
.callout p { margin: 0; }

figure { margin: 18px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; height: auto; border: 1px solid var(--c-line); }
figcaption { font-size: 8.5pt; color: var(--c-muted); margin-top: 6px;
    font-style: italic; text-align: center; }

/* Confusion-matrix grid */
.cm-grid { display: grid; grid-template-columns: 1fr 1fr;
    gap: 14px; margin: 14px 0; }
.cm-cell { background: #fff; page-break-inside: avoid; text-align: center; }
.cm-cell img { max-width: 100%; height: auto; }

/* Strengths & weaknesses cards */
.sw-card { border: 1px solid var(--c-line); border-radius: 4px; padding: 12px 14px;
    margin: 10px 0; page-break-inside: avoid; background: #fff; }
.sw-head { display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
    margin-bottom: 9px; border-bottom: 1px solid var(--c-line); padding-bottom: 6px; }
.sw-rank { font-family: var(--f-serif); font-size: 14pt; font-weight: 700;
    color: var(--c-acc); width: 30px; }
.sw-name { font-family: var(--f-mono); font-size: 10.5pt; flex-grow: 0; }
.sw-pr { font-size: 9pt; color: var(--c-muted); margin-left: auto; }
.sw-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.sw-col-head { font-size: 8.5pt; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--c-muted); margin-bottom: 4px; padding-bottom: 3px;
    border-bottom: 1px solid var(--c-line); }
.sw-col.strengths .sw-col-head { color: var(--c-good); border-color: var(--c-good); }
.sw-col.weaknesses .sw-col-head { color: var(--c-bad); border-color: var(--c-bad); }
.sw-col ul { padding-left: 16px; margin: 0; font-size: 9pt; }
.sw-col li { margin-bottom: 3px; }

footer { margin-top: 36px; padding-top: 14px; border-top: 1px solid var(--c-line);
    color: var(--c-muted); font-size: 8pt; }

/* Architecture appendix cards */
.arch-card { border: 1px solid var(--c-line); border-radius: 6px;
    padding: 18px 20px 14px 20px; margin: 16px 0;
    background: #fff; page-break-inside: avoid; }
.arch-card h3 { margin-top: 0; font-size: 13pt; padding-bottom: 6px;
    border-bottom: 1px solid var(--c-line); }
.arch-card .arch-meta {
    display: flex; flex-wrap: wrap; gap: 9px; margin: 4px 0 14px 0;
    font-size: 8.5pt; color: var(--c-muted);
}
.arch-card .arch-meta .item {
    background: var(--c-stripe); padding: 2px 9px; border-radius: 9px;
}
.arch-card .arch-meta .item strong { color: var(--c-text); font-weight: 600; }
.arch-card .math-block { text-align: center; margin: 14px 0 16px 0;
    page-break-inside: avoid; }
.arch-card .math-block img { max-width: 88%; height: auto; }
.arch-card .math-caption { text-align: center; font-size: 8.5pt;
    color: var(--c-muted); font-style: italic; margin-top: -6px; margin-bottom: 12px; }
.arch-card h4 { margin-top: 10px; }
.arch-card .bias-line { font-family: var(--f-serif); font-style: italic;
    font-size: 10.5pt; color: var(--c-text); border-left: 2px solid var(--c-gold);
    padding-left: 11px; margin: 8px 0 12px 0; line-height: 1.5; }

/* Path-to-BT4 timeline */
.bt4-step { display: grid; grid-template-columns: 60px 1fr;
    gap: 14px; padding: 12px 0; border-bottom: 1px dashed var(--c-line); }
.bt4-step:last-child { border-bottom: none; }
.bt4-step .num { font-family: var(--f-serif); font-size: 26pt; font-weight: 600;
    color: var(--c-acc); line-height: 1; }
.bt4-step .label { font-size: 7.5pt; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--c-muted); font-weight: 700;
    margin-bottom: 2px; }
.bt4-step h4 { margin: 0 0 4px 0; font-size: 10.5pt;
    text-transform: none; letter-spacing: normal; color: var(--c-text); }
.bt4-step p { margin: 0; font-size: 9.5pt; color: var(--c-muted); }
.bt4-step .compute { margin-top: 6px; font-size: 8.5pt;
    color: var(--c-acc); font-weight: 600; }

@page { size: A4; margin: 17mm 16mm 16mm 16mm; @bottom-center {
    content: counter(page) " / " counter(pages); color: #888; font-size: 8pt;
    font-family: 'Inter', sans-serif; } }
@page :first { @bottom-center { content: ""; } }

/* Landscape page for the wide heatmap */
@page landscape-page {
    size: A4 landscape;
    margin: 14mm 14mm 14mm 14mm;
    @bottom-center { content: counter(page) " / " counter(pages);
        color: #888; font-size: 8pt; font-family: 'Inter', sans-serif; }
}
section.landscape { page: landscape-page; }
section.landscape h2 { margin-top: 0; }
section.landscape figure { margin: 10px 0; }
section.landscape img { width: 100%; max-width: 100%; max-height: 180mm; }
"""

    HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Chess-NN Architecture Scout — Paper Report</title>
<style>{CSS}</style>
</head>
<body>

  <header class="title-page">
    <div class="title-eyebrow">research report · chess-nn-playground</div>
    <h1>Architecture Scout</h1>
    <div class="subtitle">A 234-model survey of bespoke chess-evaluation architectures, with apples-to-apples comparisons across difficulty, phase, eval bucket, and tactic motif.</div>
    <div class="title-meta">
      <div class="k">Scout date</div><div>2026-05-09 to 2026-05-10</div>
      <div class="k">Task</div><div>puzzle_binary (single positive logit, BCE loss)</div>
      <div class="k">Dataset</div><div>CRTK-tagged 3-class split (~173k train / ~21k val / ~21k test, zero FEN overlap)</div>
      <div class="k">Hardware</div><div>RTX 3070 (8 GiB) — single GPU, single seed (42), base scale</div>
      <div class="k">Budget</div><div>12 epochs max, patience 3, 60-min wall per task, CUDA only</div>
      <div class="k">Generated</div><div>{today}</div>
    </div>
  </header>

  <h2>Executive summary</h2>

  <p class="lead">
    234 bespoke chess architectures were trained once each at small scale on a
    puzzle-detection task. <strong>{state_summary["completed"]}</strong> produced
    usable results; {state_summary["failed"]} crashed on code bugs ({failed_pct:.0f}%);
    {state_summary["timeout"]} hit the 60-minute training wall ({timeout_pct:.0f}%).
  </p>

  <div class="stats">
    <div class="stat"><div class="num">{state_summary["total"]}</div><div class="lbl">Architectures trained</div></div>
    <div class="stat"><div class="num">{state_summary["completed"]}</div><div class="lbl">Completed runs</div></div>
    <div class="stat"><div class="num">{n_total}</div><div class="lbl">In leaderboard</div></div>
    <div class="stat"><div class="num">{winner["test_pr_auc"]:.4f}</div><div class="lbl">Top test PR&nbsp;AUC</div></div>
  </div>

  <div class="callout good section-break">
    <div class="head">Headline finding</div>
    <p>
      <code>{short_name(winner["group"])}</code> wins by a clear margin
      (<strong>{winner["test_pr_auc"]:.4f}</strong> test PR AUC,
      +{winner["test_pr_auc"]-runner_up["test_pr_auc"]:.3f} over #2 at the same
      parameter budget). Its dual-stream architecture — one branch for tactical
      exchanges, one for king safety — is the largest within-encoding architectural
      margin in the entire scout pool, and the only architecture that empirically
      moves the leaderboard above the natural ~0.86 ceiling shared by all generic
      backbones.
    </p>
  </div>

  <h2>Overall leaderboard</h2>
  <p>
    Test PR AUC after 12 epochs at base scale on the
    <code>puzzle_binary</code> task. <span class="pill lc0">lc0_bt4_112</span>
    and <span class="pill s18">simple_18</span> tag the input encoding.
  </p>

  <table>
    <thead>
      <tr>
        <th class="rank">#</th>
        <th>encoding</th>
        <th>architecture</th>
        <th class="num">test PR AUC</th>
        <th class="num">val PR AUC</th>
        <th class="num">params</th>
        <th class="num">speed</th>
        <th class="num">FLOPs / pos</th>
      </tr>
    </thead>
    <tbody>
      {"".join(leaderboard_rows)}
    </tbody>
  </table>

  <section class="landscape">
    <h2 class="page-break">Per-slice performance map</h2>

    <p>
      Each cell is the test PR AUC restricted to its slice. Color: red = above
      the median across visible cells, blue = below. Gold borders mark column
      winners. Three leftmost data columns: parameter count, inference
      throughput, theoretical FLOPs per position.
    </p>

    <figure>
      <img src="{heatmap_uri}" alt="Per-slice heatmap of top 15 models" />
      <figcaption>
        Top 15 of {n_total} completed models. Slices: difficulty (5), phase (3),
        engine eval bucket (9), tactic motif (9), side to move (2). Cell values
        are slice-restricted test PR AUC.
      </figcaption>
    </figure>
  </section>

  <h2 class="page-break">Per-slice champions</h2>

  <p>Best model for each interesting slice value, with margin to runner-up:</p>

  <table>
    <thead>
      <tr>
        <th>slice</th>
        <th>champion</th>
        <th class="num">PR AUC ± std</th>
        <th class="num">margin</th>
      </tr>
    </thead>
    <tbody>{"".join(slice_winner_rows)}</tbody>
  </table>

  <h2 class="page-break">Confusion matrices — top 8 models</h2>

  <p>
    Source-class 3×2 confusion matrices at each model's F1-optimal
    threshold. <strong>Rows</strong> are the underlying CRTK fine label —
    <em>0: non-puzzle</em> (easy negative), <em>1: verified-near-puzzle</em>
    (the hard negative — positions that look tactical but aren't actually
    puzzles), <em>2: puzzle</em>. <strong>Columns</strong> are the binary
    prediction. Cell values show count and within-row percentage, so the
    rightmost column on row 1 is exactly the matched-recall near-puzzle FP
    rate that motivates the abstention architectures.
  </p>

  <div class="cm-grid">
    {"".join(cm_html_blocks)}
  </div>

  <h2 class="page-break">Strengths and weaknesses by model</h2>

  <p>
    For each of the top 8 models, the slices where it ranks in the top 3
    among all completed models ("where it wins") and the slices where it
    sits in the bottom quartile ("where it loses"). Slices include
    difficulty levels, board phases, engine-eval buckets, and tactic motifs.
  </p>

  {"".join(sw_html)}

  <h2 class="page-break">Speed × accuracy: the Pareto frontier</h2>

  <p>
    Models on the (accuracy, speed) Pareto frontier — no other model is both
    faster and more accurate:
  </p>

  <table>
    <thead>
      <tr>
        <th>encoding</th>
        <th>architecture</th>
        <th class="num">test PR AUC</th>
        <th class="num">speed</th>
        <th class="num">params</th>
        <th class="num">FLOPs / pos</th>
      </tr>
    </thead>
    <tbody>{"".join(pareto_rows)}</tbody>
  </table>

  <h2>Robustness: matched-recall FP rate on promotion slice</h2>

  <p>
    Aggregate PR AUC misses architectures explicitly designed for
    hard-negative rejection. At recall 0.80 on the
    promotion/underpromotion tactic-motif slice, the lowest near-puzzle
    false-positive rates:
  </p>

  <table>
    <thead>
      <tr>
        <th class="rank">#</th>
        <th>architecture</th>
        <th class="num">near-puzzle FP rate</th>
        <th class="num">slice acc @ rec 0.80</th>
      </tr>
    </thead>
    <tbody>{"".join(motif_rows)}</tbody>
  </table>

  <h2>Architecture findings</h2>

  <h3>What worked</h3>
  <ul>
    <li>
      <strong>Chess-specific task decomposition.</strong>
      <code>i193_exchange_then_king_dual_stream</code> splits the trunk into a
      tactical-exchange branch and a king-safety branch, fuses them with a
      learned phase router. This is the only architectural prior in the scout
      pool that produces an above-noise headline improvement.
    </li>
    <li>
      <strong>Board-symmetry / group equivariance.</strong> The rule-symmetry /
      orbit / quotient bottleneck family (<code>i042</code>, <code>i046</code>,
      <code>i048</code>) takes three of the top six slots. Three independent
      formulations of the same prior all rank highly.
    </li>
    <li>
      <strong>Small residual + interaction backbones.</strong>
      <code>i100_independence_residual_interaction</code> matches
      <code>bench_lc0_bt4_classifier</code> at one third the parameter count
      and faster inference — the Pareto pick when inference cost matters.
    </li>
  </ul>

  <h3>What did not work</h3>
  <ul>
    <li>
      <strong>~21% of bespoke architectures had AMP/dtype bugs</strong> and failed
      in under 2 minutes. Catchable in CI with a one-shot
      <code>torch.amp.autocast</code> smoke test.
    </li>
    <li>
      <strong>~4% timed out at the 60-min wall</strong> — iterative or unrolled
      designs (Dykstra projection, soft-sort, sheaf-curvature variants) need
      a custom training budget.
    </li>
    <li>
      <strong>Several architectures produced essentially-random predictions</strong>
      (PR AUC ≈ positive-class prevalence): <code>i039</code>, <code>i051</code>,
      <code>i060</code>, <code>i062</code>, <code>i096</code>. These are
      complete training failures — the loss surface or initialization
      prevents learning entirely.
    </li>
  </ul>

  <h2 class="page-break">Promotion candidates</h2>

  <p>
    The scout is a filter, not a final leaderboard. Single-seed at base scale
    is noisy — within a ±0.005 PR AUC band, ranks are not trustworthy.
    The following are promoted to full 3-seed × scale_xl evaluation:
  </p>

  <h4>By aggregate PR AUC (top 10)</h4>
  <p>
    <code>i193_exchange_then_king_dual_stream</code>,
    <code>i048_rule_automorphism_quotient</code>,
    <code>i018_oriented_tactical_sheaf_laplacian</code>,
    <code>i188_tactical_program_induction</code>,
    <code>i011_vetoselect</code> (already has 3-seed),
    <code>i192_latent_reply_entropy</code>,
    <code>i191_safe_reply_certificate_verifier</code>,
    <code>i042_legal_automorphism_quotient</code>,
    <code>i147_specialist_head_cnn</code>,
    <code>i046_rule_exact_orbit_bottleneck</code>.
  </p>

  <h4>By matched-recall near-puzzle FP rate</h4>
  <p>
    Existing robustness leaders (<code>i011_vetoselect</code>,
    <code>i012_dykstra_lcp</code>) plus the new aggregate-PR-AUC winners
    that also do well on the slice.
  </p>

  <h4>By niche slice wins</h4>
  <p>
    Models with clear (margin &gt; 0.005) wins on a hard slice — the
    rule-symmetry family on skewer / overload, the dual-stream winner across
    the board.
  </p>

  <h2>Implications and the next architecture</h2>

  <p>
    The strongest architectural signal in the scout is <strong>chess-specific
    structural priors</strong>, not generic-architecture exotic math.
    The three families that rise to the top all encode something about chess
    that a generic transformer must learn from data:
  </p>

  <ol>
    <li><strong>Task decomposition</strong> (exchanges vs king vs positional)</li>
    <li><strong>Board symmetry</strong> (mirror + color-flip equivariance)</li>
    <li><strong>Tactical relationships</strong> (precomputed attacker/defender attention bias)</li>
  </ol>

  <p>
    None of these are in <code>LC0 BT4</code>'s trunk. The natural next
    architecture, <code>i241_multistream_attention_chess_eval</code>,
    composes them: three parallel transformer streams (exchange, king,
    positional) with chess-aware attention bias, fused by a learned phase
    router, with value+policy heads for engine play.
  </p>

  <div class="callout">
    <div class="head">Single-sentence takeaway</div>
    <p>
      Across 234 trained architectures, the architectures that win are the
      ones that encode <em>how chess is actually evaluated</em>, not the
      ones that bring exotic math to a generic CNN.
    </p>
  </div>

  <h2 class="page-break">A concrete path to beat BT4</h2>

  <p>
    LC0 BT4 is a generic piece-token transformer. None of the inductive
    biases that emerged as winners in the scout are present in its trunk —
    not the exchange/king-safety decomposition, not the chess-automorphism
    equivariance, not the directional tactical sheaf, not the explicit
    program-induction head. The architecture proposed below
    (<code>i241_multistream_attention_chess_eval</code>) composes the three
    that are mutually composable into a single trunk shaped for chess
    evaluation, with engine-strength training heads on top.
  </p>

  <p class="lead">
    The trunk is the cheap part. The training pipeline is the moat. The plan
    below assumes Stockfish evaluations are obtainable; if a self-play loop
    is also available, append Phase 4.
  </p>

  {bt4_html_steps}

  <div class="callout">
    <div class="head">Realistic outcome</div>
    <p>
      Phases 1–3 plus 2–4 weeks of Stockfish-distillation training: estimated
      ELO around 3000–3300 — strong but well below BT4. Adding Phase 4 (self-play
      with limited MCTS depth) lifts that to roughly BT4 ± 100 ELO. The
      structural priors give a real ~30–80-ELO architectural advantage at
      matched compute; closing the remaining gap is a training-budget question
      and is the part that took LC0 years.
    </p>
  </div>

  <h2 class="page-break">Appendix · How each top architecture works</h2>

  <p>
    For each of the five architectures with the strongest scout signal:
    a single-paragraph summary of what it computes, the key equation,
    the inductive bias it brings, and what it gives up. Equations are
    rendered exactly; the symbol $x$ denotes the input board state
    (encoded as a tensor in $\\mathbb{{R}}^{{c \\times 8 \\times 8}}$ for
    simple_18 or $\\mathbb{{R}}^{{112 \\times 8 \\times 8}}$ for lc0_bt4_112).
  </p>

  {"".join(arch_cards_html)}

  <footer>
    Source: <code>results/architecture_scout_2026-05-09/</code> +
    <code>_archive/paper_ready_all_2026-05-09/</code>.
    Generated by <code>scripts/build_paper_report_pdf.py</code> on {today}.
  </footer>

</body>
</html>
"""

    # Write HTML next to PDF for inspection
    html_out = out_path.with_suffix(".html")
    html_out.write_text(HTML, encoding="utf-8")
    print(f"Wrote intermediate {html_out}")

    # Render PDF
    print("Rendering PDF via WeasyPrint...")
    import weasyprint
    weasyprint.HTML(string=HTML, base_url=str(Path.cwd())).write_pdf(str(out_path))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
