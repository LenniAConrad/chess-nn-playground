"""Build extras (mean±std bars, scale curves, Pareto plot) for paper_grade_top3."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_state(state_path: Path) -> list[dict]:
    state = json.loads(state_path.read_text())
    rows = []
    for task in state.get("tasks", {}).values():
        if task.get("status") != "completed":
            continue
        run_dir = Path(task["run_dir"])
        metrics = run_dir / "metrics_final.json"
        if not metrics.exists():
            continue
        m = json.loads(metrics.read_text())
        rows.append(
            {
                "task_id": task["task_id"],
                "source": Path(task["source_config"]).parent.name
                if "ideas/registry" in task["source_config"]
                else Path(task["source_config"]).stem,
                "seed": task["seed"],
                "scale": task.get("scale_variant", "base"),
                "params": m.get("num_params"),
                "flops": m.get("estimated_flops_per_position"),
                "test_pr_auc": m.get("test_pr_auc"),
                "val_pr_auc": m.get("best_score"),
                "test_acc": m.get("test_accuracy"),
            }
        )
    return rows


def agg_mean_std(rows: list[dict], group_keys: tuple[str, ...], metric: str) -> dict:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r.get(metric) is None:
            continue
        key = tuple(r[k] for k in group_keys)
        grouped[key].append(float(r[metric]))
    out = {}
    for key, vals in grouped.items():
        out[key] = {
            "mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out


SCALE_ORDER = ["base", "scale_up", "scale_xl"]
SCALE_PRETTY = {"base": "1.0×", "scale_up": "1.5×", "scale_xl": "2.0×"}


def short_name(source: str) -> str:
    s = source.replace("_network", "").replace("bench_", "")
    if len(s) > 28:
        s = s[:25] + "..."
    return s


def plot_mean_by_model(rows: list[dict], output_path: Path) -> None:
    stats = agg_mean_std(rows, ("source",), "test_pr_auc")
    models = sorted(stats.keys(), key=lambda k: -stats[k]["mean"])
    means = [stats[m]["mean"] for m in models]
    stds = [stats[m]["std"] for m in models]
    ns = [stats[m]["n"] for m in models]
    labels = [f"{short_name(m[0])}\n(n={n})" for m, n in zip(models, ns)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color="#4C72B0", edgecolor="black", linewidth=0.6)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{mean:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("Test PR-AUC (mean ± std across seeds × scales)")
    ax.set_title("Paper-grade ranking — Test PR-AUC per model")
    ax.set_ylim(min(means) - 0.03, max(means) + 0.03)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scale_curves(rows: list[dict], output_path: Path) -> None:
    stats = agg_mean_std(rows, ("source", "scale"), "test_pr_auc")
    models = sorted({k[0] for k in stats}, key=lambda m: -max(stats.get((m, s), {"mean": -1})["mean"] for s in SCALE_ORDER))
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 4)))
    for color, model in zip(colors, models):
        xs, ys, errs = [], [], []
        for scale in SCALE_ORDER:
            s = stats.get((model, scale))
            if s is None:
                continue
            xs.append(SCALE_PRETTY[scale])
            ys.append(s["mean"])
            errs.append(s["std"])
        ax.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=5, label=short_name(model), color=color)
    ax.set_xlabel("Architecture scale multiplier")
    ax.set_ylabel("Test PR-AUC (mean ± std across 3 seeds)")
    ax.set_title("Scale-up behavior — does the model keep gaining with size?")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pareto(rows: list[dict], output_path: Path) -> None:
    stats = agg_mean_std(rows, ("source", "scale"), "test_pr_auc")
    param_stats = agg_mean_std(rows, ("source", "scale"), "params")
    models = sorted({k[0] for k in stats})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 4)))
    for color, model in zip(colors, models):
        xs, ys, errs = [], [], []
        for scale in SCALE_ORDER:
            s = stats.get((model, scale))
            p = param_stats.get((model, scale))
            if s is None or p is None or p["mean"] is None:
                continue
            xs.append(p["mean"])
            ys.append(s["mean"])
            errs.append(s["std"])
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2, capsize=5, label=short_name(model), color=color)
        for x, y, scale in zip(xs, ys, SCALE_ORDER[: len(xs)]):
            ax.annotate(SCALE_PRETTY[scale], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Test PR-AUC (mean ± std across 3 seeds)")
    ax.set_title("Pareto frontier — PR-AUC vs model size")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(linestyle="--", alpha=0.4, which="both")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_heatmap(rows: list[dict], output_path: Path) -> None:
    stats = agg_mean_std(rows, ("source", "scale"), "test_pr_auc")
    models = sorted({k[0] for k in stats}, key=lambda m: -max(stats.get((m, s), {"mean": -1})["mean"] for s in SCALE_ORDER))
    matrix = np.full((len(models), len(SCALE_ORDER)), np.nan)
    for i, model in enumerate(models):
        for j, scale in enumerate(SCALE_ORDER):
            s = stats.get((model, scale))
            if s is not None:
                matrix[i, j] = s["mean"]
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(models) * 0.55 + 1.5)))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(SCALE_ORDER)), [SCALE_PRETTY[s] for s in SCALE_ORDER])
    ax.set_yticks(range(len(models)), [short_name(m) for m in models])
    for i in range(len(models)):
        for j in range(len(SCALE_ORDER)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", color="white" if matrix[i, j] < (np.nanmin(matrix) + np.nanmax(matrix)) / 2 else "black", fontsize=9)
    ax.set_xlabel("Scale")
    ax.set_title("Test PR-AUC heatmap — model × scale")
    fig.colorbar(im, ax=ax, label="Test PR-AUC")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary_table(rows: list[dict], output_path: Path) -> None:
    stats = agg_mean_std(rows, ("source", "scale"), "test_pr_auc")
    param_stats = agg_mean_std(rows, ("source", "scale"), "params")
    flop_stats = agg_mean_std(rows, ("source", "scale"), "flops")
    lines = ["model,scale,n_seeds,test_pr_auc_mean,test_pr_auc_std,params,flops"]
    keys = sorted(stats.keys(), key=lambda k: (-stats[k]["mean"], k))
    for k in keys:
        s = stats[k]
        p = param_stats.get(k, {"mean": ""})["mean"]
        f = flop_stats.get(k, {"mean": ""})["mean"]
        lines.append(
            f"{k[0]},{k[1]},{s['n']},{s['mean']:.6f},{s['std']:.6f},{int(p) if p else ''},{int(f) if f else ''}"
        )
    output_path.write_text("\n".join(lines) + "\n")


def write_html_summary(extras_dir: Path, rows: list[dict]) -> None:
    n_runs = len(rows)
    models = sorted({r["source"] for r in rows})
    out = extras_dir / "extras_report.html"
    html = [
        "<html><head><title>Paper-Grade Top-3 + BT4 — Extras Report</title>",
        "<style>body{font-family:sans-serif;max-width:1000px;margin:2em auto;padding:0 1em}",
        "h1,h2{color:#222}img{max-width:100%;border:1px solid #ddd;margin:0.5em 0}",
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}</style></head><body>",
        "<h1>Paper-Grade Top-3 + BT4 — Extras Report</h1>",
        f"<p><b>Completed runs:</b> {n_runs} &nbsp; <b>Models:</b> {len(models)}</p>",
        "<h2>1. Headline ranking — Test PR-AUC mean ± std</h2>",
        '<img src="mean_test_pr_auc_by_model.png"/>',
        "<h2>2. Scale-up curves</h2>",
        '<img src="scale_curves.png"/>',
        "<h2>3. Pareto frontier — PR-AUC vs parameters</h2>",
        '<img src="pareto_pr_auc_vs_params.png"/>',
        "<h2>4. Heatmap — model × scale</h2>",
        '<img src="heatmap_model_x_scale.png"/>',
        "<h2>5. Per-model × scale summary table</h2>",
        '<p><a href="summary_table.csv">summary_table.csv</a></p>',
        "<h2>6. Default paper PDF</h2>",
        '<p><a href="../paper_report.pdf">paper_report.pdf</a> (per-run training curves, confusion matrices, slice reports)</p>',
        "</body></html>",
    ]
    out.write_text("\n".join(html))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-path", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_state(args.state_path)
    if not rows:
        print("No completed runs found in state; nothing to plot.")
        return

    plot_mean_by_model(rows, args.output_dir / "mean_test_pr_auc_by_model.png")
    plot_scale_curves(rows, args.output_dir / "scale_curves.png")
    plot_pareto(rows, args.output_dir / "pareto_pr_auc_vs_params.png")
    plot_heatmap(rows, args.output_dir / "heatmap_model_x_scale.png")
    write_summary_table(rows, args.output_dir / "summary_table.csv")
    write_html_summary(args.output_dir, rows)
    print(f"Wrote extras to {args.output_dir}")


if __name__ == "__main__":
    main()
