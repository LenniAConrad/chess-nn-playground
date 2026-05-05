from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chess_nn_playground.data.dataset import BINARY_MODES
from chess_nn_playground.evaluation.plots import GRID_COLOR, _plt


DEFAULT_METRICS = [
    "loss",
    "accuracy",
    "f1",
    "macro_f1",
    "pr_auc",
    "roc_auc",
    "calibration_error",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run_label(metadata: dict[str, Any], run_dir: Path) -> str:
    run_name = str(metadata.get("run_name") or run_dir.name)
    seed = metadata.get("seed")
    if seed is None:
        return run_name
    return f"{run_name} seed={seed}"


def _history_records_from_json(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    rows: list[dict[str, Any]] = []
    for split in ["train", "val"]:
        records = data.get(split, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if isinstance(record, dict):
                row = dict(record)
                row["split"] = split
                rows.append(row)
    return rows


def _history_records_from_csv(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, filename in [("train", "metrics_train.csv"), ("val", "metrics_val.csv")]:
        path = run_dir / filename
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for record in df.to_dict("records"):
            row = dict(record)
            row["split"] = split
            rows.append(row)
    return rows


def _score_from_metrics(metrics: dict[str, Any], mode: str | None) -> float:
    if mode in BINARY_MODES:
        keys = ["f1", "pr_auc", "accuracy"]
    else:
        keys = ["macro_f1", "weighted_f1", "accuracy"]
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                continue
    loss = metrics.get("loss")
    if loss is not None:
        try:
            return -float(loss)
        except Exception:
            pass
    return -math.inf


def load_training_histories(results_dir: str | Path = "results") -> pd.DataFrame:
    results_dir = Path(results_dir)
    rows: list[dict[str, Any]] = []
    run_dirs = sorted(path.parent for path in results_dir.rglob("metrics_final.json"))
    for run_dir in run_dirs:
        metadata = _read_json(run_dir / "run_metadata.json")
        if (run_dir / "metrics_history.json").exists():
            records = _history_records_from_json(run_dir / "metrics_history.json")
        else:
            records = _history_records_from_csv(run_dir)
        if not records:
            continue
        final_metrics = _read_json(run_dir / "metrics_final.json")
        score = _score_from_metrics(final_metrics, metadata.get("mode"))
        for record in records:
            row = dict(record)
            row.update(
                {
                    "run_dir": str(run_dir),
                    "run_name": metadata.get("run_name", run_dir.name),
                    "run_label": _run_label(metadata, run_dir),
                    "model_name": metadata.get("model_name"),
                    "mode": metadata.get("mode"),
                    "seed": metadata.get("seed"),
                    "best_epoch": final_metrics.get("best_epoch") or metadata.get("best_epoch"),
                    "run_score": score,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _available_metrics(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    available: list[str] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        if values.notna().any():
            available.append(metric)
    return available


def _selected_run_labels(df: pd.DataFrame, max_runs: int) -> list[str]:
    scores = (
        df[["run_label", "run_score"]]
        .drop_duplicates(subset=["run_label"])
        .sort_values("run_score", ascending=False, na_position="last")
    )
    if max_runs > 0:
        scores = scores.head(max_runs)
    return [str(value) for value in scores["run_label"].tolist()]


def _metric_title(metric: str) -> str:
    return metric.replace("_", " ").title()


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No runs found."
    try:
        return df.to_markdown(index=False)
    except Exception:
        columns = list(df.columns)
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for _, row in df.iterrows():
            values = [str(row.get(column, "")) for column in columns]
            values = [value.replace("|", "\\|").replace("\n", " ") for value in values]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)


def plot_global_training_curves(
    history: pd.DataFrame,
    output_dir: str | Path,
    *,
    max_runs: int = 24,
    metrics: list[str] | None = None,
) -> str | None:
    metrics = metrics or DEFAULT_METRICS
    available = _available_metrics(history, metrics)
    if history.empty or not available:
        return None
    selected_labels = _selected_run_labels(history, max_runs=max_runs)
    working = history[history["run_label"].isin(selected_labels)].copy()
    if working.empty:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    cols = 2
    rows = int(np.ceil(len(available) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, max(4.8, rows * 4.2)), squeeze=False)
    axes_arr = axes.reshape(-1)
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(selected_labels))))
    color_by_label = {label: colors[idx % len(colors)] for idx, label in enumerate(selected_labels)}

    for ax, metric in zip(axes_arr, available):
        for label in selected_labels:
            run_rows = working[working["run_label"] == label]
            for split, linestyle, alpha, linewidth in [("train", "-", 0.38, 1.5), ("val", "--", 0.9, 2.0)]:
                split_rows = run_rows[run_rows["split"] == split].sort_values("epoch")
                if split_rows.empty or metric not in split_rows:
                    continue
                values = pd.to_numeric(split_rows[metric], errors="coerce")
                if values.notna().sum() == 0:
                    continue
                ax.plot(
                    split_rows["epoch"],
                    values,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    color=color_by_label[label],
                    label=f"{label} {split}",
                )
        ax.set_title(_metric_title(metric), fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(_metric_title(metric))
        ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric not in {"loss", "calibration_error"}:
            ax.set_ylim(-0.02, 1.02)

    for ax in axes_arr[len(available) :]:
        ax.axis("off")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        max_legend_items = min(24, len(handles))
        fig.legend(
            handles[:max_legend_items],
            labels[:max_legend_items],
            loc="lower center",
            ncol=2,
            fontsize=7,
            frameon=False,
        )
    fig.suptitle("Training Curves Across Runs", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    output_path = output_dir / "all_training_curves.png"
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)


def plot_global_final_scores(history: pd.DataFrame, output_dir: str | Path, *, max_runs: int = 32) -> str | None:
    if history.empty:
        return None
    rows = (
        history[["run_label", "model_name", "mode", "run_score"]]
        .drop_duplicates(subset=["run_label"])
        .sort_values("run_score", ascending=False, na_position="last")
    )
    rows = rows[np.isfinite(pd.to_numeric(rows["run_score"], errors="coerce"))]
    if rows.empty:
        return None
    if max_runs > 0:
        rows = rows.head(max_runs)
    rows = rows.sort_values("run_score", ascending=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig_height = max(5.0, min(16.0, 0.38 * len(rows) + 1.8))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y = np.arange(len(rows))
    ax.barh(y, pd.to_numeric(rows["run_score"], errors="coerce"), color="#0072B2", alpha=0.86)
    ax.set_yticks(y, rows["run_label"].tolist(), fontsize=8)
    ax.set_xlabel("Best validation score")
    ax.set_title("Best Validation Score By Run", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path = output_dir / "training_final_scores.png"
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)


def write_global_training_dashboard(
    history: pd.DataFrame,
    output_dir: str | Path,
    plot_paths: list[str],
) -> tuple[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = (
        history[["run_name", "run_label", "model_name", "mode", "seed", "best_epoch", "run_score", "run_dir"]]
        .drop_duplicates(subset=["run_label"])
        .sort_values("run_score", ascending=False, na_position="last")
    )
    run_rows.to_csv(output_dir / "training_runs.csv", index=False)

    lines = [
        "# Global Training Dashboard",
        "",
        f"- Runs with training history: `{len(run_rows)}`",
        f"- Epoch rows loaded: `{len(history)}`",
        "",
        "## Plots",
        "",
    ]
    if plot_paths:
        for path in plot_paths:
            rel = Path(path).name
            lines.extend([f"![{rel}]({rel})", ""])
    else:
        lines.append("No plot artifacts were produced.")
        lines.append("")
    lines.extend(["## Runs", "", _markdown_table(run_rows), ""])
    markdown_path = output_dir / "training_dashboard.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    images = "\n".join(
        f"<figure><figcaption>{html.escape(Path(path).name)}</figcaption>"
        f"<img src='{html.escape(Path(path).name)}' alt='{html.escape(Path(path).name)}'></figure>"
        for path in plot_paths
    )
    html_path = output_dir / "training_dashboard.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Global Training Dashboard</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1240px;margin:32px auto;line-height:1.45;color:#1f2328}"
        "figure{margin:24px 0}img{max-width:100%;height:auto;border:1px solid #d0d7de;border-radius:6px}"
        "figcaption{font-weight:700;margin-bottom:8px}table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #d0d7de;padding:6px 8px;text-align:left}th{background:#f6f8fa}</style>"
        "</head><body>"
        "<h1>Global Training Dashboard</h1>"
        f"<p>Runs with training history: <code>{len(run_rows)}</code>. Epoch rows loaded: <code>{len(history)}</code>.</p>"
        f"{images}"
        "<h2>Runs</h2>"
        + (run_rows.to_html(index=False, escape=True) if not run_rows.empty else "<p>No runs found.</p>")
        + "</body></html>",
        encoding="utf-8",
    )
    return str(markdown_path), str(html_path)


def build_global_training_dashboard(
    results_dir: str | Path = "results",
    output_dir: str | Path = "reports/training",
    *,
    max_runs: int = 24,
) -> dict[str, Any]:
    history = load_training_histories(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = [
        path
        for path in [
            plot_global_training_curves(history, output_dir, max_runs=max_runs),
            plot_global_final_scores(history, output_dir, max_runs=max_runs),
        ]
        if path is not None
    ]
    markdown_path, html_path = write_global_training_dashboard(history, output_dir, plot_paths)
    return {
        "history_rows": int(len(history)),
        "run_count": int(history["run_label"].nunique()) if "run_label" in history else 0,
        "plots": plot_paths,
        "markdown": markdown_path,
        "html": html_path,
        "runs_csv": str(output_dir / "training_runs.csv"),
    }
