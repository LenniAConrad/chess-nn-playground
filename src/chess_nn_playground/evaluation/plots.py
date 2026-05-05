from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRAIN_COLOR = "#0072B2"
VAL_COLOR = "#D55E00"
BAR_COLOR = "#009E73"
GRID_COLOR = "#D0D7DE"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style_axis(ax: Any, title: str, ylabel: str, bounded: bool = False) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if bounded:
        ax.set_ylim(-0.02, 1.02)


def _plot_metric(ax: Any, train_df: pd.DataFrame, val_df: pd.DataFrame, metric_name: str) -> bool:
    plotted = False
    if metric_name in train_df and "epoch" in train_df:
        ax.plot(
            train_df["epoch"],
            train_df[metric_name],
            marker="o",
            linewidth=2.2,
            markersize=4,
            color=TRAIN_COLOR,
            label="train",
        )
        plotted = True
    if metric_name in val_df and "epoch" in val_df:
        ax.plot(
            val_df["epoch"],
            val_df[metric_name],
            marker="s",
            linewidth=2.2,
            markersize=4,
            color=VAL_COLOR,
            label="validation",
        )
        plotted = True
    if plotted:
        ax.legend(frameon=False, loc="best")
    return plotted


def plot_curves(metrics_train: pd.DataFrame, metrics_val: pd.DataFrame, output_dir: str | Path) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    metric_specs = [
        ("loss", "Loss", "Loss", False, "loss_curves.png"),
        ("accuracy", "Accuracy", "Accuracy", True, "accuracy_curves.png"),
        ("f1", "Binary F1", "F1", True, "learning_curves.png"),
        ("macro_f1", "Macro F1", "F1", True, "macro_f1_curves.png"),
        ("weighted_f1", "Weighted F1", "F1", True, "weighted_f1_curves.png"),
        ("class2_vs_class1_f1", "Puzzle vs Near-Puzzle F1", "F1", True, "class2_vs_class1_f1_curves.png"),
    ]
    plotted_specs = []
    for metric_name, title, ylabel, bounded, filename in metric_specs:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(8, 4.8))
        plotted = _plot_metric(ax, metrics_train, metrics_val, metric_name)
        if not plotted:
            plt.close(fig)
            continue
        _style_axis(ax, title, ylabel, bounded=bounded)
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
        plotted_specs.append((metric_name, title, ylabel, bounded))

    if plotted_specs:
        plt = _plt()
        cols = 2
        rows = int(np.ceil(len(plotted_specs) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, max(4, rows * 3.8)))
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, (metric_name, title, ylabel, bounded) in zip(axes_arr, plotted_specs):
            _plot_metric(ax, metrics_train, metrics_val, metric_name)
            _style_axis(ax, title, ylabel, bounded=bounded)
        for ax in axes_arr[len(plotted_specs) :]:
            ax.axis("off")
        path = output_dir / "training_dashboard.png"
        fig.suptitle("Training Progress", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_confusion_matrix(
    confusion_matrix: list[list[int]] | None,
    output_path: str | Path,
    class_names: list[str],
    title: str = "Confusion Matrix",
) -> str | None:
    if confusion_matrix is None:
        return None
    cm = np.asarray(confusion_matrix, dtype=np.int64)
    if cm.size == 0:
        return None
    row_totals = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
    plt = _plt()
    fig_width = max(7.0, 1.6 * len(class_names) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))
    im = ax.imshow(normalized, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percent of true class", rotation=270, labelpad=16)
    ax.set_xticks(range(len(class_names)), class_names, rotation=25, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted class", fontweight="bold")
    ax.set_ylabel("True class", fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = normalized[i, j] * 100.0
            text_color = "white" if normalized[i, j] >= 0.55 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold" if i == j else "normal",
            )
    totals_text = "True-class totals: " + ", ".join(
        f"{class_names[idx]}={int(total):,}" for idx, total in enumerate(row_totals[:, 0])
    )
    fig.text(0.5, 0.015, totals_text, ha="center", fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)


def plot_rectangular_confusion_matrix(
    matrix: list[list[int]] | np.ndarray | None,
    output_path: str | Path,
    row_names: list[str],
    column_names: list[str],
    title: str,
    x_label: str = "Predicted class",
    y_label: str = "True class",
) -> str | None:
    if matrix is None:
        return None
    cm = np.asarray(matrix, dtype=np.int64)
    if cm.size == 0:
        return None
    row_totals = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
    plt = _plt()
    fig_width = max(7.0, 1.8 * len(column_names) + 3.6)
    fig_height = max(5.4, 1.25 * len(row_names) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(normalized, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percent of true source class", rotation=270, labelpad=18)
    ax.set_xticks(range(len(column_names)), column_names, rotation=20, ha="right")
    ax.set_yticks(range(len(row_names)), row_names)
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(np.arange(-0.5, len(column_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_names), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = normalized[i, j] * 100.0
            text_color = "white" if normalized[i, j] >= 0.55 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )
    totals_text = "True source totals: " + ", ".join(
        f"{row_names[idx]}={int(total):,}" for idx, total in enumerate(row_totals[:, 0])
    )
    fig.text(0.5, 0.015, totals_text, ha="center", fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)


def plot_class_distribution(class_counts: dict[str, Any], output_path: str | Path) -> str | None:
    if not class_counts:
        return None
    labels = list(class_counts.keys())
    counts = [class_counts[key] for key in labels]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts, color=BAR_COLOR)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Training Class Distribution", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(count):,}", ha="center", va="bottom")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def plot_calibration(
    y_true: list[int],
    probs: list[list[float]],
    output_path: str | Path,
    positive_class: int | None = 1,
) -> str | None:
    if not y_true or len(set(y_true)) < 2:
        return None
    probs_np = np.asarray(probs)
    if positive_class is not None and probs_np.shape[1] <= positive_class:
        return None
    y_true_arr = np.asarray(y_true)
    if positive_class is None:
        predictions = probs_np.argmax(axis=1)
        y_true_np = predictions == y_true_arr
        scores = probs_np.max(axis=1)
        ylabel = "Observed accuracy"
        title = "Confidence Calibration"
    else:
        y_true_np = y_true_arr == positive_class
        scores = probs_np[:, positive_class]
        ylabel = f"Observed frequency for class {positive_class}"
        title = f"Class {positive_class} Calibration"
    bins = np.linspace(0, 1, 11)
    xs = []
    ys = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (scores > low) & (scores <= high)
        if np.any(mask):
            xs.append(float(np.mean(scores[mask])))
            ys.append(float(np.mean(y_true_np[mask])))
    if not xs:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="perfect calibration")
    ax.plot(xs, ys, marker="o", linewidth=2.2, color=TRAIN_COLOR, label="model")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="best")
    ax.grid(True, color=GRID_COLOR, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)
