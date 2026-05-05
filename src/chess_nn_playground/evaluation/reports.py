from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from chess_nn_playground.evaluation.tables import dict_to_markdown_table


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _prediction_sections(run_dir: Path) -> list[str]:
    sections: list[str] = []
    pred_path = run_dir / "predictions_val.parquet"
    if not pred_path.exists():
        return sections
    try:
        df = pd.read_parquet(pred_path)
    except Exception as exc:
        return [f"Prediction analysis unavailable: `{type(exc).__name__}: {exc}`"]
    if df.empty:
        return ["No validation predictions were produced."]
    if "confidence" not in df.columns:
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        if prob_cols:
            df["confidence"] = df[prob_cols].max(axis=1)
    if "correct" not in df.columns and {"true_label", "predicted_label"}.issubset(df.columns):
        df["correct"] = df["true_label"] == df["predicted_label"]
    correct_mask = df["correct"].astype(bool) if "correct" in df.columns else pd.Series([False] * len(df))
    table_specs = [
        ("Highest-confidence correct predictions", df[correct_mask].sort_values("confidence", ascending=False).head(5)),
        ("Highest-confidence wrong predictions", df[~correct_mask].sort_values("confidence", ascending=False).head(5)),
        ("Most uncertain predictions", df.sort_values("confidence", ascending=True).head(5)),
    ]
    for title, frame in table_specs:
        sections.extend([f"### {title}", ""])
        cols = [c for c in ["sample_id", "true_label", "predicted_label", "confidence", "label_status", "fen"] if c in frame.columns]
        if cols and not frame.empty:
            try:
                sections.append(frame[cols].to_markdown(index=False))
            except Exception:
                sections.append(frame[cols].to_string(index=False))
        else:
            sections.append("None.")
        sections.append("")
    return sections


def _slice_sections(run_dir: Path) -> list[str]:
    sections: list[str] = []
    for split in ["train", "val", "test"]:
        metrics_path = run_dir / f"slice_metrics_{split}.json"
        report_path = run_dir / f"slice_report_{split}.md"
        tagged_predictions = run_dir / f"predictions_{split}_crtk_tags.parquet"
        if not metrics_path.exists() and not report_path.exists():
            continue
        sections.extend([f"### {split.title()} Slice Analysis", ""])
        if report_path.exists():
            sections.append(f"- Full report: `{report_path.name}`")
        if tagged_predictions.exists():
            sections.append(f"- Tagged predictions: `{tagged_predictions.name}`")
        metrics = _read_json(metrics_path)
        summary = metrics.get("summary", {})
        if summary:
            sections.extend(["", "Summary:", "", dict_to_markdown_table(summary), ""])
        worst = metrics.get("worst_slices", [])[:8]
        if worst:
            rows = [
                {
                    "column": row.get("column"),
                    "slice": row.get("slice"),
                    "rows": row.get("rows"),
                    "wrong": row.get("wrong"),
                    "accuracy": row.get("accuracy"),
                    "fpr": row.get("false_positive_rate"),
                    "fnr": row.get("false_negative_rate"),
                    "fine_1_acc": row.get("fine_1_accuracy"),
                    "fine_2_acc": row.get("fine_2_accuracy"),
                }
                for row in worst
            ]
            try:
                sections.extend(["Worst slices:", "", pd.DataFrame(rows).to_markdown(index=False), ""])
            except Exception:
                sections.extend(["Worst slices:", "", pd.DataFrame(rows).to_string(index=False), ""])
    if not sections:
        return ["No CRTK slice analysis artifacts were produced.", ""]
    return sections


def _plot_sections(run_dir: Path) -> list[str]:
    plot_files = [
        ("Training dashboard", "training_dashboard.png"),
        ("Loss curves", "loss_curves.png"),
        ("Accuracy curves", "accuracy_curves.png"),
        ("Macro F1 curves", "macro_f1_curves.png"),
        ("Weighted F1 curves", "weighted_f1_curves.png"),
        ("Validation source-class to binary-output matrix", "fine_to_binary_confusion_matrix_val.png"),
        ("Test source-class to binary-output matrix", "fine_to_binary_confusion_matrix_test.png"),
        ("Validation confusion matrix", "confusion_matrix_val.png"),
        ("Test confusion matrix", "confusion_matrix_test.png"),
        ("Class distribution", "class_distribution.png"),
        ("Calibration", "calibration_plot.png"),
    ]
    lines = []
    for title, filename in plot_files:
        if (run_dir / filename).exists():
            lines.extend([f"### {title}", "", f"![{title}]({filename})", ""])
    if not lines:
        return ["No plot artifacts were produced.", ""]
    return lines


def _html_plot_gallery(run_dir: Path) -> str:
    plot_files = [
        ("Training dashboard", "training_dashboard.png"),
        ("Validation source-class to binary-output matrix", "fine_to_binary_confusion_matrix_val.png"),
        ("Test source-class to binary-output matrix", "fine_to_binary_confusion_matrix_test.png"),
        ("Validation confusion matrix", "confusion_matrix_val.png"),
        ("Test confusion matrix", "confusion_matrix_test.png"),
        ("Loss curves", "loss_curves.png"),
        ("Accuracy curves", "accuracy_curves.png"),
        ("Macro F1 curves", "macro_f1_curves.png"),
        ("Weighted F1 curves", "weighted_f1_curves.png"),
        ("Class distribution", "class_distribution.png"),
        ("Calibration", "calibration_plot.png"),
    ]
    cards = []
    for title, filename in plot_files:
        if (run_dir / filename).exists():
            cards.append(
                "<figure>"
                f"<figcaption>{html.escape(title)}</figcaption>"
                f"<img src='{html.escape(filename)}' alt='{html.escape(title)}'>"
                "</figure>"
            )
    if not cards:
        return "<p>No plot artifacts were produced.</p>"
    return "\n".join(cards)


def build_run_report(run_dir: str | Path) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    metadata = _read_json(run_dir / "run_metadata.json")
    metrics_final = _read_json(run_dir / "metrics_final.json")
    config_path = run_dir / "config_resolved.yaml"
    complexity = metadata.get("complexity", {}) if isinstance(metadata.get("complexity"), dict) else {}
    speed = metadata.get("speed", {}) if isinstance(metadata.get("speed"), dict) else metrics_final.get("speed", {})

    lines = [
        "# Run Summary",
        "",
        "## Run overview",
        "",
        f"- Run name: `{metadata.get('run_name')}`",
        f"- Timestamp: `{metadata.get('timestamp')}`",
        f"- Model: `{metadata.get('model_name')}`",
        f"- Training mode: `{metadata.get('mode')}`",
        f"- Device: `{metadata.get('device')}`",
        f"- Parameters: `{metadata.get('num_params')}`",
        f"- Dataset path: `{metadata.get('dataset_path')}`",
        "",
        "## Label explanation",
        "",
        "This run uses the configured label mode. In `puzzle_binary`, label `0` means non-puzzle or near-puzzle and label `1` means verified puzzle; source fine labels are still preserved for 3x2 diagnostics. In `coarse_binary`, label `0` means known non-puzzle and label `1` means puzzle-like. In `fine_3class`, label `0` means known non-puzzle, label `1` means verified near-puzzle, and label `2` means verified puzzle.",
        "",
        "## Dataset summary",
        "",
        "```json",
        json.dumps(metadata.get("class_counts", {}), indent=2),
        "```",
        "",
        "Source fine-label counts:",
        "",
        "```json",
        json.dumps(metadata.get("source_class_counts", {}), indent=2),
        "```",
        "",
        "## Model summary",
        "",
        f"- Model class: `{metadata.get('model_name')}`",
        f"- Parameter count: `{metadata.get('num_params')}`",
        f"- Estimated inference FLOPs/position: `{complexity.get('estimated_flops_per_position')}`",
        f"- Estimated inference MFLOPs/position: `{complexity.get('estimated_mflops_per_position')}`",
        f"- Complexity method: `{complexity.get('method')}`",
        "- Input feature planes: 12 piece planes, side-to-move, 4 castling-right planes, en-passant plane",
        "",
        "## Speed summary",
        "",
        dict_to_markdown_table(speed if isinstance(speed, dict) else {}),
        "",
        "## Validation/test results",
        "",
        dict_to_markdown_table(metrics_final),
        "",
        "## Plots",
        "",
    ]
    lines.extend(_plot_sections(run_dir))
    lines.extend(
        [
        "## Prediction analysis",
        "",
        ]
    )
    lines.extend(_prediction_sections(run_dir))
    lines.extend(
        [
            "## Benchmark slice analysis",
            "",
        ]
    )
    lines.extend(_slice_sections(run_dir))
    produced_files = [
        f"- Config: `{config_path}`",
        f"- Best checkpoint: `{metadata.get('checkpoint_best')}`",
        f"- Last checkpoint: `{metadata.get('checkpoint_last')}`",
        f"- Metric history: `{run_dir / 'metrics_history.json'}`",
        f"- Split metrics manifest: `{run_dir / 'metrics_by_split.json'}`",
    ]
    for split in ["train", "val", "test"]:
        metrics_path = run_dir / f"metrics_{split}_final.json"
        predictions_path = run_dir / f"predictions_{split}.parquet"
        if metrics_path.exists():
            produced_files.append(f"- {split.title()} metrics: `{metrics_path}`")
        if predictions_path.exists():
            produced_files.append(f"- {split.title()} predictions: `{predictions_path}`")
    lines.extend(["## Files produced", "", *produced_files, "", "## Warnings/blockers", ""])
    reasons = metrics_final.get("metric_reasons", {})
    if reasons:
        for metric, reason in reasons.items():
            lines.append(f"- `{metric}` unavailable: {reason}")
    else:
        lines.append("- No metric blockers recorded.")
    lines.append("")

    markdown_path = run_dir / "run_summary.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    html_path = run_dir / "report.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Run Report</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:1180px;margin:32px auto;line-height:1.45;color:#1f2328}"
        "pre{background:#f6f8fa;border:1px solid #d0d7de;border-radius:8px;padding:16px;overflow:auto}"
        "code{background:#f6f8fa;padding:2px 4px;border-radius:4px}"
        ".gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px;margin:24px 0}"
        "figure{margin:0;border:1px solid #d0d7de;border-radius:8px;padding:12px;background:#fff}"
        "figcaption{font-weight:700;margin-bottom:8px}"
        "img{max-width:100%;height:auto;display:block}"
        "</style>"
        "</head><body>"
        "<h1>Run Report</h1>"
        "<h2>Plots</h2><section class='gallery'>"
        + _html_plot_gallery(run_dir)
        + "</section><h2>Markdown Summary</h2><pre>"
        + html.escape(markdown_path.read_text(encoding="utf-8"))
        + "</pre></body></html>",
        encoding="utf-8",
    )

    reports_dir = Path("reports") / "latest"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "latest_report.md").write_text(markdown_path.read_text(encoding="utf-8"), encoding="utf-8")
    (reports_dir / "latest_report.html").write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    return markdown_path, html_path
