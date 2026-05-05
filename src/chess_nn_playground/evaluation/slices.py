from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


TAG_COLUMNS = [
    "crtk_difficulty",
    "crtk_phase",
    "crtk_eval_bucket",
    "crtk_eval_cp",
    "crtk_wdl",
    "crtk_to_move",
    "crtk_source",
    "crtk_tactic_motifs",
    "crtk_tactic_motif_count",
    "crtk_tag_families",
    "crtk_tag_family_count",
    "crtk_tag_count",
]
REQUIRED_SLICE_COLUMNS = {"crtk_difficulty", "crtk_phase", "crtk_tag_count"}
SIMPLE_SLICES = [
    ("crtk_difficulty", "Difficulty Performance"),
    ("crtk_phase", "Phase Performance"),
    ("crtk_eval_bucket", "Eval-Bucket Performance"),
]
PIPE_SLICES = [
    ("crtk_tactic_motifs", "Tactical Motif Performance"),
    ("crtk_tag_families", "Tag-Family Performance"),
]


def has_slice_metadata(df: pd.DataFrame) -> bool:
    return REQUIRED_SLICE_COLUMNS.issubset(df.columns) and df["crtk_tag_count"].notna().any()


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "No rows."
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _json_counts(series: pd.Series) -> dict[str, int]:
    counts = series.fillna("(missing)").value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _round(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return round(float(value), 4)


def _safe_accuracy(group: pd.DataFrame) -> float | None:
    if group.empty or "correct" not in group.columns:
        return None
    return _round(group["correct"].astype(bool).mean())


def _binary_rates(group: pd.DataFrame) -> dict[str, float | None]:
    if not {"true_label", "predicted_label"}.issubset(group.columns):
        return {}
    true = pd.to_numeric(group["true_label"], errors="coerce")
    pred = pd.to_numeric(group["predicted_label"], errors="coerce")
    if not set(true.dropna().astype(int).unique()).issubset({0, 1}):
        return {}
    if not set(pred.dropna().astype(int).unique()).issubset({0, 1}):
        return {}
    tp = int(((true == 1) & (pred == 1)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    return {
        "false_positive_rate": _round(fp / (fp + tn)) if fp + tn else None,
        "false_negative_rate": _round(fn / (fn + tp)) if fn + tp else None,
        "positive_precision": _round(tp / (tp + fp)) if tp + fp else None,
        "positive_recall": _round(tp / (tp + fn)) if tp + fn else None,
    }


def _fine_label_metrics(group: pd.DataFrame) -> dict[str, Any]:
    if "true_fine_label" not in group.columns:
        return {}
    fine = pd.to_numeric(group["true_fine_label"], errors="coerce")
    out: dict[str, Any] = {
        "fine_label_counts": {str(key): int(value) for key, value in fine.value_counts(dropna=True).sort_index().items()},
    }
    if "correct" in group.columns:
        for label in [0, 1, 2]:
            mask = fine == label
            if mask.any():
                out[f"fine_{label}_accuracy"] = _round(group.loc[mask, "correct"].astype(bool).mean())
                out[f"fine_{label}_wrong"] = int((~group.loc[mask, "correct"].astype(bool)).sum())
            else:
                out[f"fine_{label}_accuracy"] = None
                out[f"fine_{label}_wrong"] = 0
    return out


def _metric_row(slice_value: str, group: pd.DataFrame) -> dict[str, Any]:
    correct = group["correct"].astype(bool) if "correct" in group.columns else pd.Series([False] * len(group))
    row: dict[str, Any] = {
        "slice": str(slice_value),
        "rows": int(len(group)),
        "wrong": int((~correct).sum()),
        "accuracy": _safe_accuracy(group),
        "mean_confidence": _round(group["confidence"].mean()) if "confidence" in group.columns else None,
    }
    row.update(_binary_rates(group))
    row.update(_fine_label_metrics(group))
    return row


def _simple_slice_rows(df: pd.DataFrame, column: str, min_count: int = 1) -> list[dict[str, Any]]:
    if column not in df.columns:
        return []
    rows = []
    values = df[column].fillna("(missing)").replace("", "(missing)")
    for value, group in df.assign(_slice_value=values).groupby("_slice_value", dropna=False):
        if len(group) >= min_count:
            rows.append(_metric_row(str(value), group))
    return sorted(rows, key=lambda row: (row["accuracy"] if row["accuracy"] is not None else 2.0, -row["rows"]))


def _pipe_slice_rows(df: pd.DataFrame, column: str, min_count: int = 1) -> list[dict[str, Any]]:
    if column not in df.columns:
        return []
    expanded_rows = []
    for _, row in df.iterrows():
        raw_value = row.get(column)
        try:
            missing = pd.isna(raw_value)
        except TypeError:
            missing = False
        text = "" if missing or raw_value is None else str(raw_value)
        values = [value for value in text.split("|") if value] or ["(none)"]
        for value in values:
            expanded = row.to_dict()
            expanded["_slice_value"] = value
            expanded_rows.append(expanded)
    expanded = pd.DataFrame(expanded_rows)
    rows = []
    for value, group in expanded.groupby("_slice_value", dropna=False):
        if len(group) >= min_count:
            rows.append(_metric_row(str(value), group))
    return sorted(rows, key=lambda row: (row["accuracy"] if row["accuracy"] is not None else 2.0, -row["rows"]))


def _table_rows(rows: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    wanted = rows[:limit] if limit else rows
    columns = [
        "slice",
        "rows",
        "wrong",
        "accuracy",
        "mean_confidence",
        "false_positive_rate",
        "false_negative_rate",
        "positive_recall",
        "fine_0_accuracy",
        "fine_1_accuracy",
        "fine_2_accuracy",
    ]
    return [{key: row.get(key) for key in columns} for row in wanted]


def _prediction_mix(df: pd.DataFrame) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    if not {"true_label", "predicted_label"}.issubset(df.columns):
        return []
    for true_label, predicted_label in zip(df["true_label"], df["predicted_label"]):
        counter[f"{true_label}->{predicted_label}"] += 1
    return [{"true_to_pred": key, "count": int(count)} for key, count in counter.most_common()]


def _available_split_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def load_predictions_with_slice_metadata(pred_path: Path, split_path: Path | None = None) -> pd.DataFrame:
    pred = pd.read_parquet(pred_path)
    if has_slice_metadata(pred):
        return pred
    if split_path is None or not split_path.exists():
        return pred

    split_columns = _available_split_columns(split_path)
    if "sample_id" in pred.columns and "sample_id" in split_columns:
        columns = ["sample_id"] + [column for column in TAG_COLUMNS if column in split_columns]
        tagged = pd.read_parquet(split_path, columns=columns)
        if len(columns) > 1:
            return pred.merge(tagged, on="sample_id", how="left", validate="one_to_one")
    if "fen" in pred.columns and "normalized_fen" in split_columns:
        columns = ["normalized_fen"] + [column for column in TAG_COLUMNS if column in split_columns]
        tagged = pd.read_parquet(split_path, columns=columns).rename(columns={"normalized_fen": "fen"})
        if len(columns) > 1:
            return pred.merge(tagged, on="fen", how="left", validate="one_to_one")
    return pred


def build_slice_metrics(df: pd.DataFrame, min_count: int = 100, limit: int = 20) -> dict[str, Any]:
    correct = df["correct"].astype(bool) if "correct" in df.columns else pd.Series([False] * len(df))
    tagged = df["crtk_tag_count"].notna() if "crtk_tag_count" in df.columns else pd.Series([False] * len(df))
    metrics: dict[str, Any] = {
        "summary": {
            "rows": int(len(df)),
            "tagged_rows": int(tagged.sum()),
            "wrong": int((~correct).sum()),
            "accuracy": _safe_accuracy(df),
            "true_label_counts": _json_counts(df["true_label"]) if "true_label" in df.columns else {},
            "predicted_label_counts": _json_counts(df["predicted_label"]) if "predicted_label" in df.columns else {},
        },
        "error_types": _prediction_mix(df),
        "simple_slices": {},
        "pipe_slices": {},
        "worst_slices": [],
        "best_slices": [],
    }
    for column, _title in SIMPLE_SLICES:
        rows = _simple_slice_rows(df, column, min_count=min_count)
        metrics["simple_slices"][column] = rows
    for column, _title in PIPE_SLICES:
        rows = _pipe_slice_rows(df, column, min_count=min_count)
        metrics["pipe_slices"][column] = rows

    candidates: list[dict[str, Any]] = []
    for family, sections in [("simple", metrics["simple_slices"]), ("pipe", metrics["pipe_slices"])]:
        for column, rows in sections.items():
            for row in rows:
                candidate = {"family": family, "column": column, **row}
                candidates.append(candidate)
    candidates = [row for row in candidates if row.get("accuracy") is not None]
    metrics["worst_slices"] = sorted(candidates, key=lambda row: (row["accuracy"], -row["rows"]))[:limit]
    metrics["best_slices"] = sorted(candidates, key=lambda row: (row["accuracy"], row["rows"]), reverse=True)[:limit]
    return metrics


def _learning_summary(metrics: dict[str, Any]) -> list[str]:
    worst = metrics.get("worst_slices", [])
    best = metrics.get("best_slices", [])
    lines = []
    if worst:
        row = worst[0]
        lines.append(
            f"- Weakest slice: `{row['column']}={row['slice']}` with accuracy `{row['accuracy']}` over `{row['rows']}` rows."
        )
    if best:
        row = best[0]
        lines.append(
            f"- Strongest slice: `{row['column']}={row['slice']}` with accuracy `{row['accuracy']}` over `{row['rows']}` rows."
        )
    for column in ["crtk_difficulty", "crtk_phase", "crtk_tactic_motifs"]:
        source = metrics["simple_slices"].get(column) or metrics["pipe_slices"].get(column) or []
        if source:
            row = source[0]
            lines.append(f"- Needs attention on `{column}={row['slice']}`: `{row['wrong']}` wrong of `{row['rows']}`.")
    return lines or ["- No slice-level learning summary is available."]


def write_slice_report(df: pd.DataFrame, split: str, output_path: Path, min_count: int = 100, limit: int = 20) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = build_slice_metrics(df, min_count=min_count, limit=limit)
    lines = [
        f"# Prediction Slice Report: {split}",
        "",
        "CRTK tags are benchmark metadata only. They are useful for diagnosing errors by difficulty, phase, motif, and tag family.",
        "",
        "## Summary",
        "",
        _markdown_table([{"field": key, "value": value} for key, value in metrics["summary"].items()], ["field", "value"]),
        "",
        "## What This Model Appears To Learn Or Miss",
        "",
        *_learning_summary(metrics),
        "",
        "## Error Types",
        "",
        _markdown_table(metrics["error_types"], ["true_to_pred", "count"]),
    ]
    table_columns = [
        "slice",
        "rows",
        "wrong",
        "accuracy",
        "mean_confidence",
        "false_positive_rate",
        "false_negative_rate",
        "positive_recall",
        "fine_0_accuracy",
        "fine_1_accuracy",
        "fine_2_accuracy",
    ]
    for column, title in SIMPLE_SLICES:
        rows = metrics["simple_slices"].get(column, [])
        lines.extend(["", f"## {title}", "", _markdown_table(_table_rows(rows), table_columns)])
    for column, title in PIPE_SLICES:
        rows = metrics["pipe_slices"].get(column, [])
        lines.extend(
            [
                "",
                f"## Worst {title}",
                "",
                _markdown_table(_table_rows(rows, limit=limit), table_columns),
                "",
                f"## Best {title}",
                "",
                _markdown_table(_table_rows(list(reversed(rows)), limit=limit), table_columns),
            ]
        )

    if "correct" in df.columns:
        wrong = df[~df["correct"].astype(bool)].sort_values("confidence", ascending=False).head(limit)
    else:
        wrong = pd.DataFrame()
    wanted = [
        column
        for column in [
            "sample_id",
            "true_label",
            "true_fine_label",
            "predicted_label",
            "confidence",
            "crtk_difficulty",
            "crtk_phase",
            "crtk_eval_bucket",
            "crtk_tactic_motifs",
            "fen",
        ]
        if column in wrong.columns
    ]
    lines.extend(["", "## Highest Confidence Wrong Rows", "", _markdown_table(wrong[wanted].to_dict("records"), wanted)])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def write_slice_artifacts(
    run_dir: Path,
    split: str,
    pred_path: Path,
    split_path: Path | None = None,
    min_count: int = 100,
    limit: int = 20,
) -> dict[str, str] | None:
    if not pred_path.exists():
        return None
    joined = load_predictions_with_slice_metadata(pred_path, split_path)
    if not has_slice_metadata(joined):
        return None
    joined_path = run_dir / f"predictions_{split}_crtk_tags.parquet"
    report_path = run_dir / f"slice_report_{split}.md"
    metrics_path = run_dir / f"slice_metrics_{split}.json"
    joined.to_parquet(joined_path, index=False)
    metrics = write_slice_report(joined, split, report_path, min_count=min_count, limit=limit)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "predictions": str(joined_path),
        "report": str(report_path),
        "metrics": str(metrics_path),
    }
