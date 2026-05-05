#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.data.dataset import ChessPositionDataset
from chess_nn_playground.utils.logging import write_json, write_text


DEFAULT_SPLIT_DIR = Path("data/splits/crtk_sample_3class_unique_crtk_tags")
DEFAULT_REPORT_MD = Path("data/reports/benchmark_data_readiness.md")
DEFAULT_REPORT_JSON = Path("data/reports/benchmark_data_readiness.json")
EXPECTED_FINE_COUNTS = {
    "train": {0: 120000, 1: 120000, 2: 120000},
    "val": {0: 15000, 1: 15000, 2: 15000},
    "test": {0: 15000, 1: 15000, 2: 15000},
}
REQUIRED_COLUMNS = {
    "sample_id",
    "fen",
    "normalized_fen",
    "label_status",
    "coarse_label",
    "fine_label",
    "split",
    "split_group_id",
}
CRTK_COLUMNS = {
    "crtk_tags_json",
    "crtk_tag_count",
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
}


def _count_missing(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return len(df)
    values = df[column]
    if values.dtype == object:
        return int((values.isna() | values.fillna("").astype(str).eq("")).sum())
    return int(values.isna().sum())


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _sample_load_check(path: Path, mode: str) -> dict[str, Any]:
    dataset = ChessPositionDataset(path, mode=mode, cache_features=False)
    first = dataset[0]
    middle = dataset[len(dataset) // 2]
    last = dataset[len(dataset) - 1]
    shapes = [tuple(item["x"].shape) for item in [first, middle, last]]
    return {
        "mode": mode,
        "rows": len(dataset),
        "sample_feature_shapes": [list(shape) for shape in shapes],
        "sample_metadata_keys": sorted(first["metadata"].keys()),
    }


def _validate_fens(series: pd.Series) -> tuple[int, list[str]]:
    count = 0
    invalid: list[str] = []
    for fen in series.astype(str):
        try:
            chess.Board(fen)
        except ValueError:
            count += 1
            if len(invalid) < 10:
                invalid.append(fen)
    return count, invalid


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


def audit_split(split: str, path: Path, validate_fens: bool) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    issues: list[str] = []
    df = pd.read_parquet(path)
    missing_columns = sorted((REQUIRED_COLUMNS | CRTK_COLUMNS) - set(df.columns))
    if missing_columns:
        issues.append(f"{split}: missing columns {missing_columns}")

    fine_counts = Counter(int(value) for value in df["fine_label"].dropna().tolist()) if "fine_label" in df else Counter()
    expected_counts = EXPECTED_FINE_COUNTS.get(split)
    if expected_counts and dict(fine_counts) != expected_counts:
        issues.append(f"{split}: fine_label counts {dict(fine_counts)} do not match expected {expected_counts}")

    split_values = set(df["split"].dropna().astype(str).unique()) if "split" in df else set()
    if split_values != {split}:
        issues.append(f"{split}: split column values are {sorted(split_values)}, expected only {split!r}")

    missing_required = {column: _count_missing(df, column) for column in sorted(REQUIRED_COLUMNS) if column in df.columns}
    missing_crtk = {column: _count_missing(df, column) for column in sorted(CRTK_COLUMNS) if column in df.columns}
    for column, count in missing_required.items():
        if count:
            issues.append(f"{split}: {column} has {count} missing values")
    for column in ["crtk_tag_count", "crtk_difficulty", "crtk_phase", "crtk_eval_bucket", "crtk_eval_cp"]:
        if missing_crtk.get(column, 0):
            issues.append(f"{split}: {column} has {missing_crtk[column]} missing values")
    if "crtk_tag_count" in df and int((df["crtk_tag_count"].fillna(0) <= 0).sum()):
        issues.append(f"{split}: some rows have zero CRTK tags")

    invalid_fen_count = 0
    invalid_fen_examples: list[str] = []
    if validate_fens:
        invalid_fen_count, invalid_fen_examples = _validate_fens(df["normalized_fen"])
        if invalid_fen_count:
            issues.append(f"{split}: found invalid FEN examples")

    load_checks = {
        "coarse_binary": _sample_load_check(path, "coarse_binary"),
        "puzzle_binary": _sample_load_check(path, "puzzle_binary"),
        "fine_3class": _sample_load_check(path, "fine_3class"),
    }

    report = {
        "path": str(path),
        "rows": len(df),
        "columns": len(df.columns),
        "fine_label_counts": {str(key): int(value) for key, value in sorted(fine_counts.items())},
        "coarse_label_counts": _value_counts(df, "coarse_label"),
        "difficulty_counts": _value_counts(df, "crtk_difficulty"),
        "phase_counts": _value_counts(df, "crtk_phase"),
        "missing_required": missing_required,
        "missing_crtk": missing_crtk,
        "duplicate_sample_ids": int(df["sample_id"].duplicated().sum()) if "sample_id" in df else None,
        "duplicate_normalized_fens": int(df["normalized_fen"].duplicated().sum()) if "normalized_fen" in df else None,
        "invalid_fen_count": invalid_fen_count,
        "invalid_fen_examples": invalid_fen_examples,
        "load_checks": load_checks,
    }
    return report, df, issues


def _overlap_size(frames: dict[str, pd.DataFrame], column: str) -> dict[str, int]:
    sets = {split: set(df[column].dropna().astype(str)) for split, df in frames.items() if column in df}
    out: dict[str, int] = {}
    names = sorted(sets)
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            out[f"{left}__{right}"] = len(sets[left] & sets[right])
    return out


def _label_conflict_count(frames: dict[str, pd.DataFrame]) -> int:
    merged = pd.concat(
        [df[["normalized_fen", "fine_label"]] for df in frames.values()],
        ignore_index=True,
    )
    conflicts = merged.groupby("normalized_fen")["fine_label"].nunique()
    return int((conflicts > 1).sum())


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Benchmark Data Readiness",
        "",
        f"- Status: `{report['status']}`",
        f"- Split dir: `{report['split_dir']}`",
        "- CRTK metadata is benchmark metadata only and must not be used as model input.",
        "",
        "## Split Summary",
        "",
    ]
    rows = []
    for split, item in report["splits"].items():
        rows.append(
            {
                "split": split,
                "rows": item["rows"],
                "fine_counts": json.dumps(item["fine_label_counts"], sort_keys=True),
                "duplicate_fens": item["duplicate_normalized_fens"],
                "invalid_fens": item["invalid_fen_count"],
            }
        )
    lines.append(_markdown_table(rows, ["split", "rows", "fine_counts", "duplicate_fens", "invalid_fens"]))
    lines.extend(["", "## Cross Split Checks", ""])
    cross_rows = [{"check": key, "value": value} for key, value in report["cross_split"].items()]
    lines.append(_markdown_table(cross_rows, ["check", "value"]))
    lines.extend(["", "## Benchmark Metadata Coverage", ""])
    metadata_rows = []
    for split, item in report["splits"].items():
        metadata_rows.append(
            {
                "split": split,
                "difficulty": json.dumps(item["difficulty_counts"], sort_keys=True),
                "phase": json.dumps(item["phase_counts"], sort_keys=True),
            }
        )
    lines.append(_markdown_table(metadata_rows, ["split", "difficulty", "phase"]))
    if report["issues"]:
        lines.extend(["", "## Issues", ""])
        for issue in report["issues"]:
            lines.append(f"- {issue}")
    else:
        lines.extend(["", "## Issues", "", "No blocking issues found."])
    write_text("\n".join(lines) + "\n", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit benchmark splits for training and benchmark readiness.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--skip-fen-validation", action="store_true")
    args = parser.parse_args()

    issues: list[str] = []
    split_reports: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        path = args.split_dir / f"split_{split}.parquet"
        if not path.exists():
            issues.append(f"{split}: missing split file {path}")
            continue
        split_report, df, split_issues = audit_split(split, path, validate_fens=not args.skip_fen_validation)
        split_reports[split] = split_report
        frames[split] = df
        issues.extend(split_issues)

    cross_split = {
        "sample_id_overlap": _overlap_size(frames, "sample_id"),
        "split_group_id_overlap": _overlap_size(frames, "split_group_id"),
        "normalized_fen_overlap": _overlap_size(frames, "normalized_fen"),
        "label_conflicting_fens": _label_conflict_count(frames) if len(frames) == 3 else None,
    }
    for name, values in cross_split.items():
        if isinstance(values, dict):
            for pair, count in values.items():
                if count:
                    issues.append(f"{name} {pair}: {count}")
        elif values:
            issues.append(f"{name}: {values}")

    report = {
        "status": "ready" if not issues else "blocked",
        "split_dir": str(args.split_dir),
        "splits": split_reports,
        "cross_split": cross_split,
        "issues": issues,
    }
    write_json(report, args.report_json)
    write_markdown(report, args.report_md)
    print(f"Saved {args.report_json}")
    print(f"Saved {args.report_md}")
    if issues:
        for issue in issues:
            print(f"ISSUE: {issue}")
        raise SystemExit(1)
    print("Benchmark data status: ready")


if __name__ == "__main__":
    main()
