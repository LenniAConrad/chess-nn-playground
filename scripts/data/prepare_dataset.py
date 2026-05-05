#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.data.fen_utils import normalize_fen, summarize_fen_dict, validate_fen
from chess_nn_playground.data.json_loader import choose_fen, find_json_files, iter_json_records, truncate_value
from chess_nn_playground.data.schema import (
    CANONICAL_COLUMNS,
    DERIVED_COLUMNS,
    canonical_from_raw,
)
from chess_nn_playground.data.stockfish import parse_stockfish_metadata
from chess_nn_playground.data.validation import validate_canonical_dataframe
from chess_nn_playground.utils.config import load_yaml
from chess_nn_playground.utils.logging import write_json, write_text


def _load_config(path: str | None) -> dict[str, Any]:
    if path:
        return load_yaml(path)
    return {}


def _distribution(series: pd.Series, bins: int | None = None) -> dict[str, Any]:
    if series.empty:
        return {}
    if bins and pd.api.types.is_numeric_dtype(series):
        counts = pd.cut(series, bins=bins, duplicates="drop").value_counts().sort_index()
        return {str(k): int(v) for k, v in counts.items()}
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare canonical chess position dataset.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--rejected-output", default=None)
    parser.add_argument("--default-label-status", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-records-per-file", type=int, default=None)
    parser.add_argument("--raw-metadata-max-chars", type=int, default=4000)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    input_paths = args.input or cfg.get("input_paths") or ["data/exported", "data/raw"]
    output_path = Path(args.output or cfg.get("output_path", "data/processed/positions.parquet"))
    rejected_path = Path(args.rejected_output or cfg.get("rejected_output_path", "data/processed/rejected_positions.parquet"))
    default_label_status = args.default_label_status or cfg.get("default_label_status")

    files: list[Path] = []
    for input_path in input_paths:
        path = Path(input_path)
        if path.exists():
            files.extend(find_json_files(path))
    files = sorted(set(files))
    if args.max_files is not None:
        files = files[: args.max_files]
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    total_raw = 0
    missing_fen_count = 0

    stop_all = False
    for path in files:
        per_file_seen = 0
        for json_record in iter_json_records(path):
            if args.max_records is not None and total_raw >= args.max_records:
                stop_all = True
                break
            if args.max_records_per_file is not None and per_file_seen >= args.max_records_per_file:
                break
            per_file_seen += 1
            total_raw += 1
            fen_key, fen = choose_fen(json_record.record)
            if not fen:
                missing_fen_count += 1
                rejected.append(
                    {
                        "source_path": str(path),
                        "source_record_index": json_record.source_record_index,
                        "reason": "missing_fen",
                        "raw_record": json.dumps(truncate_value(json_record.record), default=str),
                    }
                )
                continue
            valid, error = validate_fen(fen)
            if not valid:
                rejected.append(
                    {
                        "source_path": str(path),
                        "source_record_index": json_record.source_record_index,
                        "fen": fen,
                        "reason": "invalid_fen",
                        "error": error,
                        "raw_record": json.dumps(truncate_value(json_record.record), default=str),
                    }
                )
                continue
            normalized = normalize_fen(fen)
            derived = summarize_fen_dict(normalized)
            derived["fen_source_field"] = fen_key
            row = canonical_from_raw(
                record=json_record.record,
                fen=fen,
                normalized_fen=normalized,
                source_path=json_record.source_path,
                source_record_index=json_record.source_record_index,
                source_kind=json_record.source_kind,
                default_status=default_label_status,
                derived=derived,
                metadata_overrides=parse_stockfish_metadata(json_record.record),
                raw_metadata_max_chars=args.raw_metadata_max_chars,
            )
            rows.append(row)
        if stop_all:
            break

    columns = CANONICAL_COLUMNS + DERIVED_COLUMNS + ["fen_source_field"]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    duplicate_count = int(df["normalized_fen"].duplicated().sum()) if not df.empty else 0
    if not df.empty:
        df = df.drop_duplicates("normalized_fen", keep="first").reset_index(drop=True)
    rejected_df = pd.DataFrame(rejected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    rejected_df.to_parquet(rejected_path, index=False)

    validation = validate_canonical_dataframe(df)
    label_counts = Counter(df["label_status"]) if not df.empty else Counter()
    report = {
        "input_paths": [str(p) for p in input_paths],
        "files": [str(p) for p in sorted(set(files))],
        "output_path": str(output_path),
        "rejected_output_path": str(rejected_path),
        "total_raw_rows": total_raw,
        "valid_rows_after_dedup": int(len(df)),
        "invalid_or_rejected_rows": int(len(rejected_df)),
        "duplicate_rows_removed": duplicate_count,
        "known_class_0_samples": int(label_counts.get("known_non_puzzle", 0)),
        "unresolved_candidate_1_2_samples": int(label_counts.get("candidate_1_or_2_unresolved", 0)),
        "verified_class_1_samples": int(label_counts.get("verified_near_puzzle", 0)),
        "verified_class_2_samples": int(label_counts.get("verified_puzzle", 0)),
        "missing_fen_count": missing_fen_count,
        "examples_rejected": rejected[:20],
        "label_status_distribution": dict(label_counts),
        "source_file_distribution": _distribution(df.get("source_file", pd.Series(dtype=object))),
        "legal_move_count_distribution": _distribution(df.get("legal_move_count", pd.Series(dtype=float)), bins=10),
        "side_to_move_distribution": _distribution(df.get("side_to_move", pd.Series(dtype=object))),
        "validation": validation,
    }
    write_json(report, "data/reports/prepare_dataset_report.json")
    lines = [
        "# Prepare Dataset Report",
        "",
        f"- Total raw rows: `{total_raw}`",
        f"- Valid rows after dedup: `{len(df)}`",
        f"- Invalid/rejected rows: `{len(rejected_df)}`",
        f"- Duplicates removed: `{duplicate_count}`",
        f"- Missing FEN count: `{missing_fen_count}`",
        f"- Output: `{output_path}`",
        f"- Rejected output: `{rejected_path}`",
        "",
        "## Label-status distribution",
        "",
    ]
    for key, value in report["label_status_distribution"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Validation", "", "```json", json.dumps(validation, indent=2), "```", ""])
    write_text("\n".join(lines), "data/reports/prepare_dataset_report.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
