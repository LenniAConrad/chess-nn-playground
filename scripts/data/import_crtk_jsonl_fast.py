#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as paj
import pyarrow.parquet as pq

import sys



from chess_nn_playground.data.schema import CANONICAL_COLUMNS, DERIVED_COLUMNS
from chess_nn_playground.utils.logging import write_json, write_text
from chess_nn_playground.utils.paths import utc_timestamp


INPUT_SCHEMA = pa.schema(
    [
        ("fen", pa.string()),
        ("label_status", pa.string()),
        ("coarse_label", pa.int64()),
        ("fine_label", pa.int64()),
        ("source_kind", pa.string()),
        ("source_file", pa.string()),
        ("source_record_index", pa.int64()),
        ("source_group_id", pa.string()),
        ("sister_group_id", pa.string()),
        ("game_id", pa.string()),
        ("position_index", pa.int64()),
        ("verification_status", pa.string()),
        ("best_move", pa.string()),
        ("pv1_cp", pa.float64()),
        ("pv2_cp", pa.float64()),
        ("pv_gap_cp", pa.float64()),
        ("pv1_mate", pa.int64()),
        ("pv2_mate", pa.int64()),
        ("stockfish_nodes", pa.int64()),
        ("stockfish_depth", pa.int64()),
        ("stockfish_version", pa.string()),
    ]
)

OUTPUT_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("fen", pa.string()),
        ("normalized_fen", pa.string()),
        ("label_status", pa.string()),
        ("coarse_label", pa.int64()),
        ("fine_label", pa.int64()),
        ("is_known_class_0", pa.bool_()),
        ("is_candidate_1_or_2", pa.bool_()),
        ("source_path", pa.string()),
        ("source_file", pa.string()),
        ("source_record_index", pa.int64()),
        ("source_kind", pa.string()),
        ("source_group_id", pa.string()),
        ("sister_group_id", pa.string()),
        ("game_id", pa.string()),
        ("position_index", pa.int64()),
        ("split_group_id", pa.string()),
        ("raw_label", pa.string()),
        ("raw_metadata_json", pa.string()),
        ("best_move", pa.string()),
        ("pv1_cp", pa.float64()),
        ("pv2_cp", pa.float64()),
        ("pv_gap_cp", pa.float64()),
        ("pv1_mate", pa.int64()),
        ("pv2_mate", pa.int64()),
        ("stockfish_nodes", pa.int64()),
        ("stockfish_version", pa.string()),
        ("verification_status", pa.string()),
        ("motif", pa.string()),
        ("game_phase", pa.string()),
        ("side_to_move", pa.string()),
        ("piece_count", pa.int64()),
        ("legal_move_count", pa.int64()),
        ("is_check", pa.bool_()),
        ("material_white", pa.int64()),
        ("material_black", pa.int64()),
        ("material_balance", pa.int64()),
        ("board_hash", pa.string()),
        ("fen_source_field", pa.string()),
        ("stockfish_depth", pa.int64()),
    ]
)

LABEL_STATUSES = [
    "known_non_puzzle",
    "candidate_1_or_2_unresolved",
    "verified_near_puzzle",
    "verified_puzzle",
]


def _column(table: pa.Table, name: str, typ: pa.DataType) -> pa.ChunkedArray:
    if name in table.column_names:
        return table[name].cast(typ)
    return pa.chunked_array([pa.nulls(table.num_rows, type=typ)])


def _scalar_array(value: Any, typ: pa.DataType, length: int) -> pa.ChunkedArray:
    return pa.chunked_array([pa.repeat(pa.scalar(value, type=typ), length)])


def _null_array(typ: pa.DataType, length: int) -> pa.ChunkedArray:
    return pa.chunked_array([pa.nulls(length, type=typ)])


def _label_fallback(label_status: pa.ChunkedArray) -> tuple[pa.ChunkedArray, pa.ChunkedArray]:
    known = pc.equal(label_status, pa.scalar("known_non_puzzle", pa.string()))
    near = pc.equal(label_status, pa.scalar("verified_near_puzzle", pa.string()))
    puzzle = pc.equal(label_status, pa.scalar("verified_puzzle", pa.string()))
    coarse = pc.if_else(known, pa.scalar(0, pa.int64()), pa.scalar(1, pa.int64()))
    fine = pc.if_else(
        known,
        pa.scalar(0, pa.int64()),
        pc.if_else(
            near,
            pa.scalar(1, pa.int64()),
            pc.if_else(puzzle, pa.scalar(2, pa.int64()), pa.scalar(None, pa.int64())),
        ),
    )
    return coarse, fine


def transform_batch(batch: pa.RecordBatch, input_path: Path) -> pa.Table:
    table = pa.Table.from_batches([batch])
    n = table.num_rows
    if n == 0:
        return pa.Table.from_arrays([pa.nulls(0, type=field.type) for field in OUTPUT_SCHEMA], schema=OUTPUT_SCHEMA)

    fen = _column(table, "fen", pa.string())
    label_status = pc.fill_null(
        _column(table, "label_status", pa.string()),
        pa.scalar("candidate_1_or_2_unresolved", pa.string()),
    )
    fallback_coarse, fallback_fine = _label_fallback(label_status)
    coarse_label = pc.fill_null(_column(table, "coarse_label", pa.int64()), fallback_coarse)
    fine_label = pc.fill_null(_column(table, "fine_label", pa.int64()), fallback_fine)

    source_file = pc.fill_null(
        _column(table, "source_file", pa.string()),
        pa.scalar(input_path.name, pa.string()),
    )
    source_record_index = _column(table, "source_record_index", pa.int64())
    source_record_index_text = pc.cast(pc.fill_null(source_record_index, pa.scalar(-1, pa.int64())), pa.string())
    sample_id = pc.binary_join_element_wise(source_file, source_record_index_text, ":")

    source_group_id = _column(table, "source_group_id", pa.string())
    sister_group_id = _column(table, "sister_group_id", pa.string())
    game_id = _column(table, "game_id", pa.string())
    split_group_id = pc.coalesce(sister_group_id, source_group_id, game_id, fen)

    source_kind = pc.fill_null(
        _column(table, "source_kind", pa.string()),
        pa.scalar("crtk_record", pa.string()),
    )

    columns: dict[str, pa.Array | pa.ChunkedArray] = {
        "sample_id": sample_id,
        "fen": fen,
        "normalized_fen": fen,
        "label_status": label_status,
        "coarse_label": coarse_label,
        "fine_label": fine_label,
        "is_known_class_0": pc.equal(coarse_label, pa.scalar(0, pa.int64())),
        "is_candidate_1_or_2": pc.equal(coarse_label, pa.scalar(1, pa.int64())),
        "source_path": _scalar_array(str(input_path), pa.string(), n),
        "source_file": source_file,
        "source_record_index": source_record_index,
        "source_kind": source_kind,
        "source_group_id": source_group_id,
        "sister_group_id": sister_group_id,
        "game_id": game_id,
        "position_index": _column(table, "position_index", pa.int64()),
        "split_group_id": split_group_id,
        "raw_label": label_status,
        "raw_metadata_json": _null_array(pa.string(), n),
        "best_move": _column(table, "best_move", pa.string()),
        "pv1_cp": _column(table, "pv1_cp", pa.float64()),
        "pv2_cp": _column(table, "pv2_cp", pa.float64()),
        "pv_gap_cp": _column(table, "pv_gap_cp", pa.float64()),
        "pv1_mate": _column(table, "pv1_mate", pa.int64()),
        "pv2_mate": _column(table, "pv2_mate", pa.int64()),
        "stockfish_nodes": _column(table, "stockfish_nodes", pa.int64()),
        "stockfish_version": _column(table, "stockfish_version", pa.string()),
        "verification_status": _column(table, "verification_status", pa.string()),
        "motif": _null_array(pa.string(), n),
        "game_phase": _null_array(pa.string(), n),
        "side_to_move": _null_array(pa.string(), n),
        "piece_count": _null_array(pa.int64(), n),
        "legal_move_count": _null_array(pa.int64(), n),
        "is_check": _null_array(pa.bool_(), n),
        "material_white": _null_array(pa.int64(), n),
        "material_black": _null_array(pa.int64(), n),
        "material_balance": _null_array(pa.int64(), n),
        "board_hash": _null_array(pa.string(), n),
        "fen_source_field": _scalar_array("fen", pa.string(), n),
        "stockfish_depth": _column(table, "stockfish_depth", pa.int64()),
    }

    return pa.Table.from_arrays([columns[field.name].cast(field.type) for field in OUTPUT_SCHEMA], schema=OUTPUT_SCHEMA)


def counts_for(table: pa.Table) -> dict[str, int]:
    counts: dict[str, int] = {}
    value_counts = pc.value_counts(table["label_status"])
    if isinstance(value_counts, pa.ChunkedArray):
        value_counts = value_counts.combine_chunks()
    for item in value_counts.to_pylist():
        counts[str(item["values"])] = int(item["counts"])
    return counts


def write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    state = dict(state)
    state["updated_at"] = utc_timestamp()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def merge_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    result = dict(left)
    for key, value in right.items():
        result[key] = int(result.get(key, 0)) + int(value)
    return result


def progress_line(rows: int, started: float, expected_rows: int | None) -> str:
    elapsed = max(time.monotonic() - started, 1e-9)
    rows_per_sec = rows / elapsed
    line = f"rows={rows:,} elapsed={elapsed / 60:.1f}m rate={rows_per_sec:,.0f} rows/s"
    if expected_rows and rows_per_sec > 0:
        remaining = max(expected_rows - rows, 0) / rows_per_sec
        line += f" eta={remaining / 3600:.2f}h"
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast import canonical crtk training JSONL into Parquet.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--report-json", default="data/reports/crtk_fast_import_report.json")
    parser.add_argument("--report-md", default="data/reports/crtk_fast_import_report.md")
    parser.add_argument("--expected-rows", type=int, default=45_002_737)
    parser.add_argument("--block-size-mb", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_path.with_suffix(output_path.suffix + ".checkpoint.json")
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")

    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_path}. Pass --overwrite or choose a new output path.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp_output.exists():
        tmp_output.unlink()

    read_options = paj.ReadOptions(block_size=args.block_size_mb * 1024 * 1024, use_threads=True)
    parse_options = paj.ParseOptions(explicit_schema=INPUT_SCHEMA, unexpected_field_behavior="ignore")
    reader = paj.open_json(input_path, read_options=read_options, parse_options=parse_options)

    started = time.monotonic()
    rows = 0
    batches = 0
    label_counts: dict[str, int] = {}
    next_progress = args.progress_every
    writer: pq.ParquetWriter | None = None
    state = {
        "created_at": utc_timestamp(),
        "input": str(input_path),
        "output": str(output_path),
        "tmp_output": str(tmp_output),
        "complete": False,
        "rows": 0,
        "batches": 0,
        "label_status_distribution": {},
        "compression": args.compression,
        "block_size_mb": args.block_size_mb,
        "deduplication": "skipped; split_group_id falls back to FEN to keep identical positions in one split",
        "fen_validation": "skipped; crtk export already filtered invalid records",
    }
    write_checkpoint(checkpoint_path, state)

    try:
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            if batch.num_rows == 0:
                continue
            if args.max_rows is not None and rows + batch.num_rows > args.max_rows:
                batch = batch.slice(0, args.max_rows - rows)
                if batch.num_rows == 0:
                    break

            table = transform_batch(batch, input_path)
            if writer is None:
                writer = pq.ParquetWriter(tmp_output, table.schema, compression=args.compression)
            writer.write_table(table)

            rows += table.num_rows
            batches += 1
            label_counts = merge_counts(label_counts, counts_for(table))
            if rows >= next_progress or args.max_rows is not None:
                print(progress_line(rows, started, args.expected_rows), flush=True)
                state.update(
                    {
                        "rows": rows,
                        "batches": batches,
                        "label_status_distribution": label_counts,
                        "complete": False,
                    }
                )
                write_checkpoint(checkpoint_path, state)
                while rows >= next_progress:
                    next_progress += args.progress_every
            if args.max_rows is not None and rows >= args.max_rows:
                break
    finally:
        if writer is not None:
            writer.close()

    tmp_output.replace(output_path)
    elapsed = time.monotonic() - started
    state.update(
        {
            "rows": rows,
            "batches": batches,
            "label_status_distribution": label_counts,
            "complete": True,
            "elapsed_seconds": elapsed,
            "rows_per_second": rows / elapsed if elapsed else None,
            "updated_at": utc_timestamp(),
        }
    )
    write_checkpoint(checkpoint_path, state)

    report = {
        **state,
        "output_size_bytes": output_path.stat().st_size if output_path.exists() else None,
        "canonical_columns": CANONICAL_COLUMNS,
        "derived_columns_present_but_not_computed": DERIVED_COLUMNS,
        "notes": (
            "Fast path for crtk canonical JSONL. It preserves labels and grouping metadata, "
            "uses fen as normalized_fen, skips Python chess validation and deduplication, "
            "and does not store raw nested engine metadata."
        ),
    }
    write_json(report, args.report_json)
    lines = [
        "# crtk Fast Import Report",
        "",
        f"- Input: `{input_path}`",
        f"- Output: `{output_path}`",
        f"- Rows: `{rows:,}`",
        f"- Elapsed minutes: `{elapsed / 60:.2f}`",
        f"- Rows/sec: `{rows / elapsed:,.0f}`" if elapsed else "- Rows/sec: `unknown`",
        f"- Compression: `{args.compression}`",
        "",
        "## Label-status distribution",
        "",
    ]
    for status in LABEL_STATUSES:
        lines.append(f"- `{status}`: {label_counts.get(status, 0):,}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- FEN validation and derived board summaries were skipped for speed.",
            "- Deduplication was skipped; identical FENs share `split_group_id = normalized_fen` when no stronger group exists.",
            "- Engine metadata is retained only for selected scalar columns, not nested `multipv`.",
            "",
        ]
    )
    write_text("\n".join(lines), args.report_md)
    print(progress_line(rows, started, args.expected_rows), flush=True)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
