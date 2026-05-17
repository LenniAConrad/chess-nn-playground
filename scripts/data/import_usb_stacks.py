#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import sys



from chess_nn_playground.data.fen_utils import normalize_fen, summarize_fen_dict, validate_fen
from chess_nn_playground.data.json_loader import choose_fen, find_json_files, iter_json_records, truncate_value
from chess_nn_playground.data.schema import CANONICAL_COLUMNS, canonical_from_raw
from chess_nn_playground.data.stockfish import parse_stockfish_metadata
from chess_nn_playground.utils.logging import write_json, write_text
from chess_nn_playground.utils.paths import utc_timestamp


STRING_COLUMNS = set(CANONICAL_COLUMNS) - {
    "coarse_label",
    "fine_label",
    "is_known_class_0",
    "is_candidate_1_or_2",
    "source_record_index",
    "position_index",
    "pv1_cp",
    "pv2_cp",
    "pv_gap_cp",
    "pv1_mate",
    "pv2_mate",
    "stockfish_nodes",
}
STRING_COLUMNS.update(
    {
        "side_to_move",
        "board_hash",
        "fen_source_field",
        "reason",
        "error",
        "raw_record",
    }
)
INT_COLUMNS = {
    "coarse_label",
    "fine_label",
    "source_record_index",
    "position_index",
    "pv1_mate",
    "pv2_mate",
    "stockfish_nodes",
    "piece_count",
    "legal_move_count",
    "material_white",
    "material_black",
    "material_balance",
}
FLOAT_COLUMNS = {"pv1_cp", "pv2_cp", "pv_gap_cp"}
BOOL_COLUMNS = {"is_known_class_0", "is_candidate_1_or_2", "is_check"}


def parts_dir_for(path: Path) -> Path:
    return path.with_name(path.name + ".parts")


def checkpoint_path_for(path: Path) -> Path:
    return path.with_name(path.name + ".checkpoint.json")


def part_path(parts_dir: Path, part_index: int) -> Path:
    return parts_dir / f"part-{part_index:06d}.parquet"


def part_files(parts_dir: Path) -> list[Path]:
    if not parts_dir.exists():
        return []
    return sorted(parts_dir.glob("part-*.parquet"))


def part_index_from_path(path: Path) -> int | None:
    try:
        return int(path.stem.split("-")[-1])
    except Exception:
        return None


def rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    df = pd.DataFrame(rows)
    for column in sorted(STRING_COLUMNS.intersection(df.columns)):
        df[column] = df[column].astype("string")
    for column in sorted(INT_COLUMNS.intersection(df.columns)):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    for column in sorted(FLOAT_COLUMNS.intersection(df.columns)):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in sorted(BOOL_COLUMNS.intersection(df.columns)):
        df[column] = df[column].astype("boolean")
    return pa.Table.from_pandas(df, preserve_index=False)


def write_part(rows: list[dict[str, Any]], parts_dir: Path, part_index: int) -> Path:
    parts_dir.mkdir(parents=True, exist_ok=True)
    target = part_path(parts_dir, part_index)
    tmp = target.with_name(target.name + ".tmp")
    pq.write_table(rows_to_table(rows), tmp, compression="zstd")
    tmp.replace(target)
    return target


def write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_timestamp()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_seen_hashes(parts_dir: Path) -> set[str]:
    seen: set[str] = set()
    for path in part_files(parts_dir):
        try:
            table = pq.read_table(path, columns=["board_hash"])
        except Exception:
            corrupt_path = path.with_suffix(path.suffix + ".corrupt")
            path.replace(corrupt_path)
            print(f"Warning: moved unreadable part to {corrupt_path}", flush=True)
            continue
        for value in table.column("board_hash").to_pylist():
            if value:
                seen.add(str(value))
    return seen


def remove_stale_tmp_files(parts_dir: Path) -> None:
    if not parts_dir.exists():
        return
    for path in parts_dir.glob("part-*.parquet.tmp"):
        path.unlink()
        print(f"Removed stale temporary part file: {path}", flush=True)


def trim_uncheckpointed_parts(parts_dir: Path, keep_count: int) -> None:
    if not parts_dir.exists():
        return
    for path in part_files(parts_dir):
        index = part_index_from_path(path)
        if index is not None and index >= keep_count:
            path.unlink()
            print(f"Removed uncheckpointed part file: {path}", flush=True)


def align_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays = []
    for field in schema:
        if field.name in table.column_names:
            arrays.append(table[field.name].cast(field.type))
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def assemble_parts(parts_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts = part_files(parts_dir)
    if not parts:
        pd.DataFrame().to_parquet(output_path, index=False)
        return

    writer: pq.ParquetWriter | None = None
    try:
        for path in parts:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            else:
                table = align_table_to_schema(table, writer.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def load_counted_state(state: dict[str, Any] | None) -> tuple[Counter[str], Counter[str], Counter[str]]:
    if state is None:
        return Counter(), Counter(), Counter()
    return (
        Counter(state.get("counts", {})),
        Counter(state.get("label_status_distribution", {})),
        Counter(state.get("source_file_distribution", {})),
    )


def flush_rows(
    rows: list[dict[str, Any]],
    parts_dir: Path,
    part_index: int,
) -> int:
    if rows:
        write_part(rows, parts_dir, part_index)
        rows.clear()
        part_index += 1
    return part_index


def clean_generated_state(output_path: Path, rejected_path: Path, checkpoint_path: Path) -> None:
    for path in [output_path, rejected_path, checkpoint_path]:
        if path.exists():
            path.unlink()
    for path in [parts_dir_for(output_path), parts_dir_for(rejected_path)]:
        if path.exists():
            shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-import the USB Stockfish stack JSON files into canonical Parquet.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=[
            "/media/lennart/USB STICK/1M_STACK",
            "/media/lennart/USB STICK/2M_STACK",
        ],
    )
    parser.add_argument("--output", default="data/processed/usb_all_positions.parquet")
    parser.add_argument("--rejected-output", default="data/processed/usb_all_rejected_positions.parquet")
    parser.add_argument("--default-label-status", default="candidate_1_or_2_unresolved")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=50000, help="Print progress after this many raw records.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-records-per-file", type=int, default=None)
    parser.add_argument("--raw-metadata-max-chars", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they already exist.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from an existing checkpoint. Use with --overwrite to start clean.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    rejected_path = Path(args.rejected_output)
    parts_dir = parts_dir_for(output_path)
    rejected_parts_dir = parts_dir_for(rejected_path)
    checkpoint_path = checkpoint_path_for(output_path)

    if args.overwrite:
        clean_generated_state(output_path, rejected_path, checkpoint_path)

    files: list[Path] = []
    print(f"Scanning input paths: {args.input}", flush=True)
    for input_path in args.input:
        files.extend(find_json_files(input_path))
    files = sorted(set(files))
    if args.max_files is not None:
        files = files[: args.max_files]
    print(
        f"Found {len(files)} JSON/JSONL files. Output: {output_path}. "
        f"Rejected output: {rejected_path}.",
        flush=True,
    )
    if not files:
        print("No JSON/JSONL files found; writing empty outputs.", flush=True)

    file_names = [str(path) for path in files]
    checkpoint = None if args.no_resume else read_checkpoint(checkpoint_path)
    resuming = checkpoint is not None and not checkpoint.get("complete", False)
    if checkpoint and checkpoint.get("complete", False):
        if output_path.exists() and rejected_path.exists():
            print(f"Import already complete for {output_path}", flush=True)
            return
        print("Checkpoint says complete, but final output is missing; rebuilding final Parquet from parts.", flush=True)
        assemble_parts(parts_dir, output_path)
        assemble_parts(rejected_parts_dir, rejected_path)
        return

    if checkpoint and checkpoint.get("files") != file_names:
        raise SystemExit(
            "Existing checkpoint was created for a different file list. "
            "Use the same command/RUN_ID to resume, or pass --overwrite to start over."
        )
    if checkpoint and checkpoint.get("default_label_status") != args.default_label_status:
        raise SystemExit(
            "Existing checkpoint used a different default label status. "
            "Use the same command/RUN_ID to resume, or pass --overwrite to start over."
        )
    if not checkpoint:
        if output_path.exists() or rejected_path.exists():
            raise SystemExit(
                f"Refusing to overwrite existing final output {output_path} or {rejected_path}. "
                "Choose a new RUN_ID or pass --overwrite."
            )
        if (parts_dir.exists() or rejected_parts_dir.exists()) and not args.overwrite:
            raise SystemExit(
                f"Found generated parts without a checkpoint: {parts_dir} or {rejected_parts_dir}. "
                "Pass --overwrite to start clean, or choose a new RUN_ID."
            )
        checkpoint = {
            "version": 2,
            "created_at": utc_timestamp(),
            "updated_at": utc_timestamp(),
            "input": args.input,
            "files": file_names,
            "output": str(output_path),
            "rejected_output": str(rejected_path),
            "parts_dir": str(parts_dir),
            "rejected_parts_dir": str(rejected_parts_dir),
            "default_label_status": args.default_label_status,
            "counts": {},
            "label_status_distribution": {},
            "source_file_distribution": {},
            "completed_files": [],
            "per_file": [],
            "current_file": None,
            "current_file_counts": {},
            "current_file_raw_rows": 0,
            "part_index": 0,
            "rejected_part_index": 0,
            "complete": False,
        }
        write_checkpoint(checkpoint_path, checkpoint)
    elif resuming:
        print(f"Resuming from checkpoint: {checkpoint_path}", flush=True)

    remove_stale_tmp_files(parts_dir)
    remove_stale_tmp_files(rejected_parts_dir)
    trim_uncheckpointed_parts(parts_dir, int(checkpoint.get("part_index", 0)))
    trim_uncheckpointed_parts(rejected_parts_dir, int(checkpoint.get("rejected_part_index", 0)))
    seen_hashes = load_seen_hashes(parts_dir)
    if seen_hashes:
        print(f"Loaded {len(seen_hashes)} already-written board hashes from part files.", flush=True)

    rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    counts, label_counts, source_counts = load_counted_state(checkpoint)
    completed_files = set(str(path) for path in checkpoint.get("completed_files", []))
    per_file: list[dict[str, Any]] = list(checkpoint.get("per_file", []))
    part_index = int(checkpoint.get("part_index", 0))
    rejected_part_index = int(checkpoint.get("rejected_part_index", 0))

    def save_state(
        current_file: str | None = None,
        current_file_counts: Counter[str] | None = None,
        current_file_raw_rows: int = 0,
        complete: bool = False,
    ) -> None:
        checkpoint.update(
            {
                "counts": dict(counts),
                "label_status_distribution": dict(label_counts),
                "source_file_distribution": dict(source_counts),
                "completed_files": sorted(completed_files),
                "per_file": per_file,
                "current_file": current_file,
                "current_file_counts": dict(current_file_counts or {}),
                "current_file_raw_rows": current_file_raw_rows,
                "part_index": part_index,
                "rejected_part_index": rejected_part_index,
                "complete": complete,
            }
        )
        write_checkpoint(checkpoint_path, checkpoint)

    stop = False
    last_progress_raw = counts["raw_rows"]
    for file_index, path in enumerate(files, start=1):
        path_text = str(path)
        if path_text in completed_files:
            print(f"Skipping completed [{file_index}/{len(files)}] {path}", flush=True)
            continue
        file_counts = Counter()
        skip_records = 0
        if checkpoint.get("current_file") == path_text:
            file_counts = Counter(checkpoint.get("current_file_counts", {}))
            skip_records = int(checkpoint.get("current_file_raw_rows", 0))
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Starting [{file_index}/{len(files)}] {path} ({size_mb:.1f} MiB)", flush=True)
        if skip_records:
            print(f"Resuming inside {path.name}: skipping {skip_records} already-checkpointed records.", flush=True)
        save_state(path_text, file_counts, int(file_counts["raw_rows"]))
        records_seen_in_file = 0
        for json_record in iter_json_records(path):
            if records_seen_in_file < skip_records:
                records_seen_in_file += 1
                continue
            if args.max_records_per_file is not None and file_counts["raw_rows"] >= args.max_records_per_file:
                break
            if args.max_records is not None and counts["raw_rows"] >= args.max_records:
                stop = True
                break
            records_seen_in_file += 1
            counts["raw_rows"] += 1
            file_counts["raw_rows"] += 1
            fen_key, fen = choose_fen(json_record.record)
            if not fen:
                counts["missing_fen"] += 1
                file_counts["missing_fen"] += 1
                rejected_rows.append(
                    {
                        "source_path": str(path),
                        "source_record_index": json_record.source_record_index,
                        "reason": "missing_fen",
                        "raw_record": json.dumps(truncate_value(json_record.record), default=str),
                    }
                )
            else:
                valid, error = validate_fen(fen)
                if not valid:
                    counts["invalid_fen"] += 1
                    file_counts["invalid_fen"] += 1
                    rejected_rows.append(
                        {
                            "source_path": str(path),
                            "source_record_index": json_record.source_record_index,
                            "fen": fen,
                            "reason": "invalid_fen",
                            "error": error,
                            "raw_record": json.dumps(truncate_value(json_record.record), default=str),
                        }
                    )
                else:
                    normalized = normalize_fen(fen)
                    derived = summarize_fen_dict(normalized)
                    board_hash = derived["board_hash"]
                    if board_hash in seen_hashes:
                        counts["duplicates"] += 1
                        file_counts["duplicates"] += 1
                    else:
                        seen_hashes.add(board_hash)
                        derived["fen_source_field"] = fen_key
                        row = canonical_from_raw(
                            record=json_record.record,
                            fen=fen,
                            normalized_fen=normalized,
                            source_path=json_record.source_path,
                            source_record_index=json_record.source_record_index,
                            source_kind=json_record.source_kind,
                            default_status=args.default_label_status,
                            derived=derived,
                            metadata_overrides=parse_stockfish_metadata(json_record.record),
                            raw_metadata_max_chars=args.raw_metadata_max_chars,
                        )
                        rows.append(row)
                        counts["valid_unique_rows"] += 1
                        file_counts["valid_unique_rows"] += 1
                        label_counts[row["label_status"]] += 1
                        source_counts[Path(row["source_file"]).name] += 1

            if len(rows) >= args.batch_size:
                part_index = flush_rows(rows, parts_dir, part_index)
                save_state(path_text, file_counts, int(file_counts["raw_rows"]))
            if len(rejected_rows) >= args.batch_size:
                rejected_part_index = flush_rows(rejected_rows, rejected_parts_dir, rejected_part_index)
                save_state(path_text, file_counts, int(file_counts["raw_rows"]))
            if args.progress_every > 0 and counts["raw_rows"] - last_progress_raw >= args.progress_every:
                last_progress_raw = counts["raw_rows"]
                print(
                    f"Progress: raw={counts['raw_rows']} unique={counts['valid_unique_rows']} "
                    f"dup={counts['duplicates']} missing_fen={counts['missing_fen']} "
                    f"invalid_fen={counts['invalid_fen']} current_file={path.name} "
                    f"current_file_raw={file_counts['raw_rows']}",
                    flush=True,
                )

        part_index = flush_rows(rows, parts_dir, part_index)
        rejected_part_index = flush_rows(rejected_rows, rejected_parts_dir, rejected_part_index)
        per_file.append({"path": str(path), **dict(file_counts)})
        completed_files.add(path_text)
        save_state(None, Counter(), 0)
        print(
            f"[{file_index}/{len(files)}] {path.name}: raw={file_counts['raw_rows']} "
            f"unique={file_counts['valid_unique_rows']} dup={file_counts['duplicates']} "
            f"total_unique={counts['valid_unique_rows']}",
            flush=True,
        )
        if stop:
            break

    if stop:
        print(
            "Stopped because --max-records was reached. Assembling output from completed/checkpointed parts.",
            flush=True,
        )

    print(f"Assembling final output from {len(part_files(parts_dir))} parts.", flush=True)
    assemble_parts(parts_dir, output_path)
    assemble_parts(rejected_parts_dir, rejected_path)
    save_state(None, Counter(), 0, complete=True)

    report = {
        "created_at": utc_timestamp(),
        "input": args.input,
        "files_seen": len(files),
        "files_completed": len(completed_files),
        "output": str(output_path),
        "rejected_output": str(rejected_path),
        "checkpoint": str(checkpoint_path),
        "parts_dir": str(parts_dir),
        "rejected_parts_dir": str(rejected_parts_dir),
        "default_label_status": args.default_label_status,
        "counts": dict(counts),
        "label_status_distribution": dict(label_counts),
        "source_file_distribution_top50": dict(source_counts.most_common(50)),
        "per_file": per_file,
        "stockfish_metadata_stored_as_metadata_only": True,
        "classification_note": (
            "USB stack records do not expose verified puzzle labels in the inspected schema. "
            "Rows are imported as unresolved candidates unless explicit labels are provided."
        ),
    }
    write_json(report, "data/reports/usb_stack_import_report.json")
    lines = [
        "# USB Stack Import Report",
        "",
        f"- Input folders/files: `{args.input}`",
        f"- Files seen: `{len(files)}`",
        f"- Files completed: `{len(completed_files)}`",
        f"- Raw rows: `{counts['raw_rows']}`",
        f"- Valid unique rows: `{counts['valid_unique_rows']}`",
        f"- Duplicates skipped: `{counts['duplicates']}`",
        f"- Missing FEN: `{counts['missing_fen']}`",
        f"- Invalid FEN: `{counts['invalid_fen']}`",
        f"- Output: `{output_path}`",
        f"- Rejected output: `{rejected_path}`",
        f"- Resume checkpoint: `{checkpoint_path}`",
        f"- Durable part files: `{parts_dir}`",
        "",
        "## Label-status distribution",
        "",
    ]
    for key, value in label_counts.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Classification note",
            "",
            report["classification_note"],
            "",
        ]
    )
    write_text("\n".join(lines), "data/reports/usb_stack_import_report.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
