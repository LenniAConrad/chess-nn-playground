#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import sys



from chess_nn_playground.utils.logging import write_json, write_text
from chess_nn_playground.utils.paths import utc_timestamp


SPLITS = ("train", "val", "test")
UINT64_MOD = 1_000_000


def split_for_hash(group_hash: np.ndarray, train_frac: float, val_frac: float) -> np.ndarray:
    buckets = np.remainder(group_hash, 10_000)
    train_cutoff = int(round(train_frac * 10_000))
    val_cutoff = train_cutoff + int(round(val_frac * 10_000))
    result = np.full(len(buckets), "test", dtype=object)
    result[buckets < val_cutoff] = "val"
    result[buckets < train_cutoff] = "train"
    return result


def stable_hash(values: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(values.fillna("").astype("string"), index=False).to_numpy(dtype=np.uint64)


def split_quota(total: int, train_frac: float, val_frac: float, test_frac: float) -> dict[str, int]:
    train = int(round(total * train_frac))
    val = int(round(total * val_frac))
    test = max(total - train - val, 0)
    return {"train": train, "val": val, "test": test}


def label_column_for_mode(mode: str) -> str:
    if mode == "coarse_binary":
        return "coarse_label"
    if mode == "fine_3class":
        return "fine_label"
    raise ValueError(f"Unsupported mode: {mode}")


def class_labels_for_mode(mode: str) -> list[int]:
    if mode == "coarse_binary":
        return [0, 1]
    if mode == "fine_3class":
        return [0, 1, 2]
    raise ValueError(f"Unsupported mode: {mode}")


def needed_columns(mode: str) -> list[str]:
    return ["sample_id", "normalized_fen", "split_group_id", label_column_for_mode(mode)]


def read_metadata_frame(batch: pa.RecordBatch) -> pd.DataFrame:
    table = pa.Table.from_batches([batch])
    return table.to_pandas(self_destruct=True)


def count_split_classes(
    input_path: Path,
    batch_size: int,
    train_frac: float,
    val_frac: float,
    mode: str,
) -> dict[str, dict[int, int]]:
    counts: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    parquet = pq.ParquetFile(input_path)
    label_column = label_column_for_mode(mode)
    started = time.monotonic()
    last_progress = started
    rows_seen = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=needed_columns(mode)):
        df = read_metadata_frame(batch)
        rows_seen += len(df)
        group_key = df["split_group_id"].where(df["split_group_id"].notna(), df["normalized_fen"])
        split_names = split_for_hash(stable_hash(group_key), train_frac, val_frac)
        labels = pd.to_numeric(df[label_column], errors="coerce")
        for split in SPLITS:
            split_mask = split_names == split
            if not split_mask.any():
                continue
            for label, count in labels[split_mask].value_counts(dropna=True).items():
                counts[split][int(label)] += int(count)
        now = time.monotonic()
        if now - last_progress >= 10:
            last_progress = now
            elapsed = max(now - started, 1e-9)
            print(
                f"counted={rows_seen:,} elapsed={elapsed / 60:.1f}m rate={rows_seen / elapsed:,.0f} rows/s",
                flush=True,
            )
    return {split: dict(counter) for split, counter in counts.items()}


def make_thresholds(
    available: dict[str, dict[int, int]],
    quotas: dict[int, dict[str, int]],
    multiplier: float,
) -> dict[tuple[str, int], int]:
    thresholds: dict[tuple[str, int], int] = {}
    for label, label_quotas in quotas.items():
        for split, quota in label_quotas.items():
            count = int(available.get(split, {}).get(label, 0))
            if count <= 0 or quota <= 0:
                thresholds[(split, label)] = 0
                continue
            rate = min(1.0, (quota / count) * multiplier)
            thresholds[(split, label)] = int(math.ceil(rate * UINT64_MOD))
    return thresholds


def make_writers(output_dir: Path, schema: pa.Schema, compression: str) -> dict[str, pq.ParquetWriter]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        split: pq.ParquetWriter(output_dir / f"split_{split}.parquet", schema, compression=compression)
        for split in SPLITS
    }


def close_writers(writers: dict[str, pq.ParquetWriter]) -> None:
    for writer in writers.values():
        writer.close()


def output_schema(input_schema: pa.Schema) -> pa.Schema:
    if "split" in input_schema.names:
        return input_schema
    return input_schema.append(pa.field("split", pa.string()))


def with_split_column(table: pa.Table, split: str) -> pa.Table:
    if "split" in table.column_names:
        index = table.column_names.index("split")
        return table.set_column(index, "split", pa.repeat(pa.scalar(split, pa.string()), table.num_rows))
    return table.append_column("split", pa.repeat(pa.scalar(split, pa.string()), table.num_rows))


def write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    state = dict(state)
    state["updated_at"] = utc_timestamp()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# crtk Sample Split Report",
        "",
        f"- Input: `{report['input']}`",
        f"- Output dir: `{report['output_dir']}`",
        f"- Mode: `{report['mode']}`",
        f"- Label column: `{report['label_column']}`",
        f"- Max per class requested: `{report['max_per_class']:,}`",
        f"- Rows written: `{report['rows_written']:,}`",
        f"- Elapsed minutes: `{report['elapsed_seconds'] / 60:.2f}`",
        f"- Batch size: `{report['batch_size']:,}`",
        f"- De-duplicate normalized FEN: `{report.get('dedupe_normalized_fen', False)}`",
        "",
        "## Rows By Split/Class",
        "",
        "```json",
        json.dumps(report["written_by_split_class"], indent=2),
        "```",
        "",
        "## Memory Safety",
        "",
        "- The script streams Parquet batches and never loads the full 45M-row file.",
        "- Output size is capped by `--max-per-class` so the current pandas-based trainer can open the split files.",
        "- Split assignment is deterministic from `split_group_id`, falling back to FEN, to avoid group leakage.",
        "- When `--dedupe-normalized-fen` is enabled, each normalized FEN can appear at most once across all splits.",
        "",
    ]
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create RAM-safe sampled train/val/test splits from large crtk Parquet.")
    parser.add_argument("--input", default="data/processed/crtk_training_20260419_180229_fast.parquet")
    parser.add_argument("--output-dir", default="data/splits/crtk_sample")
    parser.add_argument("--mode", default="fine_3class", choices=["coarse_binary", "fine_3class"])
    parser.add_argument("--max-per-class", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--sample-multiplier", type=float, default=1.10)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--report-json", default="data/reports/crtk_sample_split_report.json")
    parser.add_argument("--report-md", default="data/reports/crtk_sample_split_report.md")
    parser.add_argument("--dedupe-normalized-fen", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if abs((args.train_frac + args.val_frac + args.test_frac) - 1.0) > 1e-6:
        raise SystemExit("Split fractions must sum to 1.0")
    if args.max_per_class <= 0:
        raise SystemExit("--max-per-class must be positive")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "sample_split_checkpoint.json"
    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")
    if output_dir.exists() and any(output_dir.glob("split_*.parquet")) and not args.overwrite:
        raise SystemExit(f"Output split files already exist in {output_dir}. Pass --overwrite to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in output_dir.glob("split_*.parquet"):
            path.unlink()

    started = time.monotonic()
    class_labels = class_labels_for_mode(args.mode)
    label_column = label_column_for_mode(args.mode)
    print(f"Pass 1/2: counting split/class availability for {args.mode}...", flush=True)
    available = count_split_classes(input_path, args.batch_size, args.train_frac, args.val_frac, args.mode)
    quotas = {
        label: split_quota(args.max_per_class, args.train_frac, args.val_frac, args.test_frac)
        for label in class_labels
    }
    thresholds = make_thresholds(available, quotas, args.sample_multiplier)
    state: dict[str, Any] = {
        "created_at": utc_timestamp(),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "label_column": label_column,
        "class_labels": class_labels,
        "max_per_class": args.max_per_class,
        "available_by_split_class": available,
        "quotas": quotas,
        "thresholds": {f"{split}:{label}": value for (split, label), value in thresholds.items()},
        "dedupe_normalized_fen": args.dedupe_normalized_fen,
        "complete": False,
        "rows_written": 0,
    }
    write_checkpoint(checkpoint_path, state)

    parquet = pq.ParquetFile(input_path)
    schema = output_schema(parquet.schema_arrow)
    writers = make_writers(output_dir, schema, args.compression)
    written: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    seen_normalized_fens: set[str] = set()
    skipped_duplicate_fens = 0
    total_written = 0
    total_seen = 0
    last_progress = time.monotonic()

    print("Pass 2/2: writing sampled split files...", flush=True)
    try:
        for batch in parquet.iter_batches(batch_size=args.batch_size):
            table = pa.Table.from_batches([batch])
            meta = table.select(needed_columns(args.mode)).to_pandas(self_destruct=True)
            group_key = meta["split_group_id"].where(meta["split_group_id"].notna(), meta["normalized_fen"])
            split_names = split_for_hash(stable_hash(group_key), args.train_frac, args.val_frac)
            sample_hash = np.remainder(stable_hash(meta["sample_id"]), UINT64_MOD)
            labels = pd.to_numeric(meta[label_column], errors="coerce").to_numpy()
            total_seen += len(meta)

            for split in SPLITS:
                split_mask = split_names == split
                if not split_mask.any():
                    continue
                selected_for_split = np.zeros(len(meta), dtype=bool)
                for label in class_labels:
                    remaining = quotas[label][split] - written[split][label]
                    if remaining <= 0:
                        continue
                    threshold = thresholds[(split, label)]
                    if threshold <= 0:
                        continue
                    mask = split_mask & (labels == label) & (sample_hash < threshold)
                    selected_count = 0
                    for index in np.flatnonzero(mask):
                        if selected_count >= remaining:
                            break
                        if args.dedupe_normalized_fen:
                            normalized_fen = str(meta["normalized_fen"].iloc[index])
                            if normalized_fen in seen_normalized_fens:
                                skipped_duplicate_fens += 1
                                continue
                            seen_normalized_fens.add(normalized_fen)
                        selected_for_split[index] = True
                        selected_count += 1
                    written[split][label] += selected_count
                if selected_for_split.any():
                    selected = table.filter(pa.array(selected_for_split))
                    selected = with_split_column(selected, split)
                    writers[split].write_table(selected)
                    total_written += selected.num_rows

            now = time.monotonic()
            if now - last_progress >= 10:
                last_progress = now
                elapsed = max(now - started, 1e-9)
                print(
                    f"seen={total_seen:,} written={total_written:,} "
                    f"elapsed={elapsed / 60:.1f}m rate={total_seen / elapsed:,.0f} rows/s",
                    flush=True,
                )
                state.update(
                    {
                        "rows_seen": total_seen,
                        "rows_written": total_written,
                        "written_by_split_class": {split: dict(counter) for split, counter in written.items()},
                        "skipped_duplicate_fens": skipped_duplicate_fens,
                    }
                )
                write_checkpoint(checkpoint_path, state)

            if all(written[split][label] >= quotas[label][split] for split in SPLITS for label in class_labels):
                break
    finally:
        close_writers(writers)

    elapsed = time.monotonic() - started
    warnings = []
    for split in SPLITS:
        for label in class_labels:
            if written[split][label] < quotas[label][split]:
                warnings.append(
                    f"Only wrote {written[split][label]:,}/{quotas[label][split]:,} rows for split={split}, class={label}. "
                    "Increase --sample-multiplier or lower --max-per-class."
                )
    report = {
        **state,
        "complete": True,
        "rows_seen": total_seen,
        "rows_written": total_written,
        "written_by_split_class": {split: dict(counter) for split, counter in written.items()},
        "elapsed_seconds": elapsed,
        "batch_size": args.batch_size,
        "compression": args.compression,
        "skipped_duplicate_fens": skipped_duplicate_fens,
        "unique_normalized_fens_written": len(seen_normalized_fens) if args.dedupe_normalized_fen else None,
        "output_paths": {split: str(output_dir / f"split_{split}.parquet") for split in SPLITS},
        "warnings": warnings,
    }
    write_checkpoint(checkpoint_path, report)
    write_json(report, args.report_json)
    write_text(markdown_report(report), args.report_md)
    print(markdown_report(report), flush=True)


if __name__ == "__main__":
    main()
