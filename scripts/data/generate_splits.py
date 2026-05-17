#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys



from chess_nn_playground.data.split_utils import (
    assign_group_splits,
    filter_for_mode,
    split_manifest,
    write_split_report,
)
from chess_nn_playground.utils.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leakage-aware train/val/test splits.")
    parser.add_argument("--input", default="data/processed/positions.parquet")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--mode", default="coarse_binary", choices=["coarse_binary", "fine_3class", "class0_only_audit"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument(
        "--report-md",
        default="data/reports/split_report.md",
        help="Markdown report path for the generated split manifest",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    filtered = filter_for_mode(df, args.mode)
    warning = None
    if args.mode == "fine_3class":
        present = set(filtered["fine_label"].dropna().astype(int).tolist()) if not filtered.empty else set()
        if not {1, 2}.issubset(present):
            warning = "fine_3class is not yet possible because verified class 1 and/or class 2 samples are missing."
    if filtered.empty:
        split_df = filtered.copy()
        split_df["split"] = []
    else:
        split_df = assign_group_splits(
            filtered,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": str(output_dir / "split_train.parquet"),
        "val": str(output_dir / "split_val.parquet"),
        "test": str(output_dir / "split_test.parquet"),
    }
    for split, path in paths.items():
        split_df[split_df.get("split", pd.Series(dtype=str)) == split].to_parquet(path, index=False)
    manifest = split_manifest(split_df, args.mode, args.seed, paths)
    manifest["warning"] = warning
    write_json(manifest, output_dir / "split_manifest.json")
    write_split_report(manifest, args.report_md)
    if warning:
        with open(args.report_md, "a", encoding="utf-8") as handle:
            handle.write(f"\n## Warning\n\n{warning}\n")
    print(f"Saved splits to {output_dir}")
    if warning:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
