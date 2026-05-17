#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import sys




def run(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect mounted USB JSON data, prepare canonical Parquet, and generate coarse splits."
    )
    parser.add_argument("--usb-path", required=True, help="Mounted read-only USB path or JSON folder")
    parser.add_argument("--copy-raw", action="store_true", help="Copy JSON files into data/raw/usb_copy_<timestamp>")
    parser.add_argument("--mode", default="coarse_binary", choices=["coarse_binary", "fine_3class", "class0_only_audit"])
    parser.add_argument("--default-label-status", default=None)
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=20000)
    parser.add_argument("--max-records-per-file", type=int, default=None)
    args = parser.parse_args()

    source = Path(args.usb_path)
    if not source.exists():
        raise SystemExit(f"USB path does not exist: {source}")

    inspect_cmd = [
        sys.executable,
        "-m",
        "scripts.data.inspect_json_data",
        "--input",
        str(source),
        "--max-files",
        str(args.max_files),
    ]
    run(inspect_cmd)
    run([sys.executable, "-m", "scripts.data.data_audit", "--input", str(source), "--sample-per-file", "5"])
    prepare_cmd = [
        sys.executable,
        "-m",
        "scripts.data.prepare_dataset",
        "--input",
        str(source),
        "--output",
        "data/processed/positions.parquet",
        "--rejected-output",
        "data/processed/rejected_positions.parquet",
        "--max-files",
        str(args.max_files),
        "--max-records",
        str(args.max_records),
    ]
    if args.max_records_per_file is not None:
        prepare_cmd.extend(["--max-records-per-file", str(args.max_records_per_file)])
    if args.default_label_status:
        prepare_cmd.extend(["--default-label-status", args.default_label_status])
    run(prepare_cmd)
    run([sys.executable, "-m", "scripts.data.generate_splits", "--mode", args.mode])


if __name__ == "__main__":
    main()
