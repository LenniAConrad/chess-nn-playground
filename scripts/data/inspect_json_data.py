#!/usr/bin/env python
from __future__ import annotations

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.data.json_loader import inspect_json_paths, json_audit_markdown
from chess_nn_playground.utils.logging import write_json, write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect unknown JSON/JSONL chess data safely.")
    parser.add_argument("--input", nargs="+", required=True, help="JSON file or folder to inspect")
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--output-json", default="data/reports/json_data_audit.json")
    parser.add_argument("--output-md", default="data/reports/json_data_audit.md")
    args = parser.parse_args()

    report = inspect_json_paths(args.input, sample_size=args.sample_size, max_files=args.max_files)
    write_json(report, args.output_json)
    markdown = json_audit_markdown(report, title="JSON Data Inspection")
    write_text(markdown, args.output_md)
    print(markdown)
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_md}")


if __name__ == "__main__":
    main()
