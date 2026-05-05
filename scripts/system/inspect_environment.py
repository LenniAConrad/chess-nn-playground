#!/usr/bin/env python
from __future__ import annotations

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.utils.env import collect_environment, environment_markdown
from chess_nn_playground.utils.logging import write_json, write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a local Python, package, CUDA, and mount-point environment report.")
    parser.add_argument("--output-json", default="data/reports/environment_report.json")
    parser.add_argument("--output-md", default="data/reports/environment_report.md")
    args = parser.parse_args()

    report = collect_environment()
    markdown = environment_markdown(report)
    write_json(report, args.output_json)
    write_text(markdown, args.output_md)
    print(markdown)
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_md}")


if __name__ == "__main__":
    main()
