#!/usr/bin/env python
from __future__ import annotations

import argparse

import sys
from pathlib import Path



from chess_nn_playground.evaluation.reports import build_run_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or rebuild a run report.")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    md_path, html_path = build_run_report(args.run)
    print(f"Saved {md_path}")
    print(f"Saved {html_path}")
    print("Saved reports/latest/latest_report.md")
    print("Saved reports/latest/latest_report.html")


if __name__ == "__main__":
    main()
