#!/usr/bin/env python
from __future__ import annotations

import argparse

import sys
from pathlib import Path



from chess_nn_playground.evaluation.training_plots import build_global_training_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Build global training dashboards from stored run artifacts.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="reports/training")
    parser.add_argument("--max-runs", type=int, default=24)
    args = parser.parse_args()

    artifacts = build_global_training_dashboard(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        max_runs=args.max_runs,
    )
    print(f"Loaded {artifacts['history_rows']} epoch rows from {artifacts['run_count']} runs")
    for key in ["markdown", "html", "runs_csv"]:
        print(f"Saved {artifacts[key]}")
    for path in artifacts["plots"]:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
