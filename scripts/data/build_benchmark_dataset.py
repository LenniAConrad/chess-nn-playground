#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys



from chess_nn_playground.utils.logging import write_json, write_text
from chess_nn_playground.utils.paths import utc_timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced coarse_binary benchmark dataset.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--non-puzzles", required=True)
    parser.add_argument("--output", default="data/processed/benchmark_coarse_binary.parquet")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidates = pd.read_parquet(args.candidates)
    non_puzzles = pd.read_parquet(args.non_puzzles)
    candidates = candidates[candidates["coarse_label"] == 1].copy()
    non_puzzles = non_puzzles[non_puzzles["coarse_label"] == 0].copy()

    candidates = candidates.drop_duplicates("normalized_fen")
    non_puzzles = non_puzzles.drop_duplicates("normalized_fen")
    overlap = set(candidates["normalized_fen"]).intersection(set(non_puzzles["normalized_fen"]))
    if overlap:
        candidates = candidates[~candidates["normalized_fen"].isin(overlap)].copy()
        non_puzzles = non_puzzles[~non_puzzles["normalized_fen"].isin(overlap)].copy()

    per_class = min(len(candidates), len(non_puzzles))
    if args.max_per_class is not None:
        per_class = min(per_class, args.max_per_class)
    if per_class == 0:
        raise SystemExit("Cannot build benchmark: one class has zero rows.")

    candidates = candidates.sample(n=per_class, random_state=args.seed)
    non_puzzles = non_puzzles.sample(n=per_class, random_state=args.seed)
    combined = pd.concat([non_puzzles, candidates], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    report = {
        "created_at": utc_timestamp(),
        "candidate_path": args.candidates,
        "non_puzzle_path": args.non_puzzles,
        "output": str(output),
        "rows": int(len(combined)),
        "per_class": int(per_class),
        "overlap_removed": int(len(overlap)),
        "label_status_distribution": combined["label_status"].value_counts(dropna=False).to_dict(),
        "coarse_label_distribution": combined["coarse_label"].value_counts(dropna=False).to_dict(),
    }
    write_json(report, "data/reports/benchmark_dataset_report.json")
    lines = [
        "# Benchmark Dataset Report",
        "",
        f"- Output: `{output}`",
        f"- Rows: `{len(combined)}`",
        f"- Per class: `{per_class}`",
        f"- Overlap removed: `{len(overlap)}`",
        "",
        "## Coarse-label distribution",
        "",
    ]
    for key, value in report["coarse_label_distribution"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Label-status distribution", ""])
    for key, value in report["label_status_distribution"].items():
        lines.append(f"- `{key}`: {value}")
    write_text("\n".join(lines), "data/reports/benchmark_dataset_report.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
