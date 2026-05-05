#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.data.json_loader import (
    choose_fen,
    find_json_files,
    inspect_json_paths,
    iter_json_records,
    json_audit_markdown,
    truncate_value,
)
from chess_nn_playground.data.schema import decide_label
from chess_nn_playground.utils.logging import write_json, write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw/exported JSON data.")
    parser.add_argument("--input", nargs="*", default=["data/raw", "data/exported"])
    parser.add_argument("--sample-per-file", type=int, default=25)
    args = parser.parse_args()

    report = inspect_json_paths(args.input, sample_size=args.sample_per_file)
    label_status = Counter()
    fen_examples = []
    raw_examples = []
    missing_fen = 0
    total_sampled = 0
    for root in args.input:
        for path in find_json_files(root):
            for record in iter_json_records(path, max_records=args.sample_per_file):
                total_sampled += 1
                fen_key, fen = choose_fen(record.record)
                if fen:
                    fen_examples.append({"source": str(path), "field": fen_key, "fen": fen})
                else:
                    missing_fen += 1
                decision = decide_label(record.record)
                label_status[decision.label_status] += 1
                if len(raw_examples) < 25:
                    raw_examples.append({"source": str(path), "record": truncate_value(record.record)})

    report["data_audit"] = {
        "sampled_records": total_sampled,
        "missing_fen_in_sample": missing_fen,
        "label_status_sample_distribution": dict(label_status),
        "fen_examples": fen_examples[:50],
        "raw_examples": raw_examples,
        "stockfish_pv_node_fields_are_metadata_only": True,
    }
    write_json(report, "data/reports/data_audit.json")
    markdown = json_audit_markdown(report, title="Data Audit")
    markdown += "\n## Label-status sample distribution\n\n"
    for key, value in label_status.items():
        markdown += f"- `{key}`: {value}\n"
    markdown += "\nStockfish/PV/node/engine fields are treated as metadata or targets only, never as model inputs.\n"
    write_text(markdown, "data/reports/data_audit.md")
    print(markdown)


if __name__ == "__main__":
    main()
