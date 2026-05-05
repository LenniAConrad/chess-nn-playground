#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.utils.logging import write_json
from chess_nn_playground.utils.paths import utc_timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert plain FEN text files into JSONL raw records.")
    parser.add_argument("--input", nargs="+", required=True, help="FEN text file(s) or folders")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-status", default="known_non_puzzle")
    parser.add_argument("--source-kind", default="crtk_gen_fens")
    args = parser.parse_args()

    inputs: list[Path] = []
    for item in args.input:
        path = Path(item)
        if path.is_file():
            inputs.append(path)
        elif path.is_dir():
            inputs.extend(sorted(path.rglob("*.txt")))
            inputs.extend(sorted(path.rglob("*.fen")))
            inputs.extend(sorted(path.rglob("*.fens")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as out:
        for path in sorted(set(inputs)):
            with path.open("r", encoding="utf-8") as handle:
                for line_index, line in enumerate(handle):
                    fen = line.strip()
                    if not fen or fen.startswith("#"):
                        continue
                    record = {
                        "fen": fen,
                        "label_status": args.label_status,
                        "source_kind": args.source_kind,
                        "source_file": path.name,
                        "source_record_index": line_index,
                    }
                    out.write(json.dumps(record, ensure_ascii=True) + "\n")
                    count += 1
    manifest = {
        "created_at": utc_timestamp(),
        "inputs": [str(path) for path in sorted(set(inputs))],
        "output": str(output),
        "rows": count,
        "label_status": args.label_status,
    }
    write_json(manifest, output.with_suffix(output.suffix + ".manifest.json"))
    print(f"Wrote {count} rows to {output}")


if __name__ == "__main__":
    main()
