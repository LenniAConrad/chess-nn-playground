#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import sys



from scripts.dev.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run checkpoint predictions on a labeled split parquet.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Split parquet path")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate(Path(args.checkpoint), args.name, Path(args.input), args.device)


if __name__ == "__main__":
    main()
