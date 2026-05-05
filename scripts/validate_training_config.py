#!/usr/bin/env python
from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.training.config_validation import validate_training_config
from chess_nn_playground.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one or more training configs before running experiments.")
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--strict-warnings", action="store_true", help="Treat warnings as failures")
    parser.add_argument(
        "--static",
        action="store_true",
        help="Validate config structure without requiring the requested device to be available.",
    )
    args = parser.parse_args()

    failed = False
    for item in args.configs:
        messages = validate_training_config(load_yaml(item), item, require_device_available=not args.static)
        if not messages:
            print(f"OK: {item}")
            continue
        print(f"{item}:")
        for message in messages:
            print(f"  {message}")
        if any(message.startswith("ERROR:") for message in messages):
            failed = True
        if args.strict_warnings and any(message.startswith("WARNING:") for message in messages):
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
