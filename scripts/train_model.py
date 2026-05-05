#!/usr/bin/env python
from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.models.registry import available_models
from chess_nn_playground.training.trainer import train_from_config
from chess_nn_playground.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a registered chess puzzle-classification model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--list-models", action="store_true", help="Print registered model names and exit")
    args = parser.parse_args()
    if args.list_models:
        for name in available_models():
            print(name)
        return
    if not args.config:
        raise SystemExit("--config is required unless --list-models is used")
    config = load_yaml(args.config)
    run_dir = train_from_config(config)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
