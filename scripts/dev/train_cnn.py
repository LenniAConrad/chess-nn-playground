#!/usr/bin/env python
from __future__ import annotations

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from chess_nn_playground.training.trainer import train_from_config
from chess_nn_playground.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the simple CNN baseline.")
    parser.add_argument("--config", default="configs/_archive/legacy/cnn_baseline.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    run_dir = train_from_config(config)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
