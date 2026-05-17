#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import resource
import sys


from chess_nn_playground.models.registry import available_models
from chess_nn_playground.training.trainer import train_from_config
from chess_nn_playground.utils.config import load_yaml


def _apply_opt_in_ram_cap() -> None:
    cap_env = os.environ.get("CHESS_NN_MAX_RAM_BYTES", "").strip()
    if not cap_env:
        return
    try:
        cap_bytes = int(cap_env)
    except ValueError:
        print(f"[ram-cap] invalid CHESS_NN_MAX_RAM_BYTES={cap_env!r}, ignoring", file=sys.stderr)
        return
    if cap_bytes <= 0:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    except (ValueError, OSError):
        return
    new_hard = hard if hard != resource.RLIM_INFINITY and hard < cap_bytes else cap_bytes
    try:
        resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, new_hard))
    except (ValueError, OSError) as exc:
        print(f"[ram-cap] failed to set RLIMIT_AS={cap_bytes}: {exc}", file=sys.stderr)
        return
    print(f"[ram-cap] RLIMIT_AS set to {cap_bytes} bytes (opt-in via CHESS_NN_MAX_RAM_BYTES)", file=sys.stderr)


def main() -> None:
    _apply_opt_in_ram_cap()
    parser = argparse.ArgumentParser(description="Train a registered chess puzzle-classification model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--list-models", action="store_true", help="Print registered model names and exit")
    parser.add_argument(
        "--allow-cpu-oom-fallback",
        action="store_true",
        help="After a CUDA OOM, retry on CPU and label the run as cpu_oom_fallback_non_benchmark.",
    )
    args = parser.parse_args()
    if args.list_models:
        for name in available_models():
            print(name)
        return
    if not args.config:
        raise SystemExit("--config is required unless --list-models is used")
    config = load_yaml(args.config)
    if args.allow_cpu_oom_fallback:
        training_cfg = config.setdefault("training", {})
        if not isinstance(training_cfg, dict):
            raise SystemExit("training must be a mapping to use --allow-cpu-oom-fallback")
        training_cfg["allow_cpu_oom_fallback"] = True
    try:
        run_dir = train_from_config(config)
    except MemoryError as exc:
        print(f"[ram-cap] MemoryError aborting task: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
