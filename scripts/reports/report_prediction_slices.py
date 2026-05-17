#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys



from chess_nn_playground.evaluation.slices import write_slice_artifacts


DEFAULT_TAGGED_SPLIT_DIR = Path("data/splits/crtk_sample_3class_unique_crtk_tags")


def _run_split_path(run_dir: Path, split: str) -> Path | None:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = metadata.get("split_paths", {}).get(split)
    return Path(value) if value else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Join predictions with CRTK tags and report benchmark slices.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--tagged-split-dir",
        type=Path,
        default=None,
        help="Optional override. By default, use split paths recorded in run_metadata.json.",
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    for split in args.splits:
        pred_path = args.run_dir / f"predictions_{split}.parquet"
        split_path = _run_split_path(args.run_dir, split)
        if args.tagged_split_dir is not None:
            split_path = args.tagged_split_dir / f"split_{split}.parquet"
        if split_path is None:
            split_path = DEFAULT_TAGGED_SPLIT_DIR / f"split_{split}.parquet"
        if not pred_path.exists():
            print(f"Skipping missing {pred_path}")
            continue
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        artifacts = write_slice_artifacts(
            run_dir=args.run_dir,
            split=split,
            pred_path=pred_path,
            split_path=split_path,
            min_count=args.min_count,
            limit=args.limit,
        )
        if artifacts is None:
            print(f"Skipping {pred_path}; no CRTK slice metadata found")
            continue
        print(f"Saved {artifacts['predictions']}")
        print(f"Saved {artifacts['report']}")
        print(f"Saved {artifacts['metrics']}")


if __name__ == "__main__":
    main()
