# Data

This directory stores local data artifacts. Large Parquet files and generated splits are expected here, but they are not source code.

## Folders

- `raw/`: raw source files.
- `exported/`: CRTK training JSONL exports.
- `processed/`: imported compact Parquet datasets.
- `splits/`: train/validation/test Parquet splits used by configs.
- `reports/`: import, split, audit, and readiness reports.
- `extracted/`: temporary extracted source material.

## Canonical Split

Reliable benchmark and idea configs use:

```text
data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet
```

The CRTK columns in that split are reporting metadata only. Models must not receive CRTK tags, source labels, solution moves, engine PVs, or verification status as input features.

## Rebuild Path

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/data/import_crtk_jsonl_fast.py \
  --input data/exported/crtk_training_20260419_180229.jsonl \
  --output data/processed/crtk_training_20260419_180229_fast.parquet \
  --overwrite

PYTHONDONTWRITEBYTECODE=1 python scripts/data/make_crtk_sample_splits.py \
  --mode fine_3class \
  --input data/processed/crtk_training_20260419_180229_fast.parquet \
  --output-dir data/splits/crtk_sample_3class_unique \
  --max-per-class 150000 \
  --batch-size 200000 \
  --dedupe-normalized-fen \
  --overwrite

PYTHONDONTWRITEBYTECODE=1 python scripts/data/build_crtk_tagged_splits.py \
  --split-dir data/splits/crtk_sample_3class_unique \
  --output-dir data/splits/crtk_sample_3class_unique_crtk_tags \
  --overwrite
```
