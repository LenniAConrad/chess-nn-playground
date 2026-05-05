# Export Training Data from Stacks with CRTK

This is the short path from read-only CRTK stacks to the local split used by the trainer. Run commands from the repo root.

## 1. Point to the Stack and CRTK

```bash
export STACKS="/path/to/read-only/stacks"
export PATH="/path/to/crtk/bin:$PATH"
export CRTK_JAR="/path/to/chess-rtk/crtk.jar"
```

`crtk` must be on `PATH`. `CRTK_JAR` is only needed for the final tagging step.

## 2. Export JSONL with CRTK

Use the wrapper so the command, output path, return code, and log are saved under `data/exported/` and `data/reports/`.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/data/export_with_crtk.py \
  --input "$STACKS" \
  --output data/exported \
  --suffix .jsonl \
  --command-template '{crtk} record-to-training-jsonl --input {input} --output {output} --recursive --label-mode explicit --include-engine-metadata --include-raw-record-id --manifest {output}.manifest.json'
```

Outputs to check:

```text
data/exported/crtk_export_<timestamp>.jsonl
data/exported/export_manifest.json
data/exported/crtk_export_command.log
data/reports/export_report.md
```

The JSONL export must follow [crtk_export_contract.md](crtk_export_contract.md). In particular, engine metadata, source labels, solution moves, and verification fields are metadata only. They must not become model input features.

If the local CRTK build exposes a different export command, keep the same wrapper and adjust only `--command-template`. The command must still write one training record per JSONL line.

## 3. Import JSONL to Parquet

Replace `<export>` with the JSONL file created above.

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/data/import_crtk_jsonl_fast.py \
  --input data/exported/<export>.jsonl \
  --output data/processed/crtk_training_latest_fast.parquet \
  --report-json data/reports/crtk_training_latest_import_report.json \
  --report-md data/reports/crtk_training_latest_import_report.md \
  --overwrite
```

## 4. Build the Sampled Split

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/data/make_crtk_sample_splits.py \
  --mode fine_3class \
  --input data/processed/crtk_training_latest_fast.parquet \
  --output-dir data/splits/crtk_sample_3class_unique \
  --max-per-class 150000 \
  --batch-size 200000 \
  --dedupe-normalized-fen \
  --report-json data/reports/crtk_sample_3class_unique_split_report.json \
  --report-md data/reports/crtk_sample_3class_unique_split_report.md \
  --overwrite
```

## 5. Add CRTK Tags for Slice Reports

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/data/build_crtk_tagged_splits.py \
  --split-dir data/splits/crtk_sample_3class_unique \
  --output-dir data/splits/crtk_sample_3class_unique_crtk_tags \
  --crtk-jar "$CRTK_JAR" \
  --report-path data/reports/crtk_sample_3class_unique_crtk_tagged_report.md \
  --overwrite
```

These tag columns are for reporting and slice analysis only. The neural network encoders ignore them.

## 6. Audit Before Training

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/data/audit_benchmark_data.py \
  --split-dir data/splits/crtk_sample_3class_unique_crtk_tags \
  --report-md data/reports/benchmark_data_audit.md \
  --report-json data/reports/benchmark_data_audit.json
```

The trainer-ready paths are:

```text
data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet
```

Do not commit `data/exported/`, `data/processed/`, `data/splits/`, `data/reports/`, or any copied training data. They stay local by design.

## Fallback for Existing JSON or JSONL Stacks

If the stack files are already JSON or JSONL and CRTK is not available, the direct importer can create a canonical Parquet file with resume support:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/data/import_usb_stacks.py \
  --input "$STACKS" \
  --output data/processed/usb_all_positions.parquet \
  --rejected-output data/processed/usb_all_rejected_positions.parquet
```

Use this fallback only for raw stack ingestion. For the benchmark path, prefer the CRTK JSONL export and then run steps 3 through 6.
