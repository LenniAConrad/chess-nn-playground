# Results

Each training run writes one directory:

```text
results/{timestamp}_{run_name}/
```

Paper-ready batch runs use deterministic task directories under:

```text
results/paper_ready_all/
```

## Completed Run Contract

Only directories containing `metrics_final.json` and passing artifact validation should be used as benchmark evidence. A completed reliable run should include:

```text
checkpoint_best.pt
checkpoint_last.pt
config_resolved.yaml
metrics_train.csv
metrics_val.csv
metrics_final.json
predictions_val.parquet
predictions_test.parquet
slice_report_val.md
slice_report_test.md
training_dashboard.png
run_summary.md
report.html
artifact_manifest.json
```

Validate a run with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/validate_run_artifacts.py results/<run_dir>
```

Historical completed runs before the current artifact contract can be checked with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/validate_run_artifacts.py --allow-legacy results/<run_dir>
```

Interrupted or abandoned directories must contain `INCOMPLETE_RUN.md` and must not be used as benchmark evidence.
