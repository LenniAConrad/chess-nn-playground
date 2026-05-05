# Scripts

Run scripts from the repo root. Use `PYTHONDONTWRITEBYTECODE=1` for long jobs or catalog work if you want to avoid `__pycache__` churn.

## Primary Entrypoints

- `train_model.py`: train one YAML config.
- `run_experiment_suite.py`: run a suite or explicit config list, validate artifacts, and rebuild leaderboards.
- `run_paper_ready_all.py`: run all benchmark configs and registered ideas with resumable paper-ready settings.
- `compare_results.py`: rebuild leaderboards and comparison summaries from completed results.
- `validate_training_config.py`: statically validate config files.
- `validate_run_artifacts.py`: verify a completed run has the required metrics, plots, predictions, summaries, and checkpoints.
- `reports/build_flop_report.py`: build a tiny FLOP-only architecture report from configs without training or dataset validation.
- `reports/build_paper_report.py`: compile completed results, planned tasks, and architecture explanations into a multi-page PDF paper report.

## Utility Folders

- `data/`: import CRTK exports, build Parquet datasets, create splits, audit data, and add CRTK slice tags.
- `reports/`: build run reports, prediction slice reports, and aggregate training plots.
- `ideas/`: regenerate idea catalogs, TODOs, prompts, and promote research packets.
- `system/`: inspect the local environment and mount USB sources.
- `dev/`: older smoke/evaluation helpers that are not the canonical benchmark path.

## Common Commands

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py \
  --config configs/benchmarks/puzzle_binary/bench_lc0_bt4_classifier.yaml

PYTHONDONTWRITEBYTECODE=1 python scripts/run_experiment_suite.py \
  --suite configs/suites/network_signal_benchmark_suite.yaml \
  --dry-run

PYTHONDONTWRITEBYTECODE=1 python scripts/run_paper_ready_all.py \
  --seeds 42,43,44 \
  --scale-variants base:1,scale_up:1.5,scale_xl:2 \
  --epochs 30 \
  --min-epochs 15 \
  --patience 8 \
  --jobs 1

PYTHONDONTWRITEBYTECODE=1 python scripts/reports/plot_training_results.py \
  --results-dir results \
  --output-dir reports/training

PYTHONDONTWRITEBYTECODE=1 python scripts/reports/build_paper_report.py \
  --results-dir results \
  --state-path reports/paper_ready_all/state.json \
  --output reports/paper_ready_all/paper_report.pdf

PYTHONDONTWRITEBYTECODE=1 python scripts/reports/build_flop_report.py \
  --output-dir reports/flops
```

## Artifact Contract

The shared trainer is responsible for graphable run artifacts, including `complexity_estimate.json` for estimated inference MACs/FLOPs and `speed_summary.json` for throughput and elapsed-time analysis. Do not add model-specific one-off report formats unless the standard artifacts still exist and validate.

For interrupted paper-ready batches, rerun the same `run_paper_ready_all.py` command. The default paper-ready runner now expands each source config into `base`, `scale_up`, and `scale_xl` architecture-size variants. The resume ledger is `reports/paper_ready_all/state.json`, and training resumes from `checkpoint_last.pt` when that checkpoint exists.

The all-runner also writes `reports/paper_ready_all/status.md`; open that first to see task counts, ETA after the first completed task, failures, next tasks, logs, leaderboards, and training-dashboard paths. During execution the terminal prints numbered task start/finish lines with GPU, batch size, log path, run directory, and rough ETA. The same chronology is persisted to `reports/paper_ready_all/events.jsonl` and `reports/paper_ready_all/timeline.md`.

Default paper-ready batch caps are `base:256,scale_up:192,scale_xl:128`, chosen for a single 8GB RTX 3070. Override with `--batch-size-caps none` or custom `scale:max_batch` values only after memory calibration.

The comparison, training-dashboard, and PDF report scripts scan nested run directories. Running them with `--results-dir results` includes paper-ready runs under `results/paper_ready_all/`.
