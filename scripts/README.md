# Scripts

Install the repo in editable mode and use the console commands exposed by `pyproject.toml`:

```bash
pip install -e .[dev]
```

Use `PYTHONDONTWRITEBYTECODE=1` for long jobs or catalog work if you want to avoid `__pycache__` churn.

## Primary Entrypoints

- `chess-nn-train`: train one YAML config.
- `chess-nn-run-suite`: run a suite or explicit config list, validate artifacts, and rebuild leaderboards.
- `chess-nn-paper-ready`: run all benchmark configs and registered ideas with resumable paper-ready settings.
- `chess-nn-compare-results`: rebuild leaderboards and comparison summaries from completed results.
- `chess-nn-validate-config`: statically validate config files.
- `chess-nn-validate-run`: verify a completed run has the required metrics, plots, predictions, summaries, and checkpoints.
- `chess-nn-build-flop-report`: build a tiny FLOP-only architecture report from configs without training or dataset validation.
- `chess-nn-build-paper-report`: compile completed results, planned tasks, and architecture explanations into a multi-page PDF paper report.
- `chess-nn-build-idea-catalog`: regenerate idea indexes, research packet catalogs, and TODOs.
- `chess-nn-audit-ideas`: classify idea folders as bespoke models, shared ResearchPacketProbe variants, other shared scaffolds, or unknown, and validate metadata agreement.
- `chess-nn-audit-architectures`: validate rows marked `implemented` or `tested` are bespoke, registered, documented, and free of obvious shell markers.

## Utility Folders

- `data/`: import CRTK exports, build Parquet datasets, create splits, audit data, and add CRTK slice tags.
- `reports/`: build run reports, prediction slice reports, and aggregate training plots.
- `ideas/`: regenerate idea catalogs, TODOs, prompts, and promote research packets.
- `system/`: inspect the local environment and mount USB sources.
- `dev/`: older smoke/evaluation helpers that are not the canonical benchmark path.
- `agents/`: experimental local Claude/Codex automation. These scripts are not console entrypoints, not part of CI, and should be run only from explicit dry-run reviewed commands or disposable worktrees.

## Common Commands

```bash
chess-nn-train \
  --config configs/benchmarks/puzzle_binary/bench_lc0_bt4_classifier.yaml

chess-nn-run-suite \
  --suite configs/suites/network_signal_benchmark_suite.yaml \
  --dry-run

chess-nn-paper-ready \
  --seeds 42,43,44 \
  --scale-variants base:1,scale_up:1.5,scale_xl:2 \
  --epochs 30 \
  --min-epochs 15 \
  --patience 8 \
  --jobs 1

chess-nn-plot-training \
  --results-dir results \
  --output-dir reports/training

chess-nn-build-paper-report \
  --results-dir results \
  --state-path reports/paper_ready_all/state.json \
  --output reports/paper_ready_all/paper_report.pdf

chess-nn-build-flop-report \
  --output-dir reports/flops

chess-nn-audit-ideas \
  --sync-metadata --check

chess-nn-audit-architectures \
  --check
```

## CI Gate

The GitHub Actions workflow in `.github/workflows/ci.yml` installs `.[dev]` on Python 3.12 and runs:

- `ruff check .`
- `python -m compileall -q src scripts tests`
- package metadata and console-entrypoint smoke checks
- a stable CPU unit/regression pytest subset
- `tests/test_training_smoke.py`, which trains tiny CPU runs and validates their artifacts

The full pytest suite is intentionally not the required CI gate yet because the current repository has known registry/report/result backlog failures unrelated to the CPU smoke path.

## Artifact Contract

The shared trainer is responsible for graphable run artifacts, including `complexity_estimate.json` for estimated inference MACs/FLOPs and `speed_summary.json` for throughput and elapsed-time analysis. `speed_summary.json` includes clean synthetic forward-pass inference timings for CPU and CUDA, separate from training and dataloader-backed evaluation speed. Do not add model-specific one-off report formats unless the standard artifacts still exist and validate.

For interrupted paper-ready batches, rerun the same `chess-nn-paper-ready` command. The default paper-ready runner now expands each source config into `base`, `scale_up`, and `scale_xl` architecture-size variants. The resume ledger is `reports/paper_ready_all/state.json`, and training resumes from `checkpoint_last.pt` when that checkpoint exists.

The trunk pipeline runner also writes `reports/paper_ready_all/status.md`; open that first to see task counts, ETA after the first completed task, failures, next tasks, logs, leaderboards, and training-dashboard paths. During execution the terminal prints numbered task start/finish lines with GPU, batch size, log path, run directory, and rough ETA. The same chronology is persisted to `reports/paper_ready_all/events.jsonl` and `reports/paper_ready_all/timeline.md`.

Default paper-ready batch caps are `base:256,scale_up:192,scale_xl:128`, chosen for a single 8GB RTX 3070. Override with `--batch-size-caps none` or custom `scale:max_batch` values only after memory calibration.

The comparison, training-dashboard, and PDF report scripts scan nested run directories. Running them with `--results-dir results` includes paper-ready runs under `results/paper_ready_all/`.
