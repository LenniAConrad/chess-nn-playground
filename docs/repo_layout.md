# Repository Layout

This repo is structured as an experiment harness. Keep source code, configs, generated artifacts, and research notes separated so benchmark claims remain reproducible.

## Source And Tests

- `src/chess_nn_playground/`: importable package for data loading, board encodings, model registries, training, evaluation, and reporting.
- `pyproject.toml`: pytest and package-discovery config for the `src/` layout.
- `tests/`: unit tests, training smoke tests, config contracts, idea registry checks, and benchmark contract tests.
- `scripts/`: stable command-line entrypoints plus grouped utility folders.

## Configs

- `configs/benchmarks/puzzle_binary/`: current single-logit puzzle-vs-non-puzzle benchmark configs.
- `configs/benchmarks/fine_3class/`: 3-class non-puzzle / near-puzzle / puzzle benchmark configs.
- `configs/benchmarks/coarse_binary/`: older coarse-binary architecture comparison configs.
- `configs/suites/`: suite YAML files consumed by `scripts/run_experiment_suite.py`.
- `configs/project/`: non-training project configs.
- `configs/external/`: external engine/CLI configs that are not part of neural-network training.

## Scripts

- `scripts/train_model.py`: train one config through the shared trainer.
- `scripts/run_experiment_suite.py`: run a suite or list of configs with validation and leaderboard rebuilds.
- `scripts/run_paper_ready_all.py`: run every benchmark config and registered idea with resumable paper-ready defaults.
- `scripts/compare_results.py`: rebuild leaderboards and comparison summaries.
- `scripts/validate_training_config.py`: statically validate one or more configs.
- `scripts/validate_run_artifacts.py`: validate a completed run directory.
- `scripts/data/`: import, split, audit, and CRTK tagging utilities.
- `scripts/reports/`: plotting and report-generation utilities.
- `scripts/ideas/`: idea catalog and research-packet utilities.
- `scripts/system/`: local environment and mount helpers.
- `scripts/dev/`: older developer-only smoke/evaluation helpers.

## Data And Artifacts

- `data/processed/`: imported Parquet datasets.
- `data/splits/`: train/validation/test Parquet splits. Reliable benchmark configs use `data/splits/crtk_sample_3class_unique_crtk_tags/`.
- `data/reports/`: data import, split, and readiness reports.
- `results/`: run directories, checkpoints, metrics, predictions, plots, and local leaderboards.
- `reports/leaderboards/`: global leaderboard and seed summaries generated from completed result directories.
- `reports/latest/`: latest-run markdown and HTML snapshots.
- `reports/prompts/`: generated analysis handoff prompts.
- `reports/training/`: aggregate training dashboards and plots.
- `reports/experiment_logs/` and `reports/experiment_suites/`: suite subprocess logs, suite summaries, and generated seeded configs.
- `reports/paper_ready_all/`: resumable all-benchmarks/all-ideas plans, logs, generated configs, and state when that runner is used.
- `reports/archive/`: old one-off reports kept for provenance.
- `docs/`: stable protocols and reference documentation. Start at `docs/README.md`.

## Research Ideas

- `ideas/registry/i###_*/`: one registered idea per folder, with `idea.yaml`, math/architecture notes, `model.py`, `train.py`, `config.yaml`, report template, and run notes.
- `ideas/research/packets/classic/`: raw research packets kept for provenance and duplicate prevention.
- `ideas/registry/INDEX.md`: generated idea map.
- `ideas/registry/TODO.md`: generated idea backlog and benchmark queue.
- `ideas/docs/BENCHMARK_REPORTING.md`: required slice analysis for idea runs.

## Path Stability Rules

- Keep top-level `ideas/`, `data/`, and `results/` paths stable. Many configs and reports point to them.
- Add new benchmark configs under the appropriate `configs/benchmarks/<task>/` folder.
- Add new grouped scripts under the nearest `scripts/<purpose>/` folder, but keep primary entrypoints at `scripts/`.
- Do not use CRTK tags, source labels, tactic tags, solution moves, or verification metadata as model inputs. They are reporting fields only.
