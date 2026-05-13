# Configs

Training configs are grouped by benchmark contract. Suites reference configs by repo-relative path, so keep paths current when moving or adding files.

## Folders

- `benchmarks/puzzle_binary/`: current single-logit puzzle benchmark. Fine labels `0` and `1` train as output `0`; fine label `2` trains as output `1`.
- `benchmarks/fine_3class/`: direct 3-class classification of non-puzzle, near-puzzle, and puzzle.
- `benchmarks/coarse_binary/`: older binary architecture comparison configs.
- `suites/`: config lists for `scripts/run_experiment_suite.py`.
- `legacy/`: older configs kept for comparison, smoke tests, or migration reference.
- `project/`: non-training project configs.
- `external/`: local external-engine/CLI configs, not neural-network training configs.

## Main Suites

```text
configs/suites/network_signal_benchmark_suite.yaml
configs/suites/network_signal_fine3_benchmark_suite.yaml
configs/suites/experiment_suite.yaml
```

Use `network_signal_benchmark_suite.yaml` for the current binary puzzle benchmark, `network_signal_fine3_benchmark_suite.yaml` for direct 3-class experiments, and `experiment_suite.yaml` only when you need the older coarse-binary CNN/ResNet comparison.

Run a suite with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_experiment_suite.py \
  --suite configs/suites/network_signal_benchmark_suite.yaml \
  --skip-existing
```

Validate configs before long training:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/validate_training_config.py \
  configs/benchmarks/puzzle_binary/bench_lc0_bt4_classifier.yaml
```

## Required Defaults

Reliable benchmark configs should use:

- `device: nvidia`
- `deterministic: true`
- `training.reliability_tier: paper_grade`
- canonical tagged split paths under `data/splits/crtk_sample_3class_unique_crtk_tags/`
- full artifact validation through the shared trainer

The validator checks encoding/input-channel consistency and catches many config-contract mistakes before training starts.

The paper-ready trunk sweep runner generates seeded derived configs under `reports/paper_ready_all/generated_configs/`; keep hand-authored source configs under `configs/`.
