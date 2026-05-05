# Experimental Training Pipeline

This project is set up for repeatable chess puzzle-classification experiments. The current main benchmark is binary classification trained from all three source groups:

- class `0`: known non-puzzle
- class `1`: verified near-puzzle
- class `2`: verified puzzle

The exact benchmark goal and interpretation are written down in:

```text
docs/puzzle_binary_benchmark_goal.md
```

The required budget for reliable future training is written down in:

```text
docs/reliable_training_protocol.md
```

Do not treat 1-epoch smoke runs or 3-epoch triage runs as final evidence. A reliable benchmark run should use the canonical tagged split, the NVIDIA GPU path, a 20-epoch convergence budget, at least 10 active epochs before early stopping, and full artifact validation. Paper-grade claims require matched baselines, repeated seeds, validation-only threshold selection, slice analysis, ablations, and confidence intervals.

The model target is `puzzle_binary`:

- output `0`: predicted non-puzzle
- output `1`: predicted verified puzzle

The training mapping is:

- source class `0` -> output `0`
- source class `1` -> output `0`
- source class `2` -> output `1`

Reports include an extra rectangular `3x2` matrix:

```text
true source class 0/1/2 -> predicted binary output 0/1
```

This shows whether near-puzzles and true puzzles are being treated differently even when the model itself is binary.

## Current Data

Full converted dataset:

```text
data/processed/crtk_training_20260419_180229_fast.parquet
```

Rows:

```text
45,002,737
```

This file is the canonical large dataset. Do not feed it directly into the current pandas-backed trainer.

Canonical RAM-safe source-balanced benchmark splits:

```text
data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet
```

Create or recreate the underlying FEN-deduplicated split with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/data/make_crtk_sample_splits.py \
  --mode fine_3class \
  --input data/processed/crtk_training_20260419_180229_fast.parquet \
  --output-dir data/splits/crtk_sample_3class_unique \
  --max-per-class 150000 \
  --batch-size 200000 \
  --dedupe-normalized-fen \
  --overwrite
```

The split script streams Parquet batches. It does not load the full 45M-row dataset into RAM. Add CRTK reporting tags with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/data/build_crtk_tagged_splits.py \
  --split-dir data/splits/crtk_sample_3class_unique \
  --output-dir data/splits/crtk_sample_3class_unique_crtk_tags \
  --overwrite
```

The tagged columns are for slice reporting only and must not be used as model inputs.

## Train A Benchmark

All benchmark and idea configs should require the NVIDIA GPU path:

```yaml
device: nvidia
```

This resolves to PyTorch CUDA and fails before training if CUDA is not available. Keep `device: cpu` for explicit smoke tests only.

Small puzzle-binary CNN:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py --config configs/benchmarks/puzzle_binary/bench_cnn_signal_simple18.yaml
```

LC0 BT4-style puzzle baseline:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py --config configs/benchmarks/puzzle_binary/bench_lc0_bt4_classifier.yaml
```

List registered model names:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py --list-models
```

## Mass Training

Use the suite runner for architecture batches. It validates configs, runs each training job in a separate subprocess, assigns one visible CUDA GPU per subprocess when multiple GPUs are available, writes one log per run, validates the standard artifact set, then rebuilds the leaderboard.

Dry-run first:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_experiment_suite.py --suite configs/suites/network_signal_benchmark_suite.yaml --dry-run
```

Run the current puzzle-binary suite:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_experiment_suite.py --suite configs/suites/network_signal_benchmark_suite.yaml --skip-existing
```

Run the legacy coarse-binary suite only for the older CNN/ResNet comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_experiment_suite.py --suite configs/suites/experiment_suite.yaml --skip-existing
```

On multi-GPU machines the default is one job per visible CUDA GPU. Override explicitly when needed:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_experiment_suite.py --suite configs/suites/network_signal_benchmark_suite.yaml --skip-existing --jobs 2 --gpu-ids 0,1
```

Logs are written to:

```text
reports/experiment_logs/
```

Suite reports are written to:

```text
reports/experiment_suites/
```

Leaderboards are rebuilt at:

```text
results/leaderboard.csv
results/leaderboard.md
reports/leaderboards/global_leaderboard.md
```

Validate configs without training:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/validate_training_config.py configs/legacy/cnn_crtk_sample.yaml configs/legacy/residual_cnn_crtk_sample.yaml
```

Validate a completed run's required artifacts:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/validate_run_artifacts.py results/<run_dir>
```

## Paper-Ready All-Runner

Use `scripts/run_paper_ready_all.py` when the goal is a resumable analysis batch across every benchmark config and registered idea.

Plan first:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_paper_ready_all.py --dry-run
```

Then run with the intended convergence budget:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_paper_ready_all.py \
  --seeds 42,43,44 \
  --scale-variants base:1,scale_up:1.5,scale_xl:2 \
  --epochs 30 \
  --min-epochs 15 \
  --patience 8 \
  --jobs 1
```

The runner writes generated configs and logs under `reports/paper_ready_all/`, run outputs under `results/paper_ready_all/`, and resume state at `reports/paper_ready_all/state.json`. It also writes a timestamped JSONL event ledger at `reports/paper_ready_all/events.jsonl` and a readable chronology at `reports/paper_ready_all/timeline.md`; the terminal mirrors task start/finish progress with the GPU, batch size, log path, run directory, and rough ETA. It expands every source config into the original `base` architecture plus `scale_up` and `scale_xl` variants by increasing known architecture width, depth, and capacity fields while leaving labels, input encodings, data splits, and the training protocol fixed. The default batch caps are `base:256,scale_up:192,scale_xl:128` for single RTX 3070 readiness. Rerun the same command after an interruption; completed runs are skipped and interrupted training resumes from `checkpoint_last.pt` when possible.

Open `reports/paper_ready_all/status.md` first after a dry run or real run. It summarizes completed, pending, failed, resumable tasks, and ETA once at least one task has finished, plus the relevant logs, leaderboards, training dashboards, and `reports/paper_ready_all/paper_report.pdf`.

At the end of a non-dry run, the runner builds:

```text
results/paper_ready_all/leaderboard.md
results/paper_ready_all/leaderboard_seed_summary.md
reports/paper_ready_all/training/training_dashboard.md
reports/paper_ready_all/training/training_dashboard.html
reports/paper_ready_all/paper_report.pdf
```

The aggregate leaderboard, training-dashboard, and PDF report builders scan nested run directories. A manual `--results-dir results` rebuild therefore includes run folders inside `results/paper_ready_all/`; the mass runner uses `--results-dir results/paper_ready_all` when building the dedicated paper-ready outputs.

## Required Artifacts For Every Run

Every run through `Trainer.fit()` executes the same post-training artifact pipeline. A new architecture gets these automatically as long as it uses the shared trainer:

- `metrics_train.csv`
- `metrics_val.csv`
- `metrics_final.json`
- `complexity_estimate.json`
- `speed_summary.json`
- `artifact_manifest.json`
- `training_dashboard.png`
- individual metric curves such as `loss_curves.png`, `accuracy_curves.png`, and F1 curves
- `confusion_matrix_val.png`
- `confusion_matrix_test.png` when a test split exists
- `fine_to_binary_confusion_matrix_val.png` for binary runs with fine labels
- `fine_to_binary_confusion_matrix_test.png` for binary runs with fine labels and a test split
- `class_distribution.png`
- `calibration_plot.png`
- `predictions_val.parquet`
- `predictions_test.parquet` when a test split exists
- `run_summary.md`
- `report.html`
- `checkpoint_best.pt`
- `checkpoint_last.pt`

If a model trains through the shared trainer, curves and confusion matrices are part of the standard pipeline, not an optional manual step.

Aggregate plots across many completed runs can be rebuilt with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/reports/plot_training_results.py \
  --results-dir results \
  --output-dir reports/training
```

## Add A New Architecture

1. Add a model file under `src/chess_nn_playground/models/`.

The model must be a normal `torch.nn.Module`. It should accept board tensors shaped:

```text
(batch, input_channels, 8, 8)
```

Current benchmark encodings use `18` channels for `simple_18` and `112` channels for `lc0_bt4_112`.

It should return logits shaped:

```text
(batch, num_classes)
```

2. Add a builder function:

```python
def build_my_model_from_config(config: dict[str, Any]) -> nn.Module:
    return MyModel(
        input_channels=int(config.get("input_channels", 18)),
        num_classes=int(config.get("num_classes", 2)),
    )
```

3. Register it in `src/chess_nn_playground/models/registry.py`:

```python
from chess_nn_playground.models.my_model import build_my_model_from_config

MODEL_BUILDERS = {
    ...
    "my_model": build_my_model_from_config,
}
```

4. Create a config:

```yaml
run:
  name: my_model_puzzle_binary
  output_dir: results
seed: 42
deterministic: true
mode: puzzle_binary
device: nvidia
data:
  train_path: data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet
  val_path: data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet
  test_path: data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet
  cache_features: false
model:
  name: my_model
  input_channels: 18
  num_classes: 1
training:
  reliability_tier: paper_grade
  epochs: 20
  min_epochs: 10
  min_active_epochs: 10
  batch_size: 512
  num_workers: auto
  persistent_workers: true
  prefetch_factor: 2
  learning_rate: 0.001
  weight_decay: 0.0001
  class_weighting: balanced
  early_stopping_patience: 5
  gradient_clip_norm: 1.0
  lr_scheduler:
    name: reduce_on_plateau
    factor: 0.5
    patience: 2
    min_lr: 0.00001
  mixed_precision: true
  allow_tf32: true
  matmul_precision: high
```

5. Train it:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py --config configs/benchmarks/puzzle_binary/my_model.yaml
```

For registered research ideas under `ideas/i###_*`, keep the shared `train.py` wrapper. It calls the idea guard, which verifies:

- `idea.yaml` and `config.yaml` refer to the same idea id and slug.
- `config.yaml` uses `device: nvidia`.
- `model.name` is registered before training.
- `implementation_status` is `implemented` or `tested`.

Start model files from `ideas/idea_template/model.py` or the reusable chunks in `chess_nn_playground.models.idea_blocks` so board input shapes and classifier output shapes stay consistent.

## Scaling Rules

For now, keep `cache_features: false`. The current trainer loads split tables into pandas, so do not point it at the 45M-row file.

Scale in this order:

1. Run a small sample and verify artifacts.
2. Increase `--max-per-class`.
3. Add a streaming dataset/trainer before using the full 45M rows directly.

Use `mode: fine_3class` and `num_classes: 3` only when the architecture should directly predict non-puzzle vs near-puzzle vs puzzle. The default benchmark is binary output plus rectangular 3x2 source-class diagnostics.

This keeps experiments reproducible and avoids RAM failures while the architecture search pipeline is still evolving.

## Prepared Benchmark Configs

The current single-logit puzzle benchmark suite compares:

```text
configs/benchmarks/puzzle_binary/bench_nnue_simple18.yaml
configs/benchmarks/puzzle_binary/bench_mlp_simple18.yaml
configs/benchmarks/puzzle_binary/bench_cnn_signal_simple18.yaml
configs/benchmarks/puzzle_binary/bench_lc0_bt4_classifier.yaml
```

The 3-class benchmark suite compares:

```text
configs/benchmarks/fine_3class/bench_fine3_nnue_simple18.yaml
configs/benchmarks/fine_3class/bench_fine3_mlp_simple18.yaml
configs/benchmarks/fine_3class/bench_fine3_cnn_simple18.yaml
configs/benchmarks/fine_3class/bench_fine3_lc0_bt4_classifier.yaml
```

The legacy coarse-binary suite compares:

```text
configs/benchmarks/coarse_binary/bench_cnn_small_simple18.yaml
configs/benchmarks/coarse_binary/bench_cnn_medium_simple18.yaml
configs/benchmarks/coarse_binary/bench_cnn_deep_simple18.yaml
configs/benchmarks/coarse_binary/bench_residual_small_simple18.yaml
configs/benchmarks/coarse_binary/bench_residual_medium_simple18.yaml
configs/benchmarks/coarse_binary/bench_residual_deep_simple18.yaml
configs/benchmarks/coarse_binary/bench_cnn_small_lc0bt4.yaml
configs/benchmarks/coarse_binary/bench_residual_small_lc0bt4.yaml
```

The `simple_18` configs use the current compact FEN tensor:

```text
12 piece planes + side to move + 4 castling planes + en-passant plane
```

The `lc0_bt4_112` configs use an LC0 BT4-style 112-plane tensor:

```text
8 history slots * 13 planes + 8 auxiliary planes
```

Each history slot is:

```text
our pawns, knights, bishops, rooks, queens, king
their pawns, knights, bishops, rooks, queens, king
repetition
```

The auxiliary planes follow the modern LC0 112-plane castling/en-passant layout:

```text
queenside castling rook squares
kingside castling rook squares
zero
zero
en-passant file
halfmove clock / 100
zero
all ones
```

Because the current CRTK export contains standalone FENs and no move history, only the current-position history slot is populated. Older history and repetition planes are zero. The tensor is from the side-to-move perspective, including rank mirroring for black-to-move positions. Treat this as a BT4-style FEN benchmark until the exporter writes move history. It is for training new models from scratch, not for loading existing LC0 weights.

The older `lc0_static_112` encoding remains available for comparison, but the default suite uses `lc0_bt4_112`.
