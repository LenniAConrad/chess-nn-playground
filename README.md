# Chess NN Playground

Chess NN Playground is an experiment harness for chess puzzle classification. It imports CRTK exports, builds reproducible train/validation/test splits, trains many neural architectures through one shared trainer, and stores enough artifacts to compare runs without guessing.

The project rule is simple: benchmark results should be comparable. Baselines, new architectures, registered research ideas, reports, plots, predictions, and leaderboards all use the same data contract and artifact contract.

## What worked: read this first

- **Paper-quality report (EN):** [reports/audits/paper_report.pdf](reports/audits/paper_report.pdf) — 234-architecture single-seed scout, per-class heatmap, confusion matrices, i242 ablations, i243 proposal.
- **Paper-quality report (中文):** [reports/audits/paper_report_zh.pdf](reports/audits/paper_report_zh.pdf).
- **All scout runs in one file:** [reports/audits/scout_all_runs.json](reports/audits/scout_all_runs.json) / [.csv](reports/audits/scout_all_runs.csv) — one row per run with params, FLOPs, test PR AUC, throughput, sorted by leaderboard rank.
- **Per-class slice metrics:** [reports/audits/per_class_benchmark.json](reports/audits/per_class_benchmark.json) and matched-recall hard-negative FP at [reports/audits/matched_recall_fp_report.json](reports/audits/matched_recall_fp_report.json).
- **Models:** every bespoke architecture lives in [src/chess_nn_playground/models/](src/chess_nn_playground/models/), with the idea registry and design notes under [ideas/](ideas/).

## Current Benchmark

The main task is `puzzle_binary`:

- output `0`: non-puzzle
- output `1`: verified puzzle

The source data still has three fine labels:

- fine label `0`: known non-puzzle
- fine label `1`: verified near-puzzle
- fine label `2`: verified puzzle

For binary training, labels `0` and `1` map to output `0`; label `2` maps to output `1`. Reports include a rectangular `3x2` diagnostic matrix so near-puzzle behavior remains visible even when the model has only two outputs.

The exact benchmark target and interpretation live in [docs/puzzle_binary_benchmark_goal.md](docs/puzzle_binary_benchmark_goal.md).

## Quick Start

Create an environment and install dependencies:

```bash
git clone https://github.com/LenniAConrad/chess-nn-playground.git
cd chess-nn-playground
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

List registered model keys:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py --list-models
```

Validate the current LC0 BT4-style benchmark config:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/validate_training_config.py \
  configs/benchmarks/puzzle_binary/bench_lc0_bt4_classifier.yaml
```

Run one benchmark config:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py \
  --config configs/benchmarks/puzzle_binary/bench_cnn_signal_simple18.yaml
```

Run tests:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q
```

## Documentation Map

- [docs/README.md](docs/README.md): documentation index and maintenance rules.
- [docs/repo_layout.md](docs/repo_layout.md): current folder layout.
- [docs/experimental_training_pipeline.md](docs/experimental_training_pipeline.md): data, training, artifacts, encodings, and new-model workflow.
- [docs/reliable_training_protocol.md](docs/reliable_training_protocol.md): smoke, triage, reliable, promotion-grade, and paper-grade run standards.
- [docs/crtk_export_contract.md](docs/crtk_export_contract.md): expected CRTK export format.
- [docs/export_training_data_from_stacks.md](docs/export_training_data_from_stacks.md): short guide for exporting stack data with CRTK and rebuilding trainer-ready splits.
- [ideas/README.md](ideas/README.md): idea workspace guide.
- [ideas/registry/TODO.md](ideas/registry/TODO.md): generated implementation and benchmark backlog.
- [configs/README.md](configs/README.md): config folders and suite entrypoints.
- [scripts/README.md](scripts/README.md): command entrypoints and utility folders.

## Repository Layout

```text
src/chess_nn_playground/      Importable Python package
configs/benchmarks/           Benchmark configs grouped by task
configs/suites/               Multi-config benchmark suites
configs/_archive/legacy/               Older configs kept for comparison or smoke work
scripts/                      Stable training, validation, suite, and reporting entrypoints
scripts/data/                 CRTK import, audit, split, and tagging utilities
scripts/reports/              Plotting and report-generation utilities
scripts/ideas/                Idea catalog, prompt, and promotion utilities
ideas/                        Registered ideas and raw research packets
data/                         Local datasets, split Parquet files, and data reports
results/                      Run directories, checkpoints, metrics, predictions, plots
reports/                      Global leaderboards, suite logs, prompts, and training plots
docs/                         Protocols and reference documentation
tests/                        Unit, contract, and smoke tests
```

The package import name remains `chess_nn_playground`; only the filesystem location uses the normal `src/` layout.

## Data

Training data, split Parquet files, checkpoints, predictions, and generated run
artifacts are intentionally not committed to this repository. The trainer expects
local data under `data/`.

For the full stack-to-split workflow, use [docs/export_training_data_from_stacks.md](docs/export_training_data_from_stacks.md).

The expected local full imported dataset path is:

```text
data/processed/crtk_training_20260419_180229_fast.parquet
```

Do not point the pandas-backed trainer directly at the full 45M-row Parquet file. Use the canonical tagged split:

```text
data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet
```

Rebuild the split from the imported Parquet:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/data/make_crtk_sample_splits.py \
  --mode fine_3class \
  --input data/processed/crtk_training_20260419_180229_fast.parquet \
  --output-dir data/splits/crtk_sample_3class_unique \
  --max-per-class 150000 \
  --batch-size 200000 \
  --dedupe-normalized-fen \
  --overwrite \
  --report-json data/reports/crtk_sample_3class_unique_split_report.json \
  --report-md data/reports/crtk_sample_3class_unique_split_report.md
```

Add CRTK slice tags after the split exists:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/data/build_crtk_tagged_splits.py \
  --split-dir data/splits/crtk_sample_3class_unique \
  --output-dir data/splits/crtk_sample_3class_unique_crtk_tags \
  --report-path data/reports/crtk_sample_3class_unique_crtk_tagged_report.md \
  --overwrite
```

CRTK metadata is for reporting and slice analysis only. It must not be used as model input.

## Training

Benchmark and idea configs should use the NVIDIA path:

```yaml
device: nvidia
```

That resolves to PyTorch CUDA and fails fast if no NVIDIA GPU is visible. Use `device: cpu` only for explicit smoke tests.

Run the current binary puzzle benchmark suite:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_experiment_suite.py \
  --suite configs/suites/network_signal_benchmark_suite.yaml \
  --skip-existing
```

Run the current 3-class benchmark suite:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_experiment_suite.py \
  --suite configs/suites/network_signal_fine3_benchmark_suite.yaml \
  --skip-existing
```

Dry-run a suite before spending GPU time:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_experiment_suite.py \
  --suite configs/suites/network_signal_benchmark_suite.yaml \
  --dry-run
```

On multi-GPU machines, the suite runner assigns one CUDA device per subprocess. Pin a two-GPU run explicitly with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_experiment_suite.py \
  --suite configs/suites/network_signal_benchmark_suite.yaml \
  --skip-existing \
  --jobs 2 \
  --gpu-ids 0,1
```

## Run Everything

Use the top-level combined runner when you want one entrypoint for the full
trunk pipeline followed by the primitive pipeline:

```bash
./run_all.sh
```

It starts or attaches to the `chess-nn-run-all` tmux session, runs
`run_trunk_pipeline.sh --inside-tmux`, then runs `run_primitive_pipeline.sh`.
Arguments passed to `run_all.sh` are forwarded to the trunk pipeline. Primitive
training is controlled with `RUN_PRIMITIVE_*` variables because primitive
prototype checks, config discovery, dry-run planning, and scout training have
separate controls.

Useful controls:

```bash
RUN_ALL_SKIP_TRUNKS=1 ./run_all.sh              # primitive pipeline only
RUN_ALL_SKIP_PRIMITIVES=1 ./run_all.sh          # trunk pipeline only
RUN_ALL_CONTINUE_AFTER_TRUNK_FAILURE=1 ./run_all.sh
RUN_PRIMITIVE_TRAIN=1 RUN_PRIMITIVE_DRY_RUN=0 RUN_PRIMITIVE_GPU_IDS=0 ./run_all.sh
```

Combined logs are written under `reports/run_all/`; the trunk and primitive
status files stay in their dedicated report folders.

## Trunk Pipeline Wrapper

On a fresh GPU machine, use the tmux wrapper to install Python dependencies, detect
visible NVIDIA GPUs, verify the canonical split, and start the full resumable
trunk/registered-architecture pipeline:

```bash
./run_trunk_pipeline.sh
```

The wrapper starts or attaches to the `chess-nn-trunk-pipeline` tmux session and
then invokes `scripts/run_paper_ready_all.py` with the full default pipeline. This
is for trunk and registered-architecture runs; primitive-native architectures
with custom primitive inputs should get their own dedicated wrapper/config path.
It uses all visible NVIDIA GPUs by default, with one parallel training job per
GPU. To limit the run to the first N visible GPUs, set
`RUN_TRUNK_PIPELINE_GPU_COUNT=N`; to choose interactively inside tmux, set
`RUN_TRUNK_PIPELINE_ASK_GPU_COUNT=1`. Legacy `RUN_TRUNK_SWEEP_*` and
`RUN_ALL_*` variables still work as fallbacks.

Use this command directly when you do not need the tmux/dependency wrapper. It
trains every benchmark config and registered idea, expands each one to all
configured size variants and seeds, resumes after interruption, writes logs with
ETA, and builds the final leaderboards, dashboards, and PDF report:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_paper_ready_all.py \
  --seeds 42,43,44 \
  --scale-variants base:1,scale_up:1.5,scale_xl:2 \
  --batch-size-caps base:256,scale_up:192,scale_xl:128 \
  --epochs 30 \
  --min-epochs 15 \
  --patience 8 \
  --jobs 1 \
  --gpu-ids 0
```

It writes:

```text
results/paper_ready_all/
reports/paper_ready_all/generated_configs/
reports/paper_ready_all/logs/
reports/paper_ready_all/events.jsonl
reports/paper_ready_all/plan.md
reports/paper_ready_all/status.md
reports/paper_ready_all/timeline.md
reports/paper_ready_all/paper_report.pdf
reports/paper_ready_all/state.json
```

By default the runner performs three architecture-size sweeps for every config and seed: the original `base` size, `scale_up` at `1.5x`, and `scale_xl` at `2x`. Scaled tasks keep labels, input planes, datasets, and optimizer protocol fixed while increasing known architecture width/depth/capacity fields. The default batch caps are conservative for a single 8GB RTX 3070; use `--batch-size-caps none` only if the target GPU has more VRAM or you have already calibrated memory use.

If the process or machine stops, rerun the same command. Completed runs are skipped, interrupted runs resume from `checkpoint_last.pt`, and runs that trained but still need final reports resume into the artifact pipeline. At the end, the runner rebuilds leaderboards, aggregate training dashboards, speed summaries, and the paper-style PDF report.

Use `--dry-run` first to inspect the plan without requiring CUDA. Open `reports/paper_ready_all/status.md` first; it summarizes task counts, ETA once at least one task has finished, pending work, failures, logs, leaderboards, training dashboards, and the final PDF report path. The terminal prints numbered task progress as it starts and finishes work, including the current rough ETA; `events.jsonl` keeps the full timestamped machine-readable ledger, and `timeline.md` keeps the readable chronology of what ran when and where.

## Primitive Pipeline

Use the primitive pipeline for primitive-only prototype checks, primitive tests,
static validation, and scout training of promoted primitive configs:

```bash
./run_primitive_pipeline.sh
```

By default it runs the current primitive prototypes and any primitive-specific
pytest files, validates promoted primitive configs, and writes status/logs under
`reports/primitive_pipeline/`. It does not launch training unless requested.

Promoted primitive configs are discovered from `ideas/registry/p###_*/config*.yaml`
or the legacy handoff IDs `i244` through `i248`. To train the discovered configs:

```bash
RUN_PRIMITIVE_TRAIN=1 RUN_PRIMITIVE_DRY_RUN=0 RUN_PRIMITIVE_GPU_IDS=0 ./run_primitive_pipeline.sh
```

To train explicit configs instead of auto-discovery:

```bash
RUN_PRIMITIVE_CONFIGS="ideas/registry/p001_rule_aware_tactical_head/config.yaml" \
RUN_PRIMITIVE_TRAIN=1 RUN_PRIMITIVE_DRY_RUN=0 RUN_PRIMITIVE_GPU_IDS=0 \
./run_primitive_pipeline.sh
```

## Primitive Implementation With Claude

To hand the primitive implementation work to Claude Code using the Claude
subscription login, Opus, and maximum reasoning effort:

```bash
./run_primitive_implementation_with_claude.sh
```

The launcher defaults to:

```text
CLAUDE_MODEL=claude-opus-4-7
CLAUDE_EFFORT=max
CLAUDE_PERMISSION_MODE=acceptEdits
```

The launcher only accepts `CLAUDE_EFFORT=max` or `CLAUDE_EFFORT=xhigh` for this
task, and defaults to `max`.

If this Claude Code install exposes Opus through the moving alias instead of
the exact model string, run:

```bash
CLAUDE_MODEL=opus ./run_primitive_implementation_with_claude.sh
```

Use `CLAUDE_NONINTERACTIVE=1` to run with `claude -p` and tee the output to
`reports/primitive_implementation_with_claude/`. Use `CLAUDE_DRY_RUN=1` to
inspect the generated prompt and command without invoking Claude.

## Outputs

Every shared-trainer run should produce standard artifacts under `results/<run_name>/`:

```text
metrics_train.csv
metrics_val.csv
metrics_final.json
complexity_estimate.json
speed_summary.json
artifact_manifest.json
checkpoint_best.pt
checkpoint_last.pt
training_dashboard.png
loss_curves.png
accuracy_curves.png
confusion_matrix_val.png
confusion_matrix_test.png
fine_to_binary_confusion_matrix_val.png
fine_to_binary_confusion_matrix_test.png
predictions_val.parquet
predictions_test.parquet
slice_report_val.md
slice_report_test.md
run_summary.md
report.html
```

Leaderboards are rebuilt at:

```text
results/leaderboard.csv
results/leaderboard.md
reports/leaderboards/global_leaderboard.md
```

Rebuild aggregate training plots:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/reports/plot_training_results.py \
  --results-dir results \
  --output-dir reports/training
```

Build the multi-page paper-style PDF report manually:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/reports/build_paper_report.py \
  --results-dir results \
  --state-path reports/paper_ready_all/state.json \
  --output reports/paper_ready_all/paper_report.pdf
```

The leaderboard, training dashboard, and PDF report builders scan nested run directories, so `--results-dir results` includes paper-ready runs stored under `results/paper_ready_all/`.

Validate a completed run:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/validate_run_artifacts.py results/<run_dir>
```

## Adding A Model

1. Add reusable model code under `src/chess_nn_playground/models/`.
2. Add a builder that accepts a config dictionary and returns a `torch.nn.Module`.
3. Register the builder in `src/chess_nn_playground/models/registry.py`.
4. Add a config under `configs/benchmarks/<task>/` or the relevant `ideas/registry/i###_*/` folder.
5. Validate the config before training.
6. Add the config to a suite only after it is stable.

Useful commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py --list-models

PYTHONDONTWRITEBYTECODE=1 python scripts/validate_training_config.py \
  configs/benchmarks/puzzle_binary/my_new_model.yaml

PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py \
  --config configs/benchmarks/puzzle_binary/my_new_model.yaml
```

Keep new architectures compatible with the shared trainer unless there is a strong reason not to. That keeps curves, confusion matrices, predictions, slice reports, and leaderboards comparable.

## Idea Workflow

Registered ideas live in `ideas/registry/i###_*/`. Each folder should keep the standard scaffold:

```text
idea.yaml
math_thesis.md
architecture.md
implementation_notes.md
trainer_notes.md
ablations.md
model.py
train.py
config.yaml
report_template.md
runs/
```

The guarded idea `train.py` checks identity, implementation status, CUDA policy, model registration, and config completeness before training.

`idea.yaml implementation_kind` is the architectural honesty label. Use `bespoke_model` only when the folder is backed by a materially distinct model implementation. `implementation_status: implemented` / `tested` is reserved for bespoke code that should be trained as an architecture; `ResearchPacketProbe` wrappers must remain `probe_scaffold_only` until their markdown thesis is actually implemented.

Regenerate idea navigation and TODO files after changing idea status, importing packets, or linking results:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/build_idea_catalog.py
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/audit_implementation_kinds.py --check
PYTHONDONTWRITEBYTECODE=1 python scripts/ideas/audit_architecture_conformance.py --check
```

## Ground Rules

- Treat 1-epoch runs as smoke tests and 3-epoch runs as triage, not final evidence.
- Use the canonical tagged split for reliable comparisons.
- Keep CRTK metadata, tactic tags, source labels, solution moves, and verification metadata out of model inputs.
- Compare candidates against LC0 BT4, NNUE, VetoSelect, and Dykstra references before calling a model promising.
- Use repeated seeds and slice reports before making promotion-grade or paper-grade claims.
- Keep generated data, reports, and checkpoints out of source edits unless they are intentionally being updated.
