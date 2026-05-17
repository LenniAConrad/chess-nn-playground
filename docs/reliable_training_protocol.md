# Reliable Training And Research-Quality Protocol

This protocol defines what counts as a smoke run, a triage run, a reliable engineering run, and a paper-grade research result. Future architecture work must not treat a short or undertrained run as final evidence unless it explicitly says it is only a triage result.

## Core Rule

A result is only research-worthy if it was trained on the canonical split, on the NVIDIA GPU path, with enough post-warmup epochs to reach a real validation plateau, compared against matched baselines, repeated across seeds, and saved with all standard artifacts and statistical evidence.

Short runs are useful for catching bugs. They are not enough to decide whether an idea is good.

## Training Tiers

| Tier | Purpose | Minimum Training | Seeds | Can promote an idea? |
| --- | --- | ---: | ---: | --- |
| Smoke | Check that code, CUDA, loss, reports, and artifacts work. | 1 epoch, tiny or full split | 1 | No |
| Triage | Decide whether an implemented idea is worth more compute. | 3 epochs on the canonical split | 1 | No, unless it is an obvious large win and followed by reliable training |
| Reliable single run | Main benchmark-quality evidence for one config. | 20-epoch convergence budget, early stopping patience 5, at least 10 active epochs after warmup | 1 | Maybe, if the margin is large |
| Promotion-grade | Decide whether to call a model the new best inside this repo. | Convergence budget for each seed | 3 seeds: 42, 43, 44 | Yes for repo claims |
| Paper-grade | Evidence suitable for a research paper, preprint, or public claim. | Convergence budget plus matched baselines, ablations, and confidence intervals | At least 5 seeds preferred; 3 is the minimum | Yes, if statistically and practically meaningful |

For losses with a warmup phase, count only epochs after warmup as active evidence. Example: if `warmup_epochs: 1`, then a 20-epoch reliable run gives at most 19 active epochs, and the config should usually set `min_epochs: 11` to guarantee 10 active post-warmup epochs.

## Paper-Grade Standard

Do not describe a result as publication-quality, paper-grade, or research-paper-worthy unless all of the following are true:

- the model and all key baselines were trained under the same data split, input encoding, label mapping, optimizer family, scheduler policy, early-stopping policy, and artifact pipeline;
- training ran long enough to converge or plateau, not just long enough to produce a number;
- the final claim is based on repeated seeds, not one lucky checkpoint;
- model selection and threshold selection used validation data only;
- the test set was used only for final reporting after the model/config/threshold protocol was fixed;
- the report includes confidence intervals or seed mean/std for the primary metrics;
- the report includes matched-recall false positives and near-puzzle false positives, not only aggregate accuracy;
- the report includes worst-slice behavior and does not hide regressions on hard/equal/endgame/promotion slices;
- the report includes enough ablations to show the claimed mechanism matters;
- all negative, failed, or inconclusive runs are described honestly as such.

If any item is missing, call the result "triage", "single-seed", or "engineering evidence", not paper-grade evidence.

## Convergence Budget

Paper-grade runs should use a convergence-oriented budget rather than a fixed short budget:

```yaml
training:
  epochs: 20
  min_epochs: 10
  min_active_epochs: 10
  early_stopping_patience: 5
  class_weighting: balanced
  num_workers: auto
  persistent_workers: true
  prefetch_factor: 2
  mixed_precision: true
  allow_tf32: true
  matmul_precision: high
```

The run may stop earlier only if early stopping triggers after the minimum active-epoch requirement is satisfied. For warmup losses, there must be at least 10 active post-warmup epochs unless validation clearly plateaus earlier and this is documented.

The suite runner parallelizes across multiple NVIDIA GPUs by assigning one `CUDA_VISIBLE_DEVICES` value per subprocess. On a single-GPU machine, keep one training job active at a time; competing training processes on the same GPU usually reduce throughput and can invalidate long runs through out-of-memory failures.

Use a learning-rate schedule when the model benefits from it, but keep it matched across the candidate and the strongest comparable baselines. If the candidate gets schedule tuning, the baseline must get the same level of tuning.

Do not call a run undertrained "bad architecture evidence". If the train loss is still falling sharply, validation has not plateaued, or the run stopped because the budget was too small, report it as undertrained and rerun with the convergence budget before making a claim.

## Required Data

Use the clean tagged benchmark split for all reliable runs:

```text
data/splits/crtk_sample_3class_unique_crtk_tags/split_train.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_val.parquet
data/splits/crtk_sample_3class_unique_crtk_tags/split_test.parquet
```

Do not train reliable benchmark claims on the older untagged split unless the report says exactly why. Do not train directly on the full 45M-row Parquet until a streaming trainer exists.

For the current puzzle-binary benchmark, keep this label contract:

```text
fine label 0: known non-puzzle      -> binary label 0
fine label 1: verified near-puzzle  -> binary label 0
fine label 2: verified puzzle       -> binary label 1
```

The model and loss must not use CRTK metadata, source labels, tactic tags, solution moves, engine PVs, or verification status as input features. Those columns are for reporting only.

## Required Config Defaults

Reliable runs should use these defaults unless the run report justifies a change:

```yaml
seed: 42
deterministic: true
mode: puzzle_binary
device: nvidia
data:
  encoding: lc0_bt4_112
  cache_features: false
training:
  reliability_tier: paper_grade
  allow_cpu_oom_fallback: false
  epochs: 20
  min_epochs: 10
  min_active_epochs: 10
  early_stopping_patience: 5
  class_weighting: balanced
  num_workers: auto
  persistent_workers: true
  prefetch_factor: 2
  mixed_precision: true
  allow_tf32: true
  matmul_precision: high
```

Paper-grade runs should use the convergence budget above unless the architecture has a documented reason to require less or more.

CUDA OOM is a failed reliable run by default. `training.allow_cpu_oom_fallback: true` is only for debugging
salvage runs; if it triggers, the output is labeled `cpu_oom_fallback_non_benchmark` and should not be
cited as benchmark evidence.

Batch size is model-dependent. Use the largest stable batch size that fits GPU memory without changing the comparison target. Current LC0 BT4-style runs usually fit around `192` to `256` on the local NVIDIA GPU.

Do not make a model so large that it cannot be trained to convergence under the available compute. A larger model with poor or incomplete training is not a fair comparison against a smaller, well-trained baseline. If the candidate is much larger, report parameter count, runtime, GPU memory use, and whether it actually converged.

## Before Training

Run these checks before a long run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q src tests ideas/<idea_folder>
PYTHONDONTWRITEBYTECODE=1 python scripts/validate_training_config.py <config.yaml>
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from chess_nn_playground.ideas.implementation import validate_idea_for_training
report = validate_idea_for_training("ideas/<idea_folder>", "<config.yaml>")
print(report)
raise SystemExit(0 if report["valid"] else 1)
PY
nvidia-smi
```

For a new model or loss, also run a CUDA smoke test that creates the model, runs one forward pass, computes the loss, and calls `backward()`.

## During Training

Use the shared trainer, suite runner, paper-ready runner, or guarded idea wrapper:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/train_model.py --config <config.yaml>
PYTHONDONTWRITEBYTECODE=1 python scripts/run_experiment_suite.py --suite configs/suites/network_signal_benchmark_suite.yaml --skip-existing
PYTHONDONTWRITEBYTECODE=1 python scripts/run_paper_ready_all.py --dry-run
PYTHONDONTWRITEBYTECODE=1 python ideas/<idea_folder>/train.py --config ideas/<idea_folder>/<config.yaml>
```

Do not change the data split, label mapping, input encoding, or thresholding rule halfway through a comparison. If the run crashes, keep the partial result only as a debugging artifact, not as benchmark evidence.

## After Training

Every reliable run must pass:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/validate_run_artifacts.py results/<run_dir>
PYTHONDONTWRITEBYTECODE=1 python scripts/compare_results.py
PYTHONDONTWRITEBYTECODE=1 python scripts/reports/plot_training_results.py --results-dir results --output-dir reports/training
```

The run directory must contain:

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
run_summary.md
report.html
```

For registered ideas, also write a run note under:

```text
ideas/<idea_folder>/runs/<timestamp>_<short_name>.md
```

The note must include aggregate metrics, matched-recall false positives, worst slices, diagnostics, comparison to current references, and a decision.

For paper-grade runs, the note must also include:

- seed mean and standard deviation for primary metrics;
- confidence intervals or bootstrap intervals for matched-recall FP differences when possible;
- training time and GPU used;
- parameter count and inference-time notes;
- a convergence statement based on the learning curves;
- ablations for each major new mechanism;
- an explicit statement of whether the result is publishable, inconclusive, or only engineering evidence.

## Threshold And Comparison Rules

Do not judge high-recall puzzle-binary models only at threshold `0.5`. Always report:

- default threshold `0.5`;
- validation-best F1 threshold;
- validation-derived threshold targeting recall `0.80`;
- validation-derived threshold targeting recall `0.85`.

For this project, near-puzzle false positives are central. A model is interesting if it improves one of these without breaking the others:

- PR AUC and ROC AUC;
- test F1 at a validation-selected threshold;
- total false positives at matched recall;
- near-puzzle false positives at matched recall;
- worst-slice behavior on `hard`, `equal`, `endgame`, `mate_in_1`, `promotion`, and `underpromotion`.

## Promotion Rule

To call a model the new best inside the repo, prefer a 3-seed promotion run. Promote only if the model beats the current best on the declared primary target and does not obviously regress on the pressure slices.

Use these as practical minimum margins:

- at least `+0.003` absolute mean PR AUC or F1 across seeds, or
- at least `1%` fewer near-puzzle false positives at matched recall `0.80` or `0.85`, with similar or better precision.

If only one seed was trained, the report must say "single-seed evidence" and should avoid overclaiming unless the margin is large.

For a research-paper claim, require a stronger standard:

- at least 3 seeds, preferably 5;
- the same seed set for the candidate and all key baselines;
- confidence intervals or paired bootstrap evidence for the claimed improvement;
- a practical improvement on the benchmark's central failure mode, not only a small aggregate metric gain;
- no unexplained regression on hard/equal/near-puzzle slices.

If a model only wins because the baseline was trained for fewer epochs, with less tuning, or without the same validation protocol, the result is invalid.
