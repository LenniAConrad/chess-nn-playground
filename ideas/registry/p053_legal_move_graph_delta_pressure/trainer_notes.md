# Trainer Notes

p053 uses the shared
`chess_nn_playground.ideas.implementation.idea_train_cli` entry
point and the standard `chess_nn_playground.training.trainer`
config-driven trainer. No bespoke trainer changes are required.

## Run

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python ideas/registry/p053_legal_move_graph_delta_pressure/train.py
```

The trainer:

- Loads the canonical `crtk_sample_3class_unique_crtk_tags` splits
  (`split_train.parquet`, `split_val.parquet`,
  `split_test.parquet`) listed in `config.yaml`.
- Builds the model via the idea-local `build_model_from_config` →
  `build_legal_move_graph_delta_pressure_from_config` chain.
- Uses `BCEWithLogitsLoss` with the trainer's `balanced` class
  weighting at the puzzle-binary level (fine label 2 = puzzle, 0 /
  1 = non-puzzle).
- Honours AMP (`mixed_precision: true`), `allow_tf32: true`, and
  the `reliability_tier: paper_grade` setting (minimum 10 epochs /
  10 active epochs, gradient clip 1.0, early stopping patience 5,
  ReduceLROnPlateau scheduler).

## Inputs not used in training

- CRTK metadata, source labels, verification flags, engine
  evaluations, and principal variations are *not* consumed by the
  model. The trainer surfaces these as reporting-only columns and
  the registry validation enforces the `simple_18`-only input
  contract via `BoardTensorSpec(input_channels=18)`.
- `crtk_sample_3class_unique_crtk_tags` is the canonical fold and
  enforces source / motif uniqueness, so the trainer does not need
  bespoke per-row filtering.

## Reporting

Every training run should emit:

- the aggregate metric set (PR AUC, ROC AUC, F1, accuracy,
  calibration) on val and test;
- the fine-label diagnostic matrix (random non-puzzle FP, near-
  puzzle FP, puzzle recall);
- `slice_report_val.md` and `slice_report_test.md` keyed by
  `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
  `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-type LMGDP diagnostics:
  `lmgdp_edge_count_{P, N, B, R, Q, K}`,
  `lmgdp_post_attack_value_mean_{...}`,
  `lmgdp_capture_value_mean_{...}`, mean `primitive_gate` on
  candidates with `lmgdp_total_edge_count > 0` vs `= 0`;
- the worst-case calibration buckets (gate near 0 or near 1 with
  high `primitive_delta` magnitude).

## Ablation runs

The ablation harness wires the eight ablation modes via the
`model.ablation` key in `config.yaml`. Each ablation is a
matched-seed re-run with the same training budget and split as the
unablated `none` baseline. See `ablations.md` for the keep / drop
rule.

## CPU smoke / CI

The validator
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python \
  scripts/validate_training_config.py --static \
  ideas/registry/p053_legal_move_graph_delta_pressure/config.yaml`
exercises the config schema. The shared tests in
`tests/test_idea_registry.py::test_fully_implemented_idea_is_smoke_testable`
build the model from this folder's config and run a 2-sample CPU
forward; any breakage there means the head is not portable.

The targeted unit tests in
`tests/test_legal_move_graph_delta_pressure.py` cover the
pure-function feature paths, registry roundtrip, ablation knobs,
and forward / backward shape and gradient checks. The full
benchmark requires a GPU and is launched only via the trainer.
