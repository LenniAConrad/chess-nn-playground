# Trainer Notes

- Train with `train.py` in this folder. It calls
  `chess_nn_playground.ideas.implementation.idea_train_cli(__file__)`, which
  re-validates the scaffold and config before training.
- The shared guard requires `implementation_status: implemented` (or
  `tested`), `implementation_kind: bespoke_model`, `device: nvidia`, and
  `model.name == bt4_signed_edit_bilinear_memory_mixer` matching the folder
  slug.
- `config.yaml` is the canonical shared-trainer config: it includes `run`,
  top-level `mode`, `data`, `model`, and `training` sections. Override paths
  via `--config <path>` if needed.
- This idea reuses the shared puzzle_binary trainer
  (`chess_nn_playground.training.trainer.train_from_config`), the shared
  `bce_with_logits` loss with balanced class weighting, the shared
  ReduceLROnPlateau scheduler, and the shared report pipeline. No
  idea-specific loss or trainer extension is required.
- The mixer adds a small amount of compute per block (two `Linear(C, r)`
  projections, a global sum, a 3r -> 2r FiLM linear, a per-square
  `Linear(3r, C)` readout) over the conv baseline; the
  `training.batch_size: 256` and `training.epochs: 20` budget defined in
  `config.yaml` are the comparable budget used across the
  `a###_bt4_*_mixer` family.
- Reliability tier is `paper_grade`; runs respect `min_epochs: 10` and
  `min_active_epochs: 10` so early-stopping cannot truncate the comparison
  against the conv / attention baselines unfairly.
- Run artifacts (config snapshot, metrics, slice reports) land under
  `runs/` in this folder; the shared report builder fills in the
  `slice_report_val.md` and `slice_report_test.md` artifacts required by
  the report template and `tests/test_idea_reporting.py`.
- CPU-only smoke checks should use the static config validator
  (`scripts/validate_training_config.py --static`) plus the idea registry
  tests; do not attempt a full GPU training run from CPU.
