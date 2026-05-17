# Trainer Notes — a014 BT4 Primitive Mixer (legal_move_graph_delta)

- `train.py` calls `idea_train_cli(__file__)`. The guard checks idea /
  config / model identity, the `implementation_status`, the
  `implementation_kind`, the configured `device`, and that
  `model.name` resolves under `available_models()` before touching the
  trainer.
- Training is gated by `CLAUDE_ALLOW_TRAINING=1`. CPU smokes, static
  config validation, and import checks are allowed without it.
- Hyperparameters in `config.yaml` (lr, weight decay, schedule,
  reliability tier, gradient clip, mixed precision, early stopping)
  are held constant across the `a###_bt4_*_mixer` sweep. Do not retune
  per-sibling — the comparison is only valid if the mixer is the only
  changed variable.
- Loss is `bce_with_logits` over the single puzzle logit with
  `class_weighting: balanced`, matching the puzzle_binary contract.
- Reliability tier is `paper_grade` with `min_epochs: 10` and
  `min_active_epochs: 10`; the trainer is responsible for enforcing
  the floor.
- The shared `Trainer.fit()` artefacts (metrics, slice reports,
  confusion matrix, calibration files) must be preserved so this run is
  comparable against `bt4_conv_mixer` and `bt4_attention_mixer`.
- The mixer adds CPU-side typed-adjacency construction. If wall-clock
  blows past `1.2x` the conv-mixer baseline, record it in the report
  and decide whether to defer the fused message-pass kernel before
  re-running.
