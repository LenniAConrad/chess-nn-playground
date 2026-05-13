# Trainer Notes

- `train.py` is the standard `idea_train_cli(__file__)` wrapper.
- Pass any of the `ablations.md` modes via `model.ablation`.
- No new dataset columns are required; the head reads from the
  simple_18 board tensor only.
- The sequential scan loop is small (`max_ray_length=7`) but accounts
  for the bulk of the head's wall-clock; reduce it before reducing
  `feature_dim`.
- Mixed precision is fine; the sigmoid blocker gate stays in fp16
  cleanly.
- Default `gate_init=-1.5` biases the *output* gate closed at init so
  the trunk loss dominates the first epoch. The blocker gate is
  initialised to its natural sigmoid mean.
- The trainer surfaces the standard metrics plus the `oars_*`
  diagnostics in the per-sample predictions parquet.
