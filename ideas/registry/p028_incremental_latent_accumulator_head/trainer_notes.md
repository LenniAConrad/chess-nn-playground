# Trainer Notes

- `train.py` is the standard `idea_train_cli(__file__)` wrapper.
- Pass any of the `ablations.md` modes via `model.ablation`.
- No new dataset columns are required; the head reads from the simple_18
  board tensor only.
- The king-anchored embedding is large (~50k floats at `king_dim=16`);
  weight decay at 1e-4 keeps it generalisable on the scout split.
- Default `gate_init=-1.5` biases the primitive closed at init so the
  trunk loss dominates the first epoch.
- Mixed precision is fine; both einsums are cheap and stay in fp16
  cleanly because their inputs are binary indicators.
- The trainer surfaces the standard metrics plus the `ila_*`
  diagnostics in the per-sample predictions parquet.
