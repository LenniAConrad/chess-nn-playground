# Trainer Notes

- `train.py` is the standard `idea_train_cli(__file__)` wrapper.
- Pass any of the `ablations.md` modes via `model.ablation`.
- No new dataset columns are required; the head reads only from the
  simple_18 board tensor.
- The per-direction `gamma` parameter is clamped to `[0, 1]` inside the
  forward pass — no explicit constraint needed in the optimizer.
- Default `gate_init=-1.5` biases the primitive closed at init so the
  trunk loss dominates the first epoch.
- Mixed precision is fine; the sequential scan loop stays in fp16
  cleanly because all intermediates are bounded by `gamma`.
- The trainer surfaces the standard metrics plus the `raypool_*`
  diagnostics in the per-sample predictions parquet for slice reporting.
