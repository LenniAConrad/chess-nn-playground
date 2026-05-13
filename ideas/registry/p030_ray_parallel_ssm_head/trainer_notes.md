# Trainer Notes

- `train.py` is the standard `idea_train_cli(__file__)` wrapper.
- Pass any of the `ablations.md` modes via `model.ablation`.
- No new dataset columns are required; the head reads from the
  simple_18 board tensor only.
- The A/B projections are the dominant per-step cost in the head;
  reduce `feature_dim` before `max_ray_length` if throughput is the
  binding constraint.
- Mixed precision is fine; sigmoid-bounded A and B keep the scan
  numerically stable in fp16.
- Default `gate_init=-1.5` biases the *output* gate closed at init.
- The trainer surfaces the standard metrics plus the `ray_ssm_*`
  diagnostics in the per-sample predictions parquet.
