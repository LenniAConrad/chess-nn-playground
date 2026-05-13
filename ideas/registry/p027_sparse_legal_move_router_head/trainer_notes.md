# Trainer Notes

- `train.py` is the standard `idea_train_cli(__file__)` wrapper.
- Pass any of the `ablations.md` modes via `model.ablation`.
- No new dataset columns are required; the adjacency is computed inside
  the forward pass from the simple_18 board.
- Mixed precision is fine; the masked softmax stays in fp16 cleanly
  because we use `-inf` masking which is preserved across casts.
- Default `gate_init=-1.5` biases the primitive closed at init so the
  trunk loss dominates the first epoch.
- The trainer surfaces the standard metrics plus the `slmr_*`
  diagnostics in the per-sample predictions parquet for slice reporting.
- The (B, 64, 64) attention matmul means the head is more expensive than
  `p025`/`p028`; consider reducing `attn_dim` first if throughput is the
  binding constraint.
