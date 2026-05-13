# Trainer Notes

- `train.py` is the standard `idea_train_cli(__file__)` wrapper — no
  custom trainer. The shared puzzle_binary BCE loss is sufficient.
- Pass through any of the `ablations.md` modes via `model.ablation`. The
  per-ablation runs are matched (same seed, same split, same scaler).
- The model does not require any new dataset columns. The training
  pipeline runs untouched against
  `data/splits/crtk_sample_3class_unique_crtk_tags/` for parity with
  i193 / i248.
- The IDL embedding table is large enough (~36k params at the default
  `accumulator_dim=48`) that we keep weight decay at `1e-4`; the head
  default `gate_init=-1.5` biases the primitive closed at init so the
  trunk loss dominates the first few epochs.
- Mixed precision is fine — the einsum stays in fp16 cleanly because the
  inputs are binary and the embedding is small.
- The trainer continues to surface all standard metrics. The primitive-
  specific diagnostics (`idl_*`, `primitive_*`) are written to the
  per-sample predictions parquet for slice reporting.
