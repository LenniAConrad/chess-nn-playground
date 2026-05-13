# Trainer Notes

- Use `train.py` as-is. It calls `idea_train_cli(__file__)`, which checks
  idea/config/model identity and refuses to train unless
  `implementation_status` is `implemented` or `tested`,
  `implementation_kind: bespoke_model`, `device: nvidia`, and the
  `model.name` is registered.

- This idea does **not** introduce a new loss, a new dataset contract, or
  a new collate function. It consumes only `batch["x"]` (the simple_18
  current-board tensor) and returns a dict with `logits` plus diagnostics.

- The cross-derivative grid is built and consumed entirely inside the
  model forward; the trainer is unaware of it. Memory budget per sample
  is ~`2*(K+1)` board tensors held briefly during the grid encoder pass.
  With `batch_size=128`, `K=3`, and `tdcd_channels=48`, this fits well
  inside an 8 GiB 3070; on smaller cards reduce `batch_size` first.

- Batch size is halved relative to i193 (256 -> 128) because the grid
  encoder amplifies the effective per-sample compute by ~8x for that part
  of the network. Throughput should land at ~30-40% of i193 wall-clock per
  epoch on a 3070 according to the markdown spec's `~6-8x` estimate, and
  closer to ~2-3x on this implementation because the encoder is compact.

- Gate initialisation `gate_init = -2.0` keeps the head as a near no-op at
  the start of training. Watch `primitive_gate` in the per-position
  scalar columns: collapse to zero across the dataset means the cross-
  derivative head is not earning its keep and the keep/drop decision
  should be made on its target slice, not the aggregate metric.

- Run the matched ablations using a separate config that overrides
  `model.ablation`. Suggested file layout:
  `config_ablation_main_effects_only.yaml`,
  `config_ablation_skip_cross_derivative.yaml`, etc. Keep all other
  fields identical to the primary config so the comparison is clean.

- Diagnostics worth tracking via `report_prediction_slices`:
  `g_T_norm`, `max_dd`, `mean_dd`, `std_dd`, `primitive_gate`,
  `primitive_gate_entropy`, `saliency_entropy`, and
  `saliency_top_valid_count`. The 2-D scatter
  `(||g_T||, max DeltaDelta_k)` should show 3-cluster structure on
  held-out positions if the central hypothesis holds.
