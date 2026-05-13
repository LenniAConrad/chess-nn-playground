# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config is
paper-grade and CUDA-required, mirroring the i193 baseline.

Differences vs the i193 baseline:

- `model.name = reply_channel_capacity_network`
- New head hyperparameters: `num_candidates`, `num_replies`,
  `token_dim`, `head_hidden_dim`, `head_dropout`, `solver_iters`,
  `capacity_tau`, `gate_init`, `ablation`.

## Loss

`bce_with_logits` on the puzzle logit. No primitive-specific
auxiliary loss is required.

## Cost expectation

The RCC solver is `O(T * K * R)` per board. Defaults add ~4600 ops
per board, dwarfed by the trunk encoder. Watch `speed_summary.json`
for `train_samples_per_second`.

## Ablation runs

Promotion requires the primary falsifiers:

1. `model.ablation: row_shuffle_channel` — shuffles per-row reply
   distributions; capacity collapses.
2. `model.ablation: duplicate_rows` — all rows match row 0;
   capacity is zero by construction.
3. `model.ablation: uniform_replies` — uniform reply distribution
   per row; capacity is zero.
4. `model.ablation: entropy_only` — feeds only conditional entropy
   to the fusion head; tests whether the full capacity beats
   entropy-only diagnostics (the main novelty claim vs `i192`).

Sanity ablations:

- `model.ablation: zero_delta` — recovers the i193 baseline.
- `model.ablation: trunk_only` — strongest control.
- `model.ablation: disable_gate` — checks gate load-bearing.

## Reports

Standard idea report. Required slices:

- Aggregate validation and test PR AUC
- Near-puzzle false-positive rate at matched recall 0.80
- `crtk_eval_bucket = equal` slice
- Promotion / underpromotion near-FP versus baselines
- Highest-confidence wrong examples

Inspect `rcc_capacity_nats`, `rcc_capacity_gap`, and `primitive_gate`
to confirm that capacity is high on real puzzles and low on
near-puzzles.
