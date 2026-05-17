# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config is
paper-grade and CUDA-required, mirroring the i193 baseline so the
architecture-level comparison is matched on:

- same train/val/test split
- same encoding (`simple_18`)
- same seed
- same training budget and early-stopping policy
- same threshold-selection rule

Differences vs the i193 baseline:

- `model.name = regret_saddlepoint`
- New head hyperparameters: `num_candidates`, `num_replies`,
  `token_dim`, `head_hidden_dim`, `head_dropout`, `solver_iters`,
  `tau_p`, `tau_q`, `solver_damp`, `gate_init`, `ablation`.

## Loss

`bce_with_logits` on the puzzle logit. No primitive-specific auxiliary
loss is required.

## Cost expectation

The RSP solver adds `O(T * K * R)` per board. For default
`K = 16`, `R = 12`, `T = 24` the cost is `~4608` ops per board.
Throughput should drop by at most a few percent versus i193. Watch
`speed_summary.json` for `train_samples_per_second`.

## Ablation runs

Promotion requires the primary falsifiers:

1. `model.ablation: row_shuffle_payoff` — shuffles candidate-side
   structure.
2. `model.ablation: col_shuffle_payoff` — shuffles reply-side
   structure.
3. `model.ablation: uniform_payoff` — completely removes game
   structure.
4. `model.ablation: pure_max_min` — bypasses the regularized solver.

Sanity ablations:

- `model.ablation: zero_delta` — recovers the i193 baseline.
- `model.ablation: trunk_only` — strongest control.
- `model.ablation: disable_gate` — checks gate load-bearing.

## Reports

Standard idea report. Required slices:

- Aggregate validation and test PR AUC
- Near-puzzle false-positive rate at matched recall 0.80
- `crtk_eval_bucket = equal` slice
- Hard / very-hard slices (cyclic tactical structure)
- Highest-confidence wrong examples

Inspect `rsp_saddle_value`, `rsp_exploitability`, and `primitive_gate`
to confirm that the head fires on high-pressure tactical positions
and stays quiet on quiet positions.
