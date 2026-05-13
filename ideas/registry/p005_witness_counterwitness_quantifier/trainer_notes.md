# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config is
paper-grade and CUDA-required, mirroring the i193 baseline.

Differences vs the i193 baseline:

- `model.name = witness_counterwitness_quantifier_network`
- New head hyperparameters: `num_candidates`, `num_replies`,
  `token_dim`, `head_hidden_dim`, `head_dropout`, `tau_forall`,
  `tau_exists`, `compat_dim`, `gate_init`, `ablation`.

## Loss

`bce_with_logits` on the puzzle logit. No primitive-specific
auxiliary loss is required.

## Temperature annealing

The source packet recommends annealing `tau_forall, tau_exists` from
0.5 to 0.15 over training. The current implementation does not anneal
inside the trainer — first sweep uses fixed defaults (0.20 / 0.20).
A future improvement is to expose a temperature schedule callback if
the fixed temperature underperforms.

## Cost expectation

The WCQ operator is cheap (`O(K + K * R)` per board). Throughput
should drop by < 5% versus i193. Watch `speed_summary.json` for
`train_samples_per_second`.

## Ablation runs

Promotion requires the primary falsifiers:

1. `model.ablation: max_claim_only` — bypass the counter branch
   entirely. If this matches the unablated run, the counter side is
   not load-bearing.
2. `model.ablation: mean_counter_penalty` — replace forall-soft with
   a mean penalty. Tests whether the forall structure matters beyond
   averaging.
3. `model.ablation: random_counter_assign` — permute counter rows
   across candidates so counter scores no longer match the
   candidates they reference.
4. `model.ablation: no_counter_branch` — zero out counter scores.

Sanity ablations:

- `model.ablation: zero_delta` — recovers the i193 baseline.
- `model.ablation: trunk_only` — strongest control.
- `model.ablation: disable_gate` — checks gate load-bearing.

## Reports

Standard idea report. Required slices:

- Aggregate validation and test PR AUC
- Near-puzzle false-positive rate at matched recall 0.80
- Promotion / underpromotion near-FP (the WCQ target slices)
- Mate-in-1 near-FP
- `crtk_eval_bucket = equal` slice
- Highest-confidence wrong examples

Inspect `wcq_value`, `wcq_max_margin`, and `primitive_gate` to
confirm that high-value, low-counter-envelope positions correspond to
real puzzles and the gate fires preferentially on ambiguous tactical
texture.
