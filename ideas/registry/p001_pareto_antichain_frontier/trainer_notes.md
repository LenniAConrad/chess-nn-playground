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

- `model.name = pareto_antichain_frontier` (the legacy
  `pareto_antichain_frontier_network` alias still resolves to the same
  builder for backwards-compatible tests).
- New head hyperparameters: `num_candidates`, `token_dim`,
  `utility_channels`, `head_hidden_dim`, `head_dropout`, `tau_dim`,
  `tau_set`, `eps_margin`, `beta`, `gate_init`, `ablation`.
- Trunk hyperparameters retain their i193 names with a `trunk_` prefix
  so the builder can forward them to the wrapped trunk.

## Loss

`bce_with_logits` on the puzzle logit. No primitive-specific auxiliary
loss is required — the head learns through the main BCE signal.

## Cost expectation

The PAFR operator is `O(K^2 * C)` per board. For `K = 16`, `C = 6`
this is dwarfed by the trunk forward; throughput should drop by at
most a few percent versus i193. Watch `speed_summary.json` for
`train_samples_per_second`.

## Ablation runs

Promotion requires the primary falsifiers (in order of strength):

1. `model.ablation: shuffle_channels` — channels are decoupled from
   candidates; if this matches the unablated run, the partial-order
   structure is not load-bearing.
2. `model.ablation: single_channel` — one utility channel only; if
   this matches, the product partial order is not load-bearing.
3. `model.ablation: scalar_max` — collapses to a total order;
   verifies that the partial order beats a learned scalar.

Sanity ablations:

- `model.ablation: zero_delta` — recovers the i193 baseline.
- `model.ablation: trunk_only` — strongest control.
- `model.ablation: disable_gate` — checks whether the gate is
  load-bearing.

## Reports

Standard idea report. Required slices (see `report_template.md`):

- Aggregate validation and test PR AUC
- Near-puzzle false-positive rate at matched recall 0.80
- `crtk_eval_bucket = equal` slice
- Highest-confidence wrong examples

The diagnostic columns `pafr_frontier_width`, `pafr_frontier_entropy`,
`primitive_gate`, and `primitive_delta` should be inspected to confirm
the gate fires preferentially on positions with narrow non-dominated
frontiers (real puzzles) vs broad frontiers (near-puzzles).
