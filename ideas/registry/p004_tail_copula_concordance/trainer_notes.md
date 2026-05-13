# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config is
paper-grade and CUDA-required, mirroring the i193 baseline.

Differences vs the i193 baseline:

- `model.name = tail_copula_concordance_network`
- New head hyperparameters: `evidence_channels`, `head_hidden_dim`,
  `head_dropout`, `quantile`, `tau_rank`, `tau_tail`, `gate_init`,
  `ablation`.

## Loss

`bce_with_logits` on the puzzle logit. No primitive-specific
auxiliary loss is required.

## Cost expectation

The soft-rank stage is `O(N^2 * C)` per board. For `N = 64`,
`C = 6` this is small relative to the trunk forward. Watch
`speed_summary.json` for `train_samples_per_second`.

## Temperature sweep

The source packet flags temperature sensitivity. First sweep should
test `quantile in {0.70, 0.80, 0.90}` and
`tau_rank in {0.20, 0.35, 0.50}`. Defaults are `0.80 / 0.35 / 0.06`.

## Ablation runs

Promotion requires the primary falsifiers:

1. `model.ablation: square_shuffle` — destroys cross-site
   alignment while preserving marginals.
2. `model.ablation: channel_shuffle` — destroys cross-channel
   structure.
3. `model.ablation: rank_quantile_only` — collapse to channel-
   independent ranks (i095-style control).

Sanity ablations:

- `model.ablation: zero_delta` — recovers the i193 baseline.
- `model.ablation: trunk_only` — strongest control.
- `model.ablation: disable_gate` — checks gate load-bearing.
- `model.ablation: single_channel` — single-channel control.

## Reports

Standard idea report. Required slices:

- Aggregate validation and test PR AUC
- Near-puzzle false-positive rate at matched recall 0.80
- Hard / very-hard slices (cross-channel evidence alignment is most
  informative here)
- `crtk_eval_bucket = equal` slice
- Highest-confidence wrong examples

Inspect `tcc_tail_mean`, `tcc_channel_mass_max`, `tcc_site_mass_max`,
and `primitive_gate` to confirm that the gate fires preferentially
on positions where the upper tails of multiple channels co-locate.
