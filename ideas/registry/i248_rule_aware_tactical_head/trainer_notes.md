# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config is
paper-grade and CUDA-required, mirroring the i193 baseline so the
architecture-level comparison is matched on:

- same train/val/test split
- same encoding (`simple_18`)
- same seed
- same training budget and early-stopping policy
- same threshold-selection rule

Differences vs the i193 baseline (`ideas/registry/i193_exchange_then_king_dual_stream/config.yaml`):

- `model.name = rule_aware_tactical_head` (i248 wrapper builder)
- `model.head_hidden_dim`, `model.head_dropout`, `model.ablation` for the
  new fusion head
- All trunk hyperparameters retain their i193 names with a `trunk_` prefix
  in the config (e.g. `trunk_channels`, `trunk_depth`) so the builder can
  forward them to the wrapped `ExchangeThenKingDualStreamNetwork`.

## Loss

`bce_with_logits` on the puzzle logit. No primitive-specific auxiliary
loss is required — the head learns through the main BCE signal.

## Cost expectation

The python-chess in-forward fallback adds ~5% wall-clock at scout scale
when CPU is not the bottleneck. With `num_workers >= 4`, the data loader
absorbs the cost in parallel with the GPU forward. Watch
`speed_summary.json` for `train_samples_per_second` — if it falls below
half the i193 throughput on the matched config, switch to the precompute-
parquet path before further scouts.

## Ablation runs

Promotion of i248 requires the falsifier ablation. Use:

```yaml
model:
  ablation: shuffle_tsdp
```

with everything else matched to the unablated run. If the shuffled-
indicator run matches the unablated run on the mate_in_1 slice, the
indicators are not load-bearing and the architecture should be dropped.

Additional ablations to run if the falsifier passes:

- `model.ablation: disable_gate` — does the gate matter?
- `model.ablation: zero_delta`   — i193 baseline (sanity check)
- `model.ablation: trunk_only`   — strongest control

## Reports

Standard idea report. Required slices (see `report_template.md`):

- `crtk_tactic_motifs = mate_in_1` PR AUC (primary)
- `crtk_tactic_motifs` stalemate-trap-adjacent buckets
- aggregate validation and test PR AUC
- near-puzzle false-positive rate at matched recall
- per-slice false positives for fine label 1
- highest-confidence wrong examples

The diagnostic columns `primitive_gate`, `primitive_delta`,
`tsdp_mate_in_1`, and `tsdp_forcing_density` should be inspected to
confirm the gate fires preferentially on rule-flagged positions.
