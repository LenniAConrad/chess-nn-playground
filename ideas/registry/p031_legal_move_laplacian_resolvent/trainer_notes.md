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

- `model.name = legal_move_laplacian_resolvent` (p031 wrapper builder)
- `model.feature_dim`, `model.neumann_terms`, `model.alpha_init`,
  `model.head_hidden_dim`, `model.head_dropout`, `model.gate_init`,
  `model.ablation` for the resolvent head
- All trunk hyperparameters retain their i193 names with a `trunk_` prefix
  so the builder can forward them to the wrapped
  `ExchangeThenKingDualStreamNetwork`.

## Loss

`bce_with_logits` on the puzzle logit. No primitive-specific auxiliary
loss is required -- the head learns through the main BCE signal.

## Cost expectation

The Neumann series is implemented as ``K`` dense `(B, 64, 64) x (B, 64,
feature_dim)` matmuls. With defaults (``K=4``, ``feature_dim=32``) the
head adds ~0.5M params and a small fraction of the trunk's FLOPs. At
``B=128`` and an RTX 3070-class GPU the head wall-clock should stay
within +10% of the i193 baseline. If the throughput per epoch drops by
more than 25%, fall back to ``K=2`` before further scouts.

## Ablation runs

Promotion of p031 requires the falsifier ablation. Use:

```yaml
model:
  ablation: k1_gat_rebrand
```

with everything else matched to the unablated run. If the K=1 collapse
matches the unablated run on the target slice, the Neumann expansion is
not load-bearing and the architecture should be dropped.

Additional ablations to run if the primary falsifier passes:

- `model.ablation: uniform_piece_weights` -- per-piece weight load-bearing?
- `model.ablation: shuffle_adjacency`     -- rule-feature falsifier
- `model.ablation: zero_alpha`            -- resolvent vs constant projection
- `model.ablation: zero_delta`            -- i193 baseline (sanity check)
- `model.ablation: trunk_only`            -- strongest control
- `model.ablation: disable_gate`          -- gate load-bearing?

## Reports

Standard idea report. Required slices (see `report_template.md`):

- aggregate validation and test PR AUC
- near-puzzle false-positive rate at matched recall
- per-slice PR AUC for hard-negative / multi-hop tactical positions
- highest-confidence wrong examples

The diagnostic columns `primitive_gate`, `primitive_delta`,
`lmlpp_alpha`, `lmlpp_mean_feature_norm`, and `lmlpp_degree_mean` should
be inspected to confirm the gate / propagation fires preferentially on
tactically active positions.
