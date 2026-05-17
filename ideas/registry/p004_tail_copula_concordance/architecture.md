# Architecture

`Tail Copula Concordance Network` (p004) is an additive, gated head
on top of the existing i193 `ExchangeThenKingDualStreamNetwork`
trunk. The trunk runs once per board; the head adds a per-square
evidence projection, the differentiable soft-rank tail-copula
operator, and a small fusion MLP.

The model consumes the repository `simple_18` current-board tensor
`(B, 18, 8, 8)` and returns one puzzle logit, plus a rich per-sample
diagnostics dict.

## Mechanism

1. **i193 trunk pass.** ``run_i193_trunk_with_spatial`` returns the
   base logit, the i193 diagnostic dict, the joint pool feature
   `p_trunk`, and the spatial map `H = cat(ex_h, kg_h)`.

2. **Evidence projection.** A 1x1 conv `evidence_proj` projects the
   trunk's spatial features to `C` evidence channels at each of the
   64 squares, producing the evidence field `X in R^{B x 64 x C}`.
   No fixed tactical motifs are baked in; the channels are learned.

3. **TCC operator.** ``tail_copula_concordance`` computes the
   pairwise soft-rank `u_{n,c}` per channel, the upper-tail
   membership `m_{n,c} = sigmoid((u - q) / tau_tail)`, and the
   symmetric tail concordance matrix `Lambda in R^{B x C x C}`. It
   also returns the per-site tail mass and per-channel tail mass.

4. **Fusion head.** The flattened concordance matrix, per-channel
   tail mass, trunk pool feature, and pooled tail mean / max are fed
   through a LayerNorm + GELU MLP that produces
   `primitive_delta_raw`. A second MLP over
   `[p_trunk, tail_mean, tail_max]` produces the gate logit. Final
   contract:

   ```
   primitive_delta = primitive_gate * primitive_delta_raw
   final_logit     = base_logit + primitive_delta
   ```

   The gate bias is initialised at `-2.0` so the primitive starts as
   a near-no-op.

5. **Ablations.** Eight supported modes via `model.ablation`:

   - `none`: full architecture.
   - `square_shuffle`: shuffle squares per channel; destroys
     cross-site alignment while preserving marginals.
   - `channel_shuffle`: permute channels; destroys cross-channel
     concordance structure.
   - `rank_quantile_only`: collapse the concordance matrix to an
     identity; only marginal ranks remain (the i095-style control).
   - `single_channel`: zero out all but channel 0 — concordance is
     trivial.
   - `disable_gate`: hold the gate at 1.0.
   - `zero_delta`: hold `primitive_delta` at 0.
   - `trunk_only`: zero out features and delta.

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
Stockfish scores, principal variations, and any report-only metadata
are *not* consumed.

## Cost

| Stage | Per-sample cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder |
| Evidence projection | 1x1 conv `O(C * spatial_channels * HW)` |
| Soft-rank pairwise | `O(N^2 * C)` |
| Tail concordance | `O(N * C^2)` |
| Fusion MLPs | small |

For `N = 64` and `C = 6` the soft-rank stage does `~25k` ops per
board.

## Implementation Binding

- Registered model name: `tail_copula_concordance` (idea slug). The
  legacy registry alias `tail_copula_concordance_network` resolves to
  the same builder for backwards-compatible tests.
- Source implementation:
  `src/chess_nn_playground/models/primitives/tail_copula_concordance_network.py`
  (the module retains its `_network` filename; the class is
  `TailCopulaConcordanceNetwork`).
- Shared candidate/reply utilities:
  `src/chess_nn_playground/models/primitives/codex_reply_primitives.py`.
- Trunk source:
  `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`
  (the bespoke i193 trunk is wrapped, not reimplemented).
- Idea-local wrapper:
  `ideas/registry/p004_tail_copula_concordance/model.py`.
- Training config:
  `ideas/registry/p004_tail_copula_concordance/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/_registry_manifest.py`:
  `'tail_copula_concordance': ('chess_nn_playground.models.primitives.tail_copula_concordance_network', 'build_tail_copula_concordance_network_from_config')`.
