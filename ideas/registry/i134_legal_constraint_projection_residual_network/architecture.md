# Architecture

`Legal-Constraint Projection Residual Network` consumes the simple_18 board
tensor, lets a small CNN trunk hallucinate a soft per-square belief over the
12 piece classes plus an explicit "empty" class, alternately projects that
belief onto a soft set of basic current-board legality constraints, and
classifies one ``puzzle_binary`` logit (fine labels `0` and `1` map to
non-puzzle, fine label `2` maps to puzzle) from the projection residual.

## Input And Belief Encoder

- Input is the repo `simple_18` board tensor with shape `(batch, 18, 8, 8)`.
- A `BoardConvStem` (depth `2` by default) produces a
  `(batch, channels, 8, 8)` feature map.
- A `1x1` convolution `belief_head` projects this map into 13 logits per
  square. A per-square softmax yields the soft latent belief
  `Y in (batch, 13, 8, 8)` over the 12 piece classes
  `(P, N, B, R, Q, K, p, n, b, r, q, k)` and one explicit "empty" class.
- CRTK / source / engine metadata is ignored — only the board tensor is
  consumed by the model.

## Soft Legal-Board Constraint Set

The projection target is the intersection of four soft constraint families
that depend only on the current board (no legal-move generation, no engine
search):

- **Per-square simplex.** Each of the 64 squares carries a probability
  distribution over the 13 classes. After every other constraint we re-
  project to the simplex by clamping to non-negative values and renormalising
  per square.
- **Per-piece-class count caps.** The expected count of every piece class is
  capped at the maximum legal count given promotions: `K, k <= 1`,
  `Q, q <= 9`, `R, r <= 10`, `B, b <= 10`, `N, n <= 10`, `P, p <= 8`.
  Excess belief is shifted into the empty class on the same squares.
- **King-count normalisation.** When the expected total of `K` (or `k`)
  exceeds 1, the corresponding plane is rescaled to sum to 1 and the slack
  is shifted into empty.
- **Pawn-rank masking.** Pawn classes `P` and `p` are forced to zero on
  ranks `0` and `7`, and the masked probability is moved into the empty
  class on the same square.

## Differentiable Alternating Projection

`Y_proj` is computed as a fixed number of alternating projection sweeps
(default `3`):

```
for t in 1..projection_iters:
    Y <- piece_count_clip(Y)
    Y <- king_count_normalize(Y)
    Y <- pawn_rank_mask(Y)
    Y <- square_simplex(Y)
```

Each substep also returns a per-batch residual energy that is summed across
sweeps so the head can read which constraint family was violated most. By
default the projection is computed under a stop-gradient
(`stop_gradient_projection: true`); the residual `R = Y - Y_proj.detach()`
still flows gradients into `belief_head` through `Y`, while the projection
itself is treated as a fixed target. Setting `stop_gradient_projection:
false` makes the entire projection differentiable.

## Residual Readout And Classifier

- `R = Y - Y_proj` is the projection residual.
- A tiny `Conv -> GELU -> Conv -> GELU -> AdaptiveAvgPool` CNN over `R`
  produces a `residual_pool_channels`-wide spatial residual summary.
- Per-class residual L2 norms over the 64 squares produce a `(batch, 13)`
  vector of constraint-pressure-by-class.
- The four per-constraint residual energies (simplex / piece_count /
  king_count / pawn_rank) form a `(batch, 4)` vector.
- A pooled encoder summary (mean + max over the backbone feature map),
  total residual norm, and per-square belief entropy are concatenated.
- A two-layer GELU MLP with dropout reads the fusion vector and emits one
  ``puzzle_binary`` logit.

The forward pass returns a dict whose `logits` tensor has shape `(batch,)`
alongside diagnostics including `residual_total_norm`,
`residual_simplex_energy`, `residual_piece_count_energy`,
`residual_king_count_energy`, `residual_pawn_rank_energy`,
`belief_entropy`, `belief_empty_mass`, `projected_empty_mass`,
`residual_norm_<piece>` per piece class, `residual_norm_empty`,
`white_king_belief_total`, `black_king_belief_total`,
`white_king_projected_total`, `black_king_projected_total`,
`backbone_feature_norm`, `encoder_summary_norm`, and
`residual_map_summary_norm` for ablation analysis.

## Implementation Binding

- Registered model name: `legal_constraint_projection_residual_network`.
- Source implementation: `src/chess_nn_playground/models/legal_constraint_projection_residual_network.py`.
- Idea-local wrapper: `ideas/registry/i134_legal_constraint_projection_residual_network/model.py`.
