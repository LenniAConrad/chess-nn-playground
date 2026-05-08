# Architecture

`Critical-Square Budget Network` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and routes the puzzle
logit through an explicit *critical-square budget*: a soft mask over
the 64 squares whose total mass is fixed to a configurable budget
`K`. The mask is the operational form of the packet thesis that
puzzles hinge on a small number of critical squares (king escape
squares, line intersections, pinned-piece squares, promotion squares,
overloaded defender squares).

## Mechanism

1. **Board trunk.** `BoardConvStem(input_channels=18, channels, depth,
   use_batchnorm)` produces a feature map `feats` of shape
   `(B, channels, 8, 8)`.
2. **Deterministic critical-square priors.** Six per-square prior
   planes are derived directly from the input board planes and
   concatenated to the trunk features, one per critical-square family
   from the packet:

   - own king zone (3x3 dilation of the side-to-move king),
   - opponent king zone (3x3 dilation of the other king),
   - promotion ranks for the side to move (rank 8 for white-to-move,
     rank 1 for black-to-move),
   - own line-piece intersection landmark (rank/file co-occupancy of
     own bishops/rooks/queens),
   - opponent line-piece intersection landmark,
   - empty-square indicator (`1 - sum(piece planes)`).

   The priors are board-only and reporting-only on input -- CRTK and
   source metadata are never consumed.
3. **Saliency head.** A two-layer convolutional head reads
   `[feats, priors]` and produces per-square saliency logits
   `s in R^{B x 8 x 8}`.
4. **Budget gate.** The saliency logits are flattened, divided by a
   temperature, passed through a softmax across the 64 squares, and
   multiplied by the budget `K`:

   ```text
   mask = K * softmax(s / saliency_temperature)
   ```

   The mask is non-negative and sums to `K` per batch row. This is
   the explicit critical-square budget the packet calls for: the head
   is allowed to read at most `K` squares' worth of feature mass per
   position. Lowering the temperature concentrates the mask on fewer
   squares.
5. **Budgeted readout.** The trunk features are pooled with the mask:

   ```text
   pooled  = sum_squares(mask * feats)               # (B, channels)
   summary = (own_king_zone_mass, opp_king_zone_mass,
              promotion_mass, line_intersection_mass,
              empty_square_mass, budget_used,
              saliency_entropy, top_k_mass)
   logit   = head([pooled, summary])
   ```

   The summary scalars expose, per batch row, how much of the budget
   landed inside each prior region, the realised budget mass (a
   sanity check), the entropy of the normalised mask (a sparsity
   monitor), and the sum of the top-`K` mask entries.

At inference the model is a single-board single-logit puzzle
classifier compatible with the repository BCE-with-logits
`puzzle_binary` trainer.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer (or
`(B, num_classes)` when `num_classes > 1`):

- `logits`: `(B,)` puzzle logit.
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `saliency_logits`: `(B, 8, 8)` raw per-square saliency.
- `saliency_mask`: `(B, 8, 8)` soft mask summing to `budget`.
- `budget_used`: `(B,)` realised mask mass (equals `budget` up to
  floating-point error).
- `saliency_entropy`: `(B,)` entropy of the normalised mask.
- `top_k_mass`: `(B,)` sum of the largest `round(budget)` mask
  entries.
- `own_king_zone_mass`, `opp_king_zone_mass`, `promotion_mass`,
  `line_intersection_mass`, `empty_square_mass`: `(B,)` mass of the
  budget that lands inside each prior region.
- `trunk_energy`: `(B,)` mean-square trunk activation.

## Implementation Binding

- Registered model name: `critical_square_budget_network`
- Source implementation file: `src/chess_nn_playground/models/critical_square_budget_network.py`
- Idea-local wrapper: `ideas/i185_critical_square_budget_network/model.py`
