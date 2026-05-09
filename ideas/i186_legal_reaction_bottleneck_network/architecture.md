# Architecture

`Legal-Reaction Bottleneck Network` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and routes the puzzle
logit through an explicit *legal-reaction bottleneck* over the
side-not-to-move's piece squares. The bottleneck is the operational
form of the packet thesis that a real puzzle is not merely a position
with a threat -- it is a position where the defender's normal-looking
reactions either *fail* or are *too few*.

## Mechanism

1. **Board trunk.** `BoardConvStem(input_channels=18, channels, depth,
   use_batchnorm)` produces `feats` of shape `(B, channels, 8, 8)`.
2. **Side masks.** The side-to-move plane (`board[:, 12]`) is reduced
   to a per-row scalar in `{0, 1}`. From the white/black piece planes
   we form `own_mask` and `opp_mask`, each `(B, 1, 8, 8)`. The defender
   side is whoever is *not* to move.
3. **Defender context.** A four-plane context stack
   `[own_mask, opp_mask, all_pieces, mobility]` is concatenated to
   `feats`. `mobility` is the count of empty squares in a 3x3
   neighbourhood of each opponent-piece square, divided by 9. Pieces
   with more empty neighbours are more capable defenders; this is a
   board-only stand-in for "legal reaction quality" that the packet
   prescribes for a defender-reply graph.
4. **Reaction head.** A `3x3 -> 1x1` conv head reads
   `[feats, context]` and produces per-square reaction logits
   `r in R^{B x 8 x 8}`.
5. **Defender-reply softmax.** Reaction logits are masked to the
   opponent-piece squares (non-defender squares get `-inf`) and
   softmaxed across the 64 squares:

   ```text
   p = softmax(r / reaction_temperature, mask=opp_mask)
   ```

   `p` lives on the defender-reply graph: rows with no opponent pieces
   fall back to a uniform-over-64 distribution for stability.
6. **Effective reaction count.** The bottleneck is read out from
   `p`'s entropy:

   ```text
   K_eff = exp(H(p))
   ```

   `K_eff` is the *effective number of legal-looking reactions* in the
   position. Few effective reactions = narrow bottleneck = puzzle-like
   pressure; many effective reactions = wide bottleneck = the opponent
   has many ways to defuse the threat.
7. **Bottleneck pool.** Trunk features are pooled through `p`:

   ```text
   reaction_pool = sum_squares(p * feats)         # (B, channels)
   ```

   This is the explicit information bottleneck: the head reads at most
   `K_eff` defender squares' worth of feature mass.
8. **Threat side.** A second `3x3 -> 1x1` conv head produces threat
   logits `t in R^{B x 8 x 8}`; the threat-side scalar
   `own_piece_pressure` is the mean of `sigmoid(t)` over own-piece
   squares. A second pool `threat_pool = sum_squares(own_norm * feats)`
   summarises the threat-side feature mass.
9. **Bottleneck scalars.**

   ```text
   defense_gap   = own_piece_pressure - log1p(K_eff)
   reply_pressure = own_piece_pressure / (K_eff + 1)
   bottleneck_kl = log(defender_count) - H(p)        # >= 0
   ```

   `defense_gap` rises when threat exceeds the log of the reaction
   count -- the explicit "more pressure than reactions" signal the
   packet calls out. `reply_pressure` is the bottleneck-divided
   pressure score. `bottleneck_kl` is the KL of the reaction
   distribution from a uniform-over-defenders distribution; small
   values mean reactions are evenly spread (many ways to defuse), high
   values mean a single reaction dominates.
10. **Head.** A `LayerNorm + Linear + GELU + Linear` head consumes
    `[reaction_pool, threat_pool, summary_scalars]` and produces the
    puzzle logit. `summary_scalars` is the eight-vector
    `[K_eff, H(p), max(p), defender_count, own_piece_pressure,
    defense_gap, reply_pressure, bottleneck_kl]`.

At inference the model is a single-board single-logit puzzle
classifier compatible with the repository BCE-with-logits
`puzzle_binary` trainer.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for the
repository `puzzle_binary` BCE-with-logits trainer (or
`(B, num_classes)` when `num_classes > 1`):

- `logits`: `(B,)` puzzle logit.
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `reaction_logits`: `(B, 8, 8)` raw per-square defender reaction
  logits before masking.
- `reaction_distribution`: `(B, 8, 8)` defender-reply softmax masked
  to opponent-piece squares.
- `effective_reaction_count`: `(B,)` `K_eff = exp(H(p))`.
- `reaction_entropy`: `(B,)` `H(p)`.
- `reaction_max_strength`: `(B,)` largest mass in `p`.
- `defender_count`: `(B,)` number of opponent-piece squares.
- `own_piece_pressure`: `(B,)` mean threat-head sigmoid over own-piece
  squares.
- `defense_gap`: `(B,)` `own_piece_pressure - log1p(K_eff)`.
- `reply_pressure`: `(B,)` `own_piece_pressure / (K_eff + 1)`.
- `bottleneck_kl`: `(B,)` `KL(p || uniform-over-defenders)`.
- `trunk_energy`: `(B,)` mean-square trunk activation.

## Implementation Binding

- Registered model name: `legal_reaction_bottleneck_network`
- Source implementation file: `src/chess_nn_playground/models/legal_reaction_bottleneck_network.py`
- Idea-local wrapper: `ideas/i186_legal_reaction_bottleneck_network/model.py`
