# Architecture

`Geometry-Conditioned Board Pseudo-Likelihood Ratio Network` implements the GeomPLR packet directly as a class-conditioned square-token pseudo-likelihood model over verified `simple_18` current-board planes.

## Architecture

Input `x` has shape `[B, 18, 8, 8]`. `Simple18TokenAdapter` validates the tensor contract, maps the first 12 piece planes to a 13-token square vocabulary (`0 = empty`, `1..12 = piece type/color`), and derives only side-to-move, castling, and en-passant metadata from the remaining planes. Unknown channel semantics fail closed instead of being silently adapted.

`StaticChessRelationIndex` precomputes a fixed leave-self-out neighborhood for each square. Relations include same-rank rays, same-file rays, both diagonal-ray families, knight offsets, king-neighborhood offsets, and white/black pawn-direction offsets. The relation table stores padded neighbor indices, relation ids, distance buckets, and valid-neighbor masks. It does not call a legal-move generator, engine oracle, check detector, or source/CRTK metadata.

The network embeds square tokens, square coordinates, and metadata into a shared hidden space. For each target-square chunk, `TypedNeighborAggregator` gathers only the target square's static neighbors, adds relation and distance embeddings, applies typed gates and relation dropout, masks padded entries, and returns a context vector mixed with target coordinate and metadata embeddings. The target square's own token is never present in its prediction context.

`ClassConditionalTokenDecoder` predicts the target square token distribution twice, once under a class-0 decoder state and once under a class-1 decoder state. `PseudoLikelihoodScorer` accumulates weighted cross-entropy terms for all 64 squares, downweighting empty squares by `empty_square_weight`, and normalizes by the active board weight. This yields class-conditioned description lengths `S_0` and `S_1`.

The packet's two-class pseudo-log-likelihood scores are:

```text
class_logits = -S / softplus(score_temperature) + class_bias
```

The repository task for this idea is binary BCE with fine labels `0` and `1` mapped to non-puzzle and fine label `2` mapped to puzzle, so the configured `num_classes: 1` head returns the likelihood-ratio logit `class_logits[:, 1] - class_logits[:, 0]` with shape `[B]`. If the model is built with `num_classes: 2`, it returns the raw two-column `class_logits` matrix.

Returned diagnostics include the two class pseudo-NLL scores, the description-length ratio, the internal two-class logits, token-NLL summaries, square occupancy fractions, total token weight, and learned score temperature.

## Implementation Binding

- Registered model name: `geometry_conditioned_board_pseudo_likelihood_ratio_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/geometry_pseudolikelihood_ratio.py`
- Idea-local wrapper: `ideas/registry/i036_geometry_conditioned_board_pseudo_likelihood_ratio_network/model.py`
