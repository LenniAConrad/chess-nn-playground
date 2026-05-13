# Mathematical Thesis — p009 Legal-Move-Graph Convolution

## Operator signature

For per-square token embeddings `X ∈ R^{B × 64 × d}` and the per-piece-
type legal-move adjacency `A_r ∈ {0, 1}^{64 × 64}` (1 where the side-to-
move has an own piece of type `r ∈ {P, N, B, R, Q, K}` on `i` that can
legally move with occlusion to `j` and `j` is not occupied by own piece):

```
y_i = Σ_r ( 1 / max(1, |N_r(i)|) ) Σ_{j ∈ N_r(i)} W_r · X_j
```

with `N_r(i) = {j : A_r(X)_{i, j} = 1}` and per-type linear weights
`W_r ∈ R^{d × m}`. A LayerNorm normalises the typed sum, mean-pooling
over squares gives a `(B, m)` feature that an MLP collapses to the
primitive's scalar delta.

```
final_logit = base_logit + σ(W_gate · trunk_pool) · MLP(LN(Σ_r y^{(r)}))
```

## Why this is not just R-GCN with chess edges

- R-GCN (Schlichtkrull et al. 2018) assumes the per-type edge list is
  supplied externally; LMGConv builds `A_r` inside the op from the
  discrete board state, with `stop_gradient`.
- The aggregator is degree-normalised (GraphSAGE-style) per type to
  keep heads with very few edges (open files, few sliders) numerically
  stable; this is the file's explicit mitigation for the "unnormalised
  gradient spikes" failure mode.
- The mask is not pre-shaped 64x64 — it is a deterministic chess-rule
  function of `simple_18`.

## Claimed advantage

- Each piece type's geometry is exposed directly: knights see knight
  moves, rooks see rook moves, bishops see bishop moves. The trunk no
  longer needs to discover "knights are L-shaped" by repetition.
- Cost is `O(B · 6 · |E_r| · m)` ≈ `O(B · |E| · m)` since the per-type
  bitboards partition the same edge budget.

## What is actually proven

- Forward pass shape, gradient flow, and per-type adjacency counts
  match handcrafted FENs in the unit tests.
- The `shared_weight` ablation degenerates the typed channel into a
  generic message-passing layer; the falsifier compares against this.

## What is hypothesised

- LMGConv's 3-pp matched-recall FP improvement target is **not**
  measured (scout training gated by `CLAUDE_ALLOW_TRAINING=1`).
- `random_typed_edges` tests whether the rule-derived structure is
  load-bearing or whether the operator is essentially typed-message
  passing on noise.

## Failure cases

- Positions with zero legal moves for a given type produce zero
  contribution from that type (degree-normalised mean with a 1-floor
  on empty rows).
- LayerNorm over the typed sum can suppress dominant-type signals if
  one piece type dominates the message tensor; the `no_normalization`
  ablation tests whether normalisation is hurting more than helping.
