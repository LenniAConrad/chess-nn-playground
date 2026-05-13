# Mathematical Thesis — p007 Attack-Ray Sparse Attention

## Operator signature

Let `X ∈ R^{B × 64 × d}` be per-square token embeddings and let
`O_b ∈ {0, 1}^{64}` be the occupancy mask of the simple_18 board.
For each source square `s` and each of the 8 ray directions
`δ ∈ {N, NE, E, SE, S, SW, W, NW}`, define

```
first_blocker(b, s, δ) = argmin_{k ≥ 1, on-board} { k : O_{b, s + k·δ} = 1 }
```

The per-query key set is

```
K(b, s) = { first_blocker(b, s, δ) : δ ∈ DIRS, blocker exists } ∪ { s }
```

with a fixed cardinality of 9 slots (the 8 ray directions plus the
self-edge). Slots where no blocker exists on the ray are masked out of
the softmax. The output is

```
y_{b, s} = Σ_{k ∈ K(b, s)} softmax_k( q_{b,s}^T k_{b,s,k} / √d_q + bias_δ(k) ) · v_{b,s,k}
```

where `q`, `k`, `v` are learned linear projections of the gathered
token embeddings and `bias_δ(k)` is a learned per-slot bias allowing
the model to distinguish direction-of-ray without leaking direction
into the value tokens.

The combined model is the additive, gated head

```
final_logit = base_logit + σ(W_gate · trunk_pool) · MLP(global_mean_pool(y))
```

## Why this is not just masked attention

1. The dense 64x64 attention logit matrix is **never instantiated**;
   only a 9-slot key index tensor is.
2. The key indices are a deterministic discrete function of the input
   board occupancy, with `stop_gradient` applied — the operator's
   sparsity pattern is content-dependent in the strict sense the spec
   demands.
3. The key set per query is rule-defined (first blocker on each ray),
   not learned (as in NSA or Routing Transformer), so there is no
   top-k gradient surrogate or score-pool branch.

## Claimed advantage

- For sliding-piece tactics the relevant key set is exactly the first
  blockers on the 8 rays from each square. ARSA delivers that adjacency
  at zero conv depth.
- Cost is `O(B · 64 · K · attn_dim)` with `K = 9`, two orders of
  magnitude cheaper than dense `O(B · 64^2 · d)` attention.

## What is actually proven

- Forward pass shape, gradient flow, and 9-slot index correctness are
  unit-tested.
- For a single rook-on-empty-file FEN the operator's blocker target
  matches the expected square.

## What is hypothesised

- The "≤0.9× FP rate vs i243" target from the ARSA spec is **not** yet
  measured. Scout training is gated by `CLAUDE_ALLOW_TRAINING=1`.
- The `random_keys` ablation tests whether the rule structure (not just
  9-way sparsity) carries the lift; `uniform_attention` tests whether
  softmax weighting matters over a fixed-cardinality key set.

## Failure cases

- For empty rays (the source has no piece on any direction) the slot
  count drops below 9. The op handles this by masking those slots to
  `-inf` before softmax and falling back to the self-edge.
- The 9-slot count is fixed regardless of how many sliding pieces are
  actually on the board; positions with many empty rays will waste
  some FLOPs on padded self-edges. This is the price of keeping the
  kernel shape-static for batching.
