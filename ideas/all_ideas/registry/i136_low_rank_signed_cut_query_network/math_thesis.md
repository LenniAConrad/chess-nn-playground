# Math Thesis

Low-Rank Signed Cut Query Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2136_friday_shanghai_architecture_batch_7.md`.

Batch candidate rank: `6`.

Working thesis: Puzzle-like positions may separate the board into tense
regions: attacking mass versus defending mass, king-side versus center,
blocked wing versus open wing. A model can learn *low-rank signed cut
queries* over learned board fields and classify puzzle-likeness from per-field
imbalance statistics.

## Formal Object

Let ``F_c : {1, …, 64} → R`` be ``C = num_fields`` learned linear board fields
obtained from the simple_18 input planes via a ``1x1`` projection. For each
query pair ``k ∈ {1, …, K}`` define two rank-1 signed masks over the ``8x8``
board

```
a_k(rank, file) = tanh(r^a_k(rank)) * tanh(f^a_k(file)) ∈ [-1, 1]
b_k(rank, file) = tanh(r^b_k(rank)) * tanh(f^b_k(file)) ∈ [-1, 1]
```

so each mask is exactly the outer product of two coordinate factors, which is
the paper's "low-rank by coordinate factors" constraint.

The signed cut of pair ``k`` against field ``c`` is

```
cut_{k,c} = sum_s a_k(s) F_c(s) - sum_s b_k(s) F_c(s)
```

with absolute, squared, and mass-normalised companions

```
abs_cut_{k,c}  = |cut_{k,c}|
sq_cut_{k,c}   = cut_{k,c}^2
norm_cut_{k,c} = cut_{k,c} / (eps + sum_s |F_c(s)|).
```

A *king-anchored* variant uses ``K_king`` additional rank-1 mask pairs whose
canonical centre ``(3, 3)`` is translated to the white-king square and to the
black-king square of each board. The translation is implemented by indexing
into the 64 ``torch.roll`` shifts of each mask using the argmax of the king
piece plane.

## Classifier

The flattened tensors ``(cut, abs_cut, sq_cut, norm_cut)`` for the global
masks, their white-king and black-king anchored counterparts, per-field
mean / abs-mean / squared-mean statistics, and a small ``3x3`` convolutional
trunk over the same fields are concatenated and fed through a ``GELU`` MLP
that emits one puzzle_binary logit. All factors and convolutional weights are
trainable; only the ``64`` roll table used for king anchoring is recomputed
per forward pass from the trainable mask factors.

## Why The Cut Imbalance Matters

If a position truly separates into a "tense" pair of regions, there exists a
mask pair ``(a_k, b_k)`` whose signed cut is large in absolute value on the
fields that capture that contrast (attack mass, defender mass, blockade
density). A non-puzzle position, in contrast, has no consistent region pair
that produces a large normalised cut across the learned fields. The head can
read this imbalance signal without performing attention, graph construction,
or topology, which keeps the model orthogonal to existing puzzle_binary ideas.
