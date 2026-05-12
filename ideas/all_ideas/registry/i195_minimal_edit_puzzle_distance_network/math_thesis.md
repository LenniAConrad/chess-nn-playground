# Math Thesis

Minimal-Edit Puzzle Distance Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `10`.

Working thesis: a near-puzzle is one small edit away from being a true
puzzle.

We make this thesis literal. Let `S(x) in Delta_S^{8 x 8}` be the
input board's per-square symbol distribution and let
`{ P_k }_{k=1..K}` be a learnable bank of soft puzzle prototypes,
each `P_k in Delta_S^{8 x 8}`. The per-square edit cost is the
complement of the per-square agreement,

```
cost_k(r, f; x) = 1 - <S(x)[:, r, f], P_k[:, r, f]>   in [0, 1].
```

The total per-prototype edit distance is the soft Hamming distance

```
D_k(x) = sum_{r, f} cost_k(r, f; x)
       = 64 - sum_{r, f} <S(x)[:, r, f], P_k[:, r, f]>.
```

The *minimal-edit puzzle distance* is the (temperature-T) soft minimum
across the prototype bank:

```
D_min(x) = -T * logsumexp(-D_k(x) / T).
```

The puzzle classifier reads `D_min(x)` (and a small set of derived
diagnostics — soft prototype assignment `pi_k = softmax(-D_k / T)`,
its entropy, the per-square min-cost map, and per-prototype distance
summary statistics) to produce a single puzzle logit. Positions one
small edit away from a prototype have small `D_min` and are pushed
toward the puzzle class; positions many edits away from every
prototype are pushed toward non-puzzle.

This is the architecture's central content: a learnable puzzle
prototype bank, an explicit per-square edit cost, and a soft-min head
that ties the puzzle decision directly to the minimal-edit distance.
