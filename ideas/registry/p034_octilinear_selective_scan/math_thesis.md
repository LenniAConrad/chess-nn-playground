# Math Thesis

Source: `ideas/research/primitives/external_29_incremental_move_update_octilinear_scan.md`
(Section "primitive_oss"; "Final Ranking" pos. 1).

## Working thesis

For a position ``x`` with simple_18 board tensor and per-square seed
feature ``X in R^{B x 64 x d}``:

1. For each of eight chess ray directions ``k in {E, W, N, S, NE, NW,
   SE, SW}``, identify the per-direction scan paths -- a list of
   sequences of square indices that traverse the board along ``k``.
   Cardinal directions have 8 paths of length 8; diagonal directions
   have 15 paths of variable length (1 to 8).

2. For each direction ``k``, run a Mamba-style selective state-space
   scan along each path:

       h_t = sigmoid(A_k(x_t)) * h_{t-1} + B_k(x_t) * x_t

   where ``A_k, B_k`` are channelwise linear maps from the per-square
   feature to a per-channel gain. The ``A_k`` gate is wrapped in a
   sigmoid so the transition stays in ``(0, 1)``. The state ``h``
   propagates along the chess-rule ordering; piece occupancy enters
   only through the seed feature, so the scan can "block" by producing
   a near-zero ``A_k`` gate at a blocker.

3. Gather the per-direction final-state-per-square outputs back to a
   ``(B, 64, d)`` per-direction tensor and concatenate the 8 directions
   to ``(B, 64, 8 * d)``.

4. Fuse through ``LayerNorm + Linear + GELU`` to ``head_hidden_dim``,
   pool (own-piece-weighted mean + global mean), and project to a
   scalar gated logit delta over the i193 base logit.

## Why this matters

Mamba selective scans capture long-range causal context with O(N)
sequential cost while preserving per-token adaptivity. Mapping the
scan order to the eight chess ray directions makes the state
propagation rule-aware: a bishop on c1 looks along its a3-f6 diagonal,
a rook on h1 looks along the h-file, etc. The selectivity gate
``sigmoid(A_k(x_t))`` lets the scan attenuate or block at piece
occupancy points -- the structural blocking behaviour of chess sliding
pieces emerges from the gate's data dependence on the seed feature
(which carries piece-existence).

## What is actually proven

The scan is well-defined and stable (since ``sigmoid(A_k) in (0, 1)``
keeps the multiplicative gain in a contraction). The eight-direction
output preserves O(64 * d) per-direction work; the total per-batch
cost is O(8 * 64 * d * batch).

## What is only hypothesized

That the per-direction selectivity outperforms a single-direction
collapse (``single_direction`` ablation) and a fixed-transition
control (``fixed_transition`` ablation).

## Failure cases

1. *Hidden rebrand of weight-tied 1D recurrence*: if all eight
   directions collapse to redundant features, OSS is just a recurrent
   layer per direction with no gain from chess geometry. Tested by
   the ``single_direction`` ablation.
2. *Sequential overhead*: the scan loop in Python is slow; the
   asymptotic Mamba-style ``parallel_scan`` win is not realised
   without a Triton kernel. Throughput-bounded scout-only.
3. *Variable-length diagonals*: variable-length tracks are padded
   with -1 and masked; the masking is correct but may waste compute.

## Falsifier

- ``single_direction`` -- run only the E scan, zero the other seven.
  Tests whether the 8-direction decomposition is load-bearing.
- ``fixed_transition`` -- ``A_k`` becomes a data-independent parameter.
  Tests whether the data-dependent selectivity is load-bearing.
- ``shuffle_features`` -- batch-permute the seed features. Decouples
  rule features from position; tests rule-feature load-bearing.
