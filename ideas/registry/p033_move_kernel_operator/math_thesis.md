# Math Thesis

Source: `ideas/research/primitives/external_28_sparse_differential_accumulator_move_kernel.md`
(Section "primitive_mko"). The file lists five proposals; the first
listed is SDA. We implement MKO -- the strongest *distinct* proposal in
the file -- because SDA overlaps heavily with the existing i248 TSDP and
the registered counterfactual-move-delta networks. See `idea.yaml` for
the deferral rationale.

## Working thesis

For a position ``x`` with simple_18 board tensor:

1. Define static per-move-type reach masks ``M_t in {0, 1}^{64 x 64}`` for
   ``t in {knight, rank, file, diag, antidiag, king}``. Each mask captures
   the chess-rule geometric reach ignoring blockers: a queen at a1
   reaches every square on its rank / file / diagonal regardless of
   occupancy. ``M_knight`` is the 8-jump table, ``M_king`` is the
   adjacent-step mask, the four sliding masks come from the
   move-type-coded ``aligned`` table.
2. Project the per-square seed feature ``X in R^{B x 64 x d}`` through a
   per-type linear map ``W_t``.
3. Apply the move-type mask:

       Y_t[b, i] = sum_{j : M_t[i, j] = 1} (W_t X)[b, j]

4. Sum aggregated features across types:

       Y[b, i] = sum_t Y_t[b, i]

5. Pool ``Y`` and project to a scalar delta gated by trunk diagnostics.

## Why this matters

Standard Conv2d weights are indexed by spatial offset, so identical
chess-rule behaviour (e.g. "knight moves") must be relearned for every
square on the board. MKO ties weights across squares via the move-type
relation:

- ``W_knight`` learns "what a knight-leap-neighbour contributes" once and
  applies it at every square.
- ``W_diag`` learns "what a diagonal-aligned square contributes" once and
  applies it along every diagonal length.

This is a chess-specific weight-sharing pattern that no torch.nn op
provides. The closest analogue is depthwise-separable convolution, which
shares across channels but not across the chess-specific move-type
relation.

## What is actually proven

The construction is well-defined for any board. The shared-kernel
ablation reduces ``W_t -> W`` for all ``t`` and is the explicit "weight
sharing not load-bearing" control. The scalar-per-type ablation reduces
``W_t -> w_t * I`` (a scalar gain per type) and tests whether the
matrix-valued projection adds capacity beyond the scalar.

## What is only hypothesized

That per-type matrix projections beat both the shared kernel and the
scalar-per-type collapses on the long-range / move-type-specialised
slices.

## Failure cases

1. *Hidden rebrand of shared-kernel grid Conv*: if the unablated run
   matches ``shared_kernel``, MKO is a single conv layer with a sparse
   static mask; not a new primitive.
2. *Per-type scalar suffices*: if the unablated run matches
   ``scalar_per_type``, the matrix capacity is wasted -- the per-type
   scalar gain captures all the lift.
3. *Static reach overgenerates*: not blocker-resolved, so a queen at a1
   reaches every square on its rays regardless of occupancy. This is by
   design (matches the source primitive's framing) but may cause the
   model to over-attend to squares behind blockers.

## Falsifier

- ``shared_kernel`` -- collapse all 6 move types to one shared
  projection. Tests whether move-type weight sharing is load-bearing.
- ``scalar_per_type`` -- replace matrix per type with a scalar gain per
  type. Tests whether the matrix capacity is load-bearing.
- ``shuffle_features`` -- batch-permute the seed features so the
  rule-derived per-square input is decoupled from the position. Tests
  whether the rule features carry signal.
