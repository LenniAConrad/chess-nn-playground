# Math Thesis

Source: `ideas/research/primitives/external_25_dynamic_adjacency_rank_order_involution_gate.md`
(Section 1, "Dynamic Adjacency-Conditioned Gating").

## Working thesis

For a position ``x`` with simple_18 board tensor:

1. Build the pseudo-legal adjacency ``A(x) in {0, 1}^{64 x 64}`` as in
   the LM-LPP primitive (legal_move_graph helper).
2. Decompose ``A(x)`` by chess-rule move type ``t in T``:

       A_t(x) = A(x) ⊙ 1[move_type(i, j) = t]

   where ``T = {RANK, FILE, DIAG, ANTIDIAG, KNIGHT, KING, PAWN_PUSH, PAWN_CAPTURE}``.
3. Project the per-square seed feature ``X in R^{B x 64 x d}`` through a
   per-type linear map ``W_t: R^d -> R^d``.
4. Apply the binary mask:

       Y_t[b, i] = sum_{j : A_t[b, i, j] = 1} (W_t X)[b, j]
                 = (A_t(x) @ W_t X)[b, i]

   The aggregation matches the source primitive's defining equation

       y = (G(x) ⊙ Wx) + b

   per move-type slot.
5. Sum aggregated features across types and pool to a hidden vector.
6. Project to a scalar logit delta gated by ``sigmoid(MLP(diagnostics))``.

## Why this matters

The source primitive's framing is "hard, discrete topological constraint
directly into the kernel" -- the mask is binary by design, the gradient
of an illegal-edge cell is zero by construction. The per-type
decomposition gives the model a chance to specialise on the move-type
class that drives the position (open files vs diagonal pin lattice vs
knight outpost). The i193 trunk's general-purpose conv weights average
over move types implicitly; DAG forces the separation.

## What is actually proven

The construction is well-defined for any board. For each type slot ``t``
the masked aggregation is exactly the source primitive's operator with
``W = W_t`` and binary ``G_t``. The sum of the type slots equals applying
a single linear map to the union of move types, which the
``single_move_type`` ablation collapses to as a control.

## What is only hypothesized

That the per-type decomposition outperforms a shared kernel
(`single_move_type` ablation). This is the primary falsifier.

## Falsifier

- ``single_move_type`` -- collapse to one shared projection over the
  union of move types. If the unablated run matches the collapse, the
  per-type weight sharing is the source of the lift, not the mask
  decomposition.
- ``soft_mask`` -- replace the binary mask with sigmoid(2 * (A - 0.5))
  (still close to 0/1 but continuous). If the unablated run matches, the
  hard-mask story is not load-bearing -- the primitive collapses to a
  soft-mask attention variant.
- ``uniform_adjacency`` -- replace ``A`` with the all-ones (minus
  identity) adjacency. If the unablated run matches, the legal-move
  graph carries no signal beyond the existing i193 trunk features.
- ``shuffle_adjacency`` -- batch-permute the adjacency. Decouples rule
  indicators from position. If unablated matches, the primitive's
  rule-derived structure is not load-bearing.
