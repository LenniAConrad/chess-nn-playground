# Math Thesis

Source: `ideas/research/primitives/external_06_high_risk_legal_graph_delta_state_primitives.md`
(Section 1, "Legal-Move Laplacian Pseudoinverse Propagation"; explicitly
identified as the highest-ranked proposal in that file).

## Working thesis

For a position `x` with simple_18 board tensor `(B, 18, 8, 8)`:

1. Compute the pseudo-legal move adjacency `A(x) in {0, 1}^{64 x 64}` per
   chess-rule geometry with blocker resolution. ``A_{ij}(x) = 1`` iff some
   own-color piece on plane square ``i`` can pseudo-legally move or capture
   to plane square ``j`` (sliding pieces are blocked by the first occupied
   square on the ray; pawn pushes require the target empty; knight / king
   moves are occlusion-free; edges to own-occupied targets are dropped).
2. Multiply rows of ``A`` by a learned per-piece scalar weight
   ``w(piece(i, x))`` to obtain ``W(x) = diag(w(piece)) A(x)``.
3. Form the signed Laplacian ``L(x) = D(x) - W(x)``, where
   ``D(x) = diag(row-sum W(x))``.
4. Build per-square features ``X in R^{B x 64 x d}`` via a small linear
   projection of the simple_18 piece-existence channels + side-to-move.
5. Apply the truncated Neumann-series resolvent

       Y = sum_{k=0..K} alpha^k * L^k * X * Theta,

   where ``alpha = alpha_init * tanh(alpha_logit)`` (so ``|alpha| <
   alpha_init`` by construction) and ``Theta in R^{d x d}`` is a learned
   mixing matrix.
6. Pool ``Y`` to a (B, head_hidden_dim) summary (own-piece-weighted average
   concatenated with the global mean) and project to a scalar logit delta.
7. Final logit ``= base_logit(i193 trunk) + gate(x) * delta``.

## Why this matters

Standard self-attention sees one hop per layer; the K-truncated resolvent
captures multi-hop tactical influence (X-rays, batteries, discovered
attacks) in a single operator application. Because the adjacency is the
*rule-determined* legal-move graph, the operator preserves bounded degree
(<= 27 per square) and is sparse in expectation -- though in this first
implementation we keep the dense (B, 64, 64) matmul because PyTorch's
sparse CSR matmul is poorly tuned for ~5% density on the 8x8 grid.

## What is actually proven

The construction is well-defined for any K >= 1 and any board tensor: the
Laplacian is a square matrix with non-negative degree row-sum, so the
Neumann partial sums are finite for finite K. The mathematical claim that
``sum_k alpha^k L^k`` approximates ``(I - alpha L)^{-1}`` requires
``|alpha * lambda_max(L)| < 1`` -- spectral clipping by power-iteration is
documented in the source primitive's failure-mode catalogue but is *not*
included in the first implementation; ``alpha_init`` is conservative by
default (``0.25``) and the ``tanh`` envelope keeps the effective alpha
bounded.

## What is only hypothesized

That K >= 2 outperforms K=1 (the GAT-with-legal-mask collapse). This is
the falsifier built into the ablation list as ``k1_gat_rebrand`` and is
the primary keep / drop signal for the primitive.

## Failure cases

1. *Hidden rebrand of GAT-with-mask* (K=1). Built-in ablation.
2. *Numerical instability when alpha * lambda_max approaches 1*. Mitigated
   by ``alpha_init`` cap; not yet by power-iteration spectral clipping.
3. *Sparse-density mismatch on consumer GPUs*. The 64x64 matmul is dense
   here. Asymptotic win materializes only when the kernel uses sparse
   CSR routines; tracked as a follow-up.

## Falsifier

- ``k1_gat_rebrand`` -- forces K=1 so the resolvent collapses to a single
  hop. If the unablated run (K=4) matches K=1 on the mate / hard-negative
  slices, the Neumann expansion is not load-bearing and the primitive
  reduces to legal-mask GAT -- drop.
- ``uniform_piece_weights`` -- disables the per-piece edge weighting so
  ``w(piece) := 1``. If the unablated run matches, the per-piece weight
  is not load-bearing -- the primitive can keep but the piece-conditioned
  variant should be simplified.
- ``shuffle_adjacency`` -- in-batch permutation of the legal-move graph.
  Decouples the rule indicator from the position. If the unablated run
  matches, the rule indicators carry no signal in this trunk.
