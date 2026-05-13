# Math Thesis

Source: `ideas/research/primitives/external_39_orbit_irrep_hodge_projection_primitives.md`
(Section 2 `primitive_weighted_hodge_projector`).

## Working thesis

For a position with simple_18 board tensor:

1. Build the 8x8 grid complex with
   - 64 vertices (one per square),
   - 112 oriented edges (56 horizontal right-pointing + 56 vertical
     down-pointing), and
   - 49 unit-square faces (each a 1x1 cell).

   The vertex-edge incidence ``D_0 in {-1, 0, 1}^{64 x 112}`` has +1
   at each edge's head and -1 at its tail. The edge-face incidence
   ``D_1 in {-1, 0, 1}^{112 x 49}`` carries the cycle orientation
   (top -> right -> -bottom -> -left). With these orientations
   ``D_1^T D_0^T = 0``, i.e. ``image(D_0^T)`` and ``image(D_1)`` are
   orthogonal subspaces of the edge space (in the unweighted L2 inner
   product).
2. Pool the i193 spatial features ``S in R^{B x 2C x 8 x 8}`` and read
   per-edge endpoint features by gathering ``S`` at head and tail
   squares. ``edge_proj`` produces ``E in R^{B x 112 x d_edge}``.
3. Per-edge flow channels:

       F = edge_flow_head(E) in R^{B x 112 x C}.

   Per-edge metric (input-dependent positive weight):

       w = softplus(metric_head(E)) in R_+^{B x 112}.
4. Weighted Hodge decomposition. Let ``W_b = diag(w_b)``:

       G_b = D_0^T (D_0 W_b D_0^T + eps I_64)^{-1} D_0 W_b F_b
       R_b = F_b - G_b
       Cr_b = D_1 (D_1^T W_b D_1 + eps I_49)^{-1} D_1^T W_b R_b
       H_b = R_b - Cr_b.

   ``(G, Cr, H)`` are pairwise orthogonal in the W-weighted inner
   product. The two solves are batched
   ``torch.linalg.solve`` against SPD matrices with the eps shift
   ensuring strict positive definiteness.
5. Per-component energy summary (mean + max edge magnitude per channel,
   concatenated across the three components) and the trunk joint pool
   feed a delta MLP. Gate MLP consumes the energy scalars only.
6. Output: ``final_logit = base_logit + primitive_gate *
   primitive_delta_raw``.

## Why this matters

Tactical pressure on a chess board has natural directional structure:
direct attacks on the king look like a gradient flow ending at the
king square; circulating manoeuvres around blocked pieces look like
curl; fortress-like positions leave harmonic (divergence- and curl-
free) residuals. A conv layer or attention block has no native way to
separate these — they need to learn it from data. The Hodge primitive
gives that decomposition directly.

## What is actually proven

- ``D_1^T D_0^T == 0`` for the chosen orientations (verified by the
  construction in `_build_incidence_matrices`). Without this, the
  gradient and curl projections would not be orthogonal.
- For positive ``w``, both ``D_0 W D_0^T`` and ``D_1^T W D_1`` are
  positive semi-definite; the ``+ eps I`` shift makes them strictly
  PD so ``torch.linalg.solve`` is stable.
- The decomposition ``F = G + Cr + H`` is exact (modulo the eps-shift
  approximation; eps = 1e-2 by default).

## What is only hypothesized

That separating gradient / curl / harmonic energies contains
discriminative information for the chess-nn-playground splits beyond
what the i193 trunk and its conv layers already encode.

## Failure cases

1. *Hidden rebrand of conv*: tested by `uniform_metric` (set ``W = I``).
   With a fixed metric, the projection becomes a fixed linear map of
   the flow; if it still matches the unablated run, the input-
   dependent metric is not load-bearing.
2. *Curl is dead*: tested by `drop_curl` (zero the curl branch).
3. *Gradient is dead*: tested by `drop_gradient`.
4. *Harmonic is dead*: tested by `drop_harmonic`.
5. *Solver ill-conditioning*: ``solve_eps`` capped at 1e-2; tested by
   running the K=3 falsifier matrix with random-init weights.

## Falsifier

- `uniform_metric` — primary. Fixes ``W = I``; the decomposition is
  now a fixed linear projection. If unablated matches, the input-
  dependent metric is not load-bearing.
- `drop_curl` / `drop_gradient` / `drop_harmonic` — per-component
  ablations. At least one of these should significantly drop the
  unablated lift; otherwise the decomposition is uninformative.
- `shuffle_edge_flow` — in-batch permutation of the flow tensor
  (edge index axis). Decouples flow from board geometry.
