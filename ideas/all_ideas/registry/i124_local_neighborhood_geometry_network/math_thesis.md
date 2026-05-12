# Math Thesis

Local Neighborhood Geometry Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.

Batch candidate rank: `6`.

Working thesis: A puzzle-like position may be locally sharp: small
current-board perturbations such as removing one piece plane, masking one
square neighborhood, or reflecting a safe orientation can move its
representation more than a quiet non-puzzle position. The classifier can
measure the geometry of a deterministic neighborhood around each board.

## Formal setup

Let `phi: R^{C x 8 x 8} -> R^D` be a shared encoder applied to every view
of a board.  For a board `x` and a deterministic perturbation set
`{T_1, ..., T_V}` (with `T_1 = identity`), we form the embedding cloud
`E(x) = (phi(T_1 x), ..., phi(T_V x))` and the centred deltas
`d_i(x) = phi(T_i x) - phi(T_1 x)` for `i >= 2`.

The local-sharpness signal is the geometry of `E(x)`:

- per-view delta norms `||d_i(x)||`
- pairwise cosines `<d_i(x), d_j(x)> / (||d_i|| ||d_j||)`
- the top-K eigenvalues of the centred `V x V` Gram matrix
  `G(x) = (E(x) - mean E(x))(E(x) - mean E(x))^T / D`
- mean and max pairwise distances of `E(x)`
- anisotropy ratio `lambda_1(G) / sum_k lambda_k(G)`

The puzzle head consumes `[phi(T_1 x), geometry(E(x))]` and predicts a
puzzle logit.

## Why this should separate puzzle from non-puzzle

If puzzle-like positions sit in sharper local basins than quiet positions,
small board-only perturbations should move their embeddings farther.
Concretely:

- delta norms and mean / max pairwise distances are larger for sharp
  basins;
- the local Gram spectrum is more anisotropic (a larger top eigenvalue
  relative to the sum) when one perturbation direction dominates the local
  response;
- pairwise cosines between deltas indicate whether perturbations push the
  embedding along a single shared direction (e.g. removing one critical
  piece collapses tactical content) or along orthogonal directions
  (multiple independent local cues).

The model is *not* told that any label is invariant under any
perturbation; it only measures local response.  Some perturbations
(horizontal mirror without a side swap, piece-type group dropout) are
intentionally not chess-semantics-preserving — they act as diagnostic
probes whose response, not whose invariance, is informative.
