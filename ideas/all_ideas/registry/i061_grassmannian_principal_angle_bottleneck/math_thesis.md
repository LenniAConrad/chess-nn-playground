# Math Thesis

Grassmannian Principal-Angle Bottleneck

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2058_friday_shanghai_grassmannian_angles.md`.

## Working Thesis

Puzzle-like positions may not merely have unusual individual pieces or
local motifs. They may arrange piece-role evidence so that learned
subspaces of an embedded board align, separate, or become nearly
orthogonal in ways associated with tactical tension. The **principal
angles** between two role-gated subspaces directly measure this relative
geometry without invoking move generation, engine search, attack graphs,
or square-token self-attention.

## Setup

Let `x` be the simple_18 board tensor and `S(x) = {h_i}_{i=1}^N` an
embedding of its `N <= 32` occupied tokens (12 piece-color one-hots,
own/enemy flag, absolute and side-relative coordinates, castling and
en-passant broadcast features) through a learned token MLP to
`R^D`. Define a learned role gate `g_{r,i}(x) in [0, 1]` per token and
role `r in {1, ..., R}` and the weighted role covariance

```
mu_r(x) = sum_i g_{r,i}(x) h_i / (sum_i g_{r,i}(x) + eps)
C_r(x)  = sum_i g_{r,i}(x) (h_i - mu_r)(h_i - mu_r)^T + eps * I_D.
```

Let `Q_r(x) in R^{D x K}` be the top-`K` orthonormal eigenvectors of
`C_r(x)` (computed by `torch.linalg.eigh`). Each column-space
`span(Q_r(x))` is a point on the Grassmannian `Gr(K, D)` because the
subspace -- not the particular basis -- carries the geometric
information.

For an unordered role pair `(a, b)` with `a < b` the principal-angle
spectrum is the singular spectrum of the cross-Gram matrix:

```
M_{a,b}(x)  = Q_a(x)^T Q_b(x)               in R^{K x K}
sigma_{a,b} = svdvals(M_{a,b})              in [0, 1]^K
theta_{a,b} = arccos(clamp(sigma_{a,b}))    in [0, pi/2]^K.
```

`sigma_{a,b,1}` is the cosine of the smallest principal angle (most
aligned direction); `sigma_{a,b,K}` is the cosine of the largest
principal angle (most separated direction).

## Permutation And Basis-Rotation Invariance

`C_r(x)` is a sum over tokens, so token reordering does not change `C_r`
or `Q_r`. The angle spectrum is invariant to basis rotation inside each
subspace because

```
(Q_a U)^T (Q_b V) = U^T (Q_a^T Q_b) V
```

has the same singular values for any orthogonal `U, V in R^{K x K}`.

## What Is Actually Proven

- The angle spectra are permutation-invariant over occupied tokens.
- The angle spectra are invariant to basis rotation inside each learned
  subspace.
- Removing cross-role principal-angle features while preserving
  eigenvalue spectra (`no_cross_angles` / `eigenvalues_only` ablations)
  directly falsifies the claimed role-geometry signal.
- The `pooled_token_head` ablation matches head capacity through a
  learned mean / max / std pooling so a positive result is not just
  more capacity in the head.

## What Remains Hypothesised

- That the learned role gates discover chess-relevant subspaces.
- That puzzle-like positions have distinctive principal-angle spectra.
- That the current split is not dominated by material / source
  shortcuts.

## Counterexamples

- Labels are mostly material, phase, or source artifacts.
- Role gates collapse into identical subspaces for all positions
  (monitored via `pair_mean_angle_std`).
- Useful signal requires explicit move consequences that static
  subspace geometry cannot capture.
- Angle spectra are too coarse and lose square-specific tactics.

## Self-Critique

The model is mathematically clean but may be too abstract for chess.
Principal angles do not know about legal moves, pins, checks, or forced
lines. The bespoke implementation exposes the `no_cross_angles`,
`batch_shuffled_angles`, `eigenvalues_only`, `pooled_token_head`, and
`no_orthonormalization` falsifiers from section 9 directly through the
`model.ablation` config flag, so the experiment can decisively rule the
mechanism out if the falsifiers match the main model.
