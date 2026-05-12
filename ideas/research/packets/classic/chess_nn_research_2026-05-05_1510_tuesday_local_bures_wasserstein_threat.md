# Codex Research Packet: Bures-Wasserstein SPD Threat Manifold Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1510_tuesday_local_bures_wasserstein_threat.md`
- Generated at: 2026-05-05 15:10
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full linear-algebra architecture packet, not implemented, not benchmark results

## One-Sentence Thesis

Embed each board as a learned **Symmetric Positive Definite (SPD)** threat-covariance
matrix `Sigma in S^d_{++}` and classify puzzle-likeness using the **Bures-Wasserstein**
geometry on `S^d_{++}` -- specifically, distances and tangent-space projections relative
to learned class-conditional Frechet means `mu_0`, `mu_1` -- whose closed form requires
the operator geometric mean `Sigma^{1/2}(Sigma^{1/2} mu Sigma^{1/2})^{1/2} Sigma^{1/2}`,
a genuinely Riemannian operation that no Euclidean-cosine or Fisher-Rao packet performs.

## Why This Is A Real Linear Algebra NN Idea

The Bures-Wasserstein metric on `S^d_{++}` is

```text
d_BW(A, B)^2 = trace(A) + trace(B) - 2 trace[ (A^{1/2} B A^{1/2})^{1/2} ]
```

equivalently the 2-Wasserstein distance between centered Gaussians `N(0, A)`, `N(0, B)`.
Its tangent space at `A` is parametrized by symmetric matrices via the Lyapunov solve
`A X + X A = V`. Means under this metric (Frechet means) are *not* arithmetic averages;
they require a fixed-point iteration

```text
mu_{t+1} = ( sum_i ( mu_t^{1/2} Sigma_i mu_t^{1/2} )^{1/2} ) ... etc.
```

This geometry is provably distinct from:

- **Fisher-Rao / affine-invariant** SPD metric (Cholesky-based, scale-invariant) used in
  `i083 Fisher-Geodesic Tension`.
- **log-Euclidean** which is just the Euclidean metric on `log Sigma`.
- **plain L2 / cosine** on flattened `Sigma`.

Bures-Wasserstein is the *unique* metric on SPD that arises from the L^2 Wasserstein
geometry of centered Gaussians and treats Sigma as an ellipsoidal 2-Wasserstein object,
not as an exponential-family natural parameter. That gives it different gradients,
different means, and different tangent structure from i083.

## Target

```text
fine 0,1 -> binary 0
fine 2   -> binary 1
3x2 fine-to-binary matrix mandatory.
```

## Forbidden Inputs

Standard: no engine, no PV, no source labels, no legal-move expansion. Current-board
tensors only.

## Closest Existing Ideas And Exact Difference

### Closest registered ideas

- `i083 Fisher-Geodesic Tension Network` — uses Fisher-Rao on learned distributions over
  squares, *not* SPD-manifold Bures geometry; squares-distribution Fisher-Rao is the
  multinomial-simplex Fisher metric, mathematically distinct from Gaussian-Wasserstein
  on covariances.
- `i029-i034 entropic-OT packets` — Sinkhorn coupling between piece/target measures over
  squares, not Wasserstein-2 between Gaussian covariances.
- `i133 Orthogonal Board Moment Network` — uses moments, not the SPD manifold.
- `i058 Determinantal Volume` — `det(Sigma)`, scalar, not Riemannian distance.

### Exact difference

```text
Bures-Wasserstein is the unique Riemannian metric induced by transport between centered
Gaussians on R^d, with operator-geometric-mean closed form. No imported packet (a) puts
boards into S^d_{++}, (b) computes class Frechet means under Bures, (c) classifies via
tangent-space projection at those means. Fisher-Rao (i083) lives on a probability
manifold over squares, not on the cone of covariance matrices, and uses different
geodesics, different means, and different gradients.
```

## Mathematical Thesis

### Definitions

Build a learned **threat-covariance** for each position:

```text
F = encoder(board) in R^{n x d},  n = 64,  d = 16..32
Sigma = (1/n) F^T F + eps I_d  in S^d_{++}
```

Sigma encodes which feature directions co-vary across the 64 squares -- in chess, this
captures correlated tactical pressure (e.g., diagonal-attack feature aligned with king-
shelter feature).

Maintain class-conditional Frechet means `mu_0, mu_1 in S^d_{++}` (one per binary class)
as **learnable** SPD parameters, projected to the SPD cone after each step via
`mu <- (mu + mu^T)/2 + max(0, -lambda_min) I`.

### Bures distance

```text
d_BW(Sigma, mu)^2 = tr(Sigma) + tr(mu) - 2 tr[(Sigma^{1/2} mu Sigma^{1/2})^{1/2}]
```

Compute via two symmetric matrix square-roots, both eigendecomposable at `d <= 32`.

### Tangent log-map at `mu`

The Bures log-map is

```text
log_mu(Sigma) = T_{mu -> Sigma} - I,    where T is the optimal transport map
T = mu^{-1/2} (mu^{1/2} Sigma mu^{1/2})^{1/2} mu^{-1/2}
```

(symmetric). This embeds each position into a flat tangent space at `mu_0` (or `mu_1`).

### Readout

```text
phi_0 = log_{mu_0}(Sigma) flattened to R^{d(d+1)/2}
phi_1 = log_{mu_1}(Sigma)
d_BW0 = d_BW(Sigma, mu_0)
d_BW1 = d_BW(Sigma, mu_1)
puzzle_logit = MLP([phi_0, phi_1, d_BW0 - d_BW1, board_pool, log_det(Sigma)])
```

The signed Bures-distance gap `d_BW0 - d_BW1` is the *natural* puzzle score; the MLP
provides residual capacity around it.

## Assumptions

- Threat structure has a Gaussian-covariance interpretation: feature directions across
  squares behave like a centered Gaussian whose covariance summarizes the position.
- Puzzles cluster in a Bures-Wasserstein region distinct from non-puzzles, *more* than
  they cluster under any flat (cosine / log-Euclidean) metric.
- Class-conditional Bures means are well-defined (the dataset is balanced enough; if
  not, we use a small-batch streaming Frechet update).

## Claim / Hypothesis

If the underlying class structure of puzzle-likeness is genuinely manifold-shaped on
SPD, then a Bures-Wasserstein head will:

1. Beat the same-feature-extractor + log-Euclidean head on PR AUC and near-puzzle FPR.
2. Beat i083 Fisher-Geodesic on the same metric pair, because the relevant manifold is
   covariance-space not simplex-space.
3. The `log_euclidean_only` ablation should drop PR AUC by `>= 0.015`; the
   `frechet_means_replaced_by_arithmetic_means` ablation should drop PR AUC by
   `>= 0.01`.

## Architecture

### Components

```text
board_encoder            -> F in R^{64 x d}
sigma_builder            -> Sigma = F^T F / 64 + eps I_d
spd_root_block           -> Sigma^{1/2} via eigendecomp
bures_distance_block     -> d_BW(Sigma, mu_c) for c in {0,1}
log_map_block            -> log_{mu_c}(Sigma)
class_means              -> learnable mu_0, mu_1 in S^d_{++}
puzzle_head              -> MLP
```

### Forward pseudocode

```text
F          = board_encoder(board)
Sigma      = (F.T @ F) / 64 + eps * I(d)
S_half     = sym_sqrtm(Sigma)               # eigh-based, d x d
for c in {0, 1}:
    R      = sym_sqrtm(S_half @ mu[c] @ S_half)
    d_BW_c = sqrt( max(0, tr(Sigma) + tr(mu[c]) - 2 tr(R)) )
    T_c    = sym_sqrtm( mu_inv_half[c] @ Sigma @ mu_inv_half[c] ... )
    phi_c  = vec_sym(T_c - I_d)
gap        = d_BW_0 - d_BW_1
logit      = MLP([phi_0, phi_1, gap, log_det(Sigma), pool(F)])
```

### First config

```yaml
model:
  name: bures_wasserstein_threat_network
  input_channels: 18
  num_classes: 2
  hidden_dim: 96
  spd_dim_d: 24
  spd_floor_eps: 1.0e-3
  use_class_means: true
  frechet_lr: 1.0e-2
  frechet_update: implicit_sgd     # alt: closed_form_2class
training:
  mode: puzzle_binary
  loss: bce_with_logits
  batch_size: 512
  learning_rate: 1.0e-3
```

## Numerical / Compute Notes

- All matrix ops at `d = 24`. `eigh` cost `O(d^3) = 1.4e4` per board, negligible.
- Symmetric square root via `eigh`: `Sigma = U diag(s) U^T -> Sigma^{1/2} = U diag(sqrt(s)) U^T`.
- Use `s_clipped = max(s, eps)` to keep gradients stable.
- For the Frechet means `mu_0, mu_1`, two clean options:
  1. Learnable SPD parameters with reprojection step (`Cholesky` parametrization).
  2. Closed-form 2-Gaussian Bures mean updated EMA from class-mean Sigma per batch.
- Implicit-function autograd for the matrix square root is available via PyTorch's
  `linalg.eigh` and is differentiable as long as eigenvalues stay separated; the `eps I`
  floor guarantees this.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `log_euclidean_only` | replace Bures dist by `||log Sigma - log mu_c||_F` | tests manifold choice |
| `cosine_flat_only` | flatten Sigma, cosine to mu_c | tests need for Riemannian |
| `arithmetic_means` | replace Frechet means by arithmetic class means | tests Frechet step |
| `single_mean_only` | use one shared `mu`, not class-conditional | tests class structure |
| `random_features_F` | random fixed projection F | tests learned features |
| `i083_fisher_geodesic` | run i083 head on same F | adjacent-manifold baseline |
| `cnn_same_params` | size-matched CNN | matched-capacity |
| `i030_nuisance_orthogonal` | orthogonal-projection bottleneck | adjacent-bottleneck |

For each: full 3x2 + slice reports.

## Benchmark Targets

```text
test PR AUC      >= 0.82
test F1          >= 0.76
near-puzzle FPR  <= 0.20
puzzle recall    >= 0.78

central claim:
  log_euclidean_only drops PR AUC >= 0.015
  arithmetic_means   drops PR AUC >= 0.01
  bures beats i083 fisher_geodesic  by  >= 0.01 on PR AUC
```

## Counterexamples / Failure Modes

- Centered-Gaussian interpretation of threat features is wrong; the relevant geometry is
  not on covariances.
- `d = 24` is too small to capture separable threat directions.
- Frechet means collapse to nearly equal `mu_0 ~ mu_1` so the gap signal vanishes;
  mitigate with a `||mu_0 - mu_1||_BW`-driven separation regularizer.
- Square-root gradient instability near degenerate spectra.

## Implementation Priority

1. Encoder produces `F in R^{64 x 24}`. PSD `Sigma = F^T F / 64 + eps I`.
2. `sym_sqrtm` and `bures_dist` utility, fully differentiable.
3. Two learnable SPD class means with reprojection.
4. Distance-only head: `logit = a * (d_BW_0 - d_BW_1) + b`. Train; check signal.
5. Add tangent log-map features; combine with CNN board pool.
6. Run all 8 ablations and full slice report.

Smallest viable version:

```text
F = simple linear projection of simple_18 features, d = 16,
fixed (uniformly initialized) class means, distance-only head.
```

If that already shows lift, scale F, add tangent features, and learn the class means.
