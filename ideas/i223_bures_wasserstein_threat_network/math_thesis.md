# Math Thesis

## Working thesis

If puzzle structure has a Gaussian-covariance interpretation -- feature
directions co-varying across the 64 squares behave like a centered Gaussian
whose covariance summarizes tactical pressure -- then the puzzle vs
near-puzzle decision should be sharper in the Bures-Wasserstein geometry on
`S^d_{++}` than under any flat (cosine, log-Euclidean) metric.

## Setup

Build per-board features `F in R^{64 x d}` from a shared encoder, and the
threat covariance `Sigma(x) = (1 / 64) F^T F + eps I_d in S^d_{++}`.
Maintain learnable class Frechet means `mu_0, mu_1 in S^d_{++}`. The
Bures-Wasserstein distance is

```text
d_BW(Sigma, mu)^2 = tr(Sigma) + tr(mu) - 2 tr[(Sigma^{1/2} mu Sigma^{1/2})^{1/2}]
```

and the Bures tangent log map at `mu` is

```text
log_mu(Sigma) = T - I,
T = mu^{-1/2} (mu^{1/2} Sigma mu^{1/2})^{1/2} mu^{-1/2}.
```

## Claim

The signed Bures-distance gap `d_BW(Sigma, mu_0) - d_BW(Sigma, mu_1)` should
beat any log-Euclidean / cosine head sharing the same encoder; the operator-
geometric-mean closed form is the unique Riemannian metric induced by
2-Wasserstein transport between centered Gaussians.

## Falsifiers

- `log_euclidean_only`: replace Bures by `||log Sigma - log mu_c||_F`.
- `arithmetic_means`: replace Frechet means by arithmetic class means.
- `single_mean_only`: use one shared `mu` rather than class-conditional.
