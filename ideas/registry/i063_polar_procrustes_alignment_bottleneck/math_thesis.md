# Math Thesis

Polar-Procrustes Alignment Bottleneck

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2104_friday_shanghai_polar_procrustes.md`.

Working thesis: puzzle-likeness is tested by **the orthogonal Procrustes
alignment between learned own / opponent role matrices and the
polar-decomposition strain spectrum of their cross-covariance**.

## Setup

Let ``x`` be a current board. Extract occupied tokens, split them into
side-to-move-relative own / opponent sets, and pass each side through
a shared MLP. Pool each side into a role matrix via masked-softmax
learned role queries:

```text
X(x) in R^{R x D}     # own role matrix
Y(x) in R^{R x D}     # opponent role matrix
```

Optionally row-normalise ``X`` and ``Y``. Build the cross-covariance

```text
C(x) = X(x)^T Y(x) / R    in R^{D x D}     (matrix_space = "embedding")
C(x) = X(x) Y(x)^T / D    in R^{R x R}     (matrix_space = "role")
```

## Procrustes And Polar Decomposition

The orthogonal Procrustes problem asks for

```text
Q*(x) = argmin_{Q^T Q = I} ||X(x) Q - Y(x)||_F
```

If ``C = U Sigma V^T``, then

```text
Q*  = U V^T
H   = V Sigma V^T
C   = Q* H                              (polar decomposition)
min_Q ||X Q - Y||_F^2 = ||X||_F^2 + ||Y||_F^2 - 2 sum_i sigma_i(C)
```

so the singular spectrum of ``C`` controls how well ``Y`` can be
matched to ``X`` by any orthogonal alignment, and ``H`` is the
symmetric "strain" remaining after the alignment.

## Hypothesis

Puzzle-like positions may involve a sharply structured mismatch
between own and opponent role summaries — a tactical position is not
only "own pieces have features" or "opponent pieces have features",
but about whether the two sets can be brought into alignment by a
coherent role rotation, and where that alignment fails.

## What Is Proven

- The Procrustes residual and singular values of ``C`` are invariant
  under shared right-orthogonal coordinate changes
  ``X -> X W``, ``Y -> Y W``.
- ``Q*`` and ``H`` are uniquely determined by ``C`` when ``C`` has
  distinct nonzero singular values; the diagonal-tilt regulariser
  ``+ cross_cov_eps * diag(1, 2, ..., M) / M`` ensures backward through
  ``torch.linalg.svd`` stays finite.
- Removing Procrustes terms while preserving separate singular values
  of ``X`` and ``Y`` directly falsifies the claimed alignment signal —
  this is the ``separate_matrix_stats_only`` ablation.

## What Remains Hypothesised

- That learned role matrices correspond to chess-relevant side / role
  summaries.
- That puzzle-like positions have distinctive alignment residuals or
  strain spectra.
- That the current split is not dominated by material or source
  artifacts (controlled by ``material_only_matrices`` and the
  CRTK-tagged splits).

## Counterexamples And Self-Critique

- Labels are driven by material imbalance or source artifacts.
- Own / opponent matrices are too sparse in low-material positions.
- Tactical signal requires legal-move consequences and cannot be
  captured by static role alignment.
- Separate own / opponent spectra already contain all useful
  information.

These risks are tested by the ``material_only_matrices``,
``separate_matrix_stats_only``, ``identity_alignment_only``,
``random_orthogonal_alignment``, ``batch_shuffled_opponent``,
``role_pool_mean_only``, and ``singular_values_only`` falsifiers
exposed via ``model.ablation``.
