# Math Thesis

## Working thesis

Treat the Reynolds-style orbit *residuals* of safe board transforms as evidence,
rather than projecting them away as is done in invariant orbit models.

Let `f` be a shared encoder and `G = {g_0, ..., g_{|G|-1}}` a finite set of safe
board transforms (identity, file flip, rank flip, 180-degree rotation, color
flip). The orbit latent matrix is

```text
Z = [ f(g_0(x)); ...; f(g_{|G|-1}(x)) ]    in R^{|G| x d}
```

The Reynolds projection / orbit mean is

```text
mu(x) = (1 / |G|) sum_g f(g(x))
```

The orbit residual `R = Z - 1 mu(x)^T` carries every direction in which the
encoder fails to be invariant. Its norm spectrum (residual covariance trace and
off-diagonal norm) and the variance of per-view logits together form a
disagreement fingerprint that pure orbit-pooling models throw away.

## Claim

For an exact symmetry group `G` acting on chess-legal positions, a feature
extractor that respects the rules should produce zero residuals. Any non-zero
residual reflects either (a) a source artifact (e.g. orientation of generated
puzzles) or (b) genuine asymmetry in tactical pressure that is informative
for the binary puzzle decision.

## Falsifiers

- `invariant_mean_only`: drop residual statistics. If PR-AUC matches, the
  disagreement signal is decorative.
- `random_pseudo_orbit`: replace safe transforms with random pixel
  permutations of matched count. Should not match the safe-transform model.
- `disagreement_stopgrad`: stop gradient through the disagreement branch.
