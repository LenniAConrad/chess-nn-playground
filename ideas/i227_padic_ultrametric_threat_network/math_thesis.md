# Math Thesis

## Working thesis

Tactical structure on the board is *tree-shaped*: king-zone defenders
cluster around the king, pinned pieces and discovered-attack rays form
nested chains. The natural distance between two squares is the depth at
which their threat-relations diverge, not Euclidean displacement. A finite
p-adic ultrametric

```text
d_p(s, t) = p^{-min{i : phi_i(s) != phi_i(t)}}
```

is the canonical algebraic structure with this property; it satisfies the
strong ultrametric inequality `|x + y|_p <= max(|x|_p, |y|_p)` and gives a
tree (Bruhat-Tits-style) rather than a Euclidean space.

## Setup

A learned encoder produces a soft `phi in R^{64 x k x p}` per square. The
expected divergence depth `E[min_diff]` is computed via cumulative prefix-
match probabilities, giving the ultrametric distance matrix `D`. A learned
relation head produces a soft p-adic relation matrix whose entries pass
through a *p-adic absorption* layer that respects valuation:

```text
M_p[s, t] = sum_{i = 0}^{k - 1} alpha_i f_i(phi_i(K_p[s, t])),
alpha_i = p^{-i}.
```

Spectral features of `M_p` (eigenvalues by magnitude and Newton-polygon
slope proxies) plus the depth histogram of `D` are classification inputs.

## Claim

`d_p` decorrelates puzzle vs near-puzzle better than the matched-rank
Euclidean embedding, especially on positions where threat structure is
deeply nested.

## Falsifiers

- `euclidean_swap`: replace `d_p` with `L2` over the same digits.
- `flat_alpha`: use `alpha_i = 1` for all `i`.
- `random_phi`: freeze `phi` to random.
