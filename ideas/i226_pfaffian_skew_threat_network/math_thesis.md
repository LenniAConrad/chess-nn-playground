# Math Thesis

## Working thesis

Attacker-defender pairing has natural orientation: attacker `i` engaging
defender `j` is *signed* (like a perfect matching of a directed graph). For
a skew-symmetric `K = -K^T`, the Pfaffian `pf(K)` is the unique polynomial
satisfying `pf(K)^2 = det(K)` and
`pf(M^T K M) = det(M) pf(K)`, and its absolute value enumerates perfect
matchings *with sign*. Near-puzzle and puzzle positions can have matched
`||K||_F` but very different signed enumerator due to orientation
cancellation.

## Setup

Build `K in R^{2m x 2m}` skew-symmetric from learned upper-triangle entries.
Compute `log|pf(K)|` and a sign proxy via the eigenvalue product, plus a
sub-Pfaffian fingerprint over a fixed family of even-sized index subsets.

## Claim

`(pf(K), sign_balance)` are not derivable from spectrum or singular-value
features of `K` alone. The model should beat any spectrum-only baseline at
puzzle vs near-puzzle separation when those have matched `||K||_F`.

## Falsifiers

- `abs_only`: replace `pf` with `|pf|` and zero the sign balance.
- `det_swap`: replace `pf(K)` with `sqrt(det(K))`.
- `force_symmetric_K`: use `(E + E^T)/2`. Then `pf = 0` and the model
  collapses to a constant.
