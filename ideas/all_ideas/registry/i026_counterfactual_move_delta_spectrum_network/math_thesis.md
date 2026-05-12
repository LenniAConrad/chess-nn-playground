# Math Thesis

Counterfactual Move-Delta Spectrum Network. Source packet:
`ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0429_tuesday_los_angeles_move_delta_spectrum.md`.

## Hypothesis

A position is puzzle-like when the rule-only one-ply move-delta neighbourhood
of the side to move induces a low-dimensional, anisotropic spectrum of
learned finite-difference responses. Equivalently, a puzzle position has a
small set of candidate deltas whose latent consequences point in a sharply
preferred direction, while non-puzzles tend to spread response variance
uniformly across many candidate deltas.

## Notation

Let `x in R^{C x 8 x 8}` be a `simple_18` board tensor and let `B(x)` be its
parsed current-board state (occupancy, side to move, castling, en-passant).
Let `A(x)` be the set of pseudo-legal one-ply move deltas of the side to
move, generated from `B(x)` by current-board piece movement rules without
engine evaluation, self-check filtering, or checkmate/stalemate oracles.

For each move `a in A(x)` we attach a deterministic descriptor

```
eta(x, a) = (from(a), to(a), piece(a), captured(x, a), delta_rank(a),
             delta_file(a), promotion(a), special(a)).
```

A board stem produces square features `H(x) in R^{64 x d}` and a global
feature `g(x) in R^{d_g}`. A response network produces

```
r_theta(x, a) = psi_theta(H(x)_{from(a)}, H(x)_{to(a)},
                          H(x)_{to(a)} - H(x)_{from(a)}, g(x), eta(x, a))
              in R^k.
```

With uniform weights `w_a = 1 / |A(x)|` the masked mean and covariance are

```
r_mean(x) = sum_{a in A(x)} w_a r_theta(x, a),
K(x)      = sum_{a in A(x)} w_a (r_theta(x, a) - r_mean(x))
                                (r_theta(x, a) - r_mean(x))^T  +  eps I_k.
```

`K(x)` is permutation invariant in the move enumeration order. By Rayleigh-
Ritz,

```
lambda_1(K(x)) / trace(K(x))
  = max_{||v||_2 = 1}
        Var_{a ~ U(A(x))}[v^T r_theta(x, a)]
        / ( E_{a ~ U(A(x))}[||r_theta(x, a) - r_mean(x)||_2^2] + eps k ),
```

so the leading-eigenvalue fraction is the largest share of total centred
move-response energy captured by a single learned counterfactual direction.

## Spectral statistics fed to the head

Let `lambda_1 >= ... >= lambda_k >= 0` and `tilde lambda_i = lambda_i /
trace(K)`. The classifier consumes:

- `g(x)`,
- `r_mean(x)`, `r_max(x)`, `r_var(x)` (masked),
- the eigenvalues `lambda_1, ..., lambda_k`,
- `trace(K)`, `lambda_1 / trace(K)`, `(trace K)^2 / trace(K^2)`,
  `- sum_i tilde lambda_i log tilde lambda_i`, `||K||_F`.

## Training objective

Cross-entropy on the puzzle-binary label `Y = 1[Y_f in {1, 2}]`, optionally
with a small finite-difference bottleneck `beta * E[trace(K)]` (default
`beta = 1e-4`, must be ablated at `beta = 0`).

## Falsification

The central falsification ablation replaces the candidate moves by a
degree-preserving randomised token set (same token count and source-piece
marginals, randomised destinations and move-type descriptors). If this
ablation matches the main model, the rule-only move-delta semantics are not
carrying the claimed signal.
