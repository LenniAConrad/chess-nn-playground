# Codex Research Packet: Pfaffian Skew-Symmetric Threat Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1525_tuesday_local_pfaffian_skew_threat.md`
- Generated at: 2026-05-05 15:25
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full unorthodox-linear-algebra architecture packet, not implemented, not benchmark results

## One-Sentence Thesis

Build a learned **skew-symmetric** chess interaction operator `K = -K^T in R^{2m x 2m}`,
compute its **Pfaffian** `pf(K)` (the unique polynomial in `K`'s entries with
`pf(K)^2 = det(K)` and `pf(P^T K P) = det(P) pf(K)`), and use signed-Pfaffian patterns
of square submatrices as features: the Pfaffian is the **signed enumerator of perfect
matchings** of the skew-graph encoded by `K` and is provably distinct from any
determinant, permanent, eigenvalue, or singular-value invariant.

## Why This Is A Real And Unorthodox Linear Algebra NN Idea

For an antisymmetric `K = -K^T`, `det(K) >= 0` always, and there is a unique polynomial
`pf(K)` of degree `m` (in entries of `K in R^{2m x 2m}`) such that

```text
det(K) = pf(K)^2
```

and (Cayley's identity) for any square `M`:

```text
pf(M^T K M) = det(M) * pf(K)
```

When `K` is the signed adjacency of a graph, `|pf(K)|` equals the number of perfect
matchings (FKT theorem on planar graphs). So `pf(K)` is an *oriented* count of perfect
matchings -- something neither the determinant (which double-counts via permutation
parity) nor the permanent (which is unsigned and #P-hard) gives.

For our purposes:

- **Attacker-defender pairing** is naturally a matching: each attacker engages at most
  one defender, each defender shields at most one attacker.
- A skew-symmetric `K` with `K_{ij} > 0` if attacker `i` engages defender `j` (oriented)
  encodes that pairing.
- `pf(K)` then counts oriented engagements *modulo cancellation*; near-puzzle and puzzle
  positions can have the same `|K|`-norm but very different `pf(K)` due to sign
  cancellation.

This is fundamentally distinct from `det`, `perm`, `eigvals`, or DPP volume (i058),
because all of those are sign-symmetric in pairs of rows. The Pfaffian is the unique
linear-algebraic invariant that exposes orientation-of-matching.

## Target

```text
fine 0,1 -> binary 0
fine 2   -> binary 1
3x2 fine-to-binary matrix mandatory.
```

## Forbidden Inputs

Standard.

## Closest Existing Ideas And Exact Difference

### Closest registered

- `i058 Determinantal Tactical Volume` — uses `det`, not Pfaffian; sign-symmetric.
- `i096 Oriented Matroid Covector Bottleneck` — uses sign vectors of an oriented
  matroid; closest in *spirit* but different algebraic invariant (no polynomial
  identity), no `pf^2 = det` relation, no row-column orientation.
- `i133 Orthogonal Board Moment` — uses moments, not signed matchings.
- `i120 Sinkhorn Role Assignment` — produces a doubly-stochastic *unsigned* matching.

### Exact difference

```text
The Pfaffian is the unique polynomial invariant of skew-symmetric matrices satisfying
pf^2 = det and pf(M^T K M) = det(M) pf(K). It enumerates perfect matchings with sign,
which no imported packet computes. The signed enumeration is what makes near-puzzle
and puzzle positions separable when their |K|-norms collapse together.
```

## Mathematical Thesis

### Definitions

Build a skew-symmetric operator over `2m = 32` *paired* squares (16 attacker-relevant
squares + 16 defender-relevant squares, chosen by a soft mask):

```text
K = sum_p g_p(X_sq) (E_p - E_p^T) / 2     in R^{2m x 2m},  K = -K^T
```

with `E_p` fixed legal-geometry primitives (attacker-attacks-defender, x-ray,
discovered-line, blocker-vacation). The skew structure is enforced by construction.

### Pfaffian computation

Closed form via determinant of the upper triangle:

```text
pf(K) = sqrt(det(K))   modulo a sign chosen consistently by the algorithm
```

We compute `pf(K)` in the differentiable Faddeev / Householder-tridiagonalization style
(reduce `K` to skew-tridiagonal `T` via orthogonal congruence; then
`pf(K) = prod_i T_{2i-1, 2i} * sign(orth)`). PyTorch `torch.linalg.qr` plus a few
manually-implemented Householder reflections suffice; `m = 16` so `2m = 32` is cheap.

### Submatrix Pfaffian fingerprint

For a fixed family of submatrix index sets `I_q subset {1,...,2m}` (each of even size),
compute `pf(K[I_q, I_q])`. The signed vector of these sub-Pfaffians is the Pfaffian
fingerprint:

```text
phi_pf = (pf(K[I_q, I_q]))_q  in R^Q,    Q = 16..32
```

Choose `I_q` to correspond to chess-natural submatch families: king-zone defenders,
critical attackers, pinned pieces, x-ray pairs.

### Readout

```text
pf(K)                                    scalar (signed)
log|pf(K)|                               scalar
sub_pf_fingerprint phi_pf                R^Q
sign_balance = mean( sign(pf(K[I_q,I_q])) )
||K||_F, top-k singular values of K
```

Final:

```text
puzzle_logit = MLP([pf, log|pf|, phi_pf, sign_balance, sigma_topk, board_pool])
```

## Assumptions

- True puzzles correspond to *oriented* attacker-defender pairings whose signed
  enumeration `pf(K)` is *small in magnitude* relative to `||K||_F^m` (signs cancel),
  because no defender configuration matches all attackers in the natural orientation.
- Non-puzzles have `pf(K)` close to `||K||_F^m / m!` style maximal alignment.
- Near-puzzles can mimic non-puzzle `||K||_F` but have different `sign_balance`.

## Claim / Hypothesis

`(pf(K), sign_balance)` are not derivable from spectrum or singular-value features of
`K` alone. The model should beat any spectrum-only baseline at distinguishing puzzle
from near-puzzle when those have matched `||K||_F`.

Central falsifier:

```text
abs_only_ablation: replace pf with |pf| and sign_balance with 0.
                  if PR AUC and near-puzzle FPR don't degrade, the orientation does
                  not carry signal.
```

## Architecture

### Components

```text
board_encoder
soft_paired_square_selector  -> chooses 32 squares paired (attacker, defender)
skew_K_builder               -> K = sum_p g_p (E_p - E_p^T)/2
householder_tridiag          -> K -> T skew-tridiagonal
pfaffian_block               -> pf(T) = product of off-diagonals
sub_pfaffian_block           -> {pf(K[I_q, I_q])}
puzzle_head
```

### Forward pseudocode

```text
X_sq = board_encoder(board)
idx  = soft_paired_square_selector(X_sq)         # straight-through to top-32 pairs
K    = skew_K_builder(X_sq, idx)                  # 32 x 32, skew
T, P = householder_skew_tridiag(K)                # T skew-tridiag, P orthogonal
pf_K = product over i of T[2i-1, 2i] * det_sign(P)
phi  = sub_pfaffian_block(K, family_of_I)         # Q values
feat = [pf_K, log|pf_K| + eps, phi, sign_balance(phi), ||K||_F, sigma_topk(K)]
logit = MLP([feat, pool(X_sq)])
```

### First config

```yaml
model:
  name: pfaffian_skew_threat_network
  input_channels: 18
  num_classes: 2
  hidden_dim: 96
  paired_squares_2m: 32
  num_E_primitives: 12
  num_subsets_Q: 24
  pfaffian_algo: householder_tridiag
training:
  mode: puzzle_binary
  loss: bce_with_logits
  batch_size: 512
  learning_rate: 1.0e-3
```

## Numerical / Compute Notes

- `2m = 32`. Householder skew-tridiagonalization: `O((2m)^3) = 3.3e4` flops. Trivial.
- The Pfaffian sign needs careful tracking: each Householder step contributes `sign =
  -1`; tridiag of even size needs `m - 1` reflections to get to standard skew-tridiag.
- For `Q` sub-Pfaffians, group `I_q` so they share Householder prefixes; otherwise
  recompute per subset.
- Implicit-function / unrolled autograd on Householder is well-supported.
- For very small `|pf(K)|` (sign cancellation), the gradient of `log|pf|` blows up.
  Stabilize with `log(|pf| + eps)` and `eps = 1e-4 * ||K||_F^m`.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `abs_only` | replace `pf, sign_balance` with `|pf|, 0` | tests orientation signal |
| `det_swap` | replace `pf(K)` with `sqrt(det(K))` | unsigned magnitude only |
| `random_sign_E` | randomize signs in `(E_p - E_p^T)/2` | tests learned orientation |
| `force_symmetric_K` | use `(E_p + E_p^T)/2` (then `pf = 0` always) | sanity: must collapse |
| `single_pf_only` | drop sub-Pfaffian fingerprint, keep scalar `pf` | tests fingerprint richness |
| `random_subset_family` | randomize `{I_q}` | tests subset chess semantics |
| `cnn_same_params` | matched-capacity CNN | baseline |
| `i058_DPP_baseline` | run i058 on same K | adjacent-volume baseline |

For each: full 3x2 + slice reports.

## Benchmark Targets

```text
test PR AUC      >= 0.82
test F1          >= 0.76
near-puzzle FPR  <= 0.20
puzzle recall    >= 0.78

central claim:
  abs_only drops PR AUC >= 0.015
  random_sign_E drops PR AUC >= 0.01
  force_symmetric_K must collapse to constant prediction (sanity)
```

## Counterexamples / Failure Modes

- Tactical engagements are not naturally oriented; `pf(K)` is just `+/- |pf(K)|` with
  noise, so the orientation signal is empty.
- Sign cancellation makes gradients unstable.
- The choice of paired squares (`2m = 32`) misses critical pieces.
- Sub-Pfaffian family is uninformative.

## Implementation Priority

1. Implement differentiable Householder skew-tridiag and Pfaffian.
2. Sanity check on small examples vs `numpy / scipy`.
3. Build `K` from fixed `E_p`; test `pf(K)` distribution on a held-out batch.
4. Add learned gates `g_p` and sub-Pfaffian fingerprint.
5. Run all 8 ablations.

Smallest viable version:

```text
2m = 16, fixed E_p, no sub-Pfaffian, head = (sign(pf), log|pf|, ||K||_F).
```

Scale `2m`, learn gates, and add sub-Pfaffians if positive lift.
