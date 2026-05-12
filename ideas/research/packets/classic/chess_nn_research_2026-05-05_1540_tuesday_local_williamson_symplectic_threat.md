# Codex Research Packet: Williamson Symplectic-Eigenvalue Threat Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1540_tuesday_local_williamson_symplectic_threat.md`
- Generated at: 2026-05-05 15:40
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full unorthodox-linear-algebra architecture packet, not implemented, not benchmark results

## One-Sentence Thesis

Pair every chess square with a *symbolic conjugate momentum* (its tactical "what could
move here next"), build a learned `2n x 2n` SPD operator `M` on this position-momentum
phase space, and decompose `M` via **Williamson's normal form** `M = S^T D S` where
`S` is symplectic and `D = diag(d_1, ..., d_n, d_1, ..., d_n)`; the **symplectic
eigenvalues** `d_i` are a uniquely non-classical invariant (different from ordinary
eigenvalues) that characterize the *quantum-like uncertainty area* of each tactical
mode -- a phase-space invariant that no classical eigen-spectrum or Schur/SVD packet
exposes.

## Why This Is A Real And Unorthodox Linear Algebra NN Idea

The standard symplectic form on `R^{2n}` is

```text
J = [[ 0   I_n ]
     [-I_n  0  ]]
```

A matrix `S` is **symplectic** iff `S^T J S = J`. The symplectic group `Sp(2n, R)` is
*not* the orthogonal group; it preserves the `J`-bilinear form, not the Euclidean one.

**Williamson's theorem** (1936): every SPD `M in S^{2n}_{++}` admits a decomposition

```text
M = S^T D S,    S in Sp(2n, R),    D = diag(d_1, ..., d_n, d_1, ..., d_n),  d_i > 0
```

The values `d_i` are the **symplectic eigenvalues** of `M`. They are *invariant* under
symplectic congruence and are computed as the absolute values of the eigenvalues of
`i J M` (which come in pairs `+/- i d_i`).

Symplectic eigenvalues are central to:

- **Quantum optics** (Wigner / Husimi distributions, squeezed states).
- **Hamiltonian mechanics** (Krein collisions, parametric resonance).
- The **uncertainty principle**: `det(M) >= 1/2^n` (Heisenberg).

For chess, the bet is:

- A position has natural *position* coordinates (square occupancy) and *momentum*
  coordinates (where pieces can move next, derived from current geometry without doing
  legal-move generation).
- The covariance / interaction matrix `M` between these `2n` coordinates is SPD, and
  its symplectic spectrum captures *tactical uncertainty area* -- pairs `(square,
  next-action)` whose joint variance is small (`d_i` small) are the locked-in
  certainties; pairs with large `d_i` are the loose tactical degrees of freedom.

## Target

```text
fine 0,1 -> binary 0
fine 2   -> binary 1
3x2 fine-to-binary matrix mandatory.
```

## Forbidden Inputs

The "symbolic momentum" must come from current-board geometry only (rule-derived ray
extensions, knight-jumps, pawn pushes from current piece positions). No legal-move
generation, no engine evaluation, no source labels.

## Closest Existing Ideas And Exact Difference

### Closest registered

- `i078 Tactical Controllability Gramian` — uses Gramian of a control system; SPD but
  *not* symplectic-decomposed. Misses the symplectic structure.
- `i083 Fisher-Geodesic Tension` — uses Fisher-Rao on simplex; not phase space.
- `i061 Grassmannian Principal Angles` — subspaces; no phase-space pairing.
- `i133 Orthogonal Board Moments` — orthogonal moments, not symplectic.

### Exact difference

```text
Williamson normal form is the unique decomposition that pairs every position coordinate
with a momentum coordinate and exposes invariants under symplectic congruence Sp(2n, R)
-- a strictly larger and structurally different group than O(2n). Symplectic
eigenvalues are the only n-dimensional invariant of an SPD 2n x 2n matrix that
respects the symplectic form J. No imported packet pairs squares with rule-derived
momenta and reads off symplectic eigenvalues.
```

## Mathematical Thesis

### Definitions

Pair the 64 squares with 64 *next-action* slots (knight-jumps, ray-extension targets,
pawn-pushes derived from current geometry; pre-computed deterministic). Index the joint
phase space `{1, ..., 2n}` with `n = 64` (so `2n = 128`).

Build SPD interaction:

```text
M = G + lambda I_{2n} in S^{2n}_{++}
G = sum_p w_p(X_sq) * E_p
```

where `E_p` are fixed PSD legal-geometry primitives over the joint position-momentum
space (e.g. "square s is occupied AND ray-action a from s exists", "defender of s also
defends momentum a"), and `lambda > 0` keeps `M` SPD.

### Williamson decomposition

Computed differentiably:

```text
i J M               -> 2n x 2n complex matrix with eigenvalues in pairs +/- i d_k
abs_eigvals         -> {d_k}_{k=1..n}, the symplectic spectrum
S can be recovered, but in v1 we only need {d_k}
```

Stable algorithm:

1. Compute `M^{1/2}` (symmetric, positive).
2. Compute `K = M^{1/2} J M^{1/2}` (skew-symmetric).
3. Eigenvalues of `K` are `+/- i d_k`.
4. `d_k = sqrt(eigvals(- K K^T)_top_n)`.

All operations differentiable through `eigh`.

### Readout

```text
symplectic_spectrum  d = (d_1, ..., d_n) sorted descending
symplectic_entropy   -sum_i log(d_i)            (related to Renyi-like entropies)
heisenberg_slack     d_i - 1/2 (per-mode uncertainty deficit, can be negative if M
                                non-physical)
prod_d_i             = sqrt(det(M))
spectral_gaps        d_i - d_{i+1}
top-k symplectic eigvals
ordinary_eigvals_topk of M  (for contrast with symplectic ones)
```

Final:

```text
puzzle_logit = MLP([d_topk, sympl_entropy, heisenberg_slack_topk, gaps, board_pool])
```

## Assumptions

- Tactical structure has a natural phase-space interpretation: each square has a
  conjugate "next action" derived only from current geometry.
- Puzzles concentrate symplectic mass: a few `d_i` are very small (locked tactical
  modes) and a few are very large.
- Williamson's decomposition produces a more discriminative summary than ordinary
  eigenvalues because the symplectic group strictly contains *only* the natural Krein-
  pairing-respecting transformations.

## Claim / Hypothesis

`{d_i}` and `{eigvals(M)}` are mathematically distinct sequences that share `det(M)` =
`prod(d_i)^2` = `prod(eigvals(M))` but differ otherwise. The model should achieve
non-trivial PR-AUC lift *over* a same-params model that uses ordinary eigvals of `M`.

Central falsifier:

```text
ordinary_eigvals_swap: replace d_topk with eigvals_topk(M)
                      if PR AUC doesn't drop, the symplectic structure isn't useful.

random_J: replace J with a random skew-orthogonal of the same shape
         if PR AUC doesn't drop, J's chess-natural pairing is a non-factor.
```

## Architecture

### Components

```text
board_encoder
position_momentum_pairing  -> deterministic pairing 64 squares <-> 64 momenta
spd_M_builder              -> M in S^{2n}_{++}
matrix_sqrt_block          -> M^{1/2}
symplectic_eigvals_block   -> {d_i} via eigvals of M^{1/2} J M^{1/2}
ordinary_eigvals_block     -> sanity / ablation
puzzle_head
```

### Forward pseudocode

```text
X_sq      = board_encoder(board)
J         = build_J(n=64)                          # fixed; not learned
M         = spd_M_builder(X_sq)                    # 128 x 128 SPD
M_half    = sym_sqrtm(M)                           # eigh-based
K         = M_half @ J @ M_half                    # skew
neg_KKT   = - K @ K.T                              # SPD; eigvals = d_i^2 each twice
d_squared = topk_eigvals(neg_KKT, k=n)             # take n largest = d_i^2
d         = sqrt(d_squared.clamp_min(eps))         # n symplectic eigvals
ev_M      = eigh(M).eigvals_topk(2k=16)             # for ablation
feat      = [d_topk, -log(d).sum(), d - 0.5, gaps(d), pool(X_sq)]
logit     = MLP(feat)
```

### First config

```yaml
model:
  name: williamson_symplectic_threat_network
  input_channels: 18
  num_classes: 2
  hidden_dim: 96
  phase_n: 64
  num_E_primitives: 16
  M_floor_lambda: 1.0e-3
  topk_d: 12
training:
  mode: puzzle_binary
  loss: bce_with_logits
  batch_size: 256
  learning_rate: 5.0e-4
```

## Numerical / Compute Notes

- `2n = 128`. `eigh` of `128 x 128` SPD: `O((2n)^3) = 2.1e6` flops per board. Fine for
  `batch=256`.
- `M^{1/2}` via `eigh`: `M = U diag(s) U^T -> M^{1/2} = U diag(sqrt(s)) U^T`.
- `K K^T` is `(2n)^2 (2n) = 4.2e6` flops; eigvals on it = `eigh` again.
- The symplectic eigvals `d_i` come in pairs (numerical multiplicity 2). Smoothing:
  take `topk(d_squared, k=n)` and average adjacent paired entries.
- The fixed `J` is not learned; if the chosen pairing is bad, the `random_J` ablation
  reveals that.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `ordinary_eigvals_swap` | use ordinary eigvals(M) instead of `d_i` | tests symplectic |
| `random_J` | random skew-orthogonal in place of J | tests pairing semantics |
| `random_pairing` | random permutation pairing squares <-> momenta | tests phase-space chess sense |
| `J_eq_I_block` | replace J's `[[0,I],[-I,0]]` with `[[0,I],[I,0]]` (symmetric, not symplectic) | tests symplectic structure |
| `det_only` | use only `prod(d_i)^2 = det(M)` | tests sufficiency of det |
| `sympl_entropy_only` | use only `-sum log(d_i)` | tests sufficiency of entropy |
| `cnn_same_params` | matched CNN | baseline |
| `i078_gramian_baseline` | adjacent LA baseline | baseline |

For each: full 3x2 + slice reports.

## Benchmark Targets

```text
test PR AUC      >= 0.82
test F1          >= 0.76
near-puzzle FPR  <= 0.20
puzzle recall    >= 0.78

central claim:
  ordinary_eigvals_swap drops PR AUC >= 0.015
  random_J              drops PR AUC >= 0.01
  random_pairing        drops PR AUC >= 0.005
  beats i078 by >= 0.01 PR AUC
```

## Counterexamples / Failure Modes

- The position-momentum pairing is an artificial construction; the resulting M is just
  any old SPD and Williamson decomposition adds nothing.
- `d_i` and `eigvals(M)` correlate near 1.0 in practice on chess boards (nothing
  exploits the symplectic structure), so symplectic adds noise not signal.
- `M^{1/2}` gradient instability on near-degenerate spectra.
- `phase_n = 64` is too small to expose symplectic structure (Williamson is most useful
  on large Gaussian quantum states).

## Implementation Priority

1. Implement deterministic position-momentum pairing (knight-jumps + ray extensions).
2. Build SPD `M` with `lambda I` floor.
3. Implement `sym_sqrtm` and symplectic-eigval extraction; sanity vs `scipy` if
   available.
4. Train minimal head with `(d_topk, prod(d), entropy)`.
5. Run all 8 ablations.

Smallest viable version:

```text
phase_n = 32 (subset of squares with most occupied), 6 fixed E_p primitives,
features = (d_topk=8, prod(d), -sum log(d)).
```

If lift over CNN-same-params is positive, scale to `phase_n = 64` and full readout.
