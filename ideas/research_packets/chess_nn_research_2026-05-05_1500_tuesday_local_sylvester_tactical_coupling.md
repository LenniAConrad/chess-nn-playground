# Codex Research Packet: Tactical Sylvester Coupling Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1500_tuesday_local_sylvester_tactical_coupling.md`
- Generated at: 2026-05-05 15:00
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full linear-algebra architecture packet, not implemented, not benchmark results

## One-Sentence Thesis

Couple a learned attacker-domain operator `A` and a learned defender-domain operator `B` through the Sylvester equation `AX + XB = C`, where `C` is a board-derived target/obligation matrix; classify puzzles from properties of the unique solution `X` (Frobenius norm, singular spectrum, kernel structure) which exists iff `spec(A)` and `spec(-B)` are disjoint.

## Why This Is A Real Linear Algebra NN Idea

This is not a generalized eigenproblem (i062), Krylov projection (i076), or resolvent (i077). The core computation is the **Sylvester / Roth equation**:

```text
A X + X B = C,    A in R^{n x n},  B in R^{m x m},  C, X in R^{n x m}
```

solvable in closed form via vectorization

```text
(I_m kron A + B^T kron I_n) vec(X) = vec(C)
```

or via Bartels-Stewart / Schur decomposition. The unique solution exists iff `spec(A) cap spec(-B) = empty`. The bet: a true tactical puzzle creates a near-resonance between attacker spectrum and defender spectrum (small denominator `lambda_i(A) + mu_j(B)` for at least one pair), yielding a high-norm or low-rank-collapsing `X` that no static spatial filter sees. A near-puzzle has comparable surface pressure but no spectral resonance.

The unique formal object is:

```text
operator-coupling resonance through coupled two-sided solve, not single-operator spectrum
```

## Target

```text
fine label 0 -> binary target 0   (known non-puzzle)
fine label 1 -> binary target 0   (verified near-puzzle)
fine label 2 -> binary target 1   (verified puzzle)
```

Mandatory diagnostic: 3x2 fine-to-binary matrix.

## Forbidden Inputs

Stockfish scores, PVs, node counts, mate scores, engine moves, verification metadata, source labels, source identity, future game outcomes, full legal-move generation. Only current-board tensors and rule-derived geometry.

## Closest Existing Ideas And Exact Difference

### Closest registered ideas

- `i062 Matrix-Pencil Generalized Spectrum Bottleneck` — solves `A x = lambda B x` (one operator pair, vector-valued solution).
- `i076 Krylov Tactical Subspace` — repeated action of a single operator on seeds.
- `i077 Adaptive Tactical Resolvent` — `(z I - A)^{-1}`, single operator.
- `i078 Tactical Controllability Gramian` — `int e^{At} B B^T e^{A^T t} dt`, output-side.
- `i199 Tactical Hessian Spectrum` — eigenvalues of plain Hessian.

### Exact difference

```text
Sylvester is the unique invariant of two coupled operators acting on opposite sides of a
matrix unknown X. Its solution depends on the *cross-spectrum* sum lambda_i(A)+mu_j(B),
not a single spectrum or a generalized pair lambda_i(A)/mu_j(B). It is the smallest
linear object that exposes attacker-defender spectral resonance, and it cannot be
reduced to any of the imported single-operator linear-algebra packets without losing
the resonance signal.
```

## Mathematical Thesis

### Definitions

Let the board encode square-features `X_sq in R^{64 x d}`. Build two compact operators:

```text
A = sum_k gate^A_k(X_sq) * M^A_k     in R^{r x r}        (attacker side, low-rank r=8..16)
B = sum_k gate^B_k(X_sq) * M^B_k     in R^{r x r}        (defender side)
C = U^T  W(X_sq)  V                   in R^{r x r}        (target obligation, board-derived)
```

where `M^A_k`, `M^B_k` are fixed legal-geometry primitives projected to a learned `r`-dim
attacker / defender role basis (rays, knight, pawn, king-zone, pin-axis, defender-mass).
`U, V in R^{64 x r}` are learned role-projection bases; `W` is a 64x64 obligation field
(e.g. attacker-pressure x target-mass).

### Solve

Compute the unique solution `X` of

```text
A X + X B = C
```

via either the closed-form Bartels-Stewart (Schur of `A` and `B`, back-substitution) or
the conjugate-gradient-on-`(I kron A + B^T kron I)` map (10..20 CG iters at `r<=16`).
Both are differentiable through implicit-function autograd or unrolled solver autograd.

### Readout features

```text
sigma(X)             -- sorted singular values of X (top k)
||X||_F, ||X||_2     -- norms
rank_eps(X)          -- soft rank
trace(X^T X A)       -- attacker-projected energy
trace(X X^T B^T)     -- defender-projected energy
log |det(I + X X^T)| -- bounded log-volume
spec_resonance       -- min_{i,j} |lambda_i(A) + mu_j(B)|         (small => puzzle)
right_principal_dirs -- top-k right singular vectors of X projected back to 64 squares
```

Final:

```text
puzzle_logit = MLP([sigma_topk, log_norms, resonance_min, board_pool, role_dirs_pool])
```

## Assumptions

- Tactics arise when attacker spectrum is *resonant* with negative defender spectrum,
  i.e. `lambda_i(A) + mu_j(B) ~ 0` for some `(i, j)`. The Sylvester solution amplifies
  exactly the directions with smallest such denominator.
- A near-puzzle can have large attacker spectrum but well-separated defender spectrum,
  so `X` stays small. A true puzzle aligns the two spectra.
- Low-rank operators `r in {8, 12, 16}` are enough; the resonance signal lives in a few
  attacker-defender mode pairs.

## Claim / Hypothesis

If puzzle structure is genuinely a coupling phenomenon between attacker and defender
linear actions, then:

```text
P(puzzle | x) = sigmoid( w^T phi(X*(x)) + b )
```

with `X*` the Sylvester solution, should beat the size-matched CNN, the i001 Operator-
Basis classifier, and the i062 Matrix-Pencil packet. In particular, the Sylvester
network should distinguish near-puzzles (single-side spectrum) from puzzles (two-side
resonance) better than any single-operator spectrum model.

## Architecture

### Components

```text
board_encoder              -> X_sq in R^{64 x d}
role_basis_builder         -> U, V in R^{64 x r}
attacker_operator_builder  -> A in R^{r x r}
defender_operator_builder  -> B in R^{r x r}
obligation_builder         -> C in R^{r x r}
sylvester_solver           -> X = solve(A, B, C)
spectral_readout           -> phi(X, A, B)
puzzle_head                -> logit
```

### Forward pseudocode

```text
X_sq = board_encoder(board)
U, V = role_basis_builder(X_sq)
A    = attacker_operator_builder(X_sq, U)
B    = defender_operator_builder(X_sq, V)
C    = obligation_builder(X_sq, U, V)
A_n  = A / max(1, ||A||_2)        # spectral normalize
B_n  = B / max(1, ||B||_2)
X    = bartels_stewart(A_n, B_n, C)   # or unrolled CG, 12 iters
feat = spectral_readout(X, A_n, B_n)
logit= MLP([feat, global_pool(X_sq)])
```

### First config

```yaml
model:
  name: sylvester_tactical_coupling_network
  input_channels: 18
  num_classes: 2
  hidden_dim: 96
  rank_r: 12
  solver: bartels_stewart   # alt: unrolled_cg_12
  spectral_norm_clip: 1.0
training:
  mode: puzzle_binary
  loss: bce_with_logits
  batch_size: 512
  learning_rate: 1.0e-3
```

## Numerical / Compute Notes

- `r = 12` keeps cost at O(r^3) = 1728 flops per solve, dwarfed by the CNN trunk.
- Spectral-normalize `A` and `B` so `||A||_2 + ||B||_2 < 2` and the Sylvester operator is
  bounded away from singular.
- Use `torch.linalg.solve_sylvester` if available; otherwise hand-rolled Bartels-Stewart
  with complex Schur (or a real Schur with quasi-triangular blocks).
- Implicit-function autograd: backprop through `A X + X B = C` by solving the adjoint
  Sylvester `A^T G + G B^T = -dL/dX`. Cheap at this `r`.

## Falsification Criteria

Reject or revise the idea if any of:

- `pair_swap_ablation` (use B as A and A as B) matches full model -> coupling does not
  matter.
- `disjoint_spectra_forced` (project A and B to spectrally disjoint subspaces) matches
  full model -> resonance is not the signal.
- `rank_one_C` (replace `C` by a rank-1 obligation) does not hurt -> obligation field
  not informative.
- `resonance_only` (use only `min |lambda_i+mu_j|` as a scalar feature) matches full
  readout -> rest of `X` is decorative.

## Required Ablations

| Ablation | What it removes | Hypothesis |
|---|---|---|
| `independent_operators_only` | Replace Sylvester solve with `[A; B]` features | tests coupling |
| `swap_AB` | Use B in attacker slot and A in defender slot | tests asymmetry |
| `random_geometry_M` | Randomize fixed operator masks but keep same shape | tests chess semantics |
| `low_rank_C` | Replace C by rank-1 outer product u v^T | tests obligation richness |
| `no_spectral_normalize` | Remove `A/||A||_2` clip | tests numerical stability |
| `static_resonance_only` | Use only `min |lambda+mu|` scalar | tests rest of X |
| `cnn_same_params` | Size-matched plain CNN | matched-capacity baseline |
| `i001_basis_baseline` | Run i001 chess_operator_basis_classifier | linear-algebra baseline |

For each ablation, report the full 3x2 fine-to-binary matrix plus per-slice metrics on
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`, and `crtk_tactic_motifs`.

## Benchmark targets

```text
test PR AUC      >= 0.82
test F1          >= 0.76
near-puzzle FPR  <= 0.20  (fine label 1 -> predicted 1)
puzzle recall    >= 0.78
```

Stretch: Sylvester model improves on i062 matrix-pencil by `>= +0.02 PR AUC` while
`independent_operators_only` ablation drops at least `0.015 PR AUC` -- both required
to claim the coupling matters.

## Counterexamples / Failure Modes

- Most puzzles need only one-side pressure -> coupling never resonates.
- Learned `A`, `B` collapse to near-identical operators -> `B = -A + small` makes every
  position resonant; resonance loses discriminative power.
- Solver autograd is unstable for nearly-singular `(A, -B)` pairs -> training collapses;
  mitigate with `||A+B||_2` regularizer that keeps spectra separated.
- The `r = 12` projection is too coarse to expose resonance.

## Implementation Priority

1. Build the `r = 12` role basis `U, V` from board features.
2. Build fixed-mask operators `M^A_k`, `M^B_k` (rays, knight, pawn, king-zone, pin).
3. Wrap `torch.linalg.solve_sylvester` with implicit-function backward.
4. Train minimal version (no spectral readout; only `||X||_F` + top-3 sigma).
5. Add resonance min, rank, log-volume features.
6. Run all 8 ablations and full slice report.

Smallest viable version:

```text
r = 8, fixed M masks only, no learned gates, readout = top-3 sigma + ||X||_F
```

If that shows lift over CNN-same-params, add learned gates and full readout.
