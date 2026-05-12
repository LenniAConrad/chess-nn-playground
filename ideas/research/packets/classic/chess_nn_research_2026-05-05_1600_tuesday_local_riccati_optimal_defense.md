# Codex Research Packet: Riccati Optimal-Defense Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1600_tuesday_local_riccati_optimal_defense.md`
- Generated at: 2026-05-05 16:00
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full unorthodox-linear-algebra architecture packet, not implemented

## One-Sentence Thesis

Treat each board as an LQR control problem with attacker dynamics `A`, defender input `B`, target weighting `Q`, defender-effort weighting `R`; solve the **continuous algebraic Riccati equation** `AᵀP + PA - PBR⁻¹BᵀP + Q = 0` for the unique stabilizing `P`; classify puzzles from `P`'s spectrum, the optimal feedback gain `K = R⁻¹BᵀP`, and the **closed-loop Hamiltonian** `H = [[A, -BR⁻¹Bᵀ], [-Q, -Aᵀ]]` whose stable invariant subspace defines `P`.

## Why This Is A Real Unorthodox Linear Algebra NN Idea

The continuous algebraic Riccati equation (CARE) is **quadratic in `P`**:

```text
AᵀP + PA - PBR⁻¹BᵀP + Q = 0
```

Its unique stabilizing solution exists iff `(A, B)` is stabilizable and `(A, Q^{1/2})` detectable. Solving CARE:

```text
1.  Hamiltonian H = [[A, -B R^{-1} B^T], [-Q, -A^T]]   in R^{2n x 2n}
2.  Schur-decompose H, order eigenvalues so the n stable eigenvalues come first
3.  Read invariant subspace: V_stable = [V1; V2]  in R^{2n x n}
4.  P = V2 V1^{-1}  (unique stabilizing PSD solution)
```

This is fundamentally distinct from:

- **Lyapunov (i225)**: `AᵀP + PA = -Q`, *linear* in P; about uncontrolled stability.
- **Sylvester (i221)**: `AX + XB = C`, *linear*; couples two operators.
- **Schur complement (i222)**: pure block elimination, no quadratic term.
- **Controllability Gramian (i078)**: linear; about reachability.

The CARE encodes "what is the optimal defense given a quadratic cost?". Tactical positions correspond to **infinite optimal-defense cost** (CARE has no PSD solution), or to closed-loop systems with eigenvalues on the imaginary axis.

## Target

```text
fine 0,1 -> 0,  fine 2 -> 1.  3x2 fine-to-binary mandatory.
```

## Forbidden Inputs

Standard.

## Closest Existing Ideas And Exact Difference

- `i225 Lyapunov` — linear stability equation; CARE is quadratic and includes the LQR coupling `-PBR⁻¹BᵀP`.
- `i078 Gramian` — controllability/observability, no cost weighting Q, R.
- `i062 Pencil` — generalized eigenproblem, scalar-valued solution.

```text
CARE is the unique quadratic matrix equation whose stable solution gives the optimal LQR
feedback. The Hamiltonian matrix H has a symplectic structure (H J + J H^T = 0) that
none of the imported packets exploit.
```

## Mathematical Thesis

### Definitions

Build at low rank `r = 12`:

```text
A = -alpha I + sum gates_A * primitives_A    (attacker-flow, init Hurwitz)
B = sum gates_B * primitives_B in R^{r x m}  (defender input, m = 4)
Q = beta I + sum h_Q * (N_Q + N_Q^T)/2       PSD target weight
R = gamma I + sum h_R * (N_R + N_R^T)/2      PD effort weight
```

### Riccati solve

Form Hamiltonian, real-Schur reorder to put stable eigenvalues first, extract:

```text
H = [[A, -B R^{-1} B^T], [-Q, -A^T]]    R^{2r x 2r}
T, U = ordered_real_schur(H, criterion=stable)
V_stable = U[:, :r] = [V1; V2]
P = V2 V1^{-1}
```

Differentiable through implicit-function autograd on CARE residual.

### Readout

```text
inertia(P), top-k eigvals(P)
trace(P), log|det(P)|
optimal_gain K = R^{-1} B^T P
closed_loop A_cl = A - B K
spec(A_cl)        (must have all Re < 0 if CARE solvable)
optimal_cost J* = trace(P)         (LQR optimal cost from x_0 = unit Gaussian)
hamiltonian_eig_pairs (must be n stable + n unstable; symmetric about im axis)
care_residual_norm = ||A^T P + P A - P B R^{-1} B^T P + Q||_F  (sanity, training signal)
```

Final:

```text
puzzle_logit = MLP([eigs_P, optimal_cost, ||K||_F, log|det(P)|, hamiltonian_imag_count, board_pool])
```

## Assumptions

- Tactical positions create LQR problems where the optimal-defense cost `J* = x_0^T P x_0` is **large or infinite** (puzzles), versus moderate `J*` (non-puzzles).
- Near-puzzles have moderate `J*` but with a single mode whose closed-loop pole is close to the imaginary axis (marginal stability) — captured by `min Re spec(A_cl)`.

## Claim / Hypothesis

Optimal LQR cost + closed-loop spectral margin together give a near-sufficient statistic for puzzle-likeness. Central falsifier:

```text
linearize_swap: replace CARE with Lyapunov (drop the quadratic term -P B R^{-1} B^T P).
                if PR AUC doesn't drop, the quadratic LQR coupling is unnecessary.
```

## Architecture

```text
board_encoder
A_builder, B_builder, Q_builder, R_builder
hamiltonian_assemble       -> H in R^{2r x 2r}
ordered_real_schur          -> stable invariant subspace
care_solver                 -> P
diagnostics_block           -> spec(P), J*, K, A_cl spec
puzzle_head
```

### First config

```yaml
model:
  name: riccati_optimal_defense_network
  input_channels: 18
  num_classes: 1
  channels: 64
  hidden_dim: 96
  riccati_rank_r: 12
  defender_dim_m: 4
  q_floor_beta: 1.0e-3
  r_floor_gamma: 1.0e-3
  hurwitz_safety: 0.1
training:
  mode: puzzle_binary
  loss: bce_with_logits
  aux_loss_care_residual: 1.0e-3
```

## Numerical / Compute Notes

- `2r = 24`. Real Schur cost `O((2r)^3) = 1.4e4`. Ordered real Schur via Bartels-Stewart-style swaps; differentiable.
- Backward through CARE via implicit-function: solve adjoint Lyapunov-like equation.
- Hurwitz-clip `A` if needed; auxiliary loss penalizes `||CARE residual||_F`.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `linearize_swap` | drop `-P B R^{-1} B^T P` term (becomes Lyapunov) | tests quadratic coupling |
| `Q_eq_I` | replace Q with I | tests Q informativeness |
| `R_eq_I` | replace R with I | tests R informativeness |
| `random_geometry` | random A, B primitives | tests chess semantics |
| `J_star_only` | use only scalar J* | tests sufficiency of optimal cost |
| `cnn_same_params` | matched CNN | baseline |
| `i225_lyapunov_baseline` | run i225 on same A | adjacent baseline |
| `i078_gramian_baseline` | run i078 | adjacent baseline |

## Benchmark Targets

```text
PR AUC >= 0.82, F1 >= 0.76, near-puzzle FPR <= 0.20, puzzle recall >= 0.78
linearize_swap drops PR AUC >= 0.015 (key claim)
beats i225 by >= 0.01 PR AUC
```

## Counterexamples

- Most positions admit CARE solutions cleanly with bounded `J*`; quadratic coupling adds nothing beyond Lyapunov.
- Schur reordering is unstable for clustered eigvals.
- `r = 12, m = 4` may be too small.

## Implementation Priority

1. Build CARE solver via ordered Schur + back-substitution.
2. Verify on synthetic stable LTI vs `scipy.linalg.solve_continuous_are`.
3. Train minimal head with `(J*, top eigvals(P), spec(A_cl))`.
4. Run all 8 ablations.
