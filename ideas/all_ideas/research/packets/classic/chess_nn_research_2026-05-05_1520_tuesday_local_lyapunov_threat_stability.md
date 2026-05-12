# Codex Research Packet: Lyapunov Stability Threat Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1520_tuesday_local_lyapunov_threat_stability.md`
- Generated at: 2026-05-05 15:20
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full linear-algebra architecture packet, not implemented, not benchmark results

## One-Sentence Thesis

Treat each board as a linear dynamical system `dot x = A(board) x` and solve the
**continuous Lyapunov equation** `A^T P + P A = -Q` for a chess-derived weighting `Q`;
classify puzzle-likeness from the resulting `P` (Lyapunov stability certificate) -- its
positive-definiteness, condition number, and trace -- which exists iff every eigenvalue
of `A` has negative real part, giving a clean *binary* algebraic certificate of tactical
*stability* that is fundamentally different from the controllability Gramian (i078) and
from spectrum-only readouts.

## Why This Is A Real Linear Algebra NN Idea

The continuous Lyapunov equation

```text
A^T P + P A = -Q,    Q symmetric positive definite
```

has a unique symmetric solution `P` iff `spec(A) cap (-spec(A)) = empty`; if every
eigenvalue of `A` satisfies `Re(lambda_i) < 0`, then `P` is symmetric positive definite
(SPD) and `V(x) = x^T P x` is a Lyapunov function for `dot x = A x`, certifying
asymptotic stability. The closed-form solution is

```text
vec(P) = -(I kron A^T + A^T kron I)^{-1} vec(Q)
```

or via Bartels-Stewart on the (real) Schur form of `A`. Either is differentiable.

This is **not** the controllability Gramian (i078):

- i078 controllability solves `A W + W A^T = -B B^T` and is about *output reachability*
  of an `(A, B)` system; it requires a chosen input matrix `B`.
- Lyapunov stability solves `A^T P + P A = -Q` and is about *asymptotic stability* of
  the autonomous system `dot x = A x`; the solution `P` exists iff `A` is Hurwitz.
- The **inertia** of `P` (how many positive vs negative eigenvalues) is by Lyapunov's
  theorem equal to the inertia of `-A` (i.e. swaps signs of `Re(spec(A))`), but the
  *magnitudes* in `P` give settling-time, condition number, and direction-conditioned
  stability *that no spectrum readout can recover*.

## Target

```text
fine 0,1 -> binary 0
fine 2   -> binary 1
3x2 fine-to-binary matrix mandatory.
```

## Forbidden Inputs

Standard.

## Closest Existing Ideas And Exact Difference

### Closest registered ideas

- `i078 Tactical Controllability Gramian Network` — different equation `A W + W A^T = -BB^T`,
  different invariant (output reachability), requires a chosen input `B`.
- `i077 Adaptive Tactical Resolvent Network` — `(zI - A)^{-1}`, complex-frequency response,
  not a real symmetric quadratic Lyapunov certificate.
- `i062 Matrix-Pencil` — generalized eigenproblem, not a Lyapunov certificate.
- `i199 Tactical Hessian Spectrum` — eigenvalues of a Hessian, not the solution of a
  Lyapunov equation.

### Exact difference

```text
The Lyapunov equation produces a quadratic stability certificate P (a full SPD matrix
when A is Hurwitz). Inertia, condition number, trace, and direction-conditioned settling
time of P are unique invariants of the (autonomous) dynamics A that no controllability,
resolvent, or spectrum-only packet exposes. The Gramian (i078) needs an input B and
produces a *reachability* matrix; Lyapunov-stability needs no B and produces a
*stability* matrix.
```

## Mathematical Thesis

### Definitions

Build the chess flow operator at low rank:

```text
A = -alpha I_r + sum_k g_k(X_sq) M_k     in R^{r x r},  r = 16
```

with `alpha > 0` a learnable damping (initialized at 1.0) ensuring the *base* operator
is Hurwitz; `M_k` fixed asymmetric legal-geometry primitives (attacker rays, defender
shielding, side-to-move flow). The damping `alpha` lets us guarantee `Re(spec(A)) < 0`
at initialization and then progressively relax that as gates `g_k` increase.

Build the chess weight matrix:

```text
Q = beta I_r + sum_j h_j(X_sq) (N_j + N_j^T) / 2   PSD,  beta > 0
```

with `N_j` legal-geometry primitives anchored on king-zone, target squares, and
critical defenders. `Q` weights *which directions matter* for stability.

Solve:

```text
A^T P + P A = -Q
```

Differentiable closed-form via real Schur of `A` and Bartels-Stewart back-substitution.

### Readout

```text
inertia(P)            = soft (n_pos, n_zero, n_neg)
log|det(P)|           = total log-volume of stability ellipsoid
trace(P), trace(P Q^{-1})   (latter ~= integral_0^inf x(t)^T Q x(t) dt for x_0=I-trace)
cond(P)               = sigma_max(P) / sigma_min(P) (soft via log diff)
top-k eigvals of P
worst-direction settling proxy: lambda_max(Q) / lambda_min(P)
hurwitz_indicator     = sigmoid( -gamma * max_i Re(lambda_i(A)) )
```

Final:

```text
puzzle_logit = MLP([inertia, log_det_P, trace_P, cond_P, eigs_topk_P,
                    hurwitz_indicator, board_pool])
```

## Assumptions

- A puzzle position corresponds to dynamics that are *not* asymptotically stable for the
  defender's natural reaction direction; equivalently, `P` is ill-conditioned or fails
  to be SPD relative to a chess-natural `Q`.
- A non-puzzle has `Re(spec(A)) << 0` and a well-conditioned `P` -- pressure decays
  quickly along all chess-natural directions.
- A near-puzzle may have a single direction with poor settling but the bulk of `P` is
  fine; the *condition number* of `P` is the discriminator, not its inertia.

## Claim / Hypothesis

`cond(P)` and `lambda_min(P)` are near-sufficient statistics for puzzle-likeness *given*
the board pool features. Specifically:

1. The model beats CNN-same-params and i078 controllability Gramian on PR AUC and
   near-puzzle FPR.
2. Replacing `Q` by `I` (no chess-derived weights) drops PR AUC by `>= 0.01`.
3. Forcing `A` to be symmetric (`A <- (A + A^T) / 2`) trivializes `P = -A^{-1} Q / 2`
   and should match a spectrum-only readout: PR AUC must drop `>= 0.015` if the
   asymmetric flow is the signal.

## Architecture

### Components

```text
board_encoder            -> X_sq
flow_operator_A          -> A in R^{r x r} with damping
weight_Q                 -> Q in S^r_{++}
schur_solver             -> A = U T U^T (real Schur)
bartels_stewart          -> P solving A^T P + P A = -Q
psd_diagnostics_block    -> inertia, det, cond, eigvals
puzzle_head
```

### Forward pseudocode

```text
X_sq    = board_encoder(board)
A       = flow_operator_A(X_sq)          # damped + gated
Q       = weight_Q(X_sq)                  # PSD via L L^T + beta I
A_clipped = A - max(0, max_real_eig(A) + safety) * I    # guarantee Hurwitz at solve
T, U    = real_schur(A_clipped)
P       = bartels_stewart_solve(T, Q, U)  # solve in Schur basis, rotate back
feat    = psd_diagnostics(P, A, Q)
logit   = MLP([feat, pool(X_sq)])
```

### First config

```yaml
model:
  name: lyapunov_threat_stability_network
  input_channels: 18
  num_classes: 2
  hidden_dim: 96
  operator_rank_r: 16
  damping_alpha_init: 1.0
  hurwitz_safety: 0.1
  q_floor_beta: 1.0e-3
  num_M_primitives: 12
  num_N_primitives: 8
training:
  mode: puzzle_binary
  loss: bce_with_logits
  batch_size: 512
  learning_rate: 1.0e-3
  aux_loss_keep_hurwitz: 1.0e-3   # alpha * relu(max_real_eig(A) + 1e-3)
```

## Numerical / Compute Notes

- `r = 16`. Real Schur cost `O(r^3) = 4096`; Bartels-Stewart `O(r^3)`. Negligible.
- Use `torch.linalg.eig` for `max_real_eig` to enforce Hurwitz (or eigvals of a
  Hessenberg form -- cheaper).
- `bartels_stewart` is implementable in PyTorch with the Schur factor `T` quasi-
  triangular; back-substitution is differentiable.
- `cond(P)` via `(lambda_max - lambda_min)` of `eigh(P)`; differentiable when SPD.
- Implicit-function autograd: backprop through `A^T P + P A = -Q` by solving the adjoint
  Lyapunov `A G + G A^T = -dL/dP - (dL/dP)^T`. Cheap.
- The `hurwitz_safety` clip subtracts `(max_real_eig + safety) I` from `A` before
  solving; this makes `P` always exist. The features still see the original `A`'s
  spectrum so the model can learn how close to Hurwitz it is.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `Q_eq_I` | replace chess `Q` with identity | tests Q informativeness |
| `symmetric_A` | force `A <- (A + A^T)/2` | tests asymmetric flow |
| `random_geometry_M` | random sparse asymmetric primitives | tests chess semantics |
| `inertia_only` | drop magnitudes, keep inertia | tests sufficiency of inertia |
| `cond_only_scalar` | use only `cond(P)` scalar | tests sufficiency of cond |
| `no_aux_hurwitz_loss` | drop the keep-Hurwitz auxiliary | tests training stability |
| `cnn_same_params` | size-matched CNN | matched-capacity |
| `i078_gramian` | run i078 controllability Gramian on same A | adjacent LA baseline |
| `i077_resolvent` | run i077 on same A | adjacent LA baseline |

For each: full 3x2 + slice reports on difficulty / phase / eval-bucket / tactic-motifs.

## Benchmark Targets

```text
test PR AUC      >= 0.82
test F1          >= 0.76
near-puzzle FPR  <= 0.20
puzzle recall    >= 0.78

central claim:
  Q_eq_I       drops PR AUC >= 0.01
  symmetric_A  drops PR AUC >= 0.015
  beats i078   by  >= 0.01 PR AUC on the same operator A
```

## Counterexamples / Failure Modes

- The chess flow interpretation `dot x = A x` is meaningless for static positions; if
  Lyapunov readouts are pure noise.
- `A` collapses to nearly normal (`A A^T ~ A^T A`), making Lyapunov features redundant
  with the symmetric eigvals.
- The Hurwitz clip pushes `A` so far into the stable half-plane that `P` becomes
  near-isotropic; mitigation: the clip uses a small safety `0.1` and the `aux` loss
  rewards keeping `max_real_eig(A)` close to (but below) zero.
- Numerical instability in Bartels-Stewart for clustered eigenvalues.

## Implementation Priority

1. Build operator `A` at `r = 16` from `simple_18`; verify it is Hurwitz at init.
2. `Q = I` initial; solve Lyapunov via PyTorch's `solve_lyapunov` (or hand-rolled
   Bartels-Stewart). Verify against `scipy.linalg.solve_continuous_lyapunov` on
   randomly held-out batches.
3. Readout: inertia + cond + trace.
4. Add learned `Q` from chess primitives.
5. Run all ablations.

Smallest viable version:

```text
fixed M_k legal-geometry primitives, Q = I,
features = (cond(P), trace(P), log|det(P)|, hurwitz_indicator).
```

If lift over CNN-same-params is positive and `symmetric_A` ablation degrades, scale to
learned gates and chess-derived `Q`.
