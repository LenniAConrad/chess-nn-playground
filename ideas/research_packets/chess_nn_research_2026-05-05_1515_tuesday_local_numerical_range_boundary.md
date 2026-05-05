# Codex Research Packet: Numerical-Range Boundary Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1515_tuesday_local_numerical_range_boundary.md`
- Generated at: 2026-05-05 15:15
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full linear-algebra architecture packet, not implemented, not benchmark results

## One-Sentence Thesis

Compute the **numerical range** (field of values) `W(A) = { x* A x : x in C^n, ||x|| = 1 }`
of a learned non-symmetric chess operator `A`, sample its boundary curve in the complex
plane, and classify puzzle-likeness from boundary-shape descriptors (curvature spectrum,
non-normality gap `numr - rho`, support-function in selected directions); this is the
unique linear-algebra packet that exposes **non-normal** structure -- the gap between
spectrum and field of values -- which spectrum-only ideas fundamentally cannot see.

## Why This Is A Real Linear Algebra NN Idea

The numerical range of a complex matrix `A in C^{n x n}` is the convex compact set

```text
W(A) = { x* A x : x in C^n, ||x|| = 1 }  subset C
```

The numerical radius is `numr(A) = sup { |z| : z in W(A) }`. Always

```text
rho(A) <= numr(A) <= ||A||_2 <= 2 numr(A)
```

For **normal** `A` (i.e. `A A^* = A^* A`), `W(A)` is the convex hull of the spectrum;
for **non-normal** `A`, `W(A)` strictly contains conv(spec(A)) and the gap
`numr(A) - rho(A)` measures *transient amplification*, *pseudospectral spread*, and
*sensitivity to perturbation*.

A spectrum-only model (i062 pencils, i076 Krylov, i077 resolvent, i078 Gramian, i199
Hessian-spectrum) cannot see this gap; non-normal effects vanish under similarity
transforms that preserve eigenvalues but change `W(A)`.

The bet: tactical pressure is fundamentally non-normal. Attacker influence flowing
through ray geometry, with asymmetric defender resistance, produces operators that
*amplify transiently* before the spectrum tells you anything. The numerical range is the
right invariant.

### Boundary computation

For each angle `theta in [0, 2 pi)`, the rightmost point of `W(A)` along direction
`e^{i theta}` is

```text
max_{||x||=1} Re(e^{-i theta} x* A x) = lambda_max( H(A, theta) )
H(A, theta) = (e^{-i theta} A + e^{i theta} A^*) / 2     (Hermitian)
```

So the boundary curve is parametrized by the top eigenvalue and corresponding eigenvector
of `H(A, theta)` over `theta`. We sample `K = 16..32` angles, each requiring a
differentiable `eigh` of an `n x n` Hermitian matrix.

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

- `i001 Chess Operator Basis Classifier` — uses linear-operator features, but no
  numerical range.
- `i062, i076, i077, i078, i199` — spectrum-, Krylov-, resolvent-, Gramian-, Hessian-
  based, all are *spectrum-only* and cannot detect non-normality.
- `i138 Support-Function Envelope Network` — closest in spirit because the boundary of
  `W(A)` is the support function of `W(A)`; but i138 uses a support function over
  *boards / squares*, not over the field-of-values of an operator in C, so the formal
  object is genuinely different.

### Exact difference

```text
W(A) is a 2D set in the complex plane describing transient amplification of a learned
chess operator. Its boundary curve, curvature spectrum, and gap to spec(A) are all
non-normal invariants. No imported packet computes the field of values of a learned
chess operator and uses its boundary geometry as features.
```

## Mathematical Thesis

### Definitions

Build a possibly non-symmetric chess operator at low rank:

```text
A = sum_k g_k(X_sq) * M_k    in C^{r x r},  r = 16
```

with `M_k` fixed legal-geometry primitives (rays L/R, knight, pawn forward/back,
defender-line, pin-axis), some intentionally **asymmetric** so `A` is non-normal in
general. `g_k` are learned scalar gates from the board encoder.

For `K` angles `theta_k = 2 pi k / K`:

```text
H_k = (e^{-i theta_k} A + e^{i theta_k} A^*) / 2
mu_k = lambda_max(H_k)              # support of W(A) in direction e^{i theta_k}
v_k  = top eigenvector of H_k
z_k  = v_k^* A v_k                  # boundary point in C
```

Boundary curve: `{ z_k : k = 0, ..., K-1 }`.

### Readout

```text
boundary_real, boundary_imag       in R^K
numerical_radius numr(A) = max_k |z_k|
spectral_radius   rho(A) (top eigvalue magnitude of A)
non_normality_gap = numr(A) - rho(A)              # >= 0, key feature
crawford_number = min_k mu_k                       # boundary closest to origin
boundary_curvature = discrete second difference of (Re z_k, Im z_k)
support_function_samples mu_k                       in R^K
W_area_estimate via shoelace on (Re z_k, Im z_k)
```

Final:

```text
puzzle_logit = MLP([gap, numr, rho, crawford, curvature_topk, area, board_pool])
```

## Assumptions

- The natural chess operator is non-normal due to attacker-defender asymmetry and side-
  to-move directionality.
- Tactical positions exhibit *larger* numerical-range / spectrum gap than non-tactical
  positions, because tactics depend on transient amplification of pressure that has not
  yet shown up in the spectrum.
- A modest angle count `K = 16` and operator rank `r = 16` are enough to capture this.

## Claim / Hypothesis

The non-normality gap `numr(A) - rho(A)` correlates with puzzle-likeness *after*
controlling for board pool features. The central falsifier: forcing `A` to be normal
(`A^* A = A A^*`) via projection should shrink the gap to zero and drop PR AUC by
`>= 0.015`.

## Architecture

### Components

```text
board_encoder            -> X_sq
operator_builder_complex -> A in C^{r x r}
boundary_sampler         -> {z_k}, {mu_k}
spectrum_block           -> rho(A) and top-k eigvals (Schur form, complex)
non_normality_features
puzzle_head
```

### Forward pseudocode

```text
X_sq      = board_encoder(board)
A         = operator_builder_complex(X_sq)        # complex r x r
A_norm    = A / max(1, ||A||_2)
spec_A    = topk_eigvals(A_norm)
rho_A     = max(|spec_A|)
for k in 0..K-1:
    theta = 2 pi k / K
    H_k   = ( exp(-1j theta) * A_norm + exp(+1j theta) * A_norm.conj().T ) / 2
    mu_k, v_k = top_eigh(H_k)
    z_k   = v_k.conj() @ A_norm @ v_k
numr      = max_k |z_k|
gap       = numr - rho_A
feat      = [Re z, Im z, mu, gap, rho_A, numr, crawford_number,
             curvature(z), area(z)]
logit     = MLP([feat, pool(X_sq)])
```

### First config

```yaml
model:
  name: numerical_range_boundary_network
  input_channels: 18
  num_classes: 2
  hidden_dim: 96
  operator_rank_r: 16
  num_angles_K: 16
  spec_topk: 8
  complex_arithmetic: true
training:
  mode: puzzle_binary
  loss: bce_with_logits
  batch_size: 256          # complex operator + multi-eigh, scale carefully
  learning_rate: 5.0e-4
```

## Numerical / Compute Notes

- Each forward pass needs `K = 16` Hermitian eigendecompositions of `r x r = 16 x 16`
  matrices: `K * O(r^3) = 16 * 4096 = 6.5e4` flops per board. Cheap.
- One non-Hermitian eigendecomposition of `A` for `rho`: `O(r^3) = 4096`. Cheap.
- Use `torch.linalg.eigh` for Hermitian (cleanly differentiable; works on complex).
- Use `torch.linalg.eig` for `A`'s spectrum; backprop through complex eigvals is well-
  defined when eigenvalues are simple (regularize with a tiny non-Hermitian noise mask).
- Spectral-normalize `A` to `||A||_2 <= 1` to bound `numr <= 1` and stabilize training.
- Real-only fallback: replace `(A, A^*)` with `(A, A^T)` and lose the `i theta` rotation
  but keep the gap. Use full complex first.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `force_normal_A` | project A onto normal matrices via `A <- (A + A^*) / 2 + skew_part_with_block_orthonormal_eigvecs` | tests non-normality |
| `gap_only_scalar` | use only `numr - rho` scalar | tests sufficiency of single gap |
| `boundary_only_no_spec` | remove `rho` features, keep boundary | tests boundary alone |
| `random_M_primitives` | random sparse asymmetric primitives | tests chess semantics |
| `K_eq_4_low_resolution` | drop angles to K=4 | tests boundary resolution |
| `real_arithmetic_only` | use `(A, A^T)` Hermitian variant | tests need for complex |
| `cnn_same_params` | size-matched CNN | matched-capacity baseline |
| `i062_pencil_baseline` | run i062 (closest spectrum-only LA) | adjacent baseline |

For each: full 3x2 + slice reports.

## Benchmark Targets

```text
test PR AUC      >= 0.82
test F1          >= 0.76
near-puzzle FPR  <= 0.20
puzzle recall    >= 0.78

central claim:
  force_normal_A drops PR AUC >= 0.015
  K=4            drops PR AUC >= 0.005
  the model beats i062 by >= 0.01 PR AUC
```

## Counterexamples / Failure Modes

- Tactical pressure is approximately normal, so `numr - rho ~ 0` always; the gap signal
  is empty.
- The learned `A` collapses to a normal operator without the projection ablation,
  because nothing in the loss rewards non-normality.
  Mitigation: a small auxiliary loss `+ alpha * exp(-(numr - rho))` to keep `A`
  non-normal; report ablations with and without.
- Complex-arithmetic gradients are unstable on near-degenerate eigvalues.
- The numerical range is a *convex set*; very different chess positions might map to
  nearly the same convex set, losing discriminative power.

## Implementation Priority

1. Implement `boundary_sampler(A, K)` with differentiable `eigh`.
2. Build `A` real-symmetric first (degenerate case `numr = rho`); confirm pipeline runs.
3. Switch to non-symmetric real `A` via `(A, A^T)` Hermitian variant; check `gap > 0`.
4. Move to complex `A` once stable.
5. Run ablations, especially `force_normal_A`.

Smallest viable version:

```text
real non-symmetric A, K = 8 angles, real Hermitian variant,
features = (numr - rho, numr, rho, K boundary support values).
```

If lift over CNN-same-params is positive and `force_normal_A` ablation degrades, scale
to complex.
