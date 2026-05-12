# Codex Research Packet: Free-Probability R-Transform Spectrum Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1535_tuesday_local_free_probability_r_transform.md`
- Generated at: 2026-05-05 15:35
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full unorthodox-linear-algebra architecture packet, not implemented, not benchmark results

## One-Sentence Thesis

Treat the attacker operator `A` and defender operator `B` as **freely independent**
non-commutative random variables, predict the spectrum of `A + B` (or `A * B`) via the
**R-transform** (resp. **S-transform**) -- the free-probability analog of the
log-characteristic-function -- and use the *gap* between the actually-measured spectrum
of `A + B` and the free-probabilistic prediction `R_A oplus R_B` as a tactical-coupling
fingerprint that no classical (commutative) spectrum analysis can detect.

## Why This Is A Real And Unorthodox Linear Algebra NN Idea

In free probability (Voiculescu 1985), two non-commutative random variables `A`, `B`
are **freely independent** iff for any polynomials `p_i, q_i` with
`tau(p_i(A)) = tau(q_i(B)) = 0`, we have

```text
tau( p_1(A) q_1(B) p_2(A) q_2(B) ... ) = 0
```

where `tau` is the trace-state. Free convolution `mu_A boxplus mu_B` gives the spectral
distribution of `A + B` *under the free-independence assumption*, computed via the
**R-transform**:

```text
R_{A+B}(z) = R_A(z) + R_B(z)         (additive)
S_{AB}(z)  = S_A(z) S_B(z)            (multiplicative)
```

where `R_A(z) = G_A^{-1}(z) - 1/z` and `G_A(z) = tau((z - A)^{-1})` is the Cauchy /
resolvent transform of `A`.

For chess, this matters because:

- Classical (commuting) spectrum: `spec(A + B)` = `spec(A) + spec(B)` only if `A, B`
  commute -- which attacker and defender almost never do.
- Tensor product / Kronecker spectrum: `spec(A kron I + I kron B)` = all sums; assumes
  *full independence*.
- **Free convolution**: a single canonical "non-commutative independent" prediction.

A position is *coupled* (= a puzzle) iff the actually observed spectrum of `A + B`
deviates significantly from the free-convolution prediction. The deviation -- a
distribution-distance like the Wasserstein between empirical and predicted spectra --
is the bottleneck feature.

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

- `i062 Matrix-Pencil Generalized Spectrum` — generalized eigenproblem, classical.
- `i076 Krylov Subspace` — single operator, classical.
- `i029-i034 entropic-OT packets` — couplings via OT, not free-probability.
- `i120 Sinkhorn Role Assignment` — coupling via doubly-stochastic, not free-prob.
- `i108 TensorSketch Interaction` — uses random tensor sketches, distinct.

### Exact difference

```text
Free probability is the unique non-commutative analog of classical probability that
applies to operators. The R-transform converts addition of free operators into addition
of analytic functions, exactly as the log-characteristic function converts independent
sums of classical RVs. The deviation between the free-convolution prediction and the
empirical spectrum of A+B is a coupling fingerprint with no classical-statistics
analog. No imported packet uses free-probabilistic transforms or free convolution.
```

## Mathematical Thesis

### Definitions

Build attacker `A in R^{n x n}` and defender `B in R^{n x n}` from the current board
(`n = 64`, low-rank gates as before). Use `A_sym = (A + A^T)/2`, `B_sym = (B + B^T)/2`
so spectra are real.

### R-transform

Compute the empirical Cauchy transform on a `K`-point grid `{z_1, ..., z_K}` in the
upper half-plane (or on a real grid above the spectra):

```text
G_A(z) = (1/n) sum_i 1 / (z - lambda_i(A))
```

Invert (numerically: 1D Newton or Brent solve) to get `K_A(z) = G_A^{-1}(z)`, then

```text
R_A(z) = K_A(z) - 1/z
```

Same for `B`. Free additive convolution prediction:

```text
R_{A+B}^{predicted}(z) = R_A(z) + R_B(z)
G_{A+B}^{predicted}(z) = (R_A(z) + R_B(z) + 1/z)^{-1}     (re-invert)
```

Recover predicted spectral measure `mu_{A+B}^{pred}` via Stieltjes inversion (boundary
limit of imaginary part of `G`).

Empirical spectrum: `spec(A_sym + B_sym)`.

### Coupling distance

```text
d_couple = W_2( empirical_spec(A_sym + B_sym), mu_{A+B}^{predicted} )
        + |first_few_free_cumulants of (A+B) -
           (free_cumulants of A) - (free_cumulants of B)|
```

Free cumulants `kappa_k` are the coefficients of `R(z) = sum_k kappa_k z^{k-1}` and
satisfy `kappa_k(A + B) = kappa_k(A) + kappa_k(B)` for free `A, B`.

### Readout

```text
d_couple                                        scalar (key feature)
mismatch_kappa_topk = (kappa_k(A+B) - kappa_k(A) - kappa_k(B))_{k=1..6}
spec_overlap       = | spec(A) cap spec(B) |   approximate (kernel)
free_independence_score = exp(-beta * d_couple)
asymmetry_score    = || A - B^T ||_F            (do A and B even play same role?)
```

Final:

```text
puzzle_logit = MLP([d_couple, mismatch_kappa_topk, asymmetry_score, board_pool])
```

## Assumptions

- Attacker and defender operators are *approximately freely independent* under
  non-puzzle conditions.
- A puzzle is exactly the breaking of free independence: a tactical coupling makes the
  empirical spectrum `spec(A+B)` deviate from the free-convolution prediction.
- The first 4-6 free cumulants are sufficient to detect this deviation.

## Claim / Hypothesis

The free-cumulant mismatch `kappa_k(A+B) - kappa_k(A) - kappa_k(B)` is non-zero
*precisely* on tactical positions and approximately zero on non-tactical ones.

Central falsifier:

```text
classical_swap: replace free convolution with classical convolution (assume A, B
                commute and use spec(A) + spec(B) elementwise then sort).
                if PR AUC does not drop, free-probability is unnecessary.

permutation_freeness_check:
                replace empirical spec(A+B) with spec(A + P B P^T) for a random
                permutation P. The free-convolution prediction is permutation-invariant
                in distribution; if the network's prediction is unchanged, the model
                is using free-probability features as expected.
```

## Architecture

### Components

```text
board_encoder
operator_A_builder, operator_B_builder
sym_eig_block         -> spec(A_sym), spec(B_sym), spec(A_sym + B_sym)
cauchy_transform_block-> G_A(z), G_B(z), G_{A+B}(z) on K grid points
r_transform_block     -> invert G to K, subtract 1/z
free_cumulants_block  -> Taylor coefficients of R at z = 0
coupling_distance_block
puzzle_head
```

### Forward pseudocode

```text
X_sq    = board_encoder(board)
A_sym, B_sym = build_operators(X_sq)
ev_A    = eigh(A_sym).eigvals               # n
ev_B    = eigh(B_sym).eigvals
ev_S    = eigh(A_sym + B_sym).eigvals
G_A     = mean( 1/(grid_z - ev_A), dim=-1 ) # K
G_B     = mean( 1/(grid_z - ev_B), dim=-1 )
G_S     = mean( 1/(grid_z - ev_S), dim=-1 )
K_A     = invert_cauchy_via_newton(G_A, grid_z)
K_B     = invert_cauchy_via_newton(G_B, grid_z)
R_A     = K_A - 1/grid_z
R_B     = K_B - 1/grid_z
G_pred  = 1 / (R_A + R_B + 1/grid_z)
d_coup  = wasserstein_1d(spec_S, stieltjes_inverse(G_pred))
kappa_A, kappa_B, kappa_S = first_k_free_cumulants(R_A), ..., from char-poly of S
mismatch= kappa_S - kappa_A - kappa_B
logit   = MLP([d_coup, mismatch, asymmetry, pool(X_sq)])
```

### First config

```yaml
model:
  name: free_probability_r_transform_network
  input_channels: 18
  num_classes: 2
  hidden_dim: 96
  operator_n: 64
  cauchy_grid_K: 32
  cumulant_order: 6
  cauchy_inversion: newton
training:
  mode: puzzle_binary
  loss: bce_with_logits
  batch_size: 256
  learning_rate: 5.0e-4
```

## Numerical / Compute Notes

- Three `eigh` of `64 x 64` matrices per board: `3 * O(n^3) = 7.9e5` flops. Fine.
- Cauchy transform on `K = 32` complex grid points: `O(n K) = 2048`. Fine.
- Inverting `G_A(z) = w` for `K_A(w)` via Newton: each `z_k` independently, ~5 iters.
  Differentiable via implicit-function backward.
- Free cumulants from char-poly: extract first `m` Taylor coefficients of `R` at 0, or
  combinatorially from non-crossing partitions of moment cumulants up to order 6.
- Wasserstein-1D between sorted eigval sequences: differentiable via sorted-quantile
  approximation.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `classical_swap` | use `sort(spec(A) + spec(B))` instead of free conv | tests free-prob signal |
| `kappa_only` | drop `d_couple`, keep cumulant mismatch | tests cumulant sufficiency |
| `random_B_baseline` | use random B instead of board-derived | tests defender semantics |
| `B_eq_zero` | set B = 0; free conv collapses to A | sanity collapse |
| `swap_A_B` | swap attacker/defender | tests asymmetry |
| `A_B_commuting_force` | enforce `[A,B] = 0` via projection | tests non-commutativity |
| `cnn_same_params` | matched CNN | baseline |
| `i062_pencil_baseline` | adjacent LA baseline | baseline |

For each: full 3x2 + slice reports.

## Benchmark Targets

```text
test PR AUC      >= 0.82
test F1          >= 0.76
near-puzzle FPR  <= 0.20
puzzle recall    >= 0.78

central claim:
  classical_swap drops PR AUC >= 0.015
  A_B_commuting_force drops PR AUC >= 0.01
  beats i062 by >= 0.01 PR AUC
```

## Counterexamples / Failure Modes

- Attacker and defender on a real chess board are *never* freely independent (they
  share square-positions); the free-conv prediction is always wrong, and the
  *deviation* is dominated by this constant misspecification rather than by tactical
  coupling. Mitigation: the `random_B_baseline` ablation should still degrade.
- Newton inversion of `G` is unstable on the real spectrum.
- Free cumulants are dominated by `kappa_2` (variance) which both classes share.
- `n = 64` is too small for free-probability asymptotics (these are large-n theorems);
  finite-`n` free-prob exists but bounds are weaker.

## Implementation Priority

1. Implement Cauchy transform `G_A(z)` on a complex grid and Newton inversion. Test on
   random Hermitian.
2. Compute `R_A`, free additive convolution `R_A + R_B`, re-invert to `G_pred`.
3. Extract first 6 free cumulants from a series expansion of `R` at 0.
4. Build attacker/defender ops; train with cumulant-mismatch + Wasserstein head.
5. Run all 8 ablations.

Smallest viable version:

```text
n = 32 (subset of squares), K = 16 grid, only kappa_3 and kappa_4 mismatch as features.
```

If lift over CNN-same-params is positive, scale to `n = 64` and full Wasserstein.
