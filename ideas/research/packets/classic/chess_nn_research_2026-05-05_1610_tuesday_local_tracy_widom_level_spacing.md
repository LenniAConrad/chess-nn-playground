# Codex Research Packet: Tracy-Widom Level-Spacing Network

## File Metadata

- Filename: `chess_nn_research_2026-05-05_1610_tuesday_local_tracy_widom_level_spacing.md`
- Generated at: 2026-05-05 16:10
- Weekday: Tuesday
- Timezone: local
- Author: Claude (Opus 4.7, 1M context)
- Intended next consumer: Codex
- Status: full unorthodox-linear-algebra architecture packet

## One-Sentence Thesis

Compute the eigenvalues of a learned chess operator `H`, normalize by the local spectral density, and classify puzzle-likeness from the **nearest-neighbor level-spacing distribution** `P(s)`: tactical positions (chaotic) follow Wigner-Dyson statistics with level repulsion `P(s) ~ s^β e^{-c s²}` (β = 1, 2, 4 for GOE, GUE, GSE), while non-tactical positions (integrable) follow Poisson statistics `P(s) = e^{-s}` — a quantum-chaos signature distinct from any spectral-radius / numerical-range / RMT-mean idea.

## Why This Is A Real Unorthodox Linear Algebra NN Idea

The **Bohigas-Giannoni-Schmit conjecture** (1984): the spectral statistics of a quantum system are universally classified by symmetry class:

- **Integrable / regular**: Poisson statistics, no level repulsion: `P(s) = e^{-s}`.
- **Chaotic, time-reversal symmetric**: GOE, `P_GOE(s) = (π/2) s exp(-π s²/4)`.
- **Chaotic, no time-reversal**: GUE, `P_GUE(s) = (32/π²) s² exp(-4 s²/π)`.
- **Chaotic, Kramers degeneracy**: GSE, `P_GSE(s) = (262144/729π³) s⁴ exp(-64 s²/9π)`.

The **Tracy-Widom distribution** governs the largest-eigenvalue fluctuations of Wigner matrices and is the universal limit for many random-matrix ensembles.

Key invariants we read off:

```text
nearest-neighbor spacing histogram s_i = (lambda_{i+1} - lambda_i) / mean_local_spacing
level-spacing ratio r_i = min(s_i, s_{i-1}) / max(s_i, s_{i-1})  in [0, 1]
        <r>_Poisson = 0.386
        <r>_GOE     = 0.536
        <r>_GUE     = 0.602
spectral form factor K(t) = |sum_i exp(2 pi i lambda_i t)|^2
```

These are **bulk-spectrum** invariants — they care about *correlations* between eigenvalues, not their absolute positions. Distinct from i062 (eigvalues themselves), i076 (Krylov subspaces), i077 (resolvent), i199 (Hessian spectrum), i228 free-probability (cumulants).

## Target

```text
fine 0,1 -> 0,  fine 2 -> 1.  3x2 fine-to-binary mandatory.
```

## Forbidden Inputs

Standard.

## Closest Existing Ideas And Exact Difference

- `i062, i076, i077, i078, i199, i228` — all use eigenvalue *positions*; none use the *spacing distribution*.
- `i078 Gramian` — output reachability via integrated `e^{At}`; not level statistics.

```text
Level-spacing statistics are eigenvalue-correlation invariants. They are scale-, shift-,
and unitarily invariant after unfolding. They distinguish chaotic vs integrable
*regimes* of the operator, which no spectrum-position model exposes.
```

## Mathematical Thesis

### Definitions

Build `H = (M + M^T) / 2` Hermitian at full size `n = 64`:

```text
M = sum gates_M * primitives_chess
H = (M + M^T) / 2
```

Compute all 64 eigenvalues `lambda_1 <= ... <= lambda_64`.

### Unfolding

Local mean spacing varies; we unfold via a smooth average density estimate:

```text
N_smooth(lambda) = smoothed cumulative spectral count
unfolded eigvals: tilde_lambda_i = N_smooth(lambda_i)
```

Differentiable via a 5-point smoothing kernel.

### Level-spacing ratios

```text
s_i = tilde_lambda_{i+1} - tilde_lambda_i
r_i = min(s_i, s_{i-1}) / max(s_i, s_{i-1})        in [0, 1]
```

The mean ratio `<r>` is a single scalar that distinguishes regimes (Poisson 0.386, GOE 0.536, GUE 0.602).

### Spectral form factor

Discrete Fourier of unfolded eigvals:

```text
K(t_k) = | sum_i exp(2 pi i tilde_lambda_i t_k) |^2,    t_k in {0.05, 0.1, ..., 1.0}
```

The dip-and-ramp behavior of `K(t)` is another universal RMT signature.

### Readout

```text
spacing_histogram                         (8 bins, R^8)
mean_ratio_r                              scalar
form_factor_samples K(t_k)                R^K
nu_Poisson_loglik = sum log P_Poisson(s_i)
nu_GOE_loglik     = sum log P_GOE(s_i)
nu_GUE_loglik     = sum log P_GUE(s_i)
regime_softmax = softmax([nu_Poisson, nu_GOE, nu_GUE])
puzzle_logit = MLP([histogram, r, K, regime_softmax, board_pool])
```

## Assumptions

- Tactical positions create operators with chaotic (level-repulsing) spectra.
- Non-tactical positions create regular (no-repulsion, Poisson) spectra.
- Symmetry breakers in chess (side-to-move, castling, en-passant) determine GOE vs GUE.

## Claim / Hypothesis

The mean level-spacing ratio `<r>` is a strong puzzle-likeness signal independent of the eigvalue positions. Central falsifier:

```text
shuffle_eigvals: randomly permute the eigvals before computing spacings.
                 (Permutation destroys level-spacing structure but preserves the
                 multiset of eigvals.)
                 if PR AUC doesn't drop, level repulsion isn't the signal.
```

## Architecture

```text
board_encoder
H_builder            -> Hermitian 64 x 64
eigh_block           -> eigvals
unfolding_block      -> tilde_lambda
spacing_block        -> s_i, r_i
form_factor_block    -> K(t)
loglik_classifier    -> Poisson vs GOE vs GUE log-likelihoods
puzzle_head
```

### First config

```yaml
model:
  name: tracy_widom_level_spacing_network
  input_channels: 18
  num_classes: 1
  channels: 64
  hidden_dim: 96
  hermitian_n: 64
  num_form_factor_taps: 16
  spacing_histogram_bins: 8
training:
  mode: puzzle_binary
  loss: bce_with_logits
```

## Numerical / Compute Notes

- Single `eigh(H)` per board: `O(n^3) = 2.6e5`. Fine.
- Unfolding via 5-pt local mean of cumulative spectral count; differentiable.
- Spacing ratios are scale-invariant — no unfolding needed for `r`. We still unfold for the histogram and form factor.

## Required Ablations

| Ablation | Removes | Hypothesis |
|---|---|---|
| `shuffle_eigvals` | random permutation of eigvals | tests level-correlation signal |
| `r_only` | use only mean ratio scalar | tests sufficiency |
| `histogram_only` | use only histogram | tests histogram alone |
| `no_unfolding` | use raw spacings | tests unfolding |
| `force_real_symmetric` | always GOE class | tests symmetry-class detection |
| `cnn_same_params` | matched CNN | baseline |
| `i062_pencil_baseline` | adjacent baseline | baseline |
| `i228_free_prob_baseline` | adjacent baseline | baseline |

## Benchmark Targets

```text
PR AUC >= 0.82, F1 >= 0.76, near-puzzle FPR <= 0.20, puzzle recall >= 0.78
shuffle_eigvals drops PR AUC >= 0.015 (key claim)
beats i062 by >= 0.01 PR AUC at matched params
```

## Counterexamples

- Chess operators are too small (`n = 64`) for asymptotic RMT statistics; finite-n corrections dominate.
- The operator is too sparse / structured for chaos.
- All chess positions look chaotic at this size, no regime separation.

## Implementation Priority

1. Build Hermitian H from chess primitives, run `eigh`, plot level-spacing histogram.
2. Implement unfolding and ratio-r computation.
3. Add Poisson/GOE/GUE log-likelihoods.
4. Train minimal `(<r>, regime_softmax)` head; compare to CNN.
5. Run 8 ablations.
