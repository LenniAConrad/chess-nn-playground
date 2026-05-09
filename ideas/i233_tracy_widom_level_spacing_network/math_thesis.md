# Math Thesis

Tracy-Widom Level-Spacing Network

Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1610_tuesday_local_tracy_widom_level_spacing.md`.

## Working thesis

A learned chess Hermitian operator `H = (M + M^T) / 2` of size `n = 64` carries
a spectrum whose nearest-neighbour level statistics distinguish quantum-chaotic
positions (Wigner-Dyson, level repulsion `P(s) ~ s^beta exp(-c s^2)`) from
integrable / regular positions (Poisson, `P(s) = exp(-s)`). The bulk-spectrum
invariants we extract are scale-, shift- and unitarily invariant after
unfolding.

## Quantities computed

For each board the model computes:

- `eigvals(H)` via `torch.linalg.eigvalsh`.
- Unfolded spectrum `tilde_lambda` via a 5-point local average of the
  empirical staircase. Spacings `s_i = tilde_lambda_{i+1} - tilde_lambda_i`
  are renormalised to mean spacing 1.
- Mean nearest-neighbour spacing ratio
  `<r> = mean_i min(s_i, s_{i-1}) / max(s_i, s_{i-1})`. Reference values:
  Poisson 0.386, GOE 0.536, GUE 0.602.
- Soft (RBF) spacing histogram in 8 bins on `s in [0, 4]`.
- Spectral form factor
  `K(t_k) = | sum_i exp(2 pi i tilde_lambda_i t_k) |^2 / n`
  evaluated at `num_form_factor_taps` taps in `[0.05, 1.0]`.
- Per-spacing log-likelihoods under Poisson, GOE, and GUE surmises and the
  3-way regime softmax `softmax([nu_Poisson, nu_GOE, nu_GUE])`.

The puzzle head consumes the pooled board representation together with the
spacing histogram, mean ratio, form factor, regime log-likelihoods, and
regime softmax. Output is one puzzle logit plus the bulk-spectrum
diagnostics (mean ratio, histogram, form factor, regime softmax,
log-likelihoods).

## Distinction from neighbouring ideas

Spectrum-position ideas (`i062`, `i076`, `i077`, `i199`, `i228`) read
eigenvalue *positions* or moments. This idea reads *correlations* between
eigenvalues - the spacing distribution itself - which no spectrum-position
baseline exposes.
