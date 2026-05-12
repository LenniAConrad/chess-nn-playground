# Architecture

## Overview

`Free-Probability R-Transform Spectrum Network` treats attacker `A` and
defender `B` as freely independent operators and predicts the spectrum of
`A + B` via the free additive convolution. It classifies puzzle-likeness
from the deviation between the empirical spectrum of `A_sym + B_sym` and the
free-convolution prediction together with the free-cumulant mismatch
`kappa_k(A + B) - kappa_k(A) - kappa_k(B)`.

## Components

- Board encoder: convolutional trunk + pooled mean/max summary.
- Attacker / defender heads: linear maps to `n x n` raw matrices,
  symmetrized and spectrally normalized to give Hermitian
  `A_sym`, `B_sym`.
- `eigh` of `A_sym`, `B_sym`, and `A_sym + B_sym` gives empirical spectra.
- Empirical moments and free cumulants via the closed-form recursion
  `kappa_1 = m_1`, `kappa_2 = m_2 - m_1^2`,
  `kappa_3 = m_3 - 3 m_1 m_2 + 2 m_1^3`,
  `kappa_4 = m_4 - 4 m_1 m_3 - 2 m_2^2 + 10 m_1^2 m_2 - 5 m_1^4`.
- Free-convolution prediction: `kappa_pred = kappa_A + kappa_B`. The
  prediction's first two cumulants give a Gaussian approximation of the
  predicted measure, and the Wasserstein-1d distance to the empirical
  spectrum of `A + B` is the coupling distance.
- Mismatch features: `kappa_S - kappa_pred`, asymmetry `||A - B||_F`,
  spec overlap, free-independence score `exp(-d_couple)`.
- Classifier: pooled board features + cumulant features + scalar features
  feed an MLP head.

## Diagnostics returned by the forward pass

- `free_coupling_distance`, `free_independence_score`,
  `free_cumulant_mismatch`
- `free_cumulants_A`, `free_cumulants_B`
- `free_asymmetry`, `free_spec_overlap`, `free_predicted_std`

## Implementation Binding

- Registered model name: `free_probability_r_transform_network`
- Source implementation file: `src/chess_nn_playground/models/free_probability_r_transform.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i228_free_probability_r_transform_network/model.py`
