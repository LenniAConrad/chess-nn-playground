# Architecture

## Overview

`Sylvester Tactical Coupling Network` couples a learned attacker operator `A`
and defender operator `B` through the Sylvester equation `A X + X B = C`,
where `C` is a board-derived obligation matrix.

## Components

- Board encoder: convolutional trunk with pooled mean/max summary.
- Operator heads: linear maps to `r x r` operators `A`, `B`, `C`.
- `A` and `B` are spectrally normalized so `||A||_2, ||B||_2 <= 1`.
- Sylvester solver: vec-form solve via batched Kronecker
  `(I_r kron A + B^T kron I_r) vec(X) = vec(C)`, fully differentiable.
- Spectral readout: top-`k` singular values of `X`, Frobenius / spectral
  norms, soft rank, attacker/defender projected energies, bounded
  log-volume `log|det(I + X X^T)|`.
- Resonance readout: `min/mean |lambda_i(A) + mu_j(B)|` over the cross
  spectrum.
- Classifier: pooled board features + spectral readout + resonance summary
  feed an MLP head.

## Diagnostics returned by the forward pass

- `sylvester_frobenius`, `sylvester_spectral`, `sylvester_soft_rank`
- `sylvester_log_volume`, `sylvester_attacker_energy`,
  `sylvester_defender_energy`
- `sylvester_resonance_min`, `sylvester_resonance_mean`,
  `sylvester_singular_topk`

## Implementation Binding

- Registered model name: `sylvester_tactical_coupling_network`
- Source implementation file: `src/chess_nn_playground/models/sylvester_coupling.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i221_sylvester_tactical_coupling_network/model.py`
