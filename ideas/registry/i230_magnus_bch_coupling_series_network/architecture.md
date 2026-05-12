# Architecture

## Overview

`Magnus-BCH Operator-Coupling Series Network` builds a per-board pair of low-rank
operators `A, B in R^{r x r}` (default `r = 12`) from a compact convolutional
trunk, spectrally clamps each so `||A||_2, ||B||_2 <= spectral_clip_per_op` (so
that `||A||_2 + ||B||_2 < log 2` and the BCH series converges with safety margin),
and computes the Hall basis of nested commutators of `A, B` up to weight 4:

```text
c_2  = [A, B]
c_3a = [A, c_2]      c_3b = [B, c_2]
c_4a = [A, c_3a]     c_4b = [B, c_3a]
c_4c = [A, c_3b]     c_4d = [B, c_3b]
```

The truncated Baker-Campbell-Hausdorff log to weight 4 is

```text
Z = A + B + 1/2 c_2 + (1/12)(c_3a - c_3b) + (1/24) c_4b
```

so `c_4b` is the only weight-4 monomial with a nonzero BCH coefficient. The other
three weight-4 Hall monomials enter the feature head anyway: they are part of the
Hall basis of the free Lie algebra at weight 4 and capture iterated coupling that
the BCH log itself ignores but which is still chess-meaningful.

## Components

- Board encoder: convolutional trunk with mean+max pooling.
- Operator builders: two linear heads map the pooled board summary to raw
  `(r, r)` matrices `A_raw, B_raw`. Each is spectrally normalized via
  `svdvals(M)[..., 0]` and divided by `(sigma / spectral_clip_per_op).clamp_min(1)`.
- Hall-basis commutator block: computes `c_2, c_3a, c_3b, c_4a, c_4b, c_4c, c_4d`.
- BCH log block: evaluates the truncated BCH series at weight 4.
- Magnus feature block: emits the nine Hall-basis Frobenius norms,
  the truncated BCH log Frobenius norm `bch_log_F`, six per-pair decay ratios
  `||c_3a||/||c_2||, ||c_3b||/||c_2||, ||c_4a||/||c_3a||, ||c_4b||/||c_3a||,
  ||c_4c||/||c_3b||, ||c_4d||/||c_3b||`, and six structurally-normalized
  weight-3 / weight-4 norms `||c_k||_F / (||A||_F^a * ||B||_F^b)` for the
  monomial multiplicities `(a, b)`.
- Head: pooled board features + Magnus feature vector feed an MLP that returns
  one puzzle logit.

## Diagnostics returned by the forward pass

- `magnus_norms` (the nine Hall-basis Frobenius norms in deterministic order)
- `magnus_ratios` (the six per-pair decay ratios)
- `magnus_normalized_norms` (the six structurally-normalized weight-3/4 norms)
- `magnus_bch_log_norm`, `magnus_weight_decay_3_to_2`,
  `magnus_weight_decay_4_to_3`
- `magnus_operator_norm_A`, `magnus_operator_norm_B`,
  `magnus_commutator_norm_c2`,
  `magnus_commutator_norms_w3`, `magnus_commutator_norms_w4`

## Implementation Binding

- Registered model name: `magnus_bch_coupling_series_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/magnus_bch_coupling_series_network.py`
- Idea-local wrapper: `ideas/registry/i230_magnus_bch_coupling_series_network/model.py`
