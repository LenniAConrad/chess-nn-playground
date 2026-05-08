# Math Thesis

Loop-Frustration Curvature Network (`LFCN`)

Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0729_tuesday_new_york_frustration_curvature.md`.

## Working thesis

A current-position chess board induces a finite spin-glass graph on
its 64 squares. For each static graph edge `e = (i, j)` and replica
`k`, a board-conditioned encoder produces a coupling
`J_{e,k}(x)` that says whether the two sites prefer aligned or
anti-aligned spins. A *closed* loop `ell` is **frustrated** when its
signed coupling product is negative,

```text
P_{ell,k}(beta, x) = prod_{e in ell} tanh(beta * J_{e,k}(x)).
```

True puzzles are hypothesised to contain sharper, more localised
contradiction structure than near-puzzles. A near-puzzle may match
material, occupancy, and visible threats, but the learned interaction
graph should not produce a comparably concentrated frustrated
response.

## Loop free energy and curvature

`LFCN` does not sample spin states. It computes a stable loop free
energy

```text
A_{ell,k}(beta, x) = log(1 + eta * P_{ell,k}(beta, x)),  eta = 0.90,
```

and the centered finite-difference curvature

```text
D2A_{ell,k} = (A(beta + delta) - 2 A(beta) + A(beta - delta)) / delta^2,
```

with `delta = 0.125` and `beta - delta` clamped to keep `beta > 0`.
The physical observable is

```text
Omega_{ell,k}(x) = sigmoid(-nu * P_mid) * |D2A_{ell,k}|, nu = 4.0,
```

so frustrated cycles whose free-energy response peaks under
temperature perturbation dominate. Scattering `Omega` to its loop
vertices yields `Omega_site ∈ R^{B x K x 8 x 8}`, a saliency field
that is then summarised by physics-derived statistics for a small
classifier head.

## Falsification

The thesis is falsified or seriously weakened if the cycle-scrambled
control matches `LFCN`, if the open-chain magnitude surrogate
matches `LFCN`, if `Omega` collapses or correlates with material
count alone, or if a shuffled-label run gives non-trivial validation
performance. All of these controls are implemented as model-level
ablations on the same forward pass.

## Implementation Binding

- Registered model name: `loop_frustration_curvature_network`.
- Source implementation file: `src/chess_nn_playground/models/loop_frustration_curvature_network.py`.
- Idea-local wrapper: `ideas/i080_loop_frustration_curvature_network/model.py`.
