# Architecture

`Variational Board Action Network` implements the packet's calculus object directly:
it learns a multi-channel board field, computes finite differences on the 8x8 square
lattice, forms a differentiable action density, and feeds an Euler-Lagrange-style
residual bottleneck to the puzzle-binary classifier.

## Field Encoder

Input is the repo `simple_18` current-board tensor `(B, 18, 8, 8)`. A small
`Conv1x1 -> Conv3x3 -> Conv1x1` encoder emits:

- learned board field `u: (B, C, 8, 8)`;
- context features used by Lagrangian parameter heads.

The default config uses `field_channels: 12`, matching the packet's recommended
first experiment.

## Action And Residual

The model computes fixed forward differences with explicit reflect-style boundary
handling:

- `Dx u[i,j] = u[i,j+1] - u[i,j]` with zero boundary flux at the right edge;
- `Dy u[i,j] = u[i+1,j] - u[i,j]` with zero boundary flux at the bottom edge.

Context heads emit positive stiffness maps:

- `gx = softplus(raw_gx) + eps`;
- `gy = softplus(raw_gy) + eps`.

The implemented action density is:

```text
L = V_hat(u, x) + 0.5 * gx * (Dx u)^2 + 0.5 * gy * (Dy u)^2
```

The residual follows the packet's first recommended force-head approximation:

```text
R = force_head(u, x) - DxT(gx * Dx u) - DyT(gy * Dy u)
```

where `DxT` and `DyT` are the fixed adjoints of the finite-difference operators.

## Readout

The classifier receives:

- scalar action summaries: action value, potential energy, gradient energy, field
  energy, stiffness mean/anisotropy, and boundary flux;
- residual summaries: L1, L2, max, localization, and mask-weighted residual splits
  for king zones, occupied squares, empty squares, own pieces, opponent pieces,
  center squares, and edge squares;
- a residual-map CNN over `R`;
- a compact board CNN summary over the original board tensor.

It returns one BCE puzzle logit plus diagnostics for the action terms, residual terms,
finite-difference energies, residual map energy, and board CNN energy.

## Ablation Hooks

Supported `model.ablation` values are `cnn_only_matched`, `action_only`,
`no_gradient_terms`, `random_difference_operators`, `residual_norm_only`,
`force_head_only`, and `harmonic_control`.

## Implementation Binding

- Registered model name: `variational_board_action_network`.
- Source implementation file: `src/chess_nn_playground/models/variational_board_action.py`.
- Idea-local wrapper: `ideas/registry/i071_variational_board_action_network/model.py`.
