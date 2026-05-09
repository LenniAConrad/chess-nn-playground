# Architecture

`Low-Displacement-Rank Board Operator` is a bespoke implementation of idea
`i140`. It learns a 64x64 global square-mixing operator on the flattened
board whose Sylvester displacement `A - Z A Z^T` (with `Z` a shift on the
flattened board) is low rank by construction. The operator is built as a
sum of structured pieces and applied in stacked layers, then per-component
response statistics are read into the puzzle head.

## Pipeline

- Input: board tensor `(B, 18, 8, 8)`. CRTK / source metadata is
  reporting-only and never used as model input.
- 1x1 input projection lifts each square to `channels` features.
- `num_layers` structured-operator blocks. Each block flattens the 8x8
  board to `(B, channels, 64)` and applies

  ```text
  h_{t+1} = sigma(BN(A_t h_t + pointwise_mix(h_t)))
  ```

  where `A_t` is the 64x64 low-displacement-rank operator and
  `pointwise_mix` is a 1x1 channel-wise convolution. The activation is
  `GELU` and `BN` is an optional batch norm.

## Operator construction

Each layer constructs

```text
A = T_rank + T_file + H_diag + H_anti + U V^T
```

with explicit, interpretable parameter vectors:

- `T_rank = T_r (x) I_8`: per-file Toeplitz mixer parameterised by the
  15-vector `t_rank[r1 - r2 + 7]` that depends only on the rank
  difference between the two squares and is zero across files.
- `T_file = I_8 (x) T_f`: per-rank Toeplitz mixer parameterised by the
  15-vector `t_file[f1 - f2 + 7]`.
- `H_diag[s1, s2] = h_main[d_main(s1) + d_main(s2)]` with
  `d_main(r, f) = r - f + 7`: a Hankel-like mixer indexed by the main
  diagonal of each square (29 free params).
- `H_anti[s1, s2] = h_anti[d_anti(s1) + d_anti(s2)]` with
  `d_anti(r, f) = r + f`: the anti-diagonal Hankel-like counterpart
  (29 free params).
- `U V^T`: a small learned low-rank residual with `2 * 64 *
  low_rank_dim` parameters.

By construction `A` admits a small generator pair under the cyclic shift
`Z` on the 64-square flattening, so `A - Z A Z^T` has low rank. Total
structured-operator parameters per layer are
`15 + 15 + 29 + 29 + 2 * 64 * low_rank_dim`, well below the dense
`64 * 64 = 4096` budget.

## Diagnostics

For the final layer the head receives:

- Per-component response energies `mean_{c, s} (M h)^2` for `M` in
  `{T_rank, T_file, H_diag, H_anti, U V^T}` (5 scalars).
- The operator response residual `||A h - h||` (root-mean-square per
  sample).
- The displacement residual norm `||A - Z A Z^T||_F`, exposed as a
  scalar diagnostic confirming that the learned operator is genuinely
  low displacement rank.
- Pooled trunk features (mean and max over squares).

The head is `LayerNorm -> Linear -> GELU -> Dropout -> Linear` and
returns one puzzle logit. Outputs are exposed under `ldr_*` keys for
prediction artifacts.

## Why this is distinct

- Not a CNN: `T_rank` is global along rank but separable across files,
  so it has no 2D translation invariance. `H_diag` and `H_anti` mix
  squares purely by diagonal index, which a fixed-size convolution
  cannot express.
- Not attention: the operator weights are static low-displacement-rank
  parameters, not data-dependent pair scores.
- Not Schur-Ray or bitboard shift: there is no Woodbury complement and
  no discrete shift polynomial; the operator is a single structured
  matrix sum.

## Implementation Binding

- Registered model name: `low_displacement_rank_board_operator`
  (registered in `src/chess_nn_playground/models/registry.py`).
- Source implementation file:
  `src/chess_nn_playground/models/low_displacement_rank_board_operator.py`
  (`LowDisplacementRankBoardOperator` and
  `build_low_displacement_rank_board_operator_from_config`).
- Idea-local wrapper:
  `ideas/i140_low_displacement_rank_board_operator/model.py` calls
  `build_low_displacement_rank_board_operator_from_config`.
- The shared `ResearchPacketProbe` scaffold is no longer used by this
  idea.
