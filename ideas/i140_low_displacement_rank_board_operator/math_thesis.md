# Math Thesis

Low-Displacement-Rank Board Operator

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.

Batch candidate rank: `4`.

Working thesis: Global square mixing can be parameterized by structured
matrices instead of dense attention or convolutions. A low-displacement-rank
operator over the flattened board can express long-range interactions with
Toeplitz / Hankel-like structure and few parameters.

## Linear Algebra Object

A 64x64 matrix `A` is *low displacement rank* with respect to a shift
matrix `Z` if

```text
Delta(A) = A - Z A Z^T
```

has low rank. The bespoke implementation realises this with five named
generators acting on the row-major flattening of the 8x8 board:

```text
A = T_rank + T_file + H_diag + H_anti + U V^T
```

where

- `T_rank = T_r (x) I_8` is rank-Toeplitz (15 parameters: `t_rank[r1 - r2 + 7]`),
- `T_file = I_8 (x) T_f` is file-Toeplitz (15 parameters: `t_file[f1 - f2 + 7]`),
- `H_diag[s1, s2] = h_main[d_main(s1) + d_main(s2)]` is a Hankel-like mixer
  indexed by the main diagonal `d_main(r, f) = r - f + 7` (29 parameters),
- `H_anti[s1, s2] = h_anti[d_anti(s1) + d_anti(s2)]` is the anti-diagonal
  Hankel counterpart with `d_anti(r, f) = r + f` (29 parameters),
- `U V^T` is a low-rank residual with `2 * 64 * low_rank_dim` parameters.

Together this is `88 + 2 * 64 * low_rank_dim` free parameters per operator
layer, well below the dense `4096` budget.

## Network

`L` operator layers iterate

```text
h_{t+1} = sigma(BN(A_t h_t + pointwise_mix(h_t)))
```

with a 1x1 channel mixer per layer. The puzzle head then consumes pooled
trunk features (mean / max over squares) together with the response
statistics suggested by the source packet:

- per-component squared response energies for `T_rank`, `T_file`,
  `H_diag`, `H_anti`, and `U V^T`,
- the operator response residual `||A h - h||`,
- the displacement residual norm `||A - Z A Z^T||_F` confirming the
  low-displacement-rank property at training time.

This realises the packet sketch faithfully: structured global mixing over
the flattened board with Toeplitz / Hankel-like structure, a low-rank
correction, and operator response statistics fed to a binary head.
