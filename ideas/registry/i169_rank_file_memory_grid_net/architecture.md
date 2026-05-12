# Architecture

`Rank-File Memory Grid Net` is a board-only classifier for the
`puzzle_binary` task. It accepts the repo's simple 18-plane current-board
tensor with shape `(B, 18, 8, 8)` and returns one puzzle logit per position
together with per-block rank/file memory diagnostics.

## Trunk

A single `Conv3x3 -> BatchNorm -> GELU -> Dropout` stem projects the
18-plane board tensor to a per-square feature map
`H \in R^{B \times C \times 8 \times 8}` of width `channels`. There is no
trunk-side cross-square mixing; all global communication happens through
the rank-file memory blocks below, faithful to the thesis.

## Rank-file memory block

Let `x \in R^{B \times C \times 8 \times 8}` denote square features going
into a memory block. Each block performs three steps:

1. **Square write.** A learned linear map `W_write : R^C -> R^M` projects
   every square into a memory token

       w_{b, h, w} = W_write x_{b, h, w}.

2. **Memory aggregation.** Squares average into their rank (row `h`) and
   file (column `w`) memory, after which a learned per-rank prior
   `p_rank \in R^{8 \times M}` and per-file prior
   `p_file \in R^{8 \times M}` are added (these are the "learned memory
   vectors" the thesis names) and a `LayerNorm` is applied:

       m_rank_{b, h} = LayerNorm( mean_w w_{b, h, w} + p_rank_h )
       m_file_{b, w} = LayerNorm( mean_h w_{b, h, w} + p_file_w )

3. **Memory read.** Each square reads its rank and its file memory back,
   concatenates them, and a learned linear map produces a per-square
   update that is added residually with `LayerNorm`:

       r_{b, h, w} = W_read [ m_rank_{b, h} ; m_file_{b, w} ]
       x'_{b, h, w} = LayerNorm( x_{b, h, w} + Dropout( GELU( r_{b, h, w} ) ) )

The block is stacked `depth` times. There are no convolutions inside the
block, no axial line solves, and no attention -- the only cross-square
communication is the rank-file write/read.

## Head

After the final memory block the trunk feature map is mean-pooled across
the 8x8 grid and consumed by a small head:

    z = mean_{h, w} x'_{b, h, w}
    \hat y = Linear( GELU( Linear( LayerNorm( z ) ) ) ).

## Diagnostics

The forward pass returns a dict with `B = batch`, `D = depth`,
`R = 8` ranks, `F = 8` files, `M = memory_dim`, `C = channels`:

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` for `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `trunk_features`: `(B, C, 8, 8)` square features after the last block.
- `square_pool`: `(B, C)` mean-pooled features fed to the head.
- `rank_memory_stack`: `(B, D, 8, M)` per-block rank memories after the
  prior + LayerNorm.
- `file_memory_stack`: `(B, D, 8, M)` per-block file memories after the
  prior + LayerNorm.
- `rank_write_stack`: `(B, D, 8, M)` raw rank-direction writes (mean over
  files, before the prior is added).
- `file_write_stack`: `(B, D, 8, M)` raw file-direction writes (mean over
  ranks, before the prior is added).
- `read_stack`: `(B, D, 8, 8, C)` per-block read activations after
  `GELU` (before the residual add).
- `rank_memory_energy`: `(B, D, 8)` mean square of each rank memory.
- `file_memory_energy`: `(B, D, 8)` mean square of each file memory.
- `mean_rank_memory_energy`: `(B,)` mean over depth/rank.
- `mean_file_memory_energy`: `(B,)` mean over depth/file.
- `rank_minus_file_energy`: `(B, D)` per-block rank minus file energy.
- `rank_file_imbalance`: `(B,)` mean over depth of the above; positive
  when rank memories carry more energy than file memories.
- `depth_levels`: `(B,)` scalar tag of the configured trunk depth.
- `memory_dim_levels`: `(B,)` scalar tag of the configured memory_dim.

## Implementation Binding

- Registered model name: `rank_file_memory_grid_net`
- Source implementation file: `src/chess_nn_playground/models/rank_file_memory_grid_net.py`
- Idea-local wrapper: `ideas/registry/i169_rank_file_memory_grid_net/model.py`
