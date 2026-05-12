# Architecture

`Stripe-Selective Mixer CNN` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and returns one
puzzle logit per position, plus diagnostics that expose the
per-stripe branch energies and the per-block global-context gate.

## Mechanism

The premise of the packet is that ordinary `3x3` convolutions only
see local geometry, so near-puzzles that differ from puzzles by one
blocker on a long line are hard to separate. The architecture mixes
the local CNN with four explicit chess stripes — ranks, files,
diagonals, anti-diagonals — and lets a per-channel global gate
choose which stripe directions matter for a given board.

The packet's central layer formula is implemented verbatim:

    x_local = Conv3x3(x)
    x_rank  = rank_scan(x)
    x_file  = file_scan(x)
    x_diag  = diagonal_scan(x)
    x_anti  = anti_diagonal_scan(x)
    gate    = sigmoid(MLP(global_pool(x)))
    x_next  = x + Conv1x1([x_local,
                           gate * x_rank,
                           gate * x_file,
                           gate * x_diag,
                           gate * x_anti])

The local `Conv3x3` is never gated; only the four stripe branches
are multiplied by the per-channel sigmoid gate. The `1x1` projection
fuses the five branches back to `channels`, a residual is added, and
GELU is applied.

## Stripe scans

Each stripe scan is a `Conv2d` whose `(K, K)` kernel is constrained
by a fixed binary mask along the corresponding chess line:

- Rank scan: mask is `1` only on the centre row of the kernel
  (`mask[K // 2, :] = 1`). This is a simple sequence convolution
  along the file direction within a fixed rank.
- File scan: mask is `1` only on the centre column
  (`mask[:, K // 2] = 1`). Sequence convolution along the rank
  direction within a fixed file.
- Diagonal scan: mask is `1` only on the main diagonal of the kernel
  (`mask[i, i] = 1` for `i = 0..K-1`). Sequence convolution along
  the `(+1, +1)` direction — a bishop / queen diagonal.
- Anti-diagonal scan: mask is `1` only on the anti-diagonal
  (`mask[i, K - 1 - i] = 1`). Sequence convolution along the
  `(+1, -1)` direction.

Masks are non-trainable buffers, so the four directions share one
implementation and weights at masked-out positions stay at zero in
every forward pass. There is no recurrent machinery; the "scan" is
exactly a 1-D sequence convolution along the stripe.

The default `stripe_kernel` is `5`, which gives each stripe scan an
effective receptive field of five squares per layer. With `depth =
2` the network sees up to nine squares along every line — long
enough to span any chess line on the 8x8 board.

## Trunk

A single `Conv3x3 -> BatchNorm -> GELU` stem turns the 18-plane
board tensor into a per-square feature map of width `channels`. The
trunk is a stack of `depth` stripe-selective mixer blocks; each
block uses the formula above.

## Head

After the final block the trunk is mean+max pooled into a
`(B, 2 * channels)` descriptor and a small head emits the puzzle
logit:

    pooled = concat(mean_pool(h), max_pool(h))
    \hat y = Linear( GELU( Linear( LayerNorm(pooled) ) ) ).

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for
the repository `puzzle_binary` BCE-with-logits trainer. All tensors
are finite per batch:

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` for
  `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `trunk_features`: `(B, channels, 8, 8)` features after the last
  block.
- `trunk_energy`: `(B,)` mean-square trunk activation.
- `pooled`: `(B, 2 * channels)` descriptor fed to the head.
- `gate_history`: `(B, depth, channels)` per-block per-channel gate.
- `gate_per_block_mean`, `gate_per_block_min`, `gate_per_block_max`:
  `(B, depth)` summaries of each block's gate.
- `gate_overall_mean`: `(B,)` average gate strength across blocks.
- `local_branch_energy`, `rank_branch_energy`, `file_branch_energy`,
  `diag_branch_energy`, `antidiag_branch_energy`: `(B,)` mean-square
  per-branch activation averaged over blocks (zero for branches the
  ablation drops).
- `rank_minus_file_branch_energy`,
  `diag_minus_antidiag_branch_energy`: `(B,)` per-line-type energy
  contrasts.
- `active_stripe_count`, `stripe_kernel_levels`, `depth_levels`:
  `(B,)` scalar tags carrying the configured number of active
  stripe directions, the stripe kernel size, and the depth.
- `ablation_active`: `(B,)` flag set to `1.0` when the model is
  running a non-default ablation.

## Ablations

The packet's required ablations are exposed via `ablation`:

- `"none"` — main model.
- `"local_only"` — drop every stripe branch and keep only the local
  `Conv3x3`. Ordinary CNN control.
- `"rank_file_only"` — keep ranks and files (rook lines), drop
  diagonals and anti-diagonals.
- `"diag_only"` — keep diagonals only.
- `"random_stripes"` — replace every stripe mask with a fixed random
  `K`-position mask so the line geometry is destroyed while
  parameter count stays matched.
- `"no_global_gate"` — drop the sigmoid global-context gate so
  stripe branches are summed without selection.

## Implementation Binding

- Registered model name: `stripe_selective_mixer_cnn`
- Source implementation file: `src/chess_nn_playground/models/trunk/stripe_selective_mixer_cnn.py`
- Idea-local wrapper: `ideas/registry/i173_stripe_selective_mixer_cnn/model.py`
