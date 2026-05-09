# Architecture

`Pawn Skeleton Barrier Network` is a board-only puzzle-binary classifier that turns the pawn skeleton into deterministic barrier and distance fields and uses those fields to gate a compact board CNN. The architecture is bespoke: there is no shared `ResearchPacketProbe` wrapper.

## Input And Side Canonicalization

- Input is the repo `simple_18` board tensor with shape `(batch, 18, 8, 8)`.
- White pawns, black pawns, white king, black king, and the side-to-move plane are read from the standard simple_18 channel layout.
- Each position is canonicalized to side-to-move so the own pawns always advance toward higher ranks; the black-to-move case is rotated by flipping the rank axis. Front-spans, attack-fronts, and shelter zones can therefore share a single forward orientation.

## Deterministic Pawn Skeleton Fields

A non-trainable `PawnSkeletonFeatureBuilder` computes a 30-channel side-canonical skeleton stack from own and opponent pawn masks plus own and opponent king masks:

- own and opponent pawn masks (2 channels)
- own and opponent front spans, computed by directional cumulative sums of pawn masks excluding the current rank (2 channels)
- own and opponent attack fronts, computed by diagonally shifting pawn masks one rank forward (2 channels)
- per-file own / opponent / total pawn-count planes broadcast to `(8, 8)` (3 channels)
- open-file mask and minimum file-distance to the nearest open file (2 channels)
- isolated-pawn masks for own and opponent pawns (no friendly pawn on adjacent files) (2 channels)
- doubled-pawn masks for own and opponent pawns (more than one pawn on the same file) (2 channels)
- passed-pawn masks for own and opponent pawns (no opposing pawn on the same or adjacent file ahead) and their forward passed-lane spans (4 channels)
- own and opponent shelter zones (the 1-2 ranks in front of the king on adjacent files) and the pawns that actually occupy those zones (4 channels)
- minimum Manhattan distance to the nearest own pawn, opponent pawn, and pawn frontier (3 channels)
- king-to-shelter Manhattan distance for own and opponent kings broadcast to `(8, 8)` (2 channels)
- own and opponent 3x3 king-zone masks (2 channels)

The builder also exposes scalar pawn-structure summaries used by the head: pawn counts, isolated/doubled/passed-pawn totals, shelter-pawn count, shelter distances, pawn-frontier density, and the open-file mask.

## Board Trunk And Pawn-Conditioned Gating

A `Simple18PawnAdapter` reads the 18-channel board, a `ConvNormGelu` projects it to `channels` width, and a stack of two `ConvNormGelu` blocks projects the 30-channel pawn skeleton to the same width. A 1x1 convolution turns the projected pawn features into a sigmoid gate that is applied multiplicatively to the board features as `board * (1 + gate)`. The conditioned features pass through `depth` `ResidualBoardBlock`s with batch norm, GELU, and residual connections.

## Barrier And Shelter Pooling

The classifier sees five pooled views of the conditioned board:

- global mean and global max
- open-file masked pool (so open-file structure shows up explicitly)
- own king-zone masked pool
- opponent king-zone masked pool

It also sees the per-channel mean of the pawn skeleton stack and ten scalar pawn-structure summaries (own/opponent pawn shares, isolated / doubled / passed shares, shelter-pawn share, own and opponent shelter distance, open-file density, pawn-frontier density). All of those are concatenated and passed through a two-layer GELU MLP.

## Output

The model returns one puzzle logit for the `puzzle_binary` task (fine labels 0 and 1 map to non-puzzle, fine label 2 maps to puzzle). Diagnostic outputs include `logits`, `pawn_stack_energy`, `pawn_gate_mean`, `pawn_gate_variance`, `own_pawn_count`, `opponent_pawn_count`, `open_file_pressure`, `isolated_pawn_pressure`, `doubled_pawn_pressure`, `passed_lane_pressure`, `king_shelter_pressure`, `king_shelter_distance`, `pawn_frontier_density`, and `conditioned_board_energy`.

## Implementation Binding

- Registered model name: `pawn_skeleton_barrier_network`.
- Source implementation: `src/chess_nn_playground/models/pawn_skeleton_barrier.py`.
- Idea-local wrapper: `ideas/i126_pawn_skeleton_barrier_network/model.py`.
