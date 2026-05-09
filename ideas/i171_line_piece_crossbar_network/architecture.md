# Architecture

`Line-Piece Crossbar Network` is a board-only classifier for the
`puzzle_binary` task. It accepts the repo's simple 18-plane current-board
tensor with shape `(B, 18, 8, 8)` and returns one puzzle logit per
position together with diagnostics that expose the piece tokens, the
line tokens, and the per-line-type energies.

## Token construction

The 8x8 board carries two token populations:

- **Piece tokens**: one per square, ordered row-major as
  `s = r * 8 + c` (so `s = 0..63`). They are seeded from the
  convolutional stem features.
- **Line tokens**: 46 line tokens covering every chess line that a
  sliding piece can move along, partitioned by line type:
    - `8` ranks (rows `r = 0..7`),
    - `8` files (columns `c = 0..7`),
    - `15` diagonals (`r + c \in {0, ..., 14}`),
    - `15` anti-diagonals (`r - c + 7 \in {0, ..., 14}`).

  They are seeded from a learned per-line embedding plus a learned
  per-line-type embedding (rank / file / diag / anti-diag).

## Piece-line incidence

The deterministic 64x46 incidence matrix `I` records which lines each
square lies on:

    I_{s, l} = 1   iff   square s is on line l.

Every square lies on exactly four lines (its rank, its file, its
diagonal, its anti-diagonal), so each row of `I` has exactly four ones.
The line counts per column are 8 for ranks/files and 1..8 for
diagonals/anti-diagonals.

`I` is registered as a non-persistent buffer; it is a constant of the
chessboard and is never trained.

## Trunk

A single `Conv3x3 -> BatchNorm -> GELU -> Dropout` stem turns the
18-plane board tensor into a per-square feature map
`H \in R^{B \times C \times 8 \times 8}`. There is no further trunk-side
mixing; all global communication happens through the crossbar layers
below.

## Crossbar layer

Let `P \in R^{B \times 64 \times C}` be the piece tokens and
`L \in R^{B \times 46 \times C}` the line tokens entering a layer.
Define the row-normalized incidence weights

    A_{l, s} = I_{s, l} / sum_{s'} I_{s', l}        # (46, 64)
    B_{s, l} = I_{s, l} / sum_{l'} I_{s, l'}        # (64, 46)

so that each line aggregates a *mean* of the pieces on it and each piece
aggregates a *mean* of the four lines it lies on.

One layer performs:

    msg_p   = W_p P                                  # (B, 64, C)
    L_msg_l = sum_s A_{l, s} msg_p_s                 # (B, 46, C)
    L'      = LayerNorm( L + Dropout(GELU(L_msg)) )

    msg_l   = W_l L'                                 # (B, 46, C)
    P_msg_s = sum_l B_{s, l} msg_l_l                 # (B, 64, C)
    P'      = LayerNorm( P + Dropout(GELU(P_msg)) )

There is no convolution and no attention inside the layer; the only
cross-square communication is through the deterministic incidence.
The layer is stacked `depth` times.

## Head

After the final crossbar layer the piece tokens are mean-pooled across
the 64 squares and consumed by a small head:

    z       = mean_s P'_s
    \hat y  = Linear( GELU( Linear( LayerNorm(z) ) ) ).

## Diagnostics

The forward pass returns a dict with `B = batch`, `D = depth`,
`C = channels`, `P = 64`, `L = 46`:

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` for
  `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `trunk_features`: `(B, C, 8, 8)` square features after the stem.
- `piece_pool`: `(B, C)` mean-pooled piece tokens fed to the head.
- `piece_tokens`: `(B, P, C)` piece tokens after the last crossbar
  layer.
- `line_tokens`: `(B, L, C)` line tokens after the last crossbar layer.
- `rank_tokens`, `file_tokens`, `diag_tokens`, `antidiag_tokens`:
  per-line-type slices of `line_tokens`.
- `piece_token_history`, `line_token_history`: `(B, D, P/L, C)` tokens
  after every crossbar layer.
- `piece_message_stack`, `line_message_stack`: `(B, D, P/L, C)` raw
  per-layer messages before the residual + LayerNorm.
- `piece_energy`: `(B, P)` mean-square per piece token.
- `line_energy`: `(B, L)` mean-square per line token.
- `rank_line_energy`, `file_line_energy`, `diag_line_energy`,
  `antidiag_line_energy`: per-line-type slices of `line_energy`.
- `mean_piece_energy`, `mean_line_energy`: `(B,)` overall means.
- `rank_minus_file_line_energy`, `diag_minus_antidiag_line_energy`:
  `(B,)` per-line-type energy contrasts.
- `depth_levels`, `num_lines_levels`, `num_pieces_levels`: `(B,)`
  scalar tags carrying the configured depth, line count (46), and
  piece count (64).

## Implementation Binding

- Registered model name: `line_piece_crossbar_network`
- Source implementation file: `src/chess_nn_playground/models/line_piece_crossbar_network.py`
- Idea-local wrapper: `ideas/i171_line_piece_crossbar_network/model.py`
