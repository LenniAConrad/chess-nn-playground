# Math Thesis

Line-Piece Crossbar Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`.

Batch candidate rank: `2`.

Working thesis: Schur-Ray is mathematically powerful but more complex. A simpler line-aware architecture can create line tokens and piece tokens, then pass messages only through deterministic piece-line incidence.

## Tokens

The board carries two token populations:

- 64 *piece tokens*, one per square `s = r * 8 + c`.
- 46 *line tokens*, one per chess line that a sliding piece moves along:
  8 ranks, 8 files, 15 diagonals (`r + c` constant), and
  15 anti-diagonals (`r - c` constant).

## Piece-line incidence

A square `s` at `(r, c)` lies on exactly four lines: its rank `r`, its
file `c`, its diagonal `r + c`, and its anti-diagonal `r - c`. Define
the binary incidence matrix `I \in {0, 1}^{64 \times 46}` by
`I_{s, l} = 1` iff square `s` lies on line `l`. Every row of `I` has
exactly four 1s, and column sums match the number of squares on the
line (8 on ranks/files, 1..8 on diagonals/anti-diagonals).

`I` is a constant of the chess board, not a learned object.

## Crossbar message passing

A crossbar layer is a bipartite message round driven only by `I`.
Pieces aggregate onto lines via the column-normalized incidence and
lines aggregate back onto pieces via the row-normalized incidence:

    A_{l, s} = I_{s, l} / col_sum(l)            # (46, 64)
    B_{s, l} = I_{s, l} / row_sum(s) = I/4      # (64, 46)

Given piece tokens `P \in R^{64 \times C}` and line tokens
`L \in R^{46 \times C}` the layer does

    msg_p = W_p P
    L'    = LN( L + Dropout(GELU( A msg_p )) )
    msg_l = W_l L'
    P'    = LN( P + Dropout(GELU( B msg_l )) ).

Stacking the layer increases the *radius* of communication on the
chessboard while never letting information flow between squares that do
not share a chess line. Compared to Schur-Ray, this is a dramatically
simpler operator: it is a fixed bipartite mean-aggregator, not a line
algebra solve. Compared to attention or convolutions, it bakes the
piece-line geometry into the wiring instead of learning it.

## Why it should help on puzzle_binary

Puzzle-binary classification rewards models that can detect tactically
loaded lines (open files, exposed ranks, sharp diagonals through the
king). Routing every cross-square message through the line tokens gives
the head an explicit per-line summary, with separate slices for
ranks, files, diagonals, and anti-diagonals, that the puzzle logit and
the diagnostics can read directly.
