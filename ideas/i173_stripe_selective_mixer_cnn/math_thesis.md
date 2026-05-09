# Math Thesis

Stripe-Selective Mixer CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`.

Batch candidate rank: `4`.

Working thesis: A practical line-aware CNN may be enough to beat the
current BT4 while staying simpler than Schur-Ray. Instead of using
ordinary `3x3` convolutions only, every block mixes along the four
chess stripe directions — ranks, files, diagonals, and
anti-diagonals — and a per-channel sigmoid gate driven by the
global pool of the block input selects which stripe directions
matter for the position at hand.

The packet's central layer formula

    x_local = Conv3x3(x)
    x_rank  = rank_scan(x)
    x_file  = file_scan(x)
    x_diag  = diagonal_scan(x)
    x_anti  = anti_diagonal_scan(x)
    gate    = sigmoid(global_pool(x))
    x_next  = x + Conv1x1([x_local, gate*x_rank, gate*x_file,
                           gate*x_diag, gate*x_anti])

is implemented verbatim. The "scan" is a 1-D sequence convolution
along each stripe — a `(K, K)` `Conv2d` whose kernel is masked to be
non-zero only along the corresponding line — so the architecture
keeps the strengths of an ordinary CNN while adding exact long-range
line paths that can capture the one-blocker / one-diagonal /
one-escape-line difference that separates real puzzles from
near-puzzles.
