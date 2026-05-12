# Math Thesis

Bitboard Shift-Algebra Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2131_friday_shanghai_bitboard_shift_algebra.md`.

Working thesis: a chess board can be processed as a sparse shift algebra
over 64 squares using a small fixed family of rule-shaped square-shift
operators

```
S_north, S_south, S_east, S_west, S_ne, S_nw, S_se, S_sw,
S_knight_1 ... S_knight_8
```

where each `S_k: R^64 -> R^64` is a masked one-step displacement that
zeroes wraparound squares. From this fixed bank, the model evaluates
short shift-composition path families — orthogonal one/two/three-step
slides, diagonal one/two/three-step slides, knight jumps, king ring,
side-relative pawn captures, and a knight-then-king-ring composition —
and learns low-degree operator polynomials

```
P_h(S) x = sum_{p in P} alpha_{h,p}(x) * S_{p_m} ... S_{p_1} x
```

with board-conditioned coefficients `alpha(x)` produced from a compact
pooled board summary and normalized by `tanh / sqrt(P)` (or `softmax`
over P). A sigmoid blocker gate fuses each head's shift field with the
stem to suppress shift contributions where the current board has blockers
or noise. The classifier sees per-head shift diagnostics — pooled head
fields (mean, max, topk, occupied-square, king-zone), shift residual
`||head_field - u0||`, king-zone absolute residual, and occupied-square
energy — together with CNN and material summaries.

The thesis is that puzzle-like positions create distinctive responses
under these chess-shaped shift polynomials (forcing alignments, batteries,
knight forks, king-ring pressure) that a generic CNN does not extract,
without dense `64 x 64` attention, dynamic edge lists, or move
enumeration. The most important falsifier is `random_shift_bank`: if a
matched-sparsity random square permutation matches the chess shifts, the
mechanism is using extra capacity rather than chess geometry.
