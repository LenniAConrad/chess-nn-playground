# Math Thesis

Independence Residual Interaction Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2054_friday_shanghai_residual_inspired_batch.md`.

Batch candidate rank: `4`.

Working thesis: Some puzzle-like signals may be interactions that remain after
subtracting a simple independence explanation of board occupancy. Instead of
modeling all piece-square interactions directly, compute signed residuals:

`observed(piece, square) - expected(piece) * expected(square)`.

The implemented classifier constructs an expected piece-square occupancy tensor
from current-board piece/channel marginals, occupied-square mass, and
side-relative rank/file marginals. It then classifies from the signed residual
maps:

`r_{p,s}(x) = x_{p,s} - E_{p,s}(x)`.

The purpose is to force the head to focus on piece-square arrangements that are
not explained by material counts, global occupancy, or a low-rank rank/file
occupancy baseline.
