# Math Thesis

Piece-Token CNN Hybrid

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2109_friday_shanghai_piece_token_cnn_hybrid.md`.

Working thesis: a strong board-only puzzle_binary model can be obtained
by combining a dense 8x8 convolutional encoder of the simple_18 board
with an explicit set of occupied-piece tokens. The CNN supplies dense
spatial structure (adjacent-square local patterns and pooled global
statistics) while the token stream supplies a permutation-invariant
representation of the occupied squares enriched with piece type,
ownership relative to the side to move, normalized rank/file
coordinates, castling rights, the en-passant flag, and a deterministic
material summary. A late fusion head pools both streams, multiplies
their projections to expose a CNN-token interaction term, and
concatenates the result with the raw stream pools and the material
summary before a small MLP returns the puzzle logit.

The central falsifier asks whether explicit piece tokenization
contributes evidence beyond what a CNN of matched capacity can
extract from the same board. The folder's ablation suite
(`cnn_only_matched`, `token_only`, `no_interaction_fusion`,
`material_token_only`, `shuffle_token_coordinates`,
`single_token_layer`) targets the hybrid hypothesis directly: every
ablation either disables one stream, removes the multiplicative
interaction, or perturbs the per-token coordinate channels while
keeping the CNN trunk identical, so a positive result has to come from
the genuine combination rather than from any single stream.
