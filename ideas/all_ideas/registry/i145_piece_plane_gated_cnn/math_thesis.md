# Math Thesis

Piece-Plane Gated CNN

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2208_friday_shanghai_plain_architecture_batch.md`.

Batch candidate rank: `3`.

Working thesis: The `simple_18` channels are not arbitrary image channels. A plain CNN can respect this by first processing semantically related channel groups, then using learned gates to mix piece types and colors.

Implemented hypothesis: separate stems for white pieces, black pieces, and state planes should help the model preserve channel semantics before a learned gate mixes material and state context into a residual board CNN.
