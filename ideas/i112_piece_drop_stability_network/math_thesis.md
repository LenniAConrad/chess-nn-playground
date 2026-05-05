# Math Thesis

Piece-Drop Stability Network

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`.

Batch candidate rank: `6`.

Working thesis: Puzzle-like positions may be less stable under deterministic removal of specific safe current-board evidence groups. Instead of forcing a classifier to use sparse witnesses, measure how a small encoder's latent changes when piece groups are dropped.

This registered implementation tests the thesis through the `robustness` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.
