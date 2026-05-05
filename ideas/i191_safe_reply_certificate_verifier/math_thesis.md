# Math Thesis

Safe-Reply Certificate Verifier

Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.

Batch candidate rank: `6`.

Working thesis: Instead of proving that a position is a puzzle, try to prove that it is not a puzzle. If the model can find a cheap safe-reply certificate, the puzzle logit should go down.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.
