# Math Thesis

Counterplay Insolvency Ledger

Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_1859_tuesday_shanghai_puzzle_architecture_batch_5.md`.

Batch candidate rank: `3`.

Working thesis: Near-puzzles often fail because the defender has counterplay. A model that only measures side-to-move pressure may overcall these. Puzzlehood should depend on whether the opponent's counterthreats remain solvent after the side-to-move begins forcing play.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.
