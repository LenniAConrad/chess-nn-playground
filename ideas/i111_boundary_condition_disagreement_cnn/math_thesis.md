# Math Thesis

Boundary-Condition Disagreement CNN

Source packet: `ideas/research_packets/chess_nn_research_2026-04-24_2118_friday_shanghai_architecture_batch_3.md`.

Batch candidate rank: `5`.

Working thesis: Chess board edges matter: pawns, rooks, kings, and tactics behave differently near boundaries. A CNN's padding convention imposes a boundary assumption. Run a shared CNN under multiple boundary conditions and classify from disagreement.

This registered implementation tests the thesis through the `generic` mechanism profile in `ResearchPacketProbe`, using only board tensors and preserving all source/CRTK metadata for reporting.
