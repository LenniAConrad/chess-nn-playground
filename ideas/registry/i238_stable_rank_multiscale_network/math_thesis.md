# Math Thesis

Stable-Rank Multiscale Network

Source packet: `ideas/research/packets/classic/chess_nn_research_2026-05-05_1710_tuesday_local_stable_rank_multiscale.md`.

Working thesis: Computes stable rank ||M||_F^2 / ||M||_2^2 of a learned 64x64 chess interaction at three block scales (full, 4 quadrants, 16 sub-blocks). Continuous, differentiable, scale-equivariant; quantifies effective tactical degrees of freedom.

This idea is **implemented as a bespoke torch module** at
`src/chess_nn_playground/models/trunk/stable_rank_multiscale.py`
(class `StableRankMultiscaleNetwork`, builder `build_stable_rank_multiscale_from_config`); not routed
through the generic ResearchPacketProbe.
