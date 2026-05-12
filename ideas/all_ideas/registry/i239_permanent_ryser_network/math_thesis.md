# Math Thesis

Permanent Ryser Coupling Network

Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-05-05_1715_tuesday_local_permanent_ryser.md`.

Working thesis: Builds top-k attacker x top-k defender bilinear interaction matrix M (k=6); computes its permanent via Ryser's formula -- the unsigned count of perfect attacker-to-defender matchings. Distinct from DPP (signed determinant, i058) and Pfaffian (signed matchings of skew-graphs, i226).

This idea is **implemented as a bespoke torch module** at
`src/chess_nn_playground/models/permanent_ryser.py`
(class `PermanentRyserNetwork`, builder `build_permanent_ryser_from_config`); not routed
through the generic ResearchPacketProbe.
