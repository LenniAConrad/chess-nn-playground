# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/tactical_transport_imbalance.py`.
- Builder: `build_tactical_transport_imbalance_network_from_config`.
- Idea-local wrapper: `ideas/all_ideas/registry/i031_tactical_transport_imbalance_network/model.py` (calls the builder).
- Registry key: `tactical_transport_imbalance_network`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0512_tuesday_local_transport_imbalance.md`.
- Input contract: simple_18 current-board tensor only. CRTK, engine, verification, and source metadata are reporting-only and never consumed as model input.
- Output contract: `dict` with one puzzle logit (`logits`) and named transport diagnostics (`transport_imbalance`, `forward_transport_cost`, `reverse_transport_cost`, `transport_entropy_gap`, `transport_concentration_gap`, `transport_rank_moment_gap`).
