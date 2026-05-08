# Implementation Notes

- Central code: `src/chess_nn_playground/models/legal_reaction_bottleneck_network.py`.
- Registry key: `legal_reaction_bottleneck_network`.
- Idea-local wrapper: `ideas/i186_legal_reaction_bottleneck_network/model.py` (thin `build_model_from_config` that delegates to `build_legal_reaction_bottleneck_network_from_config`).
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0040_saturday_shanghai_puzzle_architecture_batch_3.md`.
- Batch candidate: `Legal-Reaction Bottleneck Network`.
- Board-only: side-to-move plane is the only meta-channel consulted; CRTK / source / engine / verification metadata are never used as model input.
- Distinct from idea i185 (Critical-Square Budget Network): i185 routes the puzzle logit through a single fixed-budget soft mask over all 64 squares; this idea places the bottleneck specifically on the opponent-piece *defender-reply graph*, with a *data-dependent* effective reaction count `K_eff = exp(H(p))` and an explicit threat / reaction split.
