# Implementation Notes

- Central code: `src/chess_nn_playground/models/tactical_bisimulation_puzzle_network.py`.
- Idea-local wrapper: `ideas/i075_tactical_bisimulation_puzzle_network/model.py`.
- Registry key: `tactical_bisimulation_puzzle_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0113_saturday_shanghai_tactical_bisimulation.md`.
- Inputs: `simple_18` board tensor only. The board-only `pi(a | x)` move
  proposer is the architectural stand-in for the deterministic legal /
  pseudo-legal sampler from the thesis; engine, source, verification,
  and CRTK metadata are never consumed at inference.
- Trainer: the in-tree `puzzle_binary` trainer optimises only the
  `logits` BCE term. The packet's auxiliary losses (`L_bisim`,
  `L_next`, `L_margin`) are not attached by the in-tree trainer; the
  diagnostic tensors required to compute them — `bisim_residual`,
  `prototype_distances`, `successor_*`, `move_proposal_entropy` — are
  exposed in the forward output dict so a future trainer or analysis
  can read them off prediction artifacts.
