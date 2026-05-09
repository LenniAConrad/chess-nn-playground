# Implementation Notes

- Central code: `src/chess_nn_playground/models/prototype_margin_puzzle_network.py`.
- Registered model name: `prototype_margin_puzzle_network`.
- Idea-local wrapper: `ideas/i175_prototype_margin_puzzle_network/model.py`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`.
- Batch candidate: `Prototype-Margin Puzzle Network`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- Three prototype banks (`P_random`, `P_near`, `P_puzzle`) produce per-class log-sum-exp cosine similarities; the puzzle logit is the packet's `sim_puzzle - logsumexp([sim_random, sim_near])` margin.
- Required ablations are exposed via `ablation` in the model config: `none`, `single_negative_proto`, `no_margin_logsumexp`, `random_proto_freeze`, `prototype_count_sweep`.
