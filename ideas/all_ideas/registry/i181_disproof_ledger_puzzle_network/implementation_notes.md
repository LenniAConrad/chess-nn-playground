# Implementation Notes

- Central code: `src/chess_nn_playground/models/disproof_ledger_puzzle_network.py` (`DisproofLedgerPuzzleNetwork`).
- Idea-local wrapper: `ideas/all_ideas/registry/i181_disproof_ledger_puzzle_network/model.py` calls `build_disproof_ledger_puzzle_network_from_config`.
- Registry key: `disproof_ledger_puzzle_network`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.
- Batch candidate: `Disproof-Ledger Puzzle Network`.
- Inputs are limited to the `simple_18` board tensor; engine, verification, source, and CRTK metadata are reporting-only and are never used as model input.
- The trainer multiplies `output["disproof_l1"]` by the configured `disproof_sparsity` (default `0.01`) and adds it to the loss only when `output["uses_disproof_sparsity"][0] == 1`. The near-puzzle auxiliary uses `output["max_disproof_strength"]` and is gated by `output["uses_near_disproof_aux"][0]`.
