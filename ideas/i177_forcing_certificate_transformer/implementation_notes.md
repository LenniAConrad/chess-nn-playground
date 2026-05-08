# Implementation Notes

- Central code: `src/chess_nn_playground/models/forcing_certificate_transformer.py`.
- Idea-local wrapper: `ideas/i177_forcing_certificate_transformer/model.py`.
- Registry key: `forcing_certificate_transformer`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-25_0037_saturday_shanghai_puzzle_architecture_batch_2.md`.
- Batch candidate: `Forcing-Certificate Transformer`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The chess relation matrices used for the slot-attention bias are
  precomputed once and exposed via the `relations` buffer; they cover
  `same_rank`, `same_file`, both diagonals, `knight_reach`, `king_adjacent`,
  and the two directional pawn attacks.
- Required ablations: `none`, `no_relation_bias`, `no_global_residual`,
  `uniform_slot_attention`, `single_slot`.
