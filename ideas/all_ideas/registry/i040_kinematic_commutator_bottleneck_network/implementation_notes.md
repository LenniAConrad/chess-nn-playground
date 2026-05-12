# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/kinematic_commutator_bottleneck.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i040_kinematic_commutator_bottleneck_network/model.py`.
- Registry key: `kinematic_commutator_bottleneck_network` (registered in
  `src/chess_nn_playground/models/registry.py`; intentionally removed
  from `RESEARCH_PACKET_MODEL_NAMES`).
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0728_tuesday_local_kinematic_commutator.md`.
- Input contract: empty-square mask and side-to-move are derived
  deterministically from the simple_18 piece planes (channels 0..11)
  and the side-to-move plane (channel 12). No legal moves, attack
  maps, engine metadata, CRTK source labels, or verification metadata
  are consumed as input.
- Operator bank: 8 slider directions (N, S, E, W, NE, NW, SE, SW) with
  current-board line-of-sight blocker gating, the knight leaper, the
  king one-step adjacency, and two pawn-attack flavours selected per
  batch element by the side-to-move scalar (12 operators total).
- Slider operators are applied without materialising the dense
  `(B, 64, 64)` matrix: the 7-step recurrence
  `r_{k+1} = M_d (E * r_k)` is iterated and the partial sums are
  accumulated, so memory stays linear in batch and hidden dim.
- Default operator pair set is the deterministic lexicographic list
  with `i < j` truncated to `num_operator_pairs` entries; the chunked
  evaluator processes `pair_chunk_size` pairs at a time so the
  conceptual `(B, P, d, 64)` commutator stack is never materialised
  in full.
- Default hyperparameters follow the markdown architecture spec
  (`hidden_dim=48`, `num_operator_pairs=28`, `pair_chunk_size=4`,
  `first_order_branch_dim=24`, `dropout=0.10`) and can be overridden
  via the idea config.
