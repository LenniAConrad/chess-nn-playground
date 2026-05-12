# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/soft_formal_concept_closure.py`.
- Idea-local wrapper: `ideas/registry/i057_soft_formal_concept_closure_network/model.py`.
- Registry key: `soft_formal_concept_closure_network` (registered in
  `src/chess_nn_playground/models/registry.py`).
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0922_tuesday_los_angeles_concept_closure.md`.
- Board-only input: `simple_18` tensor; no engine, verification, source, CRTK,
  or label metadata is consumed. The `Simple18BoardAdapter` enforces this and
  fails closed for non-`simple_18` encodings unless explicitly disabled.
- Rule attributes are computed deterministically from the current board only:
  pseudo-legal attacks (pawn, knight, bishop, rook, queen, king) and slider
  rays use iterative shifted-mask propagation with current-occupancy blockers.
  No legal-move enumeration or engine call.
- Central row/column-preserving rewire control is exposed as
  `semantic_rewire_ablation: true` in the model config; it operates on the
  binary incidence matrix with a deterministic, seedable bipartite double-edge
  swap that preserves both row and column sums.
- `marginal_only_ablation: true` collapses the closure path to attribute
  column marginals and globals.
- Default configuration: `num_concepts=64`, `attr_embedding_dim=32`,
  `probe_embedding_dim=16`, `concept_hidden_dim` derived from `hidden_dim`,
  `tau_extent=tau_closure=0.15`, `intent_temperature=1.0`.
