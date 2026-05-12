# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/empty_square_opportunity_network.py`.
- Registry key: `empty_square_opportunity_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.
- Batch candidate: `Empty-Square Opportunity Network`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The implementation derives occupied and empty masks from piece planes, applies
  separate occupied-square and empty-square branches, emits learned opportunity
  maps from the empty branch, and fuses occupied/empty pooled features with product
  and absolute-difference interactions.
