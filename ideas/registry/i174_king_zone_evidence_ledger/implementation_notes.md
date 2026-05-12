# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/king_zone_evidence_ledger.py`
  (`KingZoneEvidenceLedger`, `EvidenceLedger`, `BoardTrunk`).
- Idea-local wrapper: `ideas/registry/i174_king_zone_evidence_ledger/model.py`.
- Registry key: `king_zone_evidence_ledger`.
- Source packet:
  `ideas/research/packets/classic/chess_nn_research_2026-04-25_0031_saturday_shanghai_puzzle_binary_challengers.md`
  (Idea 5, "King-Zone Evidence Ledger", batch rank 5).
- The model is board-only and never consumes engine, verification,
  source, or CRTK metadata as input.
- King anchors are read from the `simple_18` board planes (5 = white
  king, 11 = black king), then re-keyed to own/opp using the
  side-to-move plane (12). For all-zero king planes the anchor falls
  back to the centre square so degenerate FENs do not crash the
  forward pass.
- The packet's `slot = slot + gated_pool(...)` rule is realised as
  slot-conditioned soft attention with a sigmoid gate and LayerNorm,
  iterated `ledger_layers` times.
- Ablations follow the packet table exactly: `no_king_relative`,
  `random_king_anchor`, `global_slots_only`, and the no-op
  `slot_count_sweep` flag for runs that vary `num_slots`.
