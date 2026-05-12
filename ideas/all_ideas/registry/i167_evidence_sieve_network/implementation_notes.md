# Implementation Notes

- Central code: `src/chess_nn_playground/models/evidence_sieve_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i167_evidence_sieve_network/model.py` (calls
  `build_evidence_sieve_network_from_config`).
- Registry key: `evidence_sieve_network`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2216_friday_shanghai_architecture_batch_11.md`.
- Batch candidate: `Evidence Sieve Network`.
- This is intentionally board-only and does not consume engine, verification,
  source, or CRTK metadata as input.
- The forward pass returns a dict including `logits`, the per-stage channel
  and spatial masks, the selected evidence tensor at each stage, and a set
  of scalar selection/energy/entropy diagnostics for downstream tooling.
