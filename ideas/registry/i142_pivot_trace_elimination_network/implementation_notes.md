# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/pivot_trace_elimination_network.py`.
- Idea-local wrapper: `ideas/registry/i142_pivot_trace_elimination_network/model.py`
  (calls `build_pivot_trace_elimination_network_from_config`).
- Registry key: `pivot_trace_elimination_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2204_friday_shanghai_architecture_batch_8.md`.
- Batch candidate: `Pivot Trace Elimination Network` (rank 6).
- Hyperparameters follow the packet's "start tiny" guidance: `K = 12`
  groups, `D = 32` group dim, `lambda = 0.1` diagonal stabiliser. The
  elimination uses `eps = 1e-4` to keep `softplus(M_tt) + eps` away from
  zero. No learned pivoting; the canonical group order is the
  elimination order, with a fixed non-canonical permutation reserved for
  the `random_elimination_order` ablation.
- The model is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.
