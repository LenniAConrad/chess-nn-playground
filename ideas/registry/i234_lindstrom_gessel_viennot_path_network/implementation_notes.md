# Implementation Notes

- Central code:
  `src/chess_nn_playground/models/trunk/lindstrom_gessel_viennot_path_network.py`.
- Registry key: `lindstrom_gessel_viennot_path_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-05-05_1615_tuesday_local_lindstrom_gessel_viennot_path.md`.
- Board-only architecture; engine, verification, source, and CRTK
  metadata are reporting-only and are never consumed as model input.
- Edge weights are derived from per-square left/right embeddings, masked
  to a strict upper-triangular DAG under the row-major square order, and
  row-normalised so `alpha * W` is contractive. The Neumann series is
  truncated at `neumann_steps` terms, which is the maximum directed path
  length the model accumulates.
- `num_paths` controls the size of the LGV path matrix `M`; values
  `>= 2` give a non-trivial determinant. The implementation regularises
  `slogdet` with `det_eps * I` to keep the log-magnitude finite when the
  DAG admits no full non-intersecting matching for the soft-selected
  sources and targets.
- The shared `ResearchPacketProbe` scaffold is no longer used; the
  bespoke `LindstromGesselViennotPathNetwork` is registered directly in
  `chess_nn_playground/models/registry.py`.
