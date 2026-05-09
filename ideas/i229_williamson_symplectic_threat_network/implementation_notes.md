# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/williamson_symplectic_threat_network.py`.
- Idea-local wrapper: `ideas/i229_williamson_symplectic_threat_network/model.py`
  delegates to `build_williamson_symplectic_threat_network_from_config`.
- Registered model name: `williamson_symplectic_threat_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1540_tuesday_local_williamson_symplectic_threat.md`.
- Input is the current-board `simple_18` tensor only; CRTK / source / engine
  metadata is reporting-only and never enters the model.
- Symplectic spectrum is computed via the
  `M^{1/2} J M^{1/2} -> eigvalsh(K^T K)` route so all operations are
  differentiable through `eigh`. Paired entries are averaged before
  taking square roots to mitigate the multiplicity-2 splitting that the
  numerics produce around `+/- i d_i`.
- Default phase dimension is `n = phase_n = 32` (so `M` is `64 x 64`).
  The original packet recommends scaling to `n = 64` once the smaller
  configuration shows lift; both are supported through `config["model"]`.
