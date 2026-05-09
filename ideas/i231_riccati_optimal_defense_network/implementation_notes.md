# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/riccati_optimal_defense_network.py`.
- Registry key: `riccati_optimal_defense_network` (registered in
  `src/chess_nn_playground/models/registry.py`).
- Idea-local wrapper: `ideas/i231_riccati_optimal_defense_network/model.py`
  delegates to `build_riccati_optimal_defense_network_from_config`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1600_tuesday_local_riccati_optimal_defense.md`.
- Input is the board tensor only; CRTK / engine / verification / source
  metadata is reporting-only and never enters the model.
- The CARE solve is performed by complex eigendecomposition of the
  Hamiltonian `H = [[A, -B R^{-1} B^T], [-Q, -A^T]]` followed by
  selecting the `r` most-stable eigenvectors and solving
  `V_1^T X^T = V_2^T` for `X = P` (Tikhonov-regularized,
  symmetrized, real part). This is differentiable via
  `torch.linalg.eig` and `torch.linalg.solve`.
- `A` is built as `-softplus(alpha) I + flow` and additionally clipped
  by `max(0, max_real(eig(A)) + hurwitz_safety) * I` so that the
  open-loop dynamics stay strictly Hurwitz before the Hamiltonian solve.
- `Q` and `R` are built as PSD/PD by construction with positive floors
  `q_floor_beta` and `r_floor_gamma`.
