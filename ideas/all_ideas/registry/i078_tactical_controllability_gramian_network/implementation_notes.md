# Implementation Notes

- Central code: `src/chess_nn_playground/models/tactical_controllability_gramian_network.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i078_tactical_controllability_gramian_network/model.py`
  (calls `build_tactical_controllability_gramian_network_from_config`).
- Registry key: `tactical_controllability_gramian_network`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-25_2004_saturday_shanghai_tactical_controllability_gramian.md`.
- Inputs are board-only: the simple_18 board tensor. CRTK/source/engine
  metadata is reporting-only and never enters the forward pass.
- The Gramian sums use the explicit recursion
  `W <- B B^T + A_hat W A_hat^T` (and `W <- C^T C + A_hat^T W A_hat`)
  authorised by the packet's solver-choice section; they avoid the
  tricky `solve_discrete_lyapunov(I - A kron A, vec(BB^T))` route.
- `W_o^{1/2}` is computed via batched `torch.linalg.eigh` with a tiny
  diagonal jitter for stability; the leading eigenvectors of `W_a`
  and `W_d` provide the principal-angle readout.
- Required ablations live on the model: `attacker_only`,
  `defender_only`, `no_observability`, `one_step_gramian`,
  `random_target_C`, `random_geometry_A`, `fixed_A_no_gates`,
  `diag_only_gramian`. The `cnn_same_params` baseline is
  trainer-side; the model only flips its `ablation_cnn_same_params`
  output flag.
