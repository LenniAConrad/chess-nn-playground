# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/toda_isospectral_flow.py`.
- Registered model name: `toda_isospectral_flow_network`.
- Idea-local wrapper: `ideas/all_ideas/registry/i235_toda_isospectral_flow_network/model.py` (calls
  `build_toda_isospectral_flow_network_from_config`).
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-05-05_1620_tuesday_local_toda_isospectral_flow.md`.
- Input contract: `simple_18` board tensor with shape `(batch, 18, 8, 8)`.
- Output: dictionary with `logits` of shape `(batch,)` plus diagnostic tensors
  (`diag_initial`, `diag_final`, `off_initial`, `off_final`, `sorting_score`,
  `max_off_diag_decay`, `mean_off_diag_decay`, `slowest_off_diag`,
  `spectral_gap_estimate`, `manakov_drift`, `manakov_drift_max`,
  `operator_frobenius_norm`).
- Flow integrator: explicit Euler on the matrix Lax form with re-symmetrisation
  at each step. Configurable `flow_steps`, `flow_dt`, and `manakov_order` keys
  in `config.yaml`.
- The model never consumes engine, verification, source, or CRTK metadata -
  those remain reporting-only artefacts.
