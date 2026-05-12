# Implementation Notes

- Registered model name: `counterfactual_move_delta_spectrum_network`.
- Source: `src/chess_nn_playground/models/counterfactual_move_delta_spectrum.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i026_counterfactual_move_delta_spectrum_network/model.py`.
- Reused primitives:
  - `Simple18BoardAdapter` and `PseudoLegalDeltaEnumerator` from
    `src/chess_nn_playground/models/move_landscape_net.py`. They provide the
    rule-only `simple_18` parser and the deterministic pseudo-legal delta
    enumerator used by both i025 and i026. Their semantics are validated by
    the i025 tests; this idea reuses them rather than re-implementing chess
    move logic.
- Bespoke components added for this idea:
  - `BoardStem`: small convolutional stem with coordinate planes that produces
    the `8x8xd_sq` square map and the global feature.
  - `MoveTokenEncoder`: gathers `H_from`, `H_to`, `H_to - H_from`, the
    broadcast global feature, and deterministic move descriptors, then
    produces per-move response vectors `r in R^k`.
  - `CounterfactualSpectrumPool`: masked mean/var/max plus the uniform-
    weighted covariance `K`, eigenvalues via `torch.linalg.eigvalsh` (cast
    to float32 for numerical stability), and spectral statistics: trace,
    leading-eigenvalue fraction, participation ratio, normalised spectral
    entropy, and Frobenius norm.
  - `MoveDeltaSpectrumHead`: LayerNorm + 2-layer MLP that consumes `g`,
    `r_mean`, `r_max`, `r_var`, eigenvalues, and spectral scalars.
- Inputs are board-only (`simple_18` 18 channels). CRTK / source / engine
  metadata is reporting-only and never reaches the model.
- Output contract matches the puzzle-binary trainer: `forward(x)` returns a
  dict with `logits` shaped `(B,)` plus diagnostic scalars.
- The optional finite-difference bottleneck `beta * E[trace(K)]` is reported
  via the `trace_penalty_beta` diagnostic. The model itself only emits
  `logits`; the trace penalty hook can be added in the trainer if/when the
  shared trainer exposes auxiliary-loss callbacks.
- Determinism: move enumeration is sorted by
  `(piece, from, special, to, promotion)` and the eigen decomposition is
  computed in float32. The model raises `ValueError` for unsupported
  encodings.
