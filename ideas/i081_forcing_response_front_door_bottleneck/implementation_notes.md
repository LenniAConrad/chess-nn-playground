# Implementation Notes

- Central code: `src/chess_nn_playground/models/forcing_response_front_door_bottleneck.py`.
- Idea-local wrapper: `ideas/i081_forcing_response_front_door_bottleneck/model.py`
  exposes `build_model_from_config(config)` and delegates to
  `build_forcing_response_front_door_bottleneck_from_config`.
- Registered model name: `forcing_response_front_door_bottleneck`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-28_0733_tuesday_new_york_forcing_response_bottleneck.md`.
- The model is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.
- The `RuleInterventionFeatureBuilder` uses `python-chess` to enumerate
  visible-board legal candidates, build the deterministic move and
  response feature vectors, and emit `move_mask`, `move_from`,
  `move_to`, ray-pooling `path_weights`, and 12 board-rule planes.
- The packet's forbidden inputs (engine scores, PVs, best moves, mate
  scores, source IDs, verification flags, fine labels) are never
  passed to the model. Fine labels and CRTK metadata remain
  reporting-only.
- Forward returns a dict whose `"logits"` is `(B,)` for the repository
  `puzzle_binary` BCE-with-logits trainer; all other entries
  (`witness_gates`, `z_c`, `fine_logits`, `masked_pred`, `defense_gap`,
  ...) are diagnostics for prediction artefacts.
- The binary head reads only the bottleneck `Z_c`. There is no path
  from pooled board features to the puzzle logit other than through
  the sparse witness gate.
