# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/piece_target_transport.py`.
- Idea-local wrapper: `ideas/all_ideas/registry/i033_piece_target_entropic_transport_bottleneck/model.py`
  (exports `build_model_from_config(config)`).
- Registered model name: `piece_target_entropic_transport_bottleneck`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0657_tuesday_los_angeles_piece_transport.md`.
- The model is intentionally board-only and never consumes engine,
  verification, source, or CRTK metadata as input.
- The transport branch only supports the `simple_18` 18-channel encoding;
  other encodings raise `ValueError` so unknown deterministic channel maps
  fail closed.
- Sinkhorn is unrolled in log domain with a clamp on the kernel exponent
  to keep mixed-precision training stable. Defaults are
  `sinkhorn_iters=16` and `sinkhorn_tau=0.15`, matching the source packet.
- Cost MLP: `(2 * type_dim + geometry_dim + direction_dim)`-input ->
  `transport_cost_hidden` -> `transport_cost_hidden` -> `transport_heads`.
  Geometry features are precomputed and registered as a buffer.
- Transport summaries: nine global statistics per direction-and-head plus
  four projected board maps (`source_cost_map`, `target_cost_map`,
  `source_conc_map`, `target_conc_map`). The fusion stack consumes the
  shallow board adapter together with the projected maps.
- `transport_ablation: cost_semantic_shuffle` is exposed for the central
  falsification ablation; `transport_ablation: none` is the default
  benchmark setting.
- The model emits a dictionary including `logits`, per-direction
  `transport_cost_*` and `transport_entropy_*` diagnostics,
  `transport_asymmetry`, `transport_low_cost_mass`, and
  `transport_bottleneck_norm`.
