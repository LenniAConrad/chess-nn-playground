# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/color_flip_orbit_evidence.py`.
- Registered builder: `build_color_flip_orbit_evidence_bottleneck_from_config` in
  `src/chess_nn_playground/models/registry.py`.
- Registry key (config `model.name`): `color_flip_orbit_evidence_bottleneck`.
- Idea-local wrapper: `ideas/registry/i047_color_flip_orbit_evidence_bottleneck/model.py`
  delegates to the registered builder.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0751_tuesday_los_angeles_color_flip_orbit.md`.
- The model is intentionally board-only and does not consume engine,
  verification, source, or CRTK metadata as input.
- The `Simple18`/`EncodingSemanticSpec` adapter fails closed on any
  encoding other than `simple_18` with 18 channels unless the caller
  explicitly opts out via `fail_closed_unknown_channels=False`.
- The same model class supports the central research-packet falsifier
  through `orbit_transform="bad_rank_color"` and the duplicated-view
  ablation through `orbit_transform="identity"`.
