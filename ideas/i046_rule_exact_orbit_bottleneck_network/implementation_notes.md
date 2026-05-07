# Implementation Notes

- Bespoke source: `src/chess_nn_playground/models/rule_exact_orbit_bottleneck.py` (`RuleExactOrbitBottleneckNet`, builder `build_rule_exact_orbit_bottleneck_from_config`).
- Idea-local wrapper: `ideas/i046_rule_exact_orbit_bottleneck_network/model.py`.
- Registry key: `rule_exact_orbit_bottleneck_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0750_tuesday_los_angeles_orbit_bottleneck.md`.
- Board-only model: it consumes the `simple_18` current-board tensor; CRTK, engine, verification, and source metadata stay reporting-only.
- Color-flip orbit is parameter-free and deterministic. The adapter fails closed when the channel schema is not `simple_18` unless explicitly opted out via `fail_closed_unknown_channels=false`.
- The packet's central falsifier (`kappa -> rank_flip_no_color`) is supported in the same class via `orbit_group="rank_flip_no_color"`; `orbit_group="identity"` gives the single-view ablation. All three ablations share the encoder and parameter count.
