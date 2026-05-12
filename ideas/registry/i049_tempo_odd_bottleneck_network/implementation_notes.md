# Implementation Notes

- Bespoke architecture: `src/chess_nn_playground/models/trunk/tempo_odd_bottleneck.py` (`TempoOddBottleneckNet`).
- Idea-local wrapper: `ideas/registry/i049_tempo_odd_bottleneck_network/model.py` calling `build_tempo_odd_bottleneck_from_config`.
- Registry key: `tempo_odd_bottleneck_network`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0755_tuesday_los_angeles_tempo_odd_bottleneck.md`.
- Input is the current `simple_18` board tensor only. The deterministic side-to-move involution `tau` and the en-passant sanitization happen inside the model; no engine, verification, source, or CRTK metadata is consumed as input.
- The bespoke implementation runs a shared encoder over `[x, tau(x)]`, applies a two-point Walsh odd/even split, and routes the odd projection through the high-capacity predictive path while the even projection is consumed only via a stop-gradient context bottleneck.
