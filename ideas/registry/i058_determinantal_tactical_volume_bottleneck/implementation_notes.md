# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/determinantal_volume.py`.
- Builder: `build_determinantal_tactical_volume_bottleneck_from_config`.
- Registry key: `determinantal_tactical_volume_bottleneck`.
- Idea-local wrapper: `ideas/registry/i058_determinantal_tactical_volume_bottleneck/model.py`
  delegates to the bespoke builder. It does not use `ResearchPacketProbe` or
  `build_research_packet_probe_from_config`.
- Source packet:
  `ideas/research/packets/classic/chess_nn_research_2026-04-24_2044_friday_shanghai_determinantal_volume.md`.
- Input: simple_18 board tensor only. CRTK / engine / source / verification
  metadata is reporting-only and never used as model input.
- The Gram-matrix log-determinant is computed in (q x q) Sylvester form for
  numerical stability and O(B * R * q^3) cost; the (N x N) form is never
  materialised at inference time.
- The diagonal-trace ablation is exposed via the config field
  `model.ablation: diagonal_trace_only`.
