# Implementation Notes

- Central code: `src/chess_nn_playground/models/research_packet_probe.py`.
- Registry key: `numerical_range_boundary_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-05-05_1515_tuesday_local_numerical_range_boundary.md`.
- This is intentionally board-only and does not consume engine, verification, source,
  or CRTK metadata as input.
- The custom linear-algebra operator described in the source packet is NOT yet
  implemented as a bespoke `nn.Module`; this folder uses the shared
  `ResearchPacketProbe` with `mechanism_family=linear_algebra` and
  `packet_profile=numerical_range_boundary_network` so that the idea passes the registry contract
  and runs on the standard puzzle_binary benchmark. To upgrade to a bespoke module,
  add `src/chess_nn_playground/models/numerical_range_boundary_network.py`, register a builder in
  `registry.py`, and update this idea's `model.py` and `config.yaml`.
