# Implementation Notes

- Central code: `src/chess_nn_playground/models/research_packet_probe.py`.
- Registry key: `sylvester_tactical_coupling_network`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-05-05_1500_tuesday_local_sylvester_tactical_coupling.md`.
- This is intentionally board-only and does not consume engine, verification, source,
  or CRTK metadata as input.
- The custom linear-algebra operator described in the source packet is NOT yet
  implemented as a bespoke `nn.Module`; this folder uses the shared
  `ResearchPacketProbe` with `mechanism_family=linear_algebra` and
  `packet_profile=sylvester_tactical_coupling_network` so that the idea passes the registry contract
  and runs on the standard puzzle_binary benchmark. To upgrade to a bespoke module,
  add `src/chess_nn_playground/models/sylvester_tactical_coupling_network.py`, register a builder in
  `registry.py`, and update this idea's `model.py` and `config.yaml`.
