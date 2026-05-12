# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/trunk/ray_language_automaton_network.py`.
- Idea-local wrapper: `ideas/registry/i039_ray_language_automaton_network/model.py`.
- Registry key: `ray_language_automaton_network` (registered in
  `src/chess_nn_playground/models/registry.py`; intentionally removed
  from `RESEARCH_PACKET_MODEL_NAMES`).
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0719_tuesday_local_ray_language_automaton.md`.
- Input contract: side-relative tokens are derived deterministically
  from the `simple_18` piece planes (channels 0..11) and the
  side-to-move plane (channel 12). No legal moves, attack maps,
  engine metadata, CRTK source labels, or verification metadata are
  consumed as input.
- Ray index buffers are precomputed at construction time as registered
  non-persistent buffers. The recurrence uses one log-sum-exp step per
  ray position with mask-gated state updates so padded positions on
  short diagonals do not contribute.
- Default hyperparameters follow the markdown architecture spec
  (`R=32` automata, `Q=8` states, `|A|=14` alphabet symbols, T<=8
  positions per ray) and can be overridden via the idea config.
