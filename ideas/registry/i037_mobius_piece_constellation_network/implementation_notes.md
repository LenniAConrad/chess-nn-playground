# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/mobius_piece_constellation.py`.
- Registry key: `mobius_piece_constellation_network`.
- Idea wrapper: `ideas/registry/i037_mobius_piece_constellation_network/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0713_tuesday_los_angeles_mobius_constellation.md`.

The model is board-only and currently supports deterministic tokenization only for `simple_18`. It reads piece planes `0:12` and state planes `12:18`; any other encoding requires an explicit `channel_map.current_piece_planes` and `channel_map.state_planes` or construction raises `ValueError`.

Tokenization uses weighted sums against learned piece, square, and joint piece-square embeddings, then masks empty squares. The elementary-symmetric interaction block computes degrees one through three by descending recurrence, so no pair or triple candidate set is materialized.

The configured `num_classes: 1` path returns a single puzzle logit for the repo's puzzle-binary contract. Setting `num_classes: 2` exposes a two-logit head for packet-style cross-entropy experiments. Degree masks and gates can be ablated through `max_degree`, `use_degree_gates`, and `normalize_by_tuple_count`.
