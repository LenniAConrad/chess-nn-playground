# Implementation Notes

- Source implementation: `src/chess_nn_playground/models/attack_hodge_sheaf.py`.
- Idea wrapper: `ideas/i023_attack_hodge_sheaf_tension_network/model.py`.
- Registry key: `attack_hodge_sheaf_tension_network`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0256_tuesday_local_attack_hodge_sheaf.md`.

This is a bespoke PyTorch model, not a `ResearchPacketProbe` wrapper. It consumes board tensors only. It does not consume Stockfish/LC0 outputs, engine search data, verifier metadata, CRTK source tags, fine labels, or unresolved-candidate fields.

The forward return is a dictionary with `logits` shaped `(B,)` for the configured `num_classes: 1` BCE trainer. Additional tensors are diagnostics for node-edge sheaf tension, face-curl Hodge tension, tactical face energies, and edge/face density.

Main implementation classes:

- `EncodingAdapter`
- `AttackComplexBuilder`
- `SquareStem`
- `EdgeInitializer`
- `FaceInitializer`
- `DiagonalLowRankMaps`
- `HodgeTensionBlock`
- `MaskedCochainPool`
- `AttackHodgeSheafNet`

Config mapping:

- `channels` maps to `d_model` when no explicit `d_model` exists.
- `hidden_dim` maps to the MLP width.
- `depth` maps to `n_layers` when no explicit layer count exists.
- `restriction_rank` can be used as an alias for `transport_rank`.
- `max_edges`, `max_faces`, `use_xray_edges`, `use_face_hodge`, and `use_energy_pool` are supported.
