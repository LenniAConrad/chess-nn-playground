# Implementation Notes

- Central code: `src/chess_nn_playground/models/hall_defect_obligation_matroid.py`.
- Registry key: `hall_defect_obligation_matroid_network`.
- Idea wrapper: `ideas/i053_hall_defect_obligation_matroid_network/model.py`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0813_tuesday_los_angeles_hall_defect.md`.

The implementation is board-only. It consumes the current `simple_18` tensor and never consumes engine evaluations, principal variations, source labels, CRTK metadata, legal move trees, or verification traces.

## Decoder And Contacts

`SafeBoardDecoder` supports `simple_18` only and raises on unknown deterministic rule encodings. It extracts current piece planes and side-to-move, then builds pseudo-legal controls with color-aware pawn attacks, leaper offsets, and blocker-aware sliding rays. The learned board adapter still sees all configured input channels.

## Obligation Builder

The obligation branch builds the packet's two side-relative roles and six strata. Defenders are ranked by obligation degree, piece value, center proximity, and stable square order when more than `D_max` candidates are available. The discarded count is emitted as a nuisance diagnostic so truncation is visible.

## Exact Zeta Layer

`HallZetaDefectLayer` computes exact subset-zeta Hall profiles over the selected defender universe. With the default `D_max=10`, each role/stratum graph uses `1024` subset bins. The implementation keeps this rule feature frozen with respect to gradients; trainable parameters live in the token encoder, board adapter, and fusion head.

## Output Contract

The packet described two-class logits, but this repo's i053 config uses the shared puzzle-binary BCE contract with `num_classes: 1`. The model therefore exposes internal `two_class_logits` and returns the puzzle margin as `output["logits"]` with shape `(B,)`.

## Ablations

Implemented ablation modes are `degree_rewire`, `count_only`, `weight_shuffle`, and `complete_neighborhood`. They are deterministic for reproducible smoke tests and run artifacts.
