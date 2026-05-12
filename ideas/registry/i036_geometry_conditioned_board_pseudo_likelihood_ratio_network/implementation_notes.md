# Implementation Notes

- Central code: `src/chess_nn_playground/models/geometry_pseudolikelihood_ratio.py`.
- Registry key: `geometry_conditioned_board_pseudo_likelihood_ratio_network`.
- Idea wrapper: `ideas/registry/i036_geometry_conditioned_board_pseudo_likelihood_ratio_network/model.py`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-21_0713_tuesday_local_geom_plr.md`.

The model is intentionally board-only. It consumes only the `simple_18` tensor described by the project board-feature encoder and never consumes engine scores, legal moves, verification data, CRTK/source metadata, or fine-label provenance fields.

The adapter fails closed unless `input_channels == 18` and `adapter == simple18_token`. It maps the 12 piece planes to the packet's 13 square tokens and raises on invalid multi-piece occupancy unless `allow_soft_tokenization` is explicitly enabled.

The static relation index is deterministic and includes rank/file/diagonal rays, knight offsets, king-neighborhood offsets, and both pawn-direction offset families. `randomize_relations: true` keeps the same relation and distance slots but replaces neighbor square identities with a deterministic same-degree randomized map for the packet's geometry falsifier. `unary_only: true` removes neighbor context while keeping coordinate, metadata, class, and token reconstruction scoring intact.

Because the shared puzzle-binary trainer uses one BCE logit for fine-label `2` versus labels `0` and `1`, the implementation returns the class-conditional pseudo-likelihood ratio `z_1 - z_0` by default. Building with `num_classes: 2` exposes the raw two-class packet logits.
