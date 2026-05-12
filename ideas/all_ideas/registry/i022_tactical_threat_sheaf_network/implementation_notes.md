# Implementation Notes

- Source implementation: `src/chess_nn_playground/models/tactical_threat_sheaf.py`.
- Idea wrapper: `ideas/all_ideas/registry/i022_tactical_threat_sheaf_network/model.py`.
- Registry key: `tactical_threat_sheaf_network`.
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-21_0255_tuesday_local_threat_sheaf.md`.

The implementation is a bespoke PyTorch model, not a `ResearchPacketProbe` wrapper. It consumes board tensors only and never uses engine scores, search output, verifier metadata, source tags, fine labels, or unresolved-candidate fields as model inputs.

The model returns a dictionary with `logits` shaped `(B,)` for `num_classes: 1`, matching the repo's `bce_with_logits` puzzle-binary trainer. Additional tensors are diagnostics, including sheaf tension, attack energy, defense energy, pin energy, contest pressure, overload pressure, gate mean, and edge density.

Config mapping:

- `hidden_dim` or `channels` maps to `d_model` when `d_model` is absent.
- `depth` maps to `num_sheaf_layers` when the explicit field is absent.
- `restriction_form`, `restriction_rank`, `use_edge_gates`, `use_contest_pool`, `use_square_embeddings`, `share_sheaf_layers`, and `max_edges` are supported.
