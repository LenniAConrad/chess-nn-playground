# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/sparse_expert_board_router.py` (`SparseExpertBoardRouter`).
- Idea-local wrapper: `ideas/all_ideas/registry/i123_sparse_expert_board_router/model.py` calls `build_sparse_expert_board_router_from_config`.
- Registry key: `sparse_expert_board_router` (registered in `chess_nn_playground.models.registry.MODEL_BUILDERS` and removed from `RESEARCH_PACKET_MODEL_NAMES`).
- Source packet: `ideas/all_ideas/research/packets/classic/chess_nn_research_2026-04-24_2124_friday_shanghai_architecture_batch_5.md`.
- Batch candidate: `Sparse Expert Board Router`.
- Input contract: `simple_18` board tensor only; CRTK/source metadata is reporting-only and never consumed by the model.
- Six experts (`local_cnn`, `dilated_cnn`, `token_mixer`, `rank_file_mixer`, `morphology_lite`, `compact_mlp_mixer`) produce hidden vectors of width `hidden_dim`. The router selects the top-`k` via masked softmax; selection counts and load-balance terms are returned as diagnostics for the trainer.
- The forward pass returns a dict with `logits` of shape `(B,)` plus routing diagnostics. The trainer can optionally use `auxiliary_loss` (balance + Switch-style + entropy) to keep routing healthy.
