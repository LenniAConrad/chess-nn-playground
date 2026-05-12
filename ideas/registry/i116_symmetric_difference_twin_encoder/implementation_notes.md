# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/symmetric_difference_twin_encoder.py`.
- Idea-local wrapper: `ideas/registry/i116_symmetric_difference_twin_encoder/model.py` (`build_model_from_config`).
- Registry key: `symmetric_difference_twin_encoder`.
- Source packet: `ideas/research/packets/classic/chess_nn_research_2026-04-24_2121_friday_shanghai_architecture_batch_4.md`.
- Batch candidate: `Symmetric Difference Twin Encoder`.
- This is intentionally board-only and does not consume engine, verification, source, or CRTK metadata as input.
- The safe transform `T` is file mirror with simple_18 kingside/queenside castling-channel swap (`13 <-> 14` and `15 <-> 16`); this keeps `T(x)` rule-faithful.
- The shared trunk is a single `_SharedBoardTrunk` module applied via `cat([x, T(x)], dim=0)` so both branches see the same weights and the same BatchNorm statistics. Untying the trunk would change the model.
- The transformed latent is aligned back to the original frame with `flip(z', dim=-1)` before comparison; element-wise comparison is otherwise meaningless.
- `preserved = (z + z_aligned) / 2` and `changed = |z - z_aligned|` are the load-bearing features the head reads; replacing them with raw `z` would collapse the model to a one-shot CNN.
