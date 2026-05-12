# Implementation Notes

- Registered model: `chess_decomposed_attention`.
- Source implementation: `src/chess_nn_playground/models/trunk/chess_decomposed_attention.py`.
- Idea wrapper: `ideas/registry/i242_chess_decomposed_attention/model.py`.
- Config: `ideas/registry/i242_chess_decomposed_attention/config.yaml`.

The implementation reuses the i193 deterministic exchange/king feature builder and adds three transformer streams plus a learned stream router. Inputs remain current-board only; CRTK metadata is reporting-only.
