# Implementation Notes

- Bespoke implementation: `src/chess_nn_playground/models/king_anchored_material_null_transport.py`.
- Builder: `build_king_anchored_material_null_transport_bottleneck_from_config`.
- Idea-local wrapper: `ideas/i032_king_anchored_material_null_transport_bottleneck/model.py` (calls the builder).
- Registry key: `king_anchored_material_null_transport_bottleneck`.
- Source packet: `ideas/research_packets/chess_nn_research_2026-04-21_0657_tuesday_local_material_ot_bottleneck.md`.
- Input contract: `simple_18` current-board tensor only. CRTK, engine, verification, and source metadata are reporting-only and never consumed as model input. Non-`simple_18` encodings raise `ValueError` at construction time.
- Output contract: `dict` with one puzzle logit (`logits`) and named transport diagnostics (`transport_residual_norm`, `forward_real_cost`, `forward_null_cost`, `reverse_real_cost`, `reverse_null_cost`, `signed_king_zone_residual`, `material_null_cost_gap`).
- Numerical stability: Sinkhorn runs in log-domain with masked source/target marginals; cost values are clamped to `[1e-4, 20]` after softplus to bound `-C/ε`. Padded candidates contribute `-1e9` to the log-kernel.
- Determinism: the null sampler is a pure function of `seed`, side-to-move, source roles, target types, and sample index. Material counts, both king squares, candidate counts, and target-role histogram are preserved by construction; only non-king source/target square coordinates are shuffled.
