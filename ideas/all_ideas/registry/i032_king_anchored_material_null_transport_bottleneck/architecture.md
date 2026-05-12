# Architecture

`King-Anchored Material-Null Transport Bottleneck` (KAMN-OTB) implements the markdown thesis directly: it estimates a per-direction entropic optimal-transport plan from current source pieces to opponent piece and king-zone targets, subtracts the descriptor expected under a deterministic material-preserving null geometry, and classifies puzzle-likeness from the residual descriptors. The architecture follows the math thesis end-to-end:

- **Simple-18 piece adapter (`Simple18PieceAdapter`).** Splits the 18-channel `simple_18` board tensor into white/black piece occupancy `[B, 2, 6, 8, 8]` and a side-to-move flag. Non-`simple_18` encodings fail closed because their channel semantics are not deterministically known.
- **Piece-target candidate builder (`PieceTargetCandidateBuilder`).** For each direction `a -> b in {stm->opp, opp->stm}` it lists up to 16 source piece slots from side `a` and up to 16 opponent piece targets, then concatenates the 9 king-zone pseudo-targets centered on the opponent king (clipped to the board). It produces `(source_roles, source_squares, source_mask)` and `(target_types, target_squares, target_mask)` tensors with permutation-invariant masks.
- **King-anchored material-null sampler (`KingAnchoredMaterialNullSampler`).** Computes deterministic seeded shuffles of non-king source/target squares while keeping the king-zone target indices and own-king source slot fixed. The shuffle is a pure function of `seed`, side-to-move, source roles, target types, and sample index, so material counts, side-to-move, both king squares, candidate counts, and target-role histogram are preserved.
- **Chess-geometry cost (`ChessGeometryCost`).** Builds a per-head softplus cost over twelve rule-independent geometry channels (Manhattan, Chebyshev, file/rank/diagonal alignment, queen-line, knight-graph distance, side-relative pawn-forward delta, role-aware piece distance, king-zone indicator, opponent high-value indicator) with role/type biases and a head bias. Costs are clamped to `[1e-4, 20]` for Sinkhorn stability.
- **Masked log-domain Sinkhorn (`MaskedLogSinkhorn`).** Solves the entropic OT subproblem `Pi_h^epsilon` with stabilized log-sum-exp updates, masked source/target marginals, default `iterations=12`, default `epsilon=0.08`. Padded candidates are kept inert via large negative log-kernel masks.
- **Transport descriptor pool (`TransportDescriptorPool`).** Pools each plan/cost pair into 15 descriptors: expected cost, normalized plan entropy, max pair mass, top-4 pair mass, king-zone target mass, value-bucket masses (pawn/minor/rook/queen/king), distance-bucket masses for ≤1, =2, =3, ≥4, and forward (toward opponent) mass. Descriptors are permutation-invariant by construction.
- **Residual assembly.** For each direction we compute `T_real - mean_K(T_null)` over `null_samples=4` material-null draws using the same cost head, then build the signed contrast `Z = [resid_fwd, resid_rev, resid_fwd - resid_rev]` of shape `[B, 3*H*Dd]`. With `H=4`, `Dd=15` the bottleneck is 180-d.
- **Classifier head.** `LayerNorm(180) -> Linear(180, hidden_dim=128) -> GELU -> Dropout -> Linear(128, 64) -> GELU -> Linear(64, num_classes)`. The forward returns one puzzle logit (squeezed when `num_classes == 1`) plus named diagnostics: `transport_residual_norm`, forward/reverse real and null cost summaries, `signed_king_zone_residual`, and `material_null_cost_gap`.

The model is board-only: CRTK, engine, verification, and source metadata are reporting-only and never consumed as input. The deterministic null sampler is used solely for descriptor centering; its shuffled boards are never used as labeled training examples.

## Implementation Binding

- Registered model name: `king_anchored_material_null_transport_bottleneck`
- Source implementation file: `src/chess_nn_playground/models/king_anchored_material_null_transport.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i032_king_anchored_material_null_transport_bottleneck/model.py`

The wrapper calls `build_king_anchored_material_null_transport_bottleneck_from_config` with the idea's `model:` config block. The registry key `king_anchored_material_null_transport_bottleneck` resolves to the same builder, so `build_model(name, model_cfg)` and the idea wrapper produce equivalent modules.
