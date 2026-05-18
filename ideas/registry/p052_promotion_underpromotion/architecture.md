# Architecture

`Promotion and Underpromotion Geometry Primitive` (p052, PUGP) is an
additive, gated head on top of the i193
`ExchangeThenKingDualStreamNetwork` trunk. The model consumes the
`simple_18` `(B, 18, 8, 8)` board tensor and returns one puzzle logit
plus a per-sample diagnostics dict.

PUGP is the geometry-first promotion primitive complementary to i246
(PFCT / `promotion_aware_head`). It does **not** re-run the trunk on
substituted boards; instead it computes a fixed board-only PUGP
feature vector through deterministic rule-derived tensor ops and
projects that vector to a scalar logit delta via a small MLP.

## Forward pass

1. **i193 trunk forward**. Emits ``base_logit`` and trunk diagnostics
   plus the joint pool feature (via `trunk_joint_features`).
2. **Side-to-move canonicalisation**. The board is vertically flipped
   and the white / black piece planes and castling-plane pairs are
   swapped when the original side to move is black, so that own pawns
   always move toward canonical row 0 and the promotion rank is
   canonical row 0. The STM plane is forced to 1.0 in canonical
   space.
3. **Per-file candidate masks**.
   - ``push_mask[f] = 1`` if own pawn lives at canonical (row 1,
     file f) and the arrival square (row 0, file f) is unoccupied.
   - ``capL_mask[f] = 1`` (source-file form) if own pawn lives at (1, f)
     and there is an enemy piece at (0, f-1). Re-keyed in the
     model to arrival-file form for pooling.
   - ``capR_mask[f]`` is the mirror.
4. **Global pawn-distance summary**. Counts of own and opp pawns at
   canonical rows 1, 2, 3.
5. **Per-arrival-square per-type attack / defense features**.
   - The sliding attack mask of a queen-like piece on (row 0, file g)
     is computed via the shared ``ray_geometry`` ray-step tables and
     a cumulative-blocker scan; the rook / bishop subsets re-use the
     same gather and select the orthogonal / diagonal directions.
   - The knight attack template is precomputed (no occlusion).
   - The king-zone overlap template is precomputed.
   - The first-blocker piece-type indicator (via the same ray gather)
     is used to count enemy sliding **attackers** of the arrival
     square; the same construction with own pieces gives
     **defenders**. Knight attackers / defenders come from the knight
     template.
   - Per-arrival-square per-type metrics:
     ``cQ, cR, cB, cN`` (gives-check), ``zQ, zR, zB, zN``
     (king-zone overlap), and a piece-agnostic ``safety = clip(d - a,
     -4, 4) / 4``.
   - ``hi_value[t]`` is the weighted sum of enemy high-value targets
     (Q=5, R=3, B/N=2, king=3) the promoted piece of type ``t``
     attacks; ``kappa_N`` is taken from ``hi_value[N]``.
6. **Per-candidate-kind tokens**. For each candidate kind ``c in
   {push, capL, capR}`` the model assembles a per-arrival-file
   ``(B, 8, 16)`` token tensor:
   ``[mask, capture_flag, edge_file, cQ, zQ, sQ, cR-cQ, zR-zQ,
   sR-sQ, cB-cQ, zB-zQ, sB-sQ, cN-cQ, zN-zQ, sN-sQ, kappa_N]``,
   masked elementwise by the per-file candidate mask.
7. **Pooling**. Per-candidate-kind tokens are sum-and-max pooled over
   the 8 arrival files; the resulting ``2 * 16 = 32``-dim vectors are
   concatenated with the global pawn summary (6) and the candidate
   counts (4) into a single ``(6 + 3 * 32 + 4) = 106``-dim feature
   vector.
8. **Delta MLP**. ``LayerNorm`` over the feature vector, concatenate
   with the trunk joint pool, and project through a 2-layer MLP to
   ``primitive_delta_raw``.
9. **Gate**. MLP over ``cat(joint, total_count, n_own_r1, n_opp_r1,
   has_capture)`` to sigmoid ``primitive_gate``; initial bias
   ``gate_init = -2.0``.
10. **Output**. ``final_logit = base_logit + primitive_gate *
    primitive_delta_raw``.

The primitive is a strict no-op on positions with no own near-promotion
pawn: every per-candidate mask is zero, the candidate counts and
candidate-token contributions vanish, and the trained gate's
negative-bias initialisation keeps the gate near zero.

## Ablation modes

| `model.ablation` | What it tests |
|---|---|
| `none` | Full PUGP feature vector (default). |
| `pseudo_only` | **Falsifier 1**. Drop legality filtering: candidates fire on all own-near-promotion pawns regardless of arrival-square occupancy. If matches `none`, exact promotion geometry is not load-bearing. |
| `no_capture` | **Falsifier 2**. Drop diagonal capture promotion candidates. If matches `none`, capture-promotion geometry is not load-bearing. |
| `queen_only` | **Falsifier 3**. Zero out rook / bishop / knight deltas and the knight-fork hint. If matches `none`, the underpromotion-hint story is false. |
| `no_attack_defense` | **Falsifier 4**. Zero out arrival-square attack/defense and gives-check / king-zone features. If matches `none`, arrival-square safety is not load-bearing. |
| `zero_delta` | Zero primitive delta. Recovers i193 baseline. |
| `trunk_only` | Same as `zero_delta`; semantic alias. |
| `disable_gate` | Pin gate at 1.0. Tests gate load-bearing. |

## Inputs not used

CRTK metadata, source labels, verification flags, engine evaluations,
and principal variations are **not** consumed.

## Cost

| Stage | Cost |
|---|---|
| i193 trunk | One forward pass through the dual-stream encoder. |
| Trunk joint refeat | One additional encoder pass (the `trunk_joint_features` helper). |
| Canonicalise | One vertical flip + 18-channel index_select. |
| Ray gather + cumulative blocker | One (B, 8, 8, 7) gather over ``occupancy``. |
| First-blocker piece extraction | One (B, 8, 8, 7, 12) gather over piece planes. |
| Knight / king-zone templates | (B, 8, 64) lookups (constant tables). |
| Per-candidate token assembly | Three ``(B, 8, 16)`` blocks, masked, sum-and-max pooled. |
| Delta / gate MLPs | Small two-layer MLPs (~30k params). |

No per-promotion-piece trunk re-run is performed, so the wall-clock
overhead is small (low single-digit percent over i193 at B=256).

## Implementation Binding

- Registered model name: `promotion_underpromotion`.
- Source implementation: `src/chess_nn_playground/models/primitives/promotion_underpromotion.py`.
- Shared helpers: `ray_geometry.build_ray_step_index`, `trunk_features.trunk_joint_features`.
- Trunk source: `src/chess_nn_playground/models/trunk/exchange_then_king_dual_stream.py`.
- Idea-local wrapper: `ideas/registry/p052_promotion_underpromotion/model.py`.
- Training config: `ideas/registry/p052_promotion_underpromotion/config.yaml`.
- Builder entry in `src/chess_nn_playground/models/_registry_manifest.py`:
  ``MODEL_SPECS['promotion_underpromotion']``.
