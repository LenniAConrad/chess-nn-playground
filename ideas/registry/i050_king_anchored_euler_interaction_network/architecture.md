# Architecture

`King-Anchored Euler Interaction Network` is a bespoke deterministic-topology
classifier. It computes king/center-anchored cubical half-plane Euler curves and
Euler additivity interaction curves over side-relative piece-role bitboards
extracted from `simple_18`, then feeds the flattened curves plus low-order
count/rule context through a small MLP head that emits a single puzzle logit.

## Forward pipeline

1. `Simple18RoleAdapter` turns the 18 input planes into 8 binary role masks
   `(B, R=8, 8, 8)` using side-to-move semantics:
   - `own_pawn`, `own_minor` (knight + bishop), `own_heavy` (rook + queen), `own_king`
   - `opp_pawn`, `opp_minor`, `opp_heavy`, `opp_king`
   It also derives a `(B, 15)` count/context vector (per-role counts, white/black
   material totals, a material differential, castling indicator bits, an
   en-passant indicator, and the side-to-move bit).
2. King anchors are extracted as the centroid of each king mask (mass-weighted,
   falling back to board centre `(3.5, 3.5)` only on malformed inputs); the
   third anchor is always the board centre. Output: `(B, A=3, 2)`.
3. `CubicalEulerCurveLayer` builds a `(B, A, U=8, T, 8, 8)` half-plane gate
   tensor from the eight king-style directions
   `[(±1, 0), (0, ±1), (±1, ±1)]` and `T=15` thresholds spanning `[-7, 7]`.
4. `EulerInteractionFeatureBuilder` sweeps each role mask through the gates,
   computes `chi = V - E + F` of the cubical closure (vertices/edges of the
   8x8 grid that touch any selected face), and stores the resulting curves
   `E_{r, a, u}(tau)`. For every default role pair it also computes
   `J_{r, s, a, u}(tau) = chi(K_r ∪ K_s) - chi(K_r) - chi(K_s)` from the
   union mask and the cached individual curves.
5. The flattened curves and their threshold first differences are concatenated
   with the `(B, 15)` context vector (when `include_count_summaries=True`) and
   passed through `EulerFeatureMLP` → two-class logits. For the
   `puzzle_binary` head, the model returns the scalar logit
   `two_class_logits[:, 1] - two_class_logits[:, 0]`.

The deterministic Euler feature extractor has no trainable parameters; only
the LayerNorm/MLP head learns. Diagnostic outputs include `role_curve_energy`,
`interaction_curve_energy`, anchor-resolved interaction pressure
(`opp_king_interaction_pressure`, `own_king_interaction_pressure`,
`center_interaction_pressure`), and `own_role_count` / `opp_role_count`.

## Default tensor shapes

- Input: `(B, 18, 8, 8)` — `simple_18`.
- Roles: `(B, 8, 8, 8)`.
- Anchors: `(B, 3, 2)`.
- Sweep gates: `(B, 3, 8, 15, 8, 8)`, processed per role/pair.
- Individual Euler curves: `(B, R=8, A=3, U=8, T=15)`.
- Interaction curves: `(B, P=8, A=3, U=8, T=15)`.
- Flat features: `(B, (R + P) * A * U * (T + (T - 1)))` = `(B, 16 * 3 * 8 * 29)`.
- Logit output: `(B,)` for `num_classes=1`, `(B, 2)` for `num_classes=2`.

## Implementation Binding

- Registered model name: `king_anchored_euler_interaction_network`
- Source implementation file: `src/chess_nn_playground/models/trunk/king_anchored_euler_interaction_network.py`
- Idea-local wrapper: `ideas/registry/i050_king_anchored_euler_interaction_network/model.py`
  (calls `build_king_anchored_euler_interaction_network_from_config`).
- Encoding adapter: `simple_18` only. The adapter fails closed if the channel
  count or encoding name does not match.
- Deterministic Euler/anchor logic and direction/threshold banks are fixed by
  configuration; only the MLP head has trainable parameters.
