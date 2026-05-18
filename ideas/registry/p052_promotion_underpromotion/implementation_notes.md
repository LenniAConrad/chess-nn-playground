# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/promotion_underpromotion.py`.
- Shared helpers:
    - `ray_geometry.build_ray_step_index`
    - `trunk_features.trunk_joint_features`
- Idea-local wrapper: `ideas/registry/p052_promotion_underpromotion/model.py`.
- Registry key: `promotion_underpromotion`.
- Source primitive: `ideas/research/primitives/external_47_promotion_underpromotion_primitive.md`.

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is consumed.
Side-to-move canonicalisation is computed analytically on the
`simple_18` tensor (vertical flip + 18-channel index permutation).

CRTK metadata, source labels, verification flags, engine scores, and
principal variations are **not** consulted.

## Canonicalisation

The permutation produced by `_build_canonicalize_perm` swaps:

- white piece planes `(P, N, B, R, Q, K)` with black `(p, n, b, r, q, k)`;
- white castling planes `(WK, WQ)` with black `(BK, BQ)`.

The STM plane is force-set to `1.0` in canonical space. The en-passant
plane is kept (vertical flip handles the geometry); it is not
load-bearing for promotion-arrival features.

## Candidate masks

`compute_per_file_candidates` returns:

- `push_mask`, `capL_mask`, `capR_mask` (source-file form).
- `n_own`, `n_opp` global summaries at canonical rows 1, 2, 3.

`_candidate_masks_with_ablation` re-keys the capture masks to
**arrival-file form** so they index the same axis as the
per-arrival-square feature tensors.

## Arrival-square attacker / defender counts

`compute_arrival_attackers_defenders` walks the shared 8-direction
ray geometry from each of the 8 arrival squares (canonical row 0,
files 0..7), gathers the first-blocker piece-type plane along each
ray, and counts sliding attackers whose type matches the ray
(rook / queen on orthogonals, bishop / queen on diagonals). Knight
attackers come from the precomputed knight template (knight attacks
are symmetric so "knights attacking u" = "knights at knight offsets
from u"). King attackers come from the precomputed
king-attack template (non-zero only on the rare case where the enemy
king sits at canonical row 1 in {f-1, f, f+1}). Pawn attackers on a
row-0 arrival square are always zero (enemy pawns move toward
canonical row 7).

The source pawn is **not** removed from the ray when computing
sliding-attacker counts. For push promotion the source pawn at
canonical `(1, f)` is the first blocker on the southward ray from
`(0, f)` (so it would be counted as a defender if it happened to be
a sliding piece; pawns are never sliding pieces, so this case does
not actually mis-count anything). For capture promotion the source
pawn sits diagonally south of the arrival square; the diagonal ray
from `(0, f')` hits it first. The pawn's presence as a blocker
prevents counting any sliding piece that sits behind it, which is a
small approximation toward conservatism (the model sees a lower
attacker count than the truly-post-move board would). This bias is
documented in the math thesis and is left for a future iteration.

## Promoted-piece attack-set features

`compute_promoted_attack_features` reuses the same ray-gather to
compute the (B, 4, 8, 64) per-arrival per-type attack mask:

- queen = sum over all 8 directions of the sliding mask, clamped to
  ``[0, 1]``;
- rook = sum over the 4 orthogonal directions;
- bishop = sum over the 4 diagonal directions;
- knight = precomputed (64,) template per arrival square.

From this:

- `check[t]`: indicator of "promoted piece of type ``t`` attacks the
  enemy king".
- `zone[t]`: count of squares in the enemy king's 3x3 zone the
  promoted piece attacks.
- `hi_value[t]`: weighted sum of enemy high-value targets the
  promoted piece attacks (Q=5, R=3, B/N=2, king=3).
- `kappa_N` = `hi_value[3]`: the knight-fork hint.

## Stop-gradient contract

All feature extraction is implemented in pure-tensor ops so autograd
runs cleanly through the cumulative-blocker scan and the per-token
masking. The trunk diagnostics fed to the gate are detached
(by the gate-MLP design); the joint pool feature used in the delta
head is not detached so the delta can co-train the trunk pool path.

## Output dict contract

The output dict follows the i193 contract, extended with:

- `logits` (rebound to `base_logit + gate * delta`)
- `base_logit`
- `primitive_delta`, `primitive_delta_raw`, `primitive_gate`,
  `primitive_gate_applied`, `primitive_gate_logit`,
  `primitive_gate_entropy`, `primitive_contribution`
- `pugp_push_count`, `pugp_capL_count`, `pugp_capR_count`,
  `pugp_total_count`
- `pugp_n_own_r1`, `pugp_n_opp_r1`
- `pugp_knight_fork_max`, `pugp_queen_check_count`,
  `pugp_queen_zone_max`
- `trunk_<name>` for every diagnostic the i193 trunk produced
- `mechanism_energy` augmented with `total_count.detach()`
- `proposal_profile_strength` = `|delta| * gate_entropy`

## Ablation modes

See `ALLOWED_ABLATIONS`. Primary geometry falsifiers are
`pseudo_only`, `no_capture`, `queen_only`, `no_attack_defense`.
`zero_delta` / `trunk_only` recover the i193 baseline; `disable_gate`
pins the gate at 1.0 for a gate-load-bearing check.

## Numerical notes

- The cumulative-blocker scan uses ``cummax`` on a {0, 1} indicator;
  output remains in {0, 1}.
- The safety score ``s = clip(d - a, -4, 4) / 4`` is bounded to
  [-1, 1].
- The candidate masks are exact 0/1 indicators (multiplicative
  gating). LayerNorm at the head input keeps the feature vector
  well-scaled across positions with very different candidate counts.

## Production upgrade path

The source markdown recommends precomputing PUGP features into
Parquet columns via a new
`scripts/data/precompute_promotion_geometry_features.py` script, with
the dataset reading them through `data.primitive_feature_columns`.
That precompute is left as a follow-up; the current implementation
computes the features inline on the simple_18 tensor inside `forward`,
which is fast enough for the additive-gated head pattern but pays a
per-batch CPU cost on the canonicalisation + ray gather. The
precompute path will move that cost outside training.

The source markdown also recommends a follow-up BT4 mixer study
analogous to `a003_bt4_promotion_aware_head_mixer`. That is deferred
until the additive side-head version (this idea) has a keep / drop
decision.
