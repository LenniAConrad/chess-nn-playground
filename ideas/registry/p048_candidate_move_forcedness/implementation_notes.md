# Implementation Notes

- Central model code: `src/chess_nn_playground/models/primitives/candidate_move_forcedness.py`.
- Shared helper: `trunk_joint_features` from
  `src/chess_nn_playground/models/primitives/trunk_features.py`.
- Move-graph helper: `compute_legal_move_graph` from
  `src/chess_nn_playground/models/primitives/legal_move_graph.py`.
- Idea-local wrapper: `ideas/registry/p048_candidate_move_forcedness/model.py`.
- Registry key: `candidate_move_forcedness`.
- Source primitive:
  `ideas/research/primitives/external_43_candidate_move_forcedness_primitive.md`.

## Inputs

Only the `simple_18` `(B, 18, 8, 8)` current-board tensor is
consumed. The pseudo-legal move adjacency, per-edge `move_type`
codes, `ray_direction` codes, and own / enemy piece masks are
computed inline via the analytic `compute_legal_move_graph` helper.

Per-square tokens are produced by a small 1x1 conv tower local to
the primitive so it can be ported to non-i193 trunks without
dragging in extra encoders.

CRTK metadata, source labels, verification flags, engine scores,
and principal variations are **not** consulted.

## Stop-gradient contract

- The move adjacency, move-type codes, ray-direction codes, and the
  deterministic 14-channel descriptor are computed inside
  `torch.no_grad()` and explicitly `.detach()`ed before downstream
  consumption.
- Trunk diagnostics that feed the gate MLP arrive from the trunk's
  forward; only the joint pool is *not* detached so the delta can
  co-train the i193 pool path (matches p047's convention).
- `mechanism_energy` is exported as
  `trunk_out["mechanism_energy"] + top1_score.detach()`.

## Output dict contract

The output dict follows the i193 contract, extended with:

- `logits` (rebound to `base_logit + gate * delta`)
- `base_logit`
- `primitive_delta` / `primitive_delta_raw`
- `primitive_gate` / `primitive_gate_applied` /
  `primitive_gate_logit` / `primitive_gate_entropy`
- `primitive_contribution`
- `cmf_top1_score`, `cmf_gap12`, `cmf_topk_mass`, `cmf_entropy`,
  `cmf_move_count` -- pool-scalar diagnostics
- `cmf_check_peak`, `cmf_capture_peak`, `cmf_promotion_peak`,
  `cmf_see_peak` -- per-category cat_max channels
- `trunk_<name>` for every diagnostic the i193 trunk produced
- `mechanism_energy` augmented with `top1_score.detach()`
- `proposal_profile_strength` = `|delta| * gate_entropy`
- `proposal_keyword_count` = the pool dimension

## Ablation modes

See `ALLOWED_ABLATIONS`. Primary falsifier is `deterministic_score`
(replace per-move learned score with the feature sum). Anti-pool
falsifier is `mean_pool` (mean over all legal candidates). Feature
falsifiers are `flags_only` (drop value / mobility / SEE channels)
and `no_consequence` (drop check / capture / promotion seeds).
Move-surface falsifier is `dense_edges` (all-pairs adjacency).
`zero_delta` and `trunk_only` recover the i193 baseline.

## Numerical notes

- Inactive edges are masked to `-inf` before top-k so they never
  participate in the pool. `topk_pool` replaces `-inf` with 0 for
  the per-feature pulls and tracks a `keep_mask` so the per-board
  count matches the number of legal candidates.
- The softmax pool uses `score - max(score)` for numerical
  stability, with `nan_to_num` on the final pool vector to guard
  against boards with zero legal moves (e.g. corrupt positions).
- The SEE-lite descriptor uses `max(victim - 0.5 * mover, 0)` so the
  channel is non-negative; this is *not* a true exchange search.
- Mobility shock channels are normalised by 28 (source out-degree)
  and 16 (target in-degree). These bounds are loose upper bounds on
  the maximum per-board edges expected from a single source / to a
  single target on an 8x8 board.

## Production upgrade path

- Post-move board apply. The source primitive proposes a full
  `apply_moves_batched` for every candidate. The current dense
  pilot defers this and relies on the *current-board* descriptors
  plus move-class flags. Adding a true post-move pass would let
  the head consume `threat_creation`, `escape_reduction`, and
  `evasion_scarcity` channels; deferred until the dense version's
  keep-decision is in.
- Candidate-major compaction. Packing only active candidates per
  board (rather than the dense `(B, 64, 64)` mask) would reduce
  memory and bandwidth at large batch sizes. The current
  implementation pays a constant `O(B * 64 * 64 * F)` regardless of
  active count; this is fine for the scout-scale budget.
- Castling / en-passant edges. `compute_legal_move_graph` does not
  yet emit castling or en-passant edges. The primitive degrades
  gracefully: those moves are absent from the pool. A future patch
  would add them as extra edge entries with dedicated `move_type`
  codes.

All three are deferred behind the keep-decision on the dense pilot.
