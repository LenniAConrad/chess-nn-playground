# Implementation Notes

- Central code:
  `src/chess_nn_playground/models/trunk/promotion_mate_slice_specialist.py`
  (`PromotionMateSliceSpecialist`,
  `build_promotion_mate_slice_specialist_from_config`).
- Idea-local wrapper:
  `ideas/registry/i257_promotion_mate_slice_specialist/model.py`
  (`build_model_from_config`).
- Registry key: `promotion_mate_slice_specialist`
  (added in `src/chess_nn_playground/models/_registry_manifest.py`).
- Source research packet:
  `ideas/research/packets/classic/i257_promotion_mate_slice_specialist.md`.

## What is implemented

- A compact conv encoder (default channels=32, depth=2) over the simple_18
  board concatenated with the deterministic exchange and king feature stacks
  from `DualStreamFeatureBuilder` (reused from i193 with no learned weights).
- A base trunk head that emits `base_logit` from the trunk mean/max pool plus
  the dual-stream board-level summary.
- A deterministic candidate gather over near-promotion own pawns: top-K
  positive entries from the union of the one-push and two-push slabs
  selected by the side-to-move plane. Default `K = 4`.
- A promotion fanout branch with a type embedding for `{Q, R, B, N}` and
  per-type analytic attack-delta masks, per-(candidate, type) score, softmax
  type attention, mask-weighted aggregation, and a bounded tanh delta with
  a sigmoid gate masked off when no candidate exists.
- An underpromotion divergence branch built on the same per-type scores,
  emitting `agg_margin = mean(max(R, B, N) - Q)` and a bounded gated delta.
- A king-zone forcing-witness branch that pools the trunk activations
  through the enemy-king and enemy-king-zone masks and combines them with
  six deterministic king scalars (check count, escape count, in-zone attack
  count, ray-to-zone count, capture-in-zone value, zone-balance ratio).
- A tiny promotion-mate joint overlap branch over the three summary
  vectors, bounded by `joint_delta_bound = 0.75` and structurally masked
  off unless both upstream branches are active.
- Nine config-driven ablations: `none`, `trunk_only`, `copy_baseline_fanout`,
  `uniform_type_attention`, `zero_under_margin`, `no_mate_witness`,
  `no_joint_branch`, `disable_gate`, `force_zero_gate`.

## What is intentionally not implemented yet

- The packet's `oriented_tactical_sheaf_fast` (i249) trunk substitution. The
  packet recommends keeping the i018/i249 sheaf trunk intact and adding the
  specialist branches above it. That requires a `forward_features` refactor
  of the i249 module which is out of scope for this promotion. The current
  conv trunk uses the same per-square interface, so swapping in i249 only
  needs to expose `trunk_squares: (B, 64, channels)`; the heads do not
  need any other change.
- The packet's extended loss
  `L = L_BCE + lambda_gate * sum_k E[gate_k] + lambda_kd * KL(...) +
  lambda_near * L_near + lambda_slice * L_slice`. Both gate-sparsity and
  slice-restricted ranking terms require trainer support for pair-aware
  batches and a custom auxiliary-loss hook. The architecture already
  exports `promotion_gate`, `underpromotion_gate`, `mate_gate`,
  `joint_gate`, and per-sample deltas, so plugging the trainer extension in
  later is purely additive: change the loss callable, not the model.
- The chess-explained slice-weighted curriculum. The shared sampler is used
  so this idea does not perturb other runs in the queue. Switching to the
  packet's slice-weighted curriculum is a trainer-side change.
- Exact one-ply legal-move enumeration for the mate branch. The current
  implementation uses six deterministic witness scalars derived from the
  dual-stream feature builder; the packet's TSDP-style precomputed exact
  rule cache is a planned follow-up tied to the i248 TSDP cache pipeline
  rather than to this architecture promotion.
- Hard-concrete gates (Louizos et al.). The current implementation uses a
  plain sigmoid gate with a structural mask. Swapping to a stochastic
  hard-concrete gate with an `L0` expectation penalty is a one-class swap
  inside each branch and is documented as a planned follow-up.
- Validation-only temperature scaling and explicit `t_{0.80} / t_{0.85}`
  threshold freezing. These belong in the reporting pipeline; the model
  exports calibrated-style diagnostics so post-hoc calibration is
  straightforward.

## Trunk choice

The research packet recommends the i249 sheaf trunk as the safest place to
add specialists because i249 documents the same numerics as i018 with a
faster execution path. The bespoke trunk here is a small self-contained
conv stack rather than a direct call into i249. The reasons are:

- The packet itself flags the first deployment target as the smaller `C1`
  variant where simple_18 + the dual-stream feature builder is enough to
  expose the specialist surface without a `forward_features` refactor.
- This idea must plug into the shared trainer without coupling to a
  potentially-changing i249 internal API surface; a self-contained conv
  trunk avoids the drift risk.
- Promoting the heads onto the i249 trunk is one localised refactor that
  can land in a follow-up without touching the head interface.

If a future iteration wants the higher-accuracy `i249` parent target, the
conv encoder can be swapped for `OrientedTacticalSheafFast` (after exposing
`forward_features`) with the same per-square interface; the heads do not
need any other change.

## Numerical guards

- Each branch delta is bounded by `Delta_k * tanh(...)`. By construction
  the contribution `gate_k * delta_k` lies in `[-Delta_k, Delta_k]`. With
  the default `delta_bound = 1.5` for the three primary branches and
  `joint_delta_bound = 0.75` for the joint branch, the total deviation
  from `base_logit` is bounded by `3 * 1.5 + 0.75 = 5.25` in the worst
  case; in practice gates rarely saturate.
- Structural masks zero a branch when its prerequisites do not hold (no
  candidate pawns for promotion / underpromotion; no checking moves or
  in-zone attack for mate; both required for joint). This protects against
  branches firing on positions where they cannot meaningfully contribute.
- The promotion gather is deterministic and tensor-only: `topk` on the
  candidate mask, then `gather` for per-square / per-square-statistic
  lookups. There is no python-chess fallback and no engine search.
- All masked softmaxes fall back to a uniform distribution when a mask
  sums to zero so entropy / expectations stay finite (matches the
  numerical contract used by i256).

## Output contract

The forward returns a dict containing at minimum:

- `logits` (B,): the final puzzle logit (`base + sum_k gate_k * delta_k`).
- `base_logit` (B,): the trunk-only logit (gate-free baseline).
- `promotion_delta`, `underpromotion_delta`, `mate_delta`, `joint_delta`
  (B,): the per-branch bounded deltas.
- `promotion_gate`, `underpromotion_gate`, `mate_gate`, `joint_gate` (B,):
  the per-branch gates (sigmoid * structural mask).
- `promotion_candidate_count`, `promotion_best_type`,
  `promotion_type_entropy`, `underpromotion_margin`: promotion-side
  diagnostics.
- `mate_witness_count`, `escape_square_count`, `checking_move_count`,
  `king_pressure`, `mating_special_count`: mate-side diagnostics.
- `mechanism_energy`: trunk activation energy proxy.

These per-sample scalars are picked up automatically by the shared
trainer's prediction-parquet writer and surface in `predictions_*.parquet`
columns. The matched-recall slice report can attribute slice wins to the
responsible branch by inspecting per-sample gate and delta columns.
