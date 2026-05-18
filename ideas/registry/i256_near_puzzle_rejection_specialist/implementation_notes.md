# Implementation Notes

- Central code:
  `src/chess_nn_playground/models/trunk/near_puzzle_rejection_specialist.py`
  (`NearPuzzleRejectionSpecialist`,
  `build_near_puzzle_rejection_specialist_from_config`).
- Idea-local wrapper:
  `ideas/registry/i256_near_puzzle_rejection_specialist/model.py`
  (`build_model_from_config`).
- Registry key: `near_puzzle_rejection_specialist`
  (added in `src/chess_nn_playground/models/_registry_manifest.py`).
- Source research packet:
  `ideas/research/packets/classic/i256_near_puzzle_rejection_specialist.md`.

## What is implemented

- A compact conv encoder (default channels=32, depth=2) over the simple_18
  board concatenated with the deterministic exchange and king feature stacks
  from `DualStreamFeatureBuilder` (reused from i193 with no learned weights).
- A per-square `claim` MLP and `reply_escape` MLP that produce the forcedness
  gap field over the 64 board squares. Aggregation is masked softmax over the
  side-to-move attacker mask.
- A per-square `overload_score_head` aggregated with masked softmax over the
  own-piece mask.
- A `king_escape_head` that pools the trunk features through the enemy-king
  and enemy-king-zone masks and combines them with the deterministic king
  feature mean and the dual-stream board-level summary.
- A `candidate_concentration_head` that consumes only the four scalar gap
  statistics (`max_forcedness_gap`, `top2_forcedness_gap`,
  `forcedness_gap_entropy`, normalised candidate count).
- A `raw_claim_head` and a `veto_head` that combine trunk pools with the
  chess-explained scalar bundle. The final logit is
  `raw_claim_logit - softplus(veto_logit)`.
- Seven config-driven ablations
  (`none`, `no_forcedness_gap`, `no_reply_envelope`, `no_overload_head`,
  `no_king_escape_head`, `no_concentration_head`, `trunk_only`).

## What is intentionally not implemented yet

- The packet's `L_gap_rank` pairwise margin term and `L_veto` BCE auxiliary on
  high-`raw_claim` near-puzzles. Both require trainer support for pair-aware
  batches and a custom auxiliary-loss hook. The architecture already exports
  `raw_claim_logit`, `reply_veto_logit`, and `max_forcedness_gap` per sample,
  so plugging the trainer extension in later is purely additive: change the
  loss callable, not the model.
- The chess-explained near-puzzle curriculum. The shared sampler is used so
  this idea does not perturb other runs in the queue. Switching to the
  packet's slice-weighted curriculum is a trainer-side change.
- Explicit bounded reply-family enumeration (recapture, escape, interposition,
  defend-target, promotion-stop, counter-threat). The current reply head is a
  per-square MLP; the deterministic exchange / king feature stack already
  contains attacker / defender / value / king-zone / check / escape planes
  that approximate those reply families. Promoting them into explicit family
  scores is the next architectural iteration and is out of scope for this
  promotion.
- Validation-only temperature scaling and explicit
  `t_{0.80}` / `t_{0.85}` threshold freezing. These belong in the reporting
  pipeline; the model exports calibrated-style diagnostics
  (`raw_claim_logit`, `reply_veto_logit`) so post-hoc calibration is
  straightforward.
- The `compile_model` / `inference_autocast_dtype` knobs added by i249. The
  forward path is small and static-shape, so `torch.compile` should work, but
  the current implementation deliberately stays on the eager path until
  numerical-equivalence checks have been wired (matching i249's procedure).

## Trunk choice

The research packet recommends the C1 `student_full` parent as the first
deployment target because the BT4-class fast student is 6x faster than i018 on
the CPU harness. The bespoke trunk here is a small self-contained conv stack
rather than a direct call into the `ExchangeThenKingDualStreamNetwork`, so the
i256 model cannot drift when the i193 module is edited. The deterministic
feature builder used by i193 (`DualStreamFeatureBuilder`) *is* reused because
it is purely deterministic geometry and is the cleanest source of the
attacker / defender / king-zone planes the specialist heads consume.

If a future iteration wants the higher-accuracy `P3_i018_full` parent target,
the conv encoder can be swapped for `OrientedTacticalSheafNet` with the same
per-square interface (`trunk_squares: (B, 64, channels)`); the heads do not
need any other change.

## Numerical guard

The architecture only subtracts via `softplus`, which is non-negative, so
`final_logit <= raw_claim_logit` for every batch. Any future edit that breaks
this inequality (for example replacing `softplus` with raw `veto_logit`) is a
contract change and should fail the rejection-identity falsifier in
`math_thesis.md`. A unit test in `tests/test_idea_i256_near_puzzle_rejection_specialist.py`
asserts this invariant.

## Output contract

The forward returns a dict containing at minimum:

- `logits` (B,): the final puzzle logit.
- `raw_claim_logit` (B,) and `reply_veto_logit` (B,): the rejection identity.
- `max_forcedness_gap`, `top2_forcedness_gap`, `forcedness_gap_entropy`,
  `effective_candidate_count`, `selected_candidate_count`: per-sample
  forcedness diagnostics.
- `defender_overload`, `king_escape_pressure`, `claim_mass`,
  `reply_escape_mass`, `own_piece_count`, `mechanism_energy`: per-sample
  pooled diagnostics.

These per-sample scalars are picked up automatically by the shared trainer's
prediction-parquet writer and surface in `predictions_*.parquet` columns.
