# Implementation Notes

- Central code: `src/chess_nn_playground/models/trunk/candidate_move_forcedness_sheaf.py`
  (`CandidateMoveForcednessSheafNet`, `CandidateMoveBuilder`,
  `MoveLocalSheafSummary`, `CandidateMoveEncoder`, `TopKMovePool`,
  `build_candidate_move_forcedness_sheaf_from_config`).
- Idea-local wrapper: `ideas/registry/i251_candidate_move_forcedness_sheaf/model.py`
  (`build_model_from_config`).
- Registry key: `candidate_move_forcedness_sheaf`.
- Parent idea: `i018 oriented_tactical_sheaf_laplacian`.

## What changed vs i018

`CandidateMoveForcednessSheafNet` subclasses `OrientedTacticalSheafNet`
and adds four submodules: `CandidateMoveBuilder`,
`MoveLocalSheafSummary`, `CandidateMoveEncoder`, and `TopKMovePool`,
plus the small `delta_head` and `gate_head` MLPs that produce the
additive logit residual. Only the `forward` method is overridden, and
only to:

1. run the i018 trunk forward exactly to produce `base_logits`, the
   per-square states `h`, and the standard i018 diagnostic bundle;
2. enumerate a bounded pseudo-legal candidate set from the canonical
   `piece_state` and `occupancy`;
3. score each candidate with the shared per-move encoder;
4. pool the top-`k` scored moves with a learned-temperature softmax and
   derive 11 forcedness scalars and a continuous top-move-kind summary;
5. compute `final_logits = base_logits + sigmoid(gate(features)) *
   delta(features)`;
6. append 14 candidate-move diagnostics to the standard i018 diagnostic
   dictionary (`candidate_base_logits`, `candidate_delta_logits`,
   `candidate_gate`, `candidate_entropy`, `candidate_top1_mass`,
   `candidate_gap`, `candidate_check_mass`, `candidate_promotion_mass`,
   `candidate_underpromotion_mass`, `candidate_pin_mass`,
   `candidate_capture_mass`, `candidate_king_zone_mass`,
   `candidate_overflow_count`, `candidate_count`).

The adapter, incidence builder, encoder, diffusion block, triad pool,
and readout are inherited unchanged. The move branch reads from the
trunk's relation masks and final per-square states but never modifies
them, so the i018 falsifier (`scramble_relations: true`) remains a
clean test of the typed topology even with i251's new branch.

## Why this is bespoke, not a probe variant

This is a bespoke architecture extension, not a `ResearchPacketProbe`
wrapper. The new candidate builder, sheaf summary, encoder, pool, and
gated delta head are implemented as their own `nn.Module`s in
`candidate_move_forcedness_sheaf.py`. The idea-local `model.py` calls
the new `build_candidate_move_forcedness_sheaf_from_config` builder,
not `build_research_packet_probe_from_config`.

## Identity at zero-init

The final linear layer of both `delta_head` and `gate_head` is
zero-initialized, and `move_encoder.score_head` is zero-initialized so
all valid candidates start with the same score. At init this gives

```text
delta(features) = 0,
gate(features)  = sigmoid(0) = 0.5,
final_logit     = base_logit + 0.5 * 0 = base_logit.
```

Local CPU check on a 4-sample batch: copying i018 weights into the
shared parameters (57 tensors) and leaving the new heads at zero init
gives a max logit difference of `0.0` exactly. With
`disable_move_branch: true` the difference is also `0.0` regardless of
the move-branch parameter values.

## Module shapes and budget

Base scale (`channels=64`, `hidden_dim=96`, `depth=2`, `stalk_dim=8`,
`max_candidates=96`, `top_k=8`):

- i018 parent: ~91k parameters.
- i251 (this idea, base scale): ~116k parameters (measured by counting
  `model.parameters()`). The added cost is dominated by the per-move
  encoder MLP plus the small delta/gate heads. The candidate builder
  and sheaf summary are parameter-free.

## Optional knobs

- `max_candidates` (default 96): upper bound on enumerated candidates;
  set higher if `candidate_overflow_count` shows clipping is hurting.
- `top_k` (default 8): pool budget after scoring; the bottleneck.
- `move_embed_dim` (default 48): per-move embedding width.
- `move_hidden_dim` (default 64): per-move MLP hidden width.
- `delta_hidden_dim` (default 48): delta head hidden width.
- `gate_hidden_dim` (default 24): gate head hidden width.
- `softmax_temperature` (default 1.0): initial pool temperature
  (learned thereafter; clamped to `[1e-2, 1e2]`).
- `flat_move_pool` (default false): force uniform pool weights on
  valid candidates. Falsifier knob.
- `disable_move_branch` (default false): skip the move branch entirely
  and emit zero-valued candidate diagnostics. Falsifier knob.

## Behaviour with the `scramble_relations` falsifier

`scramble_relations: true` (inherited from i018) is preserved. When
enabled, the i018 relation masks are randomly column-permuted per
`(batch, relation)` before being multiplied by the trunk forward. The
candidate builder reads from `piece_state`, `occupancy`, and the
unscrambled `pin_mask`/`our_attack`/`them_attack`; only the i018
diffusion sees the scrambled masks. The `psi_j` move-local sheaf
summary reads from the scrambled masks though, so the move branch
sees what the trunk sees.

## Pseudo-legal builder caveats

The default builder is pseudo-legal and not strictly legal. It does
*not* enumerate castling, en-passant, or filter out moves that leave
the king in check. This matches the i251 packet's recommendation to
keep the default builder leakage-safe and on the same footing as the
move-landscape research; a `legal_light` filter is a documented
follow-up in `ablations.md`.

Promotion is tagged when a pawn move reaches the canonical back rank;
underpromotion is tagged when the same move is also a capture
(`promo_q & is_capture`). The default builder only emits the queen
promotion slot; an explicit knight-promotion expansion is left as a
follow-up rather than wired into the default path because the slot
budget is bounded and the diagnostic `candidate_underpromotion_mass`
already exposes the conditional signal.

## Numerical guard

The audit should re-run any time the move branch is edited:

- shared-weights eval-mode `logits` max abs diff with
  `disable_move_branch: true`: must be `0.0` exactly modulo platform FP
  semantics;
- shared-weights eval-mode `logits` max abs diff at zero-init with the
  default branch: should be `0.0` (or under `1e-5`) on a small batch.

If either guard fails, the change is no longer a strict extension of
i018 and the implementation_kind / status should be re-evaluated.
