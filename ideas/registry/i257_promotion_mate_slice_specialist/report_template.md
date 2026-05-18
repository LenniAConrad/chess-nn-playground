# Report Template

## Run

- Result path:
- Config: `ideas/registry/i257_promotion_mate_slice_specialist/config.yaml`
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`
- Paired i193 baseline path (same split, seeds, scale):
- Paired i249 baseline path (same split, seeds, scale), if available:

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration:

## Bounded-delta Identity Check

The model defines
`final_logit = base_logit + sum_k gate_k * delta_k`
with `delta_k = Delta_k * tanh(...)` and `gate_k = sigmoid(...) *
structural_mask_k`. Then `|final - base| <= sum_k Delta_k`.

- Empirical `max(|final_logit - base_logit|)` over the test set
  (must be `<=` `3 * delta_bound + joint_delta_bound`,
  default `5.25`):
- Empirical `mean(promotion_gate)`, `mean(underpromotion_gate)`,
  `mean(mate_gate)`, `mean(joint_gate)`:
- Empirical `mean(|promotion_delta|)`, `mean(|underpromotion_delta|)`,
  `mean(|mate_delta|)`, `mean(|joint_delta|)`:

The headline diagnostic is that gates should be *higher* on slice positives
than on the rest of the dataset for the respective branch.

## Operating-point Table (recall 0.80 and 0.85, validation thresholds)

| Metric | recall 0.80 | recall 0.85 |
|---|---:|---:|
| Puzzle recall | | |
| Precision | | |
| Total FP | | |
| Near-puzzle FP | | |
| Near-puzzle FP rate | | |
| Far / random FP rate | | |
| Mean `base_logit` on accepted positives | | |
| Mean `promotion_gate * promotion_delta` on accepted positives | | |
| Mean `mate_gate * mate_delta` on accepted positives | | |

The matched-recall report should compare each row to the paired i193 parent
baseline on the same split, seeds, and scale.

## Required Slice Report

For each slice -- `crtk_eval_bucket = equal`, `crtk_difficulty = hard`,
`crtk_difficulty = very_hard`, `crtk_tactic_motifs = promotion`,
`crtk_tactic_motifs = underpromotion`, `crtk_tactic_motifs = mate_in_1`,
and each `crtk_phase` bucket -- at both recall `0.80` and `0.85`:

| Column | Meaning |
|---|---|
| `n` | Slice size |
| `puzzle_recall` | Recall preservation on the slice |
| `near_FP_rate` | Core rejection metric |
| `far_FP_rate` | Whether the model is becoming broadly conservative |
| `precision` | Practical acceptance quality |
| `accuracy@recall` | Continuity with the repo's audit style |
| `mean_promotion_gate` | Promotion branch fire rate on the slice |
| `mean_underpromotion_margin` | Underpromotion branch evidence |
| `mean_mate_gate` | Mate branch fire rate on the slice |
| `mean_mating_special_count` | Joint overlap evidence |

These let the report show *why* a slice win happened, not just that it
happened.

## Specialist Diagnostics

- Mechanism family: `promotion_mate` (promotion fanout + king-zone forcing)
- Packet profile: `promotion_mate_slice_specialist`
- Mean `base_logit`:
- Mean `promotion_delta`, `underpromotion_delta`, `mate_delta`,
  `joint_delta`:
- Mean `promotion_gate`, `underpromotion_gate`, `mate_gate`, `joint_gate`:
- Mean `promotion_candidate_count`:
- Mean `promotion_best_type`:
- Mean `promotion_type_entropy`:
- Mean `underpromotion_margin`:
- Mean `mate_witness_count`:
- Mean `escape_square_count`:
- Mean `checking_move_count`:
- Mean `king_pressure`:
- Mean `mating_special_count`:
- Mean `mechanism_energy`:

## Ablation Sweep

Run each `model.ablation` setting on the same seeds / scale / split:

| Ablation | PR-AUC | promo PR-AUC | under PR-AUC | mate PR-AUC | near_FP @ 0.80 | Notes |
|---|---:|---:|---:|---:|---:|---|
| `none` | | | | | | full specialist |
| `trunk_only` | | | | | | parent floor |
| `copy_baseline_fanout` | | | | | | falsifies promotion fanout |
| `uniform_type_attention` | | | | | | falsifies selective type weighting |
| `zero_under_margin` | | | | | | falsifies underpromotion margin |
| `no_mate_witness` | | | | | | falsifies mate-witness scalars |
| `no_joint_branch` | | | | | | falsifies joint overlap |
| `disable_gate` | | | | | | tests gate as rejection control |
| `force_zero_gate` | | | | | | should equal `trunk_only` closely |

## Keep / Drop Decision

- [ ] Bounded-delta guard `|final_logit - base_logit| <= sum_k Delta_k` holds on the test set.
- [ ] `none` beats `trunk_only` on at least one of `promotion`,
      `underpromotion`, or `mate_in_1` slice PR-AUC.
- [ ] At least one chess-semantic ablation loses most of the slice gain
      (the responsible branch is load-bearing).
- [ ] Aggregate PR-AUC remains within ~0.005 of the matched i193 parent
      baseline.
- [ ] Matched-recall near-puzzle FP rate at recall 0.80 / 0.85 is no
      worse than the parent baseline.

If any box fails: drop the responsible branch, or keep i193 / i249 as the
canonical parent and do not promote i257 over it.

## Conclusions

- What the specialist appears able to learn (vs i193):
- What the specialist appears unable to learn (vs i193):
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / wire up the deferred
  loss-side ablations / scale to i249 trunk / distill into student).
