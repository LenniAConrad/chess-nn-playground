# Report Template

## Run

- Result path:
- Config: `ideas/registry/i256_near_puzzle_rejection_specialist/config.yaml`
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`
- Paired i193 baseline path (same split, seeds, scale):

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration:

## Rejection Identity Check

The model defines `final_logit = raw_claim_logit - softplus(reply_veto_logit)`.

- Empirical `max(final_logit - raw_claim_logit)` over the test set (must be `<= 0`):
- Empirical `mean(softplus(reply_veto_logit))` (overall):
- Empirical `mean(softplus(reply_veto_logit))` (verified near-puzzle slice):

The third row is the headline diagnostic: the veto should be *larger* on
verified near-puzzles than on the rest of the negative class.

## Operating-point Table (recall 0.80 and 0.85, validation thresholds)

| Metric | recall 0.80 | recall 0.85 |
|---|---:|---:|
| Puzzle recall | | |
| Precision | | |
| Total FP | | |
| Near-puzzle FP | | |
| Near-puzzle FP rate | | |
| Far / random FP rate | | |
| Mean `raw_claim_logit` on accepted positives | | |
| Mean `softplus(reply_veto_logit)` on rejected near-puzzles | | |

The matched-recall report should compare each row to the paired i193 parent
baseline on the same split, seeds, and scale.

## Required Weak-slice Report

For each slice — `crtk_eval_bucket = equal`, `crtk_difficulty = hard`,
`crtk_difficulty = very_hard`, `crtk_tactic_motifs = promotion`,
`crtk_tactic_motifs = underpromotion`, `crtk_tactic_motifs = mate_in_1`, and
each `crtk_phase` bucket — at both recall `0.80` and `0.85`:

| Column | Meaning |
|---|---|
| `n` | Slice size |
| `puzzle_recall` | Recall preservation on the weak slice |
| `near_FP_rate` | Core rejection metric |
| `far_FP_rate` | Whether the model is becoming broadly conservative |
| `precision` | Practical acceptance quality |
| `accuracy@recall` | Continuity with the repo's audit style |
| `mean_max_forcedness_gap` | Mechanism visibility |
| `median_effective_candidate_count` | Reply / candidate-pool behaviour |
| `dominant_veto_head_share` | Which head contributed most to the rejection |

`dominant_veto_head_share` is derived from the per-sample diagnostics
(`defender_overload`, `king_escape_pressure`, `concentration_score` proxied by
`effective_candidate_count`, `reply_escape_mass`) and lets the report show
*why* a slice win happened, not just that it happened.

## Specialist Diagnostics

- Mechanism family: `king_path` (decision-time reply / king pressure)
- Packet profile: `near_puzzle_rejection_specialist`
- Mean `raw_claim_logit`:
- Mean `reply_veto_logit`:
- Mean `max_forcedness_gap`:
- Mean `top2_forcedness_gap`:
- Mean `effective_candidate_count`:
- Mean `defender_overload`:
- Mean `king_escape_pressure`:
- Mean `claim_mass`:
- Mean `reply_escape_mass`:
- Mean `mechanism_energy`:

## Ablation Sweep

Run each `model.ablation` setting on the same seeds / scale / split:

| Ablation | PR-AUC | near_FP @ 0.80 | near_FP @ 0.85 | Notes |
|---|---:|---:|---:|---|
| `none` | | | | full specialist |
| `trunk_only` | | | | parent floor |
| `no_forcedness_gap` | | | | falsifies forcedness story |
| `no_reply_envelope` | | | | falsifies reply story |
| `no_overload_head` | | | | falsifies overload story |
| `no_king_escape_head` | | | | falsifies `mate_in_1` story |
| `no_concentration_head` | | | | falsifies concentration story |

## Keep / Drop Decision

- [ ] Rejection-identity guard `final_logit <= raw_claim_logit` holds on the test set.
- [ ] `none` beats `trunk_only` on near-puzzle FP rate at both recall points.
- [ ] At least one chess-semantic ablation loses most of the near-FP gain (the responsible head is load-bearing).
- [ ] Aggregate PR-AUC remains within ~0.003 of the matched i193 parent baseline.
- [ ] No slice regression on `equal`, `hard`, `very_hard`, `promotion`, `underpromotion`, `mate_in_1` beyond noise.

If any box fails: drop the responsible head, or keep i193 as the canonical
parent and do not promote i256 over it.

## Conclusions

- What the specialist appears able to learn (vs i193):
- What the specialist appears unable to learn (vs i193):
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / wire up `L_gap_rank` / `L_veto` /
  scale to i018 parent):
