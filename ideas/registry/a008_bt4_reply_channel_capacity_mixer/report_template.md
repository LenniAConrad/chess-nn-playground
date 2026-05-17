# Report Template

## Run

- Result path:
- Config: `ideas/registry/a008_bt4_reply_channel_capacity_mixer/config.yaml`
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Do not stop at an aggregate
confusion matrix. Every promoted idea must report:

- aggregate metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
  `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives for fine
  label `2`;
- confidence/calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and motifs;
- a short conclusion describing what the model appears able and unable to
  learn.

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration (ECE / Brier):

## Architecture-Specific Diagnostics

- Mechanism family: `bt4_mixer` with `mixer=reply_channel_capacity`
- Tower: shared BT4 residual stack from `bt4_primitive_mixer` (stem conv,
  N residual+SE blocks, value head)
- Per-block RCC solver:
  `P_{kr} = softmax_r(L_{kr} / tau)`,
  `marginal_r = sum_k q_k P_{kr}`,
  `per_row_k = sum_r P_{kr} (log P_{kr} - log marginal_r)`,
  `q_new = softmax_k(per_row)` unrolled for `iters` steps,
  with `capacity_nats = sum_k q_k * per_row_k`,
  `H(reply | candidate) = -sum_k q_k sum_r P_{kr} log P_{kr}`,
  `H(reply) = -sum_r marginal_r log marginal_r`,
  `capacity_gap = H(reply) - H(reply | candidate)`.
- Capacity distributions (`capacity_nats`, `capacity_bits`) on:
  - true puzzles (fine label `2`)
  - near-puzzles (fine label `1`)
  - clear non-puzzles (fine label `0`)
- Capacity-gap distribution `H(reply) - H(reply | candidate)` on the same
  three buckets (collapsed-channel signal vs sharp-channel signal).
- Per-candidate `q*_k` distribution on tactical vs quiet slices
  (`crtk_tactic_motifs` non-empty vs empty); flag any collapse onto a
  single candidate as evidence of low-rank transition tables.
- Transition-table `P` row/column rank and reply-marginal `r` Shannon
  entropy to verify the table is not collapsing to a degenerate all-rows-
  equal regime (sanity check for the `duplicate_rows` ablation).

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
  - `near_puzzle` false-positive rate at recall 0.80
  - high `crtk_difficulty` buckets where candidate-controlled reply
    distributions distinguish forcing tactics from decoys
  - `crtk_tactic_motifs` containing forcing motifs where the strongest
    candidate's reply distribution is sharply distinct from decoys
- Slices where this idea is expected to fail:
  - `crtk_eval_bucket = equal` quiet positions (no robust channel to
    measure capacity over)
  - stalemate-vs-checkmate discrimination (rule-exact, not surrogate)
- Ablation that should erase the slice-level gain: `row_shuffle_channel`
  must lose >= 50% of the near-puzzle FP rate lift; `duplicate_rows` and
  `entropy_only` must each lose >= 80% of the lift.
- Minimum useful slice-level improvement: -3 percentage points on the
  `near_puzzle` false-positive rate at recall 0.80 over the conv mixer
  baseline at matched budget.

## Baseline Comparison Table

| Mixer | aggregate PR AUC | near_puzzle FPR @ recall 0.80 | mean capacity_nats (true puzzle) | mean capacity_nats (near puzzle) | step time vs conv |
|---|---|---|---|---|---|
| `conv` (baseline) | | | n/a | n/a | 1.00x |
| `attention` (baseline) | | | n/a | n/a | |
| `reply_channel_capacity` (this idea) | | | | | |
| `reply_channel_capacity` + `row_shuffle_channel` | | | | | |
| `reply_channel_capacity` + `duplicate_rows` | | | | | |
| `reply_channel_capacity` + `entropy_only` | | | | | |
| `reply_channel_capacity` + `uniform_q_init_only` | | | | | |
| `reply_channel_capacity` + `low_tau` | | | | | |
| `reply_channel_capacity` + `high_tau` | | | | | |

## Keep / Drop Decision

- [ ] aggregate PR AUC delta vs `conv` baseline >= -0.005
- [ ] `near_puzzle` FPR @ recall 0.80 improved by >= 3 percentage points
- [ ] `row_shuffle_channel` loses >= 50% of the near-puzzle FPR lift
- [ ] `duplicate_rows` loses >= 80% of the near-puzzle FPR lift
- [ ] `entropy_only` loses >= 80% of the near-puzzle FPR lift
- [ ] `uniform_q_init_only` loses >= 30% of the lift
- [ ] Step time within 25% of the conv baseline

If any box fails: drop this mixer from the BT4 mixer family.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, `crtk_difficulty`, `crtk_phase`,
  `crtk_tactic_motifs`):
- Recommended next step (promote / drop / re-run with larger tower):
