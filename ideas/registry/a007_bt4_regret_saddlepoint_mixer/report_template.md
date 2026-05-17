# Report Template

## Run

- Result path:
- Config: `ideas/registry/a007_bt4_regret_saddlepoint_mixer/config.yaml`
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

- Mechanism family: `bt4_mixer` with `mixer=regret_saddlepoint`
- Tower: shared BT4 residual stack from `bt4_primitive_mixer` (stem conv,
  N residual+SE blocks, value head)
- Per-block RSP solver:
  `p_new = softmax(A q / tau_p)`,
  `q_new = softmax(-p_new^T A / tau_q)`,
  `(p, q) <- (1 - damp)(p, q) + damp(p_new, q_new)` unrolled for `iters` steps,
  with `value = p^T A q`,
  `attacker_regret = max_i (A q)_i - value`,
  `defender_regret = value - min_j (p^T A)_j`,
  `exploitability = attacker_regret + defender_regret`.
- Saddle `value` and `exploitability` distributions on:
  - true puzzles (fine label `2`)
  - near-puzzles (fine label `1`)
  - clear non-puzzles (fine label `0`)
- Attacker entropy `H(p)` and defender entropy `H(q)` on the same three
  buckets (collapse-to-pure-strategy signal vs uniform-mixture signal).
- Per-candidate `p_k` distribution on tactical vs quiet slices
  (`crtk_tactic_motifs` non-empty vs empty).
- Payoff table `A` Frobenius-norm and row/column rank diagnostics to
  verify the table is not collapsing to a uniform constant (sanity check
  for the `uniform_payoff` ablation).

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
  - `near_puzzle` false-positive rate at recall 0.80
  - high `crtk_difficulty` buckets where saddle-game discrimination matters
  - `crtk_tactic_motifs` containing forcing motifs where the defender's
    best response is decisive
- Slices where this idea is expected to fail:
  - `crtk_eval_bucket = equal` quiet positions (no robust saddle to find)
  - stalemate-vs-checkmate discrimination (rule-exact, not surrogate)
- Ablation that should erase the slice-level gain: `row_shuffle_payoff`
  and `col_shuffle_payoff` must each lose >= 50% of the near-puzzle FP
  rate lift; `uniform_payoff` must lose >= 80% of the lift.
- Minimum useful slice-level improvement: -3 percentage points on the
  `near_puzzle` false-positive rate at recall 0.80 over the conv mixer
  baseline at matched budget.

## Baseline Comparison Table

| Mixer | aggregate PR AUC | near_puzzle FPR @ recall 0.80 | mean exploitability (true puzzle) | mean exploitability (near puzzle) | step time vs conv |
|---|---|---|---|---|---|
| `conv` (baseline) | | | n/a | n/a | 1.00x |
| `attention` (baseline) | | | n/a | n/a | |
| `regret_saddlepoint` (this idea) | | | | | |
| `regret_saddlepoint` + `row_shuffle_payoff` | | | | | |
| `regret_saddlepoint` + `col_shuffle_payoff` | | | | | |
| `regret_saddlepoint` + `uniform_payoff` | | | | | |
| `regret_saddlepoint` + `pure_max_min` | | | | | |
| `regret_saddlepoint` + `single_iter` | | | | | |

## Keep / Drop Decision

- [ ] aggregate PR AUC delta vs `conv` baseline >= -0.005
- [ ] `near_puzzle` FPR @ recall 0.80 improved by >= 3 percentage points
- [ ] `row_shuffle_payoff` loses >= 50% of the near-puzzle FPR lift
- [ ] `col_shuffle_payoff` loses >= 50% of the near-puzzle FPR lift
- [ ] `uniform_payoff` loses >= 80% of the near-puzzle FPR lift
- [ ] `pure_max_min` or `single_iter` loses >= 30% of the lift
- [ ] Step time within 25% of the conv baseline

If any box fails: drop this mixer from the BT4 mixer family.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, `crtk_difficulty`, `crtk_phase`,
  `crtk_tactic_motifs`):
- Recommended next step (promote / drop / re-run with larger tower):
