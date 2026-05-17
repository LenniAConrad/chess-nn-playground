# Report Template

## Run

- Result path:
- Config: `ideas/registry/a006_bt4_pareto_antichain_frontier_mixer/config.yaml`
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

- Mechanism family: `bt4_mixer` with `mixer=pareto_antichain_frontier`
- Tower: shared BT4 residual stack from `bt4_primitive_mixer` (stem conv,
  N residual+SE blocks, value head)
- Per-block PAFR reducer:
  `p_{ij} = prod_c sigmoid((U_{ic} - U_{jc} - eps) / tau_dim)`,
  `log pi_j = sum_{i!=j} log(1 - p_{ij})`,
  `alpha = softmax((log pi + beta * mean_c U) / tau_set)`
- Frontier width (`sum_j pi_j`) and entropy (`-sum_j alpha_j log alpha_j`)
  on:
  - true puzzles (fine label `2`)
  - near-puzzles (fine label `1`)
  - clear non-puzzles (fine label `0`)
- Per-candidate alpha distribution on tactical vs quiet slices
  (`crtk_tactic_motifs` non-empty vs empty).
- Utility channel correlation matrix `corr(U_{:,:,c}, U_{:,:,c'})` to
  verify the channels are not collapsing to a 1-D order (sanity check
  for the `single_channel` ablation).

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
  - `near_puzzle` false-positive rate at recall 0.80
  - high `crtk_difficulty` buckets where partial-order signals matter
  - `crtk_tactic_motifs = mate_in_1` where the frontier is expected to
    concentrate on one candidate
- Slices where this idea is expected to fail:
  - `crtk_eval_bucket = equal` quiet positions (frontier is uninformative)
  - stalemate-vs-checkmate discrimination (rule-exact, not surrogate)
- Ablation that should erase the slice-level gain: `scalar_max`
  (collapse to total order) and `shuffle_channels` (permute
  channel-to-candidate binding) must each lose >= 50% of the near-puzzle
  FP rate lift.
- Minimum useful slice-level improvement: -3 percentage points on the
  `near_puzzle` false-positive rate at recall 0.80 over the conv mixer
  baseline at matched budget.

## Baseline Comparison Table

| Mixer | aggregate PR AUC | near_puzzle FPR @ recall 0.80 | frontier width (true puzzle) | frontier width (near puzzle) | step time vs conv |
|---|---|---|---|---|---|
| `conv` (baseline) | | | n/a | n/a | 1.00x |
| `attention` (baseline) | | | n/a | n/a | |
| `pareto_antichain_frontier` (this idea) | | | | | |
| `pareto_antichain_frontier` + `scalar_max` | | | | | |
| `pareto_antichain_frontier` + `shuffle_channels` | | | | | |
| `pareto_antichain_frontier` + `single_channel` | | | | | |
| `pareto_antichain_frontier` + `uniform_frontier` | | | | | |

## Keep / Drop Decision

- [ ] aggregate PR AUC delta vs `conv` baseline >= -0.005
- [ ] `near_puzzle` FPR @ recall 0.80 improved by >= 3 percentage points
- [ ] `scalar_max` loses >= 50% of the near-puzzle FPR lift
- [ ] `shuffle_channels` loses >= 50% of the near-puzzle FPR lift
- [ ] `single_channel` underperforms `none` by >= 30% of the lift
- [ ] Step time within 25% of the conv baseline

If any box fails: drop this mixer from the BT4 mixer family.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, `crtk_difficulty`, `crtk_phase`,
  `crtk_tactic_motifs`):
- Recommended next step (promote / drop / re-run with larger tower):
