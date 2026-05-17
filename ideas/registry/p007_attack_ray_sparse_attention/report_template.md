# Report Template — p007 Attack-Ray Sparse Attention

## Run

- Result path:
- Config: `ideas/registry/p007_attack_ray_sparse_attention/config.yaml`
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Every promoted idea must
require aggregate metrics plus the fine-label diagnostic matrix,
`slice_report_val.md`, `slice_report_test.md`, performance by
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`, per-slice false
positives for fine label `1` and false negatives for fine label `2`,
confidence/calibration by slice, the highest-confidence wrong examples
(FEN, `crtk_difficulty`, `crtk_phase`, motifs), and a short
keep/drop conclusion.

## Aggregate Metrics

- Accuracy / F1 / ROC AUC / PR AUC / Calibration:

## Architecture-Specific Diagnostics

- Mechanism family: `ray_attention`
- Primitive: ARSA (attack-ray sparse attention over first-blocker keys)
- `primitive_gate` mean/max on sliding-piece tactical positives vs quiet
- `arsa_attention_entropy` distribution (high entropy = uniform; low =
  routing concentrated on a single ray neighbour)
- `arsa_self_weight` mean — high values would indicate the operator
  is mostly attending to its own square (failure mode)

## Slice Findings

- Target slice: `crtk_tactic_motifs in {pin, skewer, x_ray, discovered_attack}`
- Watch slice: aggregate FP rate at matched recall
- Watch slice: `crtk_eval_bucket = equal` — must not regress
- Performance must be broken out by `crtk_difficulty` and `crtk_phase`
  buckets so we can tell whether the lift is concentrated on the easy
  end of the puzzle distribution or on a particular game-phase.

## Ablation Comparison Table

| Ablation | target slice PR AUC | aggregate PR AUC | gate mean on positives | self-edge weight mean |
|---|---|---|---|---|
| `none` | | | | |
| `random_keys` | | | | |
| `uniform_attention` | | | | |
| `no_blocker_mask` | | | | |
| `zero_delta` | | | | |
| `disable_gate` | | | | |
| `trunk_only` | | | | |

## Keep / Drop Decision

- [ ] Aggregate PR AUC delta from i193 >= -0.005
- [ ] target slice lift >= +0.04
- [ ] `random_keys` ablation loses >= 70% of the lift
- [ ] `uniform_attention` ablation does NOT match `none`
- [ ] Throughput drop versus i193 < 30%

If any box fails: drop p007.

## Conclusions

- What the operator appears able to learn:
- What it appears unable to learn:
- Highest-confidence wrong examples (FEN, `crtk_difficulty`,
  `crtk_phase`, motifs):
- Recommended next step:
