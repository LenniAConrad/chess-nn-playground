# Report Template

## Run

- Result path:
- Config: `ideas/registry/i258_relation_masked_attention_i018/config.yaml`
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`
- Paired matched-budget i018 baseline path (same split, seeds, scale):
- Paired i242 reference (for attention-cost discussion only):

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration:

## Identity-Recovery Check

The graft is designed so that `force_gate: 0.0` should recover the
matched-budget i018 baseline closely (the readout `hidden_dim` is the
only deliberate difference).

- Empirical `max(|logits_force_gate_0 - logits_attention_disabled|)`
  over the test set (should be near zero to a few times 1e-6):
- Empirical `mean(attention_gate_mean)` on the test set:
- Empirical `mean(attention_delta_norm)` on the test set:
- Empirical `mean(attention_entropy)` on the test set:

## Operating-point Table (recall 0.80 and 0.85, validation thresholds)

| Metric | recall 0.80 | recall 0.85 |
|---|---:|---:|
| Puzzle recall | | |
| Precision | | |
| Total FP | | |
| Near-puzzle FP | | |
| Near-puzzle FP rate | | |
| Far / random FP rate | | |
| Mean `attention_gate_mean` on accepted positives | | |
| Mean `attention_king_share` on accepted positives | | |

The matched-recall report should compare each row to the paired
matched-budget i018 baseline on the same split, seeds, and scale.

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
| `mean_attention_gate_mean` | Graft fire rate on the slice |
| `mean_attention_king_share` | Graft's king-zone attention mass on the slice |

These let the report show *why* a slice win happened, not just that it
happened.

## Specialist Diagnostics

- Mechanism family: `sheaf`
- Packet profile: `relation_masked_attention_i018`
- Mean `mechanism_energy`:
- Mean `attention_entropy`:
- Mean `attention_king_share`:
- Mean `attention_gate_mean`:
- Mean `attention_delta_norm`:
- Mean `attention_neighbor_count`:
- Mean `attention_relation_bias_norm`:
- Mean `sheaf_tension`:
- Mean `pin_pressure`:

## Ablation Sweep

Run each ablation on the same seeds / scale / split. See `ablations.md`
for details on each flag.

| Ablation | Aggregate PR-AUC | hard PR-AUC | equal PR-AUC | promotion PR-AUC | mate PR-AUC | near_FP @ 0.80 | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| A0 `attention_disabled`  | | | | | | | matched-budget i018 floor |
| A1 `global` neighborhood | | | | | | | falsifies chess constraint |
| A2 `scramble_relations`  | | | | | | | falsifies typed relation masks |
| A3 `relation` (default)  | | | | | | | primary i258 design |
| A4 `king_zone`           | | | | | | | tactical specialist |
| A5 `candidate`           | | | | | | | move-targeted reweighting |
| A6 `force_gate=0`        | | | | | | | should equal A0 |

## Keep / Drop Decision

- [ ] Identity check: A6 reproduces A0 closely (`max |logits_A6 - logits_A0|` near zero).
- [ ] A3 beats A0 on aggregate test PR-AUC by `>= 0.003` over three seeds,
      or matched-recall near-puzzle FP at recall `0.80` / `0.85` improves
      without aggregate regression.
- [ ] A3 beats A1 (`global`) by `>= 0.003`.
- [ ] A3 beats A2 (`scramble_relations: true`) by `>= 0.010`.
- [ ] Aggregate PR-AUC remains within `0.005` of the matched i018 parent
      baseline (no silent regression from the readout reduction).
- [ ] Train/inference slowdown stays within `15%` of the i018 baseline.

If any box fails: keep i018 / i249 as the canonical parent and do not
promote i258 over it. If only one neighborhood mode passes, keep only
that mode and drop the other neighborhood flags from the default config.

## Conclusions

- What the graft appears able to learn (vs i018):
- What the graft appears unable to learn (vs i018):
- Highest-confidence wrong examples (FEN, difficulty, phase, motifs):
- Recommended next step (promote / drop / wire up the deferred
  loss-side ablations / scale to a 2-block graft / swap encoder to
  i249-fast).
