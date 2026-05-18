# Ablations

i255 is a distillation study, not a new trunk family. The ablation
ladder is intentionally short and ordered to give the question
"what part of the distillation actually earned its keep" a clean
attribution.

## Primary Ablation Ladder

| ID  | Configuration                                                       | Hypothesis |
|-----|---------------------------------------------------------------------|------------|
| A1  | supervised BT4 only (base, `simple_18`, BCE)                         | The real non-distilled baseline. |
| A2  | + calibrated logit KD                                                | How much does plain Hinton-style KD buy by itself? |
| A3  | + scalar diagnostics (6 mandatory)                                   | Do explanation-style scalar targets reduce the teacher gap? |
| A4  | + 12-d relation density                                              | Does typed tactical mass help beyond scalar diagnostics? (Same loss term as A3; the head's 18-d output already includes both.) |
| A5  | + 8 summary planes                                                   | Does spatial teacher structure help enough to justify the head? |
| A6  | + readout matching (`readout_dim=64`)                                | Is compact feature KD useful or unnecessary? |
| A7  | + near-puzzle emphasis (`fine_label`-aware up-weight)                | Does loss-time emphasis cut matched-recall FPs? |
| A8  | canonicalization on / off                                            | Is the teacher's mover orientation a cheap inductive bias? |
| A9  | base vs scale_up (`channels=96`, `num_blocks=6`)                     | Where is the latency-quality Pareto point? |
| A10 | `simple_18` vs `lc0_bt4_112`                                         | Is benchmark comparability worth the input width overhead? |

A1, A8, A9, and A10 are runnable today on plain BCE. The rest depend on
the future `i018_bt4_distill` loss (see `trainer_notes.md`).

## Falsifier Rows

The student's auxiliary heads are training-time signals, so the
falsifiers are loss-side, not architecture-side:

| ID  | Configuration                            | Hypothesis |
|-----|------------------------------------------|------------|
| F1  | A5 with summary-plane targets shuffled across batch | Spatial supervision is load-bearing; should drop PR-AUC. |
| F2  | A3 with scalar diagnostics shuffled across batch    | Scalar supervision is load-bearing; should drop PR-AUC. |
| F3  | A6 with teacher readout replaced by random vectors   | Readout matching is load-bearing; should drop PR-AUC. |
| F4  | KD weight `lambda_kd` set to 0 in the full stack    | The headline distillation effect should collapse. |

Each falsifier must produce a clean PR-AUC drop versus the matched
intact row (at the matched seed) to count as a passed falsifier. A
falsifier that does NOT drop PR-AUC means that loss term was
decorative.

## Keep / Drop Rule

Keep i255 as a canonical distillation student if all of these hold:

- At least one row in the ladder lifts PR-AUC by `+0.005` or more vs
  the supervised BT4 baseline (A1), without slice regressions on
  hard / equal / endgame / mate_in_1 / promotion / underpromotion.
- The same row also cuts near-puzzle false positives at recall 0.80
  by `1%` or more vs the supervised BT4 baseline.
- Batch-1 CPU latency stays under the research-markdown gate
  (`<= 1.2 ms` for `base`, `<= 1.6 ms` for `scale_up`).
- F4 collapses the headline distillation effect (`lambda_kd=0` should
  recover the supervised baseline, plus or minus seed noise).

Drop the distillation conclusion (not the architecture) if:

- Distillation only helps in `scale_up`, not in `base`. The whole
  point is BT4-class latency; if only the wider student benefits, the
  deployment story is weaker.
- Distillation helps headline PR-AUC but worsens the equal / hard /
  very_hard / mate_in_1 / promotion / underpromotion slices. That is
  teacher-blind-spot inheritance; the response is slice-aware
  hard-negative weighting (A7), not a more exotic student.

Drop the architecture (`i255` itself) if A1 the supervised baseline
beats every distilled row, AND F1 / F2 / F3 do not produce clean
drops. That would mean the auxiliary heads add no information and
are pure overhead.
