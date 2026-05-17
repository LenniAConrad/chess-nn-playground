# Idea Report Template — a014 BT4 Primitive Mixer (legal_move_graph_delta)

## Run

- Result path:
- Config: `ideas/registry/a014_bt4_legal_move_graph_delta_mixer/config.yaml`
- Seeds:
- GPU:
- Training budget:
- Reporting standard: `ideas/docs/BENCHMARK_REPORTING.md`
- Validation slice report: `slice_report_val.md`
- Test slice report: `slice_report_test.md`

## Aggregate Metrics

- Accuracy:
- F1:
- ROC AUC:
- PR AUC:
- Calibration (ECE, MCE):
- Wall-clock per epoch versus `bt4_conv_mixer`:

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Do not stop at an aggregate
confusion matrix. This idea is part of the controlled
`a###_bt4_*_mixer` sweep, so every promoted result must report:

- aggregate metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
  `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives for
  fine label `2`;
- confidence / calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and
  motifs;
- a short conclusion describing what the model appears able and unable
  to learn.

## Sibling Comparison Table

| Mixer | Aggregate PR AUC | CRTK class-1 FP at matched recall | mate_in_1 PR AUC | wall-clock / epoch |
|---|---|---|---|---|
| `bt4_conv_mixer` | | | | |
| `bt4_attention_mixer` | | | | |
| `bt4_legal_move_graph_delta_mixer` | | | | |

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline
  (e.g. `crtk_tactic_motifs = fork` or `discovered_attack`, positions
  with high typed mobility on knights and sliders):
- Slices where this idea is expected to fail
  (e.g. `crtk_phase = endgame` with very low mobility, or pawn-storm
  positions where typed adjacency is sparse):
- Ablation that should erase the slice-level gain
  (`random_typed_edges`, `shared_weight`, or `no_normalization` in the
  primitive folder):
- Minimum useful slice-level improvement:

## Sweep-Level Keep / Drop

- [ ] Aggregate PR AUC over `bt4_conv_mixer` >= +0.005
- [ ] Lift over `bt4_attention_mixer` >= lift over `bt4_conv_mixer`
- [ ] CRTK class-1 matched-recall FP rate <= `bt4_conv_mixer`
- [ ] Wall-clock per epoch within 1.2x of `bt4_conv_mixer`
- [ ] Primitive `random_typed_edges` ablation loses the lift

If any box fails: drop `a014` from the architecture-study sweep.

## Conclusions

- What the model appears able to learn:
- What the model appears unable to learn:
- Highest-confidence wrong examples (FEN, `crtk_difficulty`,
  `crtk_phase`, motifs):
- Recommended next step (promote / drop / re-run with fused
  message-pass kernel):
