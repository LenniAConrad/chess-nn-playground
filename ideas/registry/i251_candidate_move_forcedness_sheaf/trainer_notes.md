# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config is
paper-grade, CUDA-required, and matched to the i018 baseline so the
comparison is on:

- same train / val / test split (`crtk_sample_3class_unique_crtk_tags`)
- same encoding (`simple_18`)
- same seeds (42, 43, 44)
- same epochs / batch_size / lr / weight_decay / class_weighting / loss
- same threshold-selection rule and early-stopping policy

Differences vs `ideas/registry/i018_oriented_tactical_sheaf_laplacian/config.yaml`:

- `model.name = candidate_move_forcedness_sheaf` (i251 builder).
- `training.monitor: pr_auc` is set explicitly so the trainer never
  falls back to F1, accuracy, or negative loss for paper-grade
  comparisons.
- New candidate-move branch knobs:
  `max_candidates`, `top_k`, `move_embed_dim`, `move_hidden_dim`,
  `delta_hidden_dim`, `gate_hidden_dim`, `softmax_temperature`,
  `flat_move_pool`, `disable_move_branch`.
  All defaults preserve the design called out in the source packet
  (`max_candidates=96`, `top_k=8`, `move_embed_dim=48`, etc.).

If you change a trunk hyperparameter (`channels`, `hidden_dim`,
`depth`, `stalk_dim`, `dropout`, `use_batchnorm`), change it on i018
too and re-run both -- the move-branch comparison is only honest when
both nets share trunk geometry.

## Loss

`bce_with_logits` on the puzzle logit. No i251-specific auxiliary loss.
The mechanism-energy diagnostics from i018 are emitted unchanged, plus
the candidate-move diagnostics:

- `candidate_base_logits`: i018-equivalent base logit before fusion.
- `candidate_delta_logits`: pre-gate delta head output.
- `candidate_gate`: post-sigmoid gate value.
- `candidate_entropy`: pool-weight entropy (lower = more forced).
- `candidate_top1_mass`: largest pool weight.
- `candidate_gap`: top1 - top2 raw score gap.
- `candidate_check_mass`, `candidate_capture_mass`,
  `candidate_pin_mass`, `candidate_king_zone_mass`,
  `candidate_promotion_mass`, `candidate_underpromotion_mass`:
  pool-weighted flag masses over the top-k moves.
- `candidate_overflow_count`: 1 if the candidate budget was saturated.
- `candidate_count`: number of valid candidates after enumeration.

## Cost expectation

- Parameter count is about `+25k` over i018 at base scale (about 116k
  vs 91k). The runtime cost is dominated by the per-move encoder and
  the pool; expect a moderate increase in wall-clock vs i018, not a
  multiplier. The trunk runs once per batch as in i018; the move
  branch is `O(B*K)` for `K = max_candidates`.
- Test PR-AUC at zero init must match i018 within FP noise on the same
  seed/scale, since `final_logit = base_logit + 0.5 * 0` at init.
- After training, treat the model as a meaningful improvement over
  i018 only under the decision rule in `math_thesis.md`.

## Benchmark plan

3 seeds (42, 43, 44) x 3 scales (`base`, `scale_up:1.5`, `scale_xl:2`),
identical training hyperparameters to the i018 paper-grade runs
already in `results/paper_grade_top3/`. Report mean and standard
deviation of test PR-AUC, near-puzzle false positives at
validation-derived recall `0.80` and `0.85`, and the candidate
diagnostics alongside the matching i018 entries.

## Reports

Standard idea report (see `report_template.md`). The slice analysis is
inherited from i018's reporting contract and must include
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. The new i251-specific
analysis layer is a top-k move breakdown per board, grouped by kind
(check, capture, promotion, pin-aligned, king-zone-entry), which is
only meaningful for this idea.
