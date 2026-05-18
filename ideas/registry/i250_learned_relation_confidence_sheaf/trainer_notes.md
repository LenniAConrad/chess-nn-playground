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

- `model.name = learned_relation_confidence_sheaf` (i250 builder).
- `training.monitor: pr_auc` is set explicitly so the trainer never falls
  back to F1, accuracy, or negative loss for paper-grade comparisons.
- New confidence-head knobs:
  `confidence_context_dim`, `confidence_hidden_dim`,
  `confidence_group_count`, `confidence_floor`,
  `normalize_confidence_within_relation`, `flat_confidence`.
  All defaults preserve the design called out in the source packet.

If you change a trunk hyperparameter (`channels`, `hidden_dim`, `depth`,
`stalk_dim`, `dropout`, `use_batchnorm`), change it on i018 too and re-run
both -- the confidence comparison is only honest when both nets share
trunk geometry.

## Loss

`bce_with_logits` on the puzzle logit. No i250-specific auxiliary loss.
The mechanism-energy diagnostics from i018 are emitted unchanged, plus
five new confidence-attribution scalars:

- `confidence_mean`: per-batch mean over relations of the normalized
  confidence's mean over active edges.
- `confidence_max`: per-batch max over relations of the per-edge
  normalized confidence.
- `confidence_std`: per-batch mean over relations of the normalized
  confidence's std over active edges.
- `pin_edge_confidence`: per-batch mean of the normalized confidence on
  the pin relation.
- `king_zone_confidence`: per-batch mean of the normalized confidence on
  the two king-zone relations.

## Cost expectation

- Parameter count is about `+7k` over i018 at base scale (about 98k vs
  91k). The runtime cost is dominated by the new per-edge feature
  evaluation; expect a moderate increase in wall-clock vs i018, not a
  multiplier.
- Test PR-AUC at zero init should match i018 within FP32 reduction noise
  on the same seed/scale, since `alpha_hat = 1` everywhere at init.
- After training, treat the model as a meaningful improvement over i018
  only under the decision rule in `math_thesis.md`.

## Benchmark plan

3 seeds (42, 43, 44) x 3 scales (`base`, `scale_up:1.5`, `scale_xl:2`),
identical training hyperparameters to the i018 paper-grade runs already
in `results/paper_grade_top3/`. Report mean and standard deviation of
test PR-AUC, near-puzzle false positives at validation-derived recall
`0.80` and `0.85`, and the five new confidence-attribution scalars
alongside the matching i018 entries.

## Reports

Standard idea report (see `report_template.md`). The slice analysis is
inherited from i018's reporting contract and must include
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. The new i250-specific
analysis layer is a top-k confident-edge breakdown per board, grouped by
relation, which is only meaningful for this idea.
