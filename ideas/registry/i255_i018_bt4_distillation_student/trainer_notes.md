# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The default config
trains the BT4 distillation student on plain supervised BCE - this is
the honest baseline the research markdown's ablation ladder starts
from.

Differences vs `ideas/registry/i018_oriented_tactical_sheaf_laplacian/config.yaml`:

- `model.name = i018_bt4_distillation_student` (i255 builder).
- `model` is the BT4-shaped student trunk, not the i018 sheaf trunk.
  No `sheaf_layers`, no `stalk_dim`; instead `channels`, `num_blocks`,
  `value_channels`, `value_hidden`, `se_channels`, `canonicalize`,
  `diagnostic_dim`, `summary_plane_dim`, `readout_dim`.
- `training.learning_rate = 0.0008` (slightly higher than i018's 0.0007;
  the BT4-class backbone tolerates a hotter LR; matches the BT4
  defaults the repo's paper-grade comparison uses).
- `training.batch_size = 192` (matches the i253 paper-grade preset and
  the BT4 reference training).
- `training.loss = bce_with_logits` for the default trainable baseline.
  See "Distillation loss" below for the planned future loss name.

## Loss

Default: `bce_with_logits` on `outputs["logits"]`. The trainer ignores
the other output keys (`pooled_features`, `diagnostic_logits`,
`summary_plane_logits`, `readout_features`) under this loss, but the
heads still train through the supervised path because they share
parameters in the BT4 backbone.

Planned distillation loss (NOT bundled in this packet):
`i018_bt4_distill`, implementing the research-markdown objective:

```
L = lambda_sup  * BCEWithLogits(z_s, y)
  + lambda_kd   * T_t^2 * KL_Bern(p_t || sigma(z_s / T_t))
  + lambda_diag * Huber(hat_d_s - hat_d_t)
  + lambda_plane* SmoothL1(P_s, P_t)
  + lambda_read * L1(W_s r_s - LayerNorm(r_t))
  + lambda_brier* (sigma(z_s) - y)^2
  + lambda_rank * pairwise_logistic_rank(z_s, sign(z_t^i - z_t^j))
```

When that loss lands, the only changes needed will be:

1. add `i018_bt4_distill` to the trainer's loss registry;
2. teach the data path to read cached teacher targets
   (`teacher_logit`, `teacher_diagnostics`, `teacher_summary_planes`,
   optionally `teacher_readout`);
3. flip `config.yaml`'s `training.loss` to `i018_bt4_distill`.

No model changes are needed - this packet's model already emits every
output key the loss needs.

## Optimizer schedule

Stay close to the BT4 paper-grade defaults:

- AdamW, `lr=0.0008`, `weight_decay=0.0001`.
- ReduceLROnPlateau (`factor=0.5`, `patience=2`, `min_lr=1e-5`).
- Gradient clip norm 1.0.
- Mixed precision on (`mixed_precision: true`, `allow_tf32: true`,
  `matmul_precision: high`).
- Early stopping on `pr_auc`, patience 5.
- `min_epochs=10`, `min_active_epochs=10`, `epochs=20`.

The research markdown's distillation schedule (warm start with BCE +
light diagnostic loss, then full distillation, then hard-negative
phase) is a *training-recipe* concern that lives in the future
`i018_bt4_distill` loss, not in the model. The student backbone
already trains stably under the BT4 default recipe.

## Decision rule

The research markdown's promotion gate is reproduced in
`report_template.md`:

| metric                              | base target | scale_up target |
|-------------------------------------|------------:|----------------:|
| PR-AUC                              | >= 0.875    | >= 0.880        |
| near-puzzle FP @ recall 0.80        | <= 0.16     | <= 0.155        |
| puzzle recall                       | >= 0.80     | >= 0.80         |
| batch-1 CPU latency                 | <= 1.2 ms   | <= 1.6 ms       |

If the `base` student clears the gate, ship `base`. If `base` stalls on
quality but `scale_up` still lands under the latency cap, ship
`scale_up`. Either way, the winning row must not collapse on the
equal / hard / very_hard / mate_in_1 / promotion / underpromotion
slices that the repo already flags as i018 stress points.

## Cost expectation

- `base` trunk param count: 453,159 (about 5x the BT4 baseline at the
  same channels because of the diagnostic / plane / readout heads, but
  still well inside BT4 deployment shape).
- `scale_up` trunk param count: ~1.18M.
- Trunk arithmetic is dense `Conv2d` on an 8x8 grid, so wall-clock per
  step should be comparable to the BT4 baseline at the same scale.
- The plane head is one 1x1 conv; the diagnostic head is a tiny MLP;
  the readout head is a single Linear. Inference cost stays BT4-class.

## Benchmark plan

Following the research markdown's ablation ladder:

| Tranche                       | Rows                                                             | Seeds | Runs |
|-------------------------------|------------------------------------------------------------------|------:|-----:|
| Supervised baseline           | `base`, `scale_up` x `simple_18`                                 |     3 |    6 |
| Plain calibrated logit KD     | + `lambda_kd > 0`                                                |     3 |    6 |
| + scalar diagnostics          | + `lambda_diag > 0`                                              |     3 |    6 |
| + 12-d relation density       | (same loss; diagnostic_dim already includes the 12 dims)         |     3 |    6 |
| + 8 summary planes            | + `lambda_plane > 0`                                             |     3 |    6 |
| + readout matching            | + `lambda_read > 0`, `model.readout_dim=64`                      |     3 |    6 |
| + near-puzzle emphasis        | + class-weighted BCE on `source_class==1`                        |     3 |    6 |
| canonicalization on/off       | `model.canonicalize=false`                                       |     3 |    2 |
| `simple_18` vs `lc0_bt4_112`  | `data.encoding=lc0_bt4_112`, `model.input_channels=112`          |     3 |    2 |
| **Total**                     |                                                                  |       | **46** |

Seeds 42 / 43 / 44 per cell. The tranches that require the
distillation loss (rows 2-6 and the near-puzzle emphasis row) cannot
be run until the `i018_bt4_distill` loss lands; the supervised
baseline, canonicalization ablation, and encoding ablation can be run
today.

## Reports

Standard idea report (see `report_template.md`). The slice analysis
must include `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. The promotion gate is
matched-recall near-puzzle FP rate; PR-AUC alone is not enough.
