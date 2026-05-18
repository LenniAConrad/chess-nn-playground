# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config is
paper-grade, CUDA-required, and matched to the i018 baseline so the
comparison is on:

- same train / val / test split (`crtk_sample_3class_unique_crtk_tags`)
- same encoding (`simple_18`)
- same seeds (42, 43, 44)
- same epochs / batch_size / lr / weight_decay / class_weighting / loss
- same threshold-selection rule and early-stopping policy

Differences vs `ideas/registry/i018_oriented_tactical_sheaf_laplacian/config.yaml`
and `ideas/registry/i249_oriented_tactical_sheaf_fast/config.yaml`:

- `model.name = pin_xray_overload_sheaf` (i252 builder).
- `training.monitor: pr_auc` is set explicitly so the trainer never
  falls back to F1, accuracy, or negative loss for paper-grade
  comparisons.
- Three new falsifier knobs:
  `scramble_relations`, `scramble_new_only`, `family_collapse`. All
  default to `false`; set one at a time per falsifier run.

If you change a trunk hyperparameter (`channels`, `hidden_dim`,
`depth`, `stalk_dim`, `dropout`, `use_batchnorm`), change it on i018
and i249 too and re-run all three -- the comparison is only honest when
the three nets share trunk geometry.

## Loss

`bce_with_logits` on the puzzle logit. No i252-specific auxiliary loss.
The mechanism-energy diagnostics from i018 are emitted unchanged, plus
five new pressure diagnostics:

- `xray_pressure`: mean density across both x-ray planes.
- `skewer_pressure`: mean density across both skewer planes.
- `discovered_pressure`: mean density across both discovered-attack
  planes.
- `pinned_defender_pressure`: mean density across both
  attacks-against-pieces-with-pinned-defender planes.
- `overload_pressure`: mean density across both attacks-on-overloaded
  -defender planes.

## Cost expectation

- Parameter count is about `+6k` over i018 at base scale. Wall-clock
  cost per batch grows slightly because the diffusion block now runs
  with `R = 22` planes (linear in `R`) and the template-bank clear
  matvec adds about `B * 64 * 2576 = 165k * B` extra mul-add per
  batch. On 8 GB GPUs the relation tensor at batch `256` is about
  `256 * 22 * 64 * 64 * 4 bytes = 92 MB` in FP32, manageable.
- At init the model is *not* numerically equivalent to i018 (different
  relation count). Use `family_collapse: true` or
  `scramble_new_only: true` for ablation comparisons aligned with the
  i018 falsifier protocol.

## Benchmark plan

3 seeds (42, 43, 44) x 3 scales (`base`, `scale_up:1.5`, `scale_xl:2`),
identical training hyperparameters to the i018 / i249 paper-grade runs
already in `results/paper_grade_top3/`. Report mean and standard
deviation of test PR-AUC, near-puzzle false positives at
validation-derived recall `0.80` and `0.85`, and the four target motif
slices (`pin`, `skewer`, `overload`, `discovered_attack`). Required
falsifier comparisons:

- F2 `scramble_relations: true` -- topology scramble.
- F3 `scramble_new_only: true` -- new-plane scramble only.
- F4 `family_collapse: true` -- collapse new planes.

## Reports

Standard idea report (see `report_template.md`). The slice analysis is
inherited from i018's reporting contract and must include
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. The new i252-specific
analysis layer is per-family pressure deltas (x-ray, skewer,
discovered, pinned-defender, overload) sliced by `crtk_tactic_motifs`.
