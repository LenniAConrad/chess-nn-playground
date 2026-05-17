# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config is paper-grade,
CUDA-required, and matched to the i018 baseline so the comparison is on:

- same train / val / test split (`crtk_sample_3class_unique_crtk_tags`)
- same encoding (`simple_18`)
- same seeds (42, 43, 44)
- same epochs / batch_size / lr / weight_decay / class_weighting / loss
- same threshold-selection rule and early-stopping policy

Differences vs `ideas/registry/i018_oriented_tactical_sheaf_laplacian/config.yaml`:

- `model.name = oriented_tactical_sheaf_fast` (i249 builder).
- Execution-only knobs: `compile_model`, `compile_mode`, `return_diagnostics`,
  and `inference_autocast_dtype`. The first three preserve logits; FP16 eval
  autocast is the fastest local path and introduces small floating-point drift.

If you change a trunk hyperparameter (`channels`, `hidden_dim`, `depth`,
`stalk_dim`, `dropout`, `use_batchnorm`), change it on i018 too and re-run
both — the speed comparison is only honest when both nets share trunk
geometry.

## Loss

`bce_with_logits` on the puzzle logit. No i249-specific auxiliary loss.
The mechanism-energy diagnostics from i018 are emitted unchanged.

## Cost expectation

- Numerically equivalent to i018 (see `architecture.md` and `math_thesis.md`).
- Wall-clock should be **strictly faster** than i018 on the same GPU.
  Watch `speed_summary.json` (`train_samples_per_second`,
  `fit_elapsed_seconds`). If i249's `samples_per_second` is not above
  i018's at matched config, the variant is providing no benefit and should
  not be promoted over i018.
- If `torch.compile` mis-handles your PyTorch / CUDA combo, set
  `model.compile_model: false`. The algebraic block alone still delivers most
  of the speedup.
- Keep `model.return_diagnostics: true` for paper-grade reports. Set it to
  false only for serving/inference paths that consume logits and do not write
  diagnostic prediction columns.
- Use `config.yaml` for exact i018-equivalence audits
  (`inference_autocast_dtype: none`).
- Use `config_eval_fp16.yaml` for the lower-precision eval/serving comparison
  (`inference_autocast_dtype: float16`).

## Benchmark plan

3 seeds (42, 43, 44) x 3 scales (`base`, `scale_up:1.5`, `scale_xl:2`),
identical training hyperparameters to the i018 paper-grade runs already in
`results/paper_grade_top3/`. Report `samples_per_second` and
`fit_elapsed_seconds` next to test PR-AUC, side-by-side with the matching
i018 entries.

## Reports

Standard idea report (see `report_template.md`). The slice analysis is
inherited from i018's reporting contract and must include
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`, `crtk_tactic_motifs`,
and `crtk_tag_families`. No i249-specific slice hypotheses exist beyond
i018's; the speed comparison is the new column.
