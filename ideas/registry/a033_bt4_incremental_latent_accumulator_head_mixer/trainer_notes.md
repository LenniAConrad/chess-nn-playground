# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The trainer is
the shared `train_from_config` path; the guard checks idea/config/
model identity (`idea_id`, `slug`, `device: nvidia`, registered
`model.name`, `implementation_status` in `{implemented, tested}`,
`implementation_kind: bespoke_model`) before any optimizer step.

## Matched-baseline contract

This idea is one of a sweep of `a###_bt4_*_mixer` ideas plus `conv`
and `attention` baselines, all sharing:

- the same train/val/test split
  (`data/splits/crtk_sample_3class_unique_crtk_tags/`)
- the same `simple_18` encoding
- the same seed (`42`)
- the same training budget (epochs, batch size, optimizer,
  scheduler, early-stopping, reliability tier)
- the same loss (`bce_with_logits`) and class-weighting strategy

The only variable that may differ across these runs is `model.mixer`.
Do not alter the optimizer protocol or data contract in this folder;
if a comparison requires a change, change it in the shared
`bt4_primitive_mixer` tower or in the mixer factory so every sibling
inherits it.

## Cost expectation

The `incremental_latent_accumulator_head` mixer computes, per BT4
block, two `Linear(C -> latent_dim)` projections of the per-square
channel features (one for the global stream, one for the king
stream), a board-wide sum over the 64 tokens to build
`h_global, h_king in R^{B x latent_dim}`, a `Conv2d(C -> 1)`
saliency map followed by a 64-way softmax and an
`einsum(anchor_w, king_anchor_table)` to compute the anchor row, a
broadcast of the two latents back to `(B, 64, latent_dim)`, and a
`phi` MLP `LayerNorm -> Linear -> GELU -> Linear` over `(2 *
latent_dim + C) -> C` for the per-square lift. Dominant per-block
cost is the two `Linear(C -> latent_dim)` projections (`O(64 * C *
latent_dim)` per sample) plus the `phi` MLP (`O(64 * (2 *
latent_dim + C) * C)` per sample); no `O(64^2)` token-pair matmul
is required. Expect the per-block FLOP count to sit at the same
big-O as the `conv` baseline (both are `O(64 * C^2)` to within a
constant), with a small constant-factor overhead from the saliency
conv and the broadcast-then-concat. The shared trainer logs
`train_samples_per_second` to `speed_summary.json`; if throughput
on matched hardware falls below ~40% of the `conv` baseline, drop
`model.num_blocks` or `model.channels` for the matched comparison
and re-run the conv baseline at the same shrunken capacity.

## Reports

Standard idea report. Required slices follow `report_template.md`
and the shared `ideas/docs/BENCHMARK_REPORTING.md` contract:
aggregate + fine-label diagnostic confusion matrix; per-
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families` slices;
`slice_report_val.md` and `slice_report_test.md`; per-slice false
positives for fine label 1 and false negatives for fine label 2;
highest-confidence wrong examples with FEN; calibration by slice.
