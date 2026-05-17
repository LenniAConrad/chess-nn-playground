# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The trainer is the
shared `train_from_config` path; the guard checks idea/config/model
identity (`idea_id`, `slug`, `device: nvidia`, registered `model.name`,
`implementation_status` in `{implemented, tested}`,
`implementation_kind: bespoke_model`) before any optimizer step.

## Matched-baseline contract

This idea is one of a sweep of `a###_bt4_*_mixer` ideas plus `conv` and
`attention` baselines, all sharing:

- the same train/val/test split
  (`data/splits/crtk_sample_3class_unique_crtk_tags/`)
- the same `simple_18` encoding
- the same seed (`42`)
- the same training budget (epochs, batch size, optimizer,
  scheduler, early-stopping, reliability tier)
- the same loss (`bce_with_logits`) and class-weighting strategy

The only variable that may differ across these runs is `model.mixer`.
Do not alter the optimizer protocol or data contract in this folder; if
a comparison requires a change, change it in the shared
`bt4_primitive_mixer` tower or in the mixer factory so every sibling
inherits it.

## Cost expectation

The DPA mixer is heavier than the `conv` baseline: each block builds
a per-edge bilinear message over the alignment mask (~840 ordered
aligned pairs on the 8x8 board) projected through `pair_dim` features,
in addition to a LayerNorm, the first-order projection, the
`delta_square_gate` MLP, and the fuse Linear. Memory per block scales
as `O(B * 64 * 64 * pair_dim)` for the intermediate `msg` tensor; with
`pair_dim=32` and `batch_size=256` this is roughly 33 MB per block at
fp32. The shared trainer logs `train_samples_per_second` to
`speed_summary.json`; if throughput falls below ~50% of the `conv`
baseline on matched hardware, investigate before relying on matched-
budget comparisons. The capacity-matched A4 ablation should hold the
parameter delta in scope when reporting.

## Reports

Standard idea report. Required slices follow `report_template.md` and
the shared `ideas/docs/BENCHMARK_REPORTING.md` contract: aggregate +
fine-label diagnostic confusion matrix; per-`crtk_difficulty`,
`crtk_phase`, `crtk_eval_bucket`, `crtk_tactic_motifs`, and
`crtk_tag_families` slices; `slice_report_val.md` and
`slice_report_test.md`; per-slice false positives for fine label 1 and
false negatives for fine label 2; highest-confidence wrong examples
with FEN; calibration by slice.
