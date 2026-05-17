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

The `delta_event_legal_routing` mixer is more expensive per call than a
3x3 conv because it builds an event-conditioned delta routing field.
The shared trainer logs `train_samples_per_second` to
`speed_summary.json`; if throughput on matched hardware falls below
~40% of the `conv` baseline, drop `model.num_blocks` or `model.channels`
for the matched comparison and re-run the conv baseline at the same
shrunken capacity.

## Reports

Standard idea report. Required slices follow `report_template.md` and
the shared `ideas/docs/BENCHMARK_REPORTING.md` contract: aggregate +
fine-label diagnostic confusion matrix; per-`crtk_difficulty`,
`crtk_phase`, `crtk_eval_bucket`, `crtk_tactic_motifs`, and
`crtk_tag_families` slices; `slice_report_val.md` and
`slice_report_test.md`; per-slice false positives for fine label 1 and
false negatives for fine label 2; highest-confidence wrong examples
with FEN; calibration by slice.
