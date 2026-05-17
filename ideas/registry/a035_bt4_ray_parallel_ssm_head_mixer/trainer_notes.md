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

The `ray_parallel_ssm_head` mixer computes, per BT4 block: two 1x1
convs `A_proj`/`B_proj` with `C -> NUM_DIRECTIONS * C` (each
`O(NUM_DIRECTIONS * C^2 * 64)` per sample); an iterated selective
state update per direction
`h = A * shifted_h + B * x` repeated `max_ray_length = 7` times
across `NUM_DIRECTIONS = 8` directions, each step a shift + two
multiplies + an add (`O(8 * 7 * C * 64)` per sample); a per-direction
broadcast of the learned `C[d]` vector (`O(8 * C * 64)`); and a
final `Conv2d(C -> C, 1x1)` output projection (`O(C^2 * 64)` per
sample). Dominant per-block cost is the `A_proj`/`B_proj` linear
projections (`O(2 * 8 * 64 * C^2)` per sample), comparable to the
`conv` baseline's two 3x3 convs (`O(18 * 64 * C^2)` per sample), and
substantially cheaper than the dense `attention` baseline's
`O(64 * 64 * C)` token-pair matmul plus `O(64 * C^2)` projections.
The iterated scan adds a small constant factor from the seven
sequential shift / multiply / add steps, which is the only
sequential dependency in the mixer; everything else is parallel.
The shared trainer logs `train_samples_per_second` to
`speed_summary.json`; if throughput on matched hardware falls below
~40% of the `conv` baseline (the sequential scan loop kicks in when
`max_ray_length` is large), drop `model.num_blocks` or
`model.channels` for the matched comparison and re-run the conv
baseline at the same shrunken capacity.

## Reports

Standard idea report. Required slices follow `report_template.md`
and the shared `ideas/docs/BENCHMARK_REPORTING.md` contract:
aggregate + fine-label diagnostic confusion matrix; per-
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families` slices;
`slice_report_val.md` and `slice_report_test.md`; per-slice false
positives for fine label 1 and false negatives for fine label 2;
highest-confidence wrong examples with FEN; calibration by slice.
