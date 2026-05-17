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

The `move_kernel_operator` mixer computes, per BT4 block: LayerNorm
over the per-square feature vector (`O(C)` per square); `T = 6`
independent per-type linear projections `W_t Z` (`O(T * C^2)` per
square, so `O(64 * T * C^2)` per sample); `T = 6` dense
`(B, 64, 64) x (B, 64, C)` batched matmuls
`Y_t = M_t (W_t Z)` (`O(T * 64^2 * C)` per sample, dominating the
per-block cost for typical `C = 64..128`); a sum over types (free
relative to the matmuls); and the final `out_proj`
(`O(C^2 * 64)` per sample) linear projection. Dominant per-block
cost is the `T = 6` dense matmuls (`O(6 * 64^2 * C)` per sample),
roughly `6x` more flops than the dense `attention` baseline's
`O(64^2 * C)` token-pair matmul plus `O(64 * C^2)` projections, and
substantially more expensive than the `conv` baseline's two 3x3
convs (`O(18 * 64 * C^2)` per sample). The `T` matmuls are
*parallel* across types (no sequential dependency), so the wall-
clock cost on a GPU with enough memory is closer to one large
batched-einsum than to `T` sequential matmuls. The shared trainer
logs `train_samples_per_second` to `speed_summary.json`; if
throughput on matched hardware falls below ~50% of the `conv`
baseline, drop `model.num_blocks` or `model.channels` for the
matched comparison and re-run the conv baseline at the same
shrunken capacity.

## Reports

Standard idea report. Required slices follow `report_template.md`
and the shared `ideas/docs/BENCHMARK_REPORTING.md` contract:
aggregate + fine-label diagnostic confusion matrix; per-
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families` slices;
`slice_report_val.md` and `slice_report_test.md`; per-slice false
positives for fine label 1 and false negatives for fine label 2;
highest-confidence wrong examples with FEN; calibration by slice.
