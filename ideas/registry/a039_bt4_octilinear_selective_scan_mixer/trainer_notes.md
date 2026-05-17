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

The `octilinear_selective_scan` mixer computes, per BT4 block: an
input `LayerNorm(C)` on the per-square channel features; eight pairs
of `Linear(C -> C)` projections producing the per-(square, direction,
channel) `A_k = sigmoid(A_proj[k](x))` selectivity gate and the
`B_k = B_proj[k](x)` injection coefficient (each pair contributes
`O(C^2 * 64)` per sample, total `O(2 * 8 * C^2 * 64)` per sample);
the per-direction sequential scan
`h_t = sigmoid(A_k(x_t)) * h_{t-1} + B_k(x_t) * x_t` over the eight
scan-path tables (cardinal directions: 8 paths of length 8;
diagonal directions: up to 15 paths of variable length 1..8), each
step a per-channel multiply-accumulate (`O(8 * 64 * C)` per sample
across the 8 directions, with the scan body running across all paths
within a direction in parallel via the path-table indexing); and a
fuse stage `LayerNorm(8*C) -> Linear(8*C -> C) -> GELU` over the
concatenated 8-direction outputs (`O(8 * C^2 * 64)` per sample for
the linear). Dominant per-block cost is the per-direction A / B
projections (`O(NUM_DIRECTIONS * C^2)` per token), comparable to the
`conv` baseline's two 3x3 convs (`O(18 * C^2)` per token) at small
`C` and substantially cheaper than the dense `attention` baseline's
`O(64 * 64 * C)` token-pair matmul plus `O(64 * C^2)` projections.
The sequential scan loop in Python is the only sequential dependency
in the mixer; everything else is parallel over (batch, square,
channel). The shared trainer logs `train_samples_per_second` to
`speed_summary.json`; if throughput on matched hardware falls below
~40% of the `conv` baseline (the per-direction Python loop kicks in
with the eight outer-iterated direction passes), drop
`model.num_blocks` or `model.channels` for the matched comparison
and re-run the conv baseline at the same shrunken capacity. The
source primitive notes that the asymptotic Mamba parallel-scan win
is not realised without a Triton kernel; the per-block budget should
budget for the sequential overhead until a fused scan kernel ships.

## Reports

Standard idea report. Required slices follow `report_template.md`
and the shared `ideas/docs/BENCHMARK_REPORTING.md` contract:
aggregate + fine-label diagnostic confusion matrix; per-
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families` slices;
`slice_report_val.md` and `slice_report_test.md`; per-slice false
positives for fine label 1 and false negatives for fine label 2;
highest-confidence wrong examples with FEN; calibration by slice.
