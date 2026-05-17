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

The `sparse_legal_graph_transition` mixer computes, per BT4 block:
an input `LayerNorm(C)` on the per-square channel features; three
`Linear(C -> d_edge)` projections producing `W_self X_i`,
`W_neighbor X_j`, and `W_interact (X_i (.) X_j)` over the explicit
`(B, 64, 64, C)` pair tensor (the `W_interact` projection dominates
at `O(64 * 64 * C * d_edge)` per sample, while `W_self` and
`W_neighbor` are each `O(64 * C * d_edge)` per sample); the
broadcast-add of the three terms with a `ReLU` and `LayerNorm(d_edge)`
applied per-edge (`O(64 * 64 * d_edge)` per sample for the
elementwise add + nonlinearity + per-edge norm); the
degree-normalised aggregation einsum `bij,bijd->bid` (`O(64 * 64 *
d_edge)` per sample, with the hard binary mask making the einsum
sparse-effective but dense-in-memory); and the `Linear(d_edge -> C)`
back-projection (`O(64 * C * d_edge)` per sample). With `d_edge =
C` the dominant per-block cost is the `W_interact` projection
`O(64 * 64 * C^2) = O(4096 C^2)` per sample, asymptotically the same
scale as dense attention's `O(64 * 64 * C)` pair matmul plus `O(64
* C^2)` projections but with a heavier per-edge MLP body. The pair
tensor is `O(B * 64 * 64 * C)` activation memory; at default sizes
(`B = 256`, `C = 64`) this is `~256 MiB` per block (FP32) before the
masked einsum. The shared trainer logs `train_samples_per_second`
to `speed_summary.json`; if throughput on matched hardware falls
below ~40% of the `conv` baseline (the dense pair-MLP is the
expected bottleneck), drop `model.num_blocks` or `model.channels`
for the matched comparison and re-run the conv baseline at the
same shrunken capacity. If the pair tensor OOMs at the matched
batch size on small-VRAM hardware, the matched comparison must
either drop batch size for all siblings or drop `model.channels`
across siblings to fit; do not silently break the matched-baseline
contract.

## Reports

Standard idea report. Required slices follow `report_template.md`
and the shared `ideas/docs/BENCHMARK_REPORTING.md` contract:
aggregate + fine-label diagnostic confusion matrix; per-
`crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families` slices;
`slice_report_val.md` and `slice_report_test.md`; per-slice false
positives for fine label 1 and false negatives for fine label 2;
highest-confidence wrong examples with FEN; calibration by slice.
