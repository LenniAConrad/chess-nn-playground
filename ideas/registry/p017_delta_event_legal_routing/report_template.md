# Idea Report Template — Delta-Event Legal-Move Routing (p017)

- Extra report sections:
  - Distribution of ``primitive_active_count`` across val / test splits.
  - Mean ``primitive_gate`` conditional on the gate context (high vs
    low ``mechanism_energy``, high vs low ``stream_disagreement``).
  - Top-N highest-confidence wrong examples with FEN, difficulty,
    phase, and motif tags.

- Required comparisons:
  - Aggregate val / test PR AUC vs the i193 baseline.
  - Primitive-specific falsifier ablation PR AUC on the declared
    target slices.
  - Per-slice PR AUC on all standard CRTK slices to verify the
    primitive does not regress non-target slices by more than 0.01.
  - Parameter / FLOP / wall-clock cost vs i193 baseline.

- Known blockers:
  - No CUDA-fused make/unmake delta kernel exists yet; the O(|Δ|)
    inference property is documented but not benchmarked in the scout
    training harness. The matched-recall FP rate comparison is
    therefore the primary criterion at scout scale; wall-clock benefit
    follows once an engine wrapper is built.

## Required Benchmark Reporting

Follow ``ideas/docs/BENCHMARK_REPORTING.md``. Every promoted idea must
require aggregate metrics plus the fine-label diagnostic matrix,
``slice_report_val.md``, ``slice_report_test.md``, performance by
``crtk_difficulty``, ``crtk_phase``, ``crtk_eval_bucket``,
``crtk_tactic_motifs``, and ``crtk_tag_families``, per-slice false
positives for fine label ``1``, and a short keep/drop conclusion.

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
  positions where the primitive's structural inductive bias is most
  load-bearing. See ``math_thesis.md`` for the per-primitive thesis.

- Slices where this idea is expected to fail:
  - Positions where the primitive's feature set is empty (the gate is
    structurally clamped towards 0 on those samples).
  - Slices where the i193 baseline already handles the structure
    (the primitive is then a near-no-op).

- Ablation that should erase the slice-level gain: the primary
  falsifier listed in ``ablations.md`` for this primitive.
