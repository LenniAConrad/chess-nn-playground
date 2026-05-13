# Idea Report Template — Promotion-Aware Head (i246, PFCT)

- Extra report sections:
  - `promotion_pawn_count` distribution across val / test splits.
  - Per-slice PR AUC on `crtk_tactic_motifs ∈ {promotion, underpromotion}`.
  - Mean `primitive_gate` conditional on `promotion_has_pawn`
    (must be near 0 when `has_pawn == 0`).
  - Mean `promotion_attention_entropy` conditional on the dominant type.
  - Top-N highest-confidence wrong examples that have at least one
    near-promotion pawn, with FEN, difficulty, phase, and motif tags.

- Required comparisons:
  - PR AUC on `promotion` and `underpromotion` slices: `none`
    ablation vs `copy_baseline_fanout` matched ablation vs i193 baseline.
  - Aggregate val / test PR AUC vs i193 baseline.
  - Per-slice PR AUC on all standard CRTK slices to verify the primitive
    does not regress non-target slices by more than 0.01.
  - Parameter / FLOP / wall-clock cost vs i193 baseline.

- Known blockers:
  - No precomputed promotion-counterfactual cache exists yet — the trunk
    runs `K * 4` extra times per sample inside forward. If wall-clock
    becomes a problem at larger scale, see `implementation_notes.md` for
    the dynamic-batching mitigation.

## Required Benchmark Reporting

Follow `ideas/docs/BENCHMARK_REPORTING.md`. Every promoted idea must
require:

- aggregate metrics plus the fine-label diagnostic matrix;
- `slice_report_val.md` and `slice_report_test.md`;
- performance by `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
  `crtk_tactic_motifs`, and `crtk_tag_families`;
- per-slice false positives for fine label `1` and false negatives for
  fine label `2`;
- confidence/calibration by slice;
- highest-confidence wrong examples with FEN, difficulty, phase, and
  motifs;
- a short conclusion describing what the model appears able and unable
  to learn.

## Idea-Specific Slice Hypotheses

- Target slices where this idea should beat the strongest baseline:
  - `crtk_tactic_motifs = promotion` PR AUC: target ≥ 0.720
    (vs i193 ≈ 0.652, +0.068 absolute).
  - `crtk_tactic_motifs = underpromotion` PR AUC: target ≥ 0.720.

- Slices where this idea is expected to fail:
  - Any slice where `promotion_has_pawn` is zero (the gate is
    structurally clamped to 0 there, so PFCT collapses to i193).
  - Open-endgame quiet positions where the promotion fanout is a
    no-op because the trunk already encodes Q ≫ R, B, N.

- Ablation that should erase the slice-level gain:
  - `copy_baseline_fanout` (the spec's "A1" matched falsifier). If this
    ablation matches the full architecture on the promotion slice,
    PFCT's substitution adds nothing and the proposal is rejected.

- Minimum useful slice-level improvement:
  - `promotion` slice PR AUC ≥ 0.685 (a +0.033 absolute lift over i193
    is the minimum to claim the primitive does anything). Anything
    smaller is statistical noise on a slice this size and the primitive
    should be dropped.
