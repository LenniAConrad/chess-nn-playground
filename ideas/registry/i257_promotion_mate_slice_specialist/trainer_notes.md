# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config is
paper-grade, CUDA-required, and matched to the canonical tagged split so
the comparison is honest against the i193 fast conv student parent and the
i249 sheaf parent:

- same train / val / test split (`crtk_sample_3class_unique_crtk_tags`)
- same encoding (`simple_18`)
- same `epochs`, `batch_size`, `class_weighting`, `loss`, early-stopping
  policy, and `lr_scheduler` as i193, i248, and i256
- seeds 42 / 43 / 44 for the reliable protocol

## Loss

`bce_with_logits` on the puzzle logit `final = base + sum_k gate_k *
delta_k`.

The research packet recommends an extended loss

```text
L = L_BCE
  + lambda_gate * sum_k E[gate_k]
  + lambda_kd   * T^2 KL( sigma(z_teacher / T) || sigma(z_student / T) )
  + lambda_near * L_near
  + lambda_slice * (L_prom_rank + L_mate_rank)
```

with a gate-sparsity penalty, a near-puzzle-FP margin / reweighting term,
and slice-restricted ranking losses on the main puzzle label. Those terms
require a trainer extension (pair-aware batches and a custom auxiliary
loss hook) that is intentionally not bundled with this architecture
promotion. The model already exports per-sample gate values and bounded
deltas for every branch, so plugging the extension in later is purely
additive on the trainer side.

## Sampling

The default config uses the shared sampler — *no* slice-weighted
near-puzzle curriculum. Switching to the packet's slice-weighted
curriculum (effective 2-4x on `promotion`, `underpromotion`, `mate_in_1`
positives; near-puzzle negatives at least half of all negatives) is also a
trainer-side change. Until that lands, validation matched-recall remains
the operating-point metric.

## Cost expectation

The default config
(channels=32, depth=2, hidden_dim=64, head_hidden_dim=48,
type_embed_dim=16, max_promotion_candidates=4) is intentionally lighter
than the i248 trunk so the first deployment target (C1) stays within the
research packet's expected ~10-20% GPU / ~15-30% CPU latency envelope
versus the i193 parent. If the matched-recall reliability run shows the
specialist is genuinely helpful, the i249-parent variant can be added by
swapping the conv encoder for an i249 trunk without touching the heads.

## Reports

Standard idea report (see `report_template.md`). The slice analysis must
include `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. The packet specifies that
the slice PR-AUC lifts on `promotion`, `underpromotion`, and `mate_in_1`
plus matched-recall near-puzzle FP non-regression form the primary
scoreboard. Validation-only threshold selection is therefore mandatory.

The model emits one logit, so the existing artifact pipeline (calibration,
confusion-matrix, slice reports) works without changes. The per-sample
specialist diagnostics (`base_logit`, `promotion_delta`, `mate_delta`, …,
`promotion_gate`, `mate_gate`, …) all land in the prediction parquets so
the post-hoc reports can attribute slice wins to the responsible branch.

## Smoke / CI

CPU smoke is sufficient for compile / registry / forward checks (no GPU is
required at scaffold time). Local
`tests/test_i257_promotion_mate_slice_specialist.py` covers builder
registration, forward shape, ablation safety, gradient flow, and the
bounded-delta identity `|final - base| <= sum_k Delta_k`.

The reliability run still requires `device: nvidia` per the global
idea-config contract; the guarded trainer will refuse to fall back to CPU
silently.
