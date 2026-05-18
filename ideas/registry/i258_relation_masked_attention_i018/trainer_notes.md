# Trainer Notes

Use the guarded idea `train.py` (`idea_train_cli`). The config is
paper-grade, CUDA-required, and matched to the i018 baseline so the
comparison is honest:

- same train / val / test split (`crtk_sample_3class_unique_crtk_tags`)
- same encoding (`simple_18`)
- same `epochs` (20), `batch_size` (256), `class_weighting`,
  `loss` (BCE-with-logits), early-stopping policy
  (`min_epochs=10`, `min_active_epochs=10`, `patience=5`), and
  `lr_scheduler` (`reduce_on_plateau`, `factor=0.5`,
  `patience=2`, `min_lr=1e-5`) as the matched i018 paper-grade run
- seeds 42 / 43 / 44 for the reliable promotion-grade protocol

## Loss

`bce_with_logits` on the puzzle logit. The research packet's extended
loss (`+ lambda_gate * E[gate] + lambda_slice * L_slice + lambda_near *
L_near + lambda_kd * KL`) is intentionally deferred. The model already
exports per-sample gate and bounded delta values, so plugging in the
extension later is purely additive on the trainer side.

## Monitor metric

`monitor: pr_auc`. The PR-AUC reselection audit (see
`docs/research_audit_2026-05-09.md` style audits) found that monitoring
PR-AUC instead of loss tends to lift validation PR-AUC by ~`+0.005` on
average; if the claim metric is PR-AUC, the checkpoint metric must
match it.

## Ablation matrix

Each row is a one-flag config change to `model.relation_attention` (and
optionally `model.scramble_relations`). All rows reuse the same trainer,
sampler, loss, and seeds.

| ID | Flag change | What it tests |
|---|---|---|
| A0 | `relation_attention.enabled: false` | Pure i018 baseline (matched parameter budget). |
| A1 | `relation_attention.neighborhood: global` | Generic global attention falsifier. |
| A2 | `scramble_relations: true` | Random degree-preserving rewiring of the typed masks. |
| A3 | `relation_attention.neighborhood: relation` (default) | Primary i258 design. |
| A4 | `relation_attention.neighborhood: king_zone` | High-precision tactical / mate refinement. |
| A5 | `relation_attention.neighborhood: candidate` | Move-targeted reweighting (own-piece outer product). |
| A6 | `relation_attention.force_gate: 0.0` | Gate-disabled control (graft cannot fire). |

A0 is the matched-budget i018 baseline run from the same config. The
keep / drop rule from the research packet requires A3 > A1, A3 > A2,
and A3 within seed noise of A0 on aggregate PR-AUC; deviations are
documented as falsifier wins for the responsible flag.

## Sampling

The default config uses the shared sampler. No slice-weighted
near-puzzle curriculum is bundled; switching to the packet's
slice-weighted curriculum is a trainer-side change tracked outside
this idea promotion.

## Cost expectation

The default config keeps the parameter budget within ~40 parameters of
the matched-budget i018 baseline (see `architecture.md` and
`math_thesis.md`). The attention forward adds a single
gather-and-attention pass on a 64-square graph with `K = 8`. The
research packet's expectation is single-digit arithmetic overhead and
~5-12% wall-clock slowdown versus the i018 baseline at this scale.

## Smoke / CI

CPU smoke is sufficient for compile / registry / forward checks. The
local
`tests/test_i258_relation_masked_attention_i018.py` covers builder
registration, forward shape, gradient flow, identity-recovery under
zero gate, the four neighborhood modes, and the `scramble_relations`
falsifier path.

The reliability run still requires `device: nvidia` per the global
idea-config contract; the guarded trainer refuses to fall back to CPU
silently.

## Reports

Standard idea report (see `report_template.md`). The slice analysis
must include `crtk_difficulty`, `crtk_phase`, `crtk_eval_bucket`,
`crtk_tactic_motifs`, and `crtk_tag_families`. The packet specifies
that the architecture promotion criterion is aggregate PR-AUC
non-regression versus matched i018 plus a mechanism-direction win
versus the `global` and `scramble_relations` falsifiers; matched-recall
near-puzzle FP at recall `0.80` / `0.85` is a check, not the primary
scoreboard.

Per-sample diagnostics include the six attention extras
(`attention_entropy`, `attention_king_share`, `attention_gate_mean`,
`attention_delta_norm`, `attention_neighbor_count`,
`attention_relation_bias_norm`); the post-hoc reports can use them to
attribute slice wins to attention rather than to trunk drift.
