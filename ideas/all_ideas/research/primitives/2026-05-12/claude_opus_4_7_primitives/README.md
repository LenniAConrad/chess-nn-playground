# Claude Opus 4.7 Primitives — 2026-05-12

Five neural-network primitives invented by Claude (Opus 4.7, via the
`chess-nn-playground` Claude Code session) on 2026-05-12, with explicit
benchmark-failure-mode motivation and primitive-level falsification plans.

These are **not** external research-packet imports (those live in
[external_imports/](../external_imports/)).
These are Claude-authored designs produced after reading the 20-file external
batch, the i### registry (243 entries), and the data/audit reports
([reports/audits/per_class_benchmark.md](../../../../../reports/audits/per_class_benchmark.md),
[reports/audits/matched_recall_fp_report.md](../../../../../reports/audits/matched_recall_fp_report.md),
[docs/research_audit_2026-05-09.md](../../../../../docs/research_audit_2026-05-09.md)).

## Status

All five are **proposed only**. No scout-scale falsification has been run.
The `prototypes/` directory contains local toy prototype scripts for DHPE,
PFCT, CAIO, and TSDP. Those scripts validate math signatures on
hand-crafted cases; they are not scout-scale benchmark runs.

The implementation handoff is preserved in [HANDOFF.md](HANDOFF.md), including
promotion checks and the recommended test order.

## Quick map

| # | Slug | Targets failure mode | Counterfactual axis | Architecture extension |
|---|---|---|---|---|
| 01 | `signed_hessian_resonance` (DHPE) | near-puzzle vs puzzle discrimination | piece-pair existence (2nd-order) | i245 Pair-Resonance Hessian Network |
| 02 | `tempo_defender_cross_derivative` (TDCD) | universal `equal` eval-bucket | tempo × defender cross-derivative | i244 Tempo-Defender Cross-Derivative Network |
| 03 | `promotion_fanout_counterfactual` (PFCT) | promotion/underpromotion slice (largest gap) | piece-type substitution at promotion squares | i246 Promotion-Aware Head |
| 04 | `complex_amplitude_interference` (CAIO) | structural color/tempo coherence | chess-Z2 → U(1) phase representation | i247 Complex-Amplitude Chess Network |
| 05 | `terminal_state_detection` (TSDP) | mate_in_1 slice + stalemate-trap avoidance | rule-exact terminal classification of legal-move successors | i248 Rule-Aware Tactical Head |

The five primitives are **disjoint** in mechanism and target **non-overlapping**
benchmark failure modes — they stack orthogonally in an architecture. See
[codex_primitive_stacking_strategy.md](../architecture_bridges/codex_primitive_stacking_strategy.md)
for how a hybrid architecture would combine them.

## Files

- `01_signed_hessian_resonance.md` — DHPE primitive
- `02_tempo_defender_cross_derivative.md` — TDCD primitive
- `03_promotion_fanout_counterfactual.md` — PFCT primitive
- `04_complex_amplitude_interference.md` — CAIO primitive
- `05_terminal_state_detection.md` — TSDP primitive
- `MANIFEST.md` — provenance and authorship details
- `HANDOFF.md` — Claude Code handoff, validation order, and implementation hints
- `prototypes/` — local toy prototype implementations and checks

## Empirical motivation summary

From the benchmark reports:

| Slice | Best PR AUC | Aggregate PR AUC | Gap | Primitive targeting it |
|---|---:|---:|---:|---|
| `crtk_tactic_motifs = promotion` | 0.667 | 0.876 | **0.209** | PFCT (03) |
| `crtk_tactic_motifs = underpromotion` | 0.667 | 0.876 | 0.209 | PFCT (03) |
| `crtk_eval_bucket = equal` | 0.817 | 0.876 | 0.059 | TDCD (02) |
| `crtk_difficulty = hard` | 0.792 | 0.876 | 0.084 | DHPE (01), TDCD (02) |
| `mate_in_1` | 0.817 | 0.876 | 0.059 | DHPE (01), TSDP (05) |
| Aggregate near-puzzle FP rate | 0.128 (i193) | – | – | DHPE (01) sign-disambiguation |

CAIO (04) is the unorthodox option — it doesn't target a specific slice but
provides a structurally novel feature representation that may generalise across
slices, especially where color-symmetric / color-asymmetric structure carries
the tactical signal.

TSDP (05) is the conservative option — it uses exact `python-chess` rule
checks (rather than learned approximations) to provide rule-exact terminal
state features (mate-in-1, stalemate threat, forcing-move counts). Targets
the `mate_in_1` slice directly with negligible compute overhead.

## Provenance

Author: Claude (Opus 4.7) via `chess-nn-playground` interactive session.
Date: 2026-05-12.
External inputs read: the 20-file primitive-research batch under
[external_imports/](../external_imports/), the i### registry,
[docs/research_audit_2026-05-09.md](../../../../../docs/research_audit_2026-05-09.md),
and [reports/audits/](../../../../../reports/audits/).
