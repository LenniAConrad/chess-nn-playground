# Claude Code Handoff

This note preserves the implementation handoff for the five Claude Opus 4.7 primitive proposals in this folder.

## What Was Done

Designed and documented five proposed neural primitives that target documented benchmark failure modes. Each primitive has a markdown spec with its mathematical signature, falsifier, ablations, and a proposed `i###` architecture extension. None have been benchmarked at scout scale.

Repository verification on 2026-05-12: the folder contains five primitive markdown files and five prototype scripts. There is no dedicated TDCD prototype script in the current file set; the five scripts are `dhpe_prototype.py`, `dhpe_v2.py`, `pfct_prototype.py`, `caio_prototype.py`, and `tsdp_prototype.py`.

## Files

| File | Primitive | Proposed architecture extension |
|---|---|---|
| `01_signed_hessian_resonance.md` | DHPE | i245 Pair-Resonance Hessian Network |
| `02_tempo_defender_cross_derivative.md` | TDCD | i244 Tempo-Defender Cross-Derivative Network |
| `03_promotion_fanout_counterfactual.md` | PFCT | i246 Promotion-Aware Head |
| `04_complex_amplitude_interference.md` | CAIO | i247 Complex-Amplitude Chess Network |
| `05_terminal_state_detection.md` | TSDP | i248 Rule-Aware Tactical Head |
| `README.md` | batch map | quick map and slice-coverage table |
| `MANIFEST.md` | provenance | authorship/model attribution |
| `prototypes/` | reference code | toy math-signature checks, not trainer imports |

Cross-primitive synthesis:

- [codex_primitive_stacking_strategy.md](../architecture_bridges/codex_primitive_stacking_strategy.md): test order, hybrid architecture sketch, drop rules, and promotion rule.
- [PRIMITIVE_TRAINING_TODO.md](../../PRIMITIVE_TRAINING_TODO.md): operational plan for code promotion, tests, training, ablations, and hybrid stacking.

## Recommended Implementation Order

1. TSDP first: about 1.05x cost, precomputable in the data loader, targets the `mate_in_1` slice. If the shuffled-indicator ablation matches, drop it and move on.
2. PFCT: gated, about 1.4x cost, zero on most positions. Targets the promotion slice, currently the largest documented gap.
3. TDCD: about 8x cost. Targets the universal `equal` bucket failure.
4. DHPE: about 10x cost. Tests sub-additivity vs super-additivity of piece pairs through a signed Hessian.
5. CAIO: speculative complex-amplitude bet, about 1.5x cost. Run last.

Do not merge these into a hybrid before each primitive has been individually validated against its own ablations and pass/fail thresholds.

## Verify Before Promotion

- Novelty: grep closest existing concepts before assigning an `i###` folder, especially `i189`, `i041`, `i025-i027`, `i058`, `i066`, `i127`, and `i141`.
- CAIO autograd: complex tensor backward works in modern PyTorch, but test on the real trunk before relying on it with `torch.compile`.
- TSDP performance: per-position `python-chess` calls should be precomputed into the data loader or parquet split instead of called every training step.
- Cost estimates assume the i193 exchange/king dual-stream trunk as the shared trunk.

## Implementation Hints

The `i###_*` registry expects:

```text
idea.yaml
math_thesis.md
architecture.md
implementation_notes.md
trainer_notes.md
ablations.md
model.py
train.py
config.yaml
report_template.md
runs/
```

The primitive markdowns map roughly to `math_thesis.md` plus `architecture.md`; the rest should be scaffolded from `ideas/all_ideas/registry/template/`.

Recommended production shape:

- Use i193 exchange/king dual-stream as the initial shared trunk.
- Treat TSDP, PFCT, TDCD, and DHPE as additive heads.
- Treat CAIO as either an additive head or a trunk candidate, depending on the prototype outcome.
- Use `implementation_kind: bespoke_model` once promoted, because each primitive requires distinct model code rather than a `ResearchPacketProbe` wrapper.
