# Math Thesis

Source: `ideas/research/primitives/claude_05_terminal_state_detection.md`
(Terminal-State Detection Primitive, TSDP).

## Working thesis

For a position `x` with side-to-move `c`, let `M(x, c)` be the set of legal
moves. For each `m in M(x, c)`, exact chess rules give six boolean
indicators on `x.apply(m)` and `m` itself:

```
is_checkmate(x.apply(m))   in {0, 1}
is_stalemate(x.apply(m))   in {0, 1}
is_check(x.apply(m))       in {0, 1}    # excluding checkmate
is_promotion(m)            in {0, 1}
is_capture(m)              in {0, 1}
is_castling(m)             in {0, 1}
```

These are aggregated into an 11-dim feature vector:

```
mate_in_1, mate_count,
stalemate_threat, stalemate_count,
check_count, promotion_count, capture_count, castling_count,
total_legal_moves, forcing_density, mating_special_count
```

with `forcing_density = (check_count + capture_count) / max(total, 1)`
and `mating_special_count` the count of mating moves that are simultaneously
promotion or capture.

The architecture-level claim is additive:

```
final_logit(x) = i193_trunk(x) + gate(x) * delta(x)
```

where `gate(x), delta(x)` are small MLPs on the 11-d TSDP feature vector
concatenated with stop-gradient i193 trunk diagnostics
(`gate`, `gate_entropy`, `mechanism_energy`, `stream_disagreement`). The
rule features are stop-gradient by design: chess-rule indicators are
integer-valued and not differentiable. Gradient flow is entirely through
the trunk and the MLP head weights.

## Why this matters

The per-class puzzle_binary benchmark shows `crtk_tactic_motifs = mate_in_1`
PR AUC ~0.81 for all top models, versus aggregate ~0.876. Existing trunks
have to *learn* checkmate from raw piece-position features. TSDP feeds the
trunk the answer directly. The architecture-level question is whether the
trunk's implicit learning of mate detection is suboptimal enough that
adding an exact rule signal lifts the slice without dragging aggregate PR
AUC down.

## Falsifier

- Primitive-level: shuffling TSDP indicators across the dataset (ablation
  `shuffle_tsdp`) must lose the mate_in_1 slice lift versus the unablated
  run. If shuffled features match unablated, the rule features carry no
  signal in this trunk.
- Architecture-level: i248 must beat i193 on the mate_in_1 slice
  (>= +0.04 absolute PR AUC, per the TSDP falsifier) without regressing
  aggregate PR AUC by more than 0.005. The `equal` eval-bucket slice must
  not regress.

## Composition with other primitives

TSDP stacks orthogonally with the other four Opus 4.7 primitives:

- PFCT (promotion-fanout counterfactual): TSDP detects mate-by-promotion
  via `mating_special_count`; PFCT learns promotion fanout features.
- TDCD (tempo-defender cross-derivative): TSDP detects mate; TDCD detects
  tempo asymmetry. On a mate-in-1 position both align.
- DHPE (signed Hessian on piece pairs): different scale; orthogonal.
- CAIO (complex-amplitude interference): different representation;
  orthogonal.

The hybrid fusion plan (see `ideas/research/primitives/PRIMITIVE_TRAINING_TODO.md`)
keeps additive gated deltas independent so failed primitives can be
removed without touching the trunk.
