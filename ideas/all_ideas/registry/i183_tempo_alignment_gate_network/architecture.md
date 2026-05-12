# Architecture

`Tempo-Alignment Gate Network` is a board-only classifier for the
`puzzle_binary` task. It accepts the repository's `simple_18`
current-board tensor with shape `(B, 18, 8, 8)` and returns one
puzzle logit per position. The packet thesis is that many
near-puzzles look tactical for the *wrong* side or require a tempo
that the side to move does not have. A plain CNN absorbs the static
tactical danger regardless of who is to move; this network instead
*gates* the static tactical danger by side-to-move tempo alignment.

## Mechanism

A compact convolutional trunk turns the 18-plane board into per-square
features `H ∈ R^{B×C×8×8}`. Three signals are extracted from the
trunk and the raw board:

- **Static danger field** `d(s) ∈ R^{8×8}`: a 1x1 `danger_head` over
  `H` followed by `relu` produces an undirected per-square static
  tactical danger signal -- the "tactical danger for somebody"
  signal.
- **Side-of-attacker field** `a(s) ∈ R^{8×8}`: a 1x1 `side_head`
  over `[H ; W ; B]` (the trunk concatenated with the white/black
  occupancy summaries summed over piece planes 0-5 and 6-11)
  produces a *signed* attacker-side logit. Positive means the local
  tactic is white's, negative means black's.
- **Tempo signal** `stm ∈ {-1, +1}` is read from plane 12
  (`white_to_move`). It feeds two paths: a per-square alignment
  modulator and a global tempo gate.

### Per-square alignment

The side-to-move alignment at each square is

```
alignment(s) = sigmoid(γ · stm · a(s) + β)
```

where `γ` (`align_scale`) and `β` (`align_bias`) are learned scalars.
This collapses to ≈ 1 when the local tactic's color matches the side
to move (we have the tempo) and to ≈ 0 when the position is
tactical-looking for the wrong side.

### Global tempo gate

A small MLP over `[mean_pooled_trunk ; white_material ;
black_material ; stm]` produces

```
tempo_gate = sigmoid(MLP(...))  ∈ (0, 1)
```

The gate captures whether the position even *has* enough tempo
content for the alignment signal to matter.

### Multiplicative gate

The gated per-square pressure is

```
g(s) = tempo_gate * alignment(s) * relu(d(s))
```

This is multiplicative on purpose: the score collapses if either the
tempo gate is small (no tempo) or the alignment is small (wrong
side). The pooled scalars are

- `own_pressure  = mean_s alignment(s) * relu(d(s))`
- `opp_pressure  = mean_s (1 - alignment(s)) * relu(d(s))`
- `alignment_gap = own_pressure - opp_pressure`
- `gated_pressure = mean_s g(s)`

### Side-to-move counterfactual stream

The trunk is re-evaluated on the same board with plane 12 flipped,
producing a parallel stream with its own gated pressure. The
`flip_contrast = own_gated_pressure - flipped_gated_pressure` is the
faithful tempo intervention the markdown calls for: a real
tempo-aligned puzzle should look much weaker after the flip; an
undirected tactical position should not.

### Puzzle head

A `LayerNorm → Linear → GELU → Dropout → Linear` head reads from
`[own_pressure, opp_pressure, alignment_gap, gated_pressure,
tempo_gate, mean_danger, max_danger, own_pressure_flipped,
gated_pressure_flipped, tempo_gate_flipped, flip_contrast]` and
returns a scalar puzzle logit. The trainer uses BCE with logits.

## Why a multiplicative tempo-alignment gate

The packet calls out that the same static tactical danger signal can
look right or wrong depending on tempo. A plain additive head can let
strong static danger override a missing alignment; a multiplicative
gate makes the conjunction explicit -- if either leg is small, the
signal collapses. The `additive_gate` ablation makes this concrete by
swapping `*` for `+` and observing that the additive form happily
fires for tactical-but-wrong-side positions.

## Output Contract

Forward returns a dict whose `"logits"` entry has shape `(B,)` for
the repository `puzzle_binary` BCE-with-logits trainer (or
`(B, num_classes)` when `num_classes > 1`, with the puzzle scalar
written into the last column of a zero-padded tensor):

- `logits`: `(B,)` puzzle logit (or `(B, num_classes)` when
  `num_classes > 1`).
- `prob`: `sigmoid(logits)` when `num_classes == 1`.
- `danger_field`: `(B, 8, 8)` undirected static tactical danger.
- `side_field`: `(B, 8, 8)` signed attacker-side logit.
- `alignment_field`: `(B, 8, 8)` per-square alignment.
- `own_pressure`, `opp_pressure`, `alignment_gap`,
  `gated_pressure`, `mean_danger`, `max_danger`: `(B,)`.
- `tempo_gate`: `(B,)` ∈ `(0, 1)`.
- `own_pressure_flipped`, `gated_pressure_flipped`,
  `tempo_gate_flipped`, `flip_contrast`: `(B,)` from the
  side-to-move-flipped counterfactual stream.
- `trunk_features`: `(B, channels, 8, 8)`.
- `ablation_active`, `uses_tempo_gate`, `uses_alignment`,
  `uses_multiplicative_gate`: `(B,)` flags exposing the running
  ablation.

## Ablations

The packet's required ablations are exposed through the model:

- `"none"` -- main multiplicative tempo-alignment gate.
- `"no_tempo_gate"` (`ablation`) -- `tempo_gate ≡ 1`, so the
  alignment can no longer be tuned by the side-to-move signal.
- `"no_alignment"` (`ablation`) -- `alignment(s) ≡ 0.5`, so the
  gate cannot tell which side the local danger is for. This is the
  "undirected tactical CNN" baseline implied by the markdown.
- `"additive_gate"` (`ablation`) -- replace
  `tempo_gate * alignment * relu(d)` with
  `tempo_gate + alignment + relu(d)`, killing the multiplicative
  conjunction.

## Implementation Binding

- Registered model name: `tempo_alignment_gate_network`
- Source implementation file: `src/chess_nn_playground/models/tempo_alignment_gate_network.py`
- Idea-local wrapper: `ideas/all_ideas/registry/i183_tempo_alignment_gate_network/model.py`
